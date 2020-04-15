#

# span and head

import json
from msp.utils import zwarn, zlog

# =====
# expand from head word to span

class SpanExpander:
    pass

#
class SpanExpanderDep:
    def __init__(self, **kwargs):
        pass

    # build children and span
    def build_tree(self, heads):
        slen = len(heads)
        children = [[] for _ in range(slen)]
        spans = [None for _ in range(slen)]  # [start, end)
        for m, h in enumerate(heads):
            if m>0:
                children[h].append(m)
        # -----
        # recursively get span
        def _get_span(m):
            if spans[m] is not None:
                return spans[m]
            else:
                left, right = m, m+1
                for c in children[m]:
                    cleft, cright = _get_span(c)
                    left = min(left, cleft)
                    right = max(right, cright)
                new_span = (left, right)
                spans[m] = new_span
                return new_span
        # -----
        _get_span(0)
        assert all(z is not None for z in spans)
        return children, spans

    # #
    # def ruled_exclude(self, cur_pos, cur_label, child_pos, child_label):
    #     # if child_label == "nmod:poss":
    #     #     return True
    #     child_label0 = child_label.split(":")[0]
    #     # if cur_pos == "PROPN" and child_label0 == "det":
    #     #     return False
    #     # if child_label0 in {"fixed", "flat", "compound", "nummod", "amod"}:  # "nmod"
    #     if child_label0 in {"fixed", "flat", "compound", "nummod", "amod"}:  # "nmod"
    #         return False
    #     return True

    # todo(note): head_wid includes the ROOT offset
    def expand_span(self, head_wid, sent):
        words, uposes, heads, labels = sent.words.vals, sent.uposes.vals, sent.ud_heads.vals, sent.ud_labels.vals
        labels = [z.split(":")[0] for z in labels]
        # first get spans for all heads
        children, spans = self.build_tree(heads)
        # raise up through MWE
        while True:
            cur_children = sorted(children[head_wid])
            cur_pos, cur_label = uposes[head_wid], labels[head_wid]
            # if cur_label not in {"fixed", "flat", "compound"}:
            #     break
            break
            head_wid = heads[head_wid]
        # check all the children
        left_boundary, right_boundary = head_wid, head_wid+1
        left_children = sorted([z for z in cur_children if z<head_wid], reverse=True)
        right_children = sorted([z for z in cur_children if z>head_wid])
        for c in left_children:
            if labels[c] not in {"fixed", "flat", "compound", "nummod", "amod"}:
                break
            left_boundary = min(left_boundary, spans[c][0])
        for c in right_children:
            if labels[c] not in {"fixed", "flat", "compound", "nummod", "amod"}:
                break
            right_boundary = max(right_boundary, spans[c][1])
        # return (wid, wlen)
        return left_boundary, right_boundary-left_boundary

    def expand_span_check_reason(self, head_wid, sent, gold_left, gold_right):
        words, uposes, heads, labels = sent.words.vals, sent.uposes.vals, sent.ud_heads.vals, sent.ud_labels.vals
        labels = [z.split(":")[0] for z in labels]
        # first get spans for all heads
        children, spans = self.build_tree(heads)
        cur_children = sorted(children[head_wid])
        cur_pos, cur_label = uposes[head_wid], labels[head_wid]
        # check all the children
        left_boundary, right_boundary = head_wid, head_wid+1
        left_children = sorted([z for z in cur_children if z<head_wid], reverse=True)
        right_children = sorted([z for z in cur_children if z>head_wid])
        added_left, added_right = [], []
        for c in left_children:
            if labels[c] not in {"fixed", "flat", "compound", "nummod", "amod"}:
                break
            left_boundary = min(left_boundary, spans[c][0])
            added_left.append(c)
        for c in right_children:
            if labels[c] not in {"fixed", "flat", "compound", "nummod", "amod"}:
                break
            right_boundary = max(right_boundary, spans[c][1])
            added_right.append(c)
        # -----
        # how to make it correct
        # step 0: already correct
        if left_boundary == gold_left and right_boundary == gold_right:
            return "s0_noerr"
        # step 1: delete extra ones
        for fix_step, del_set in zip(["s1_del_mod", "s2_del_mwe"], [{"nummod", "amod"}, {"fixed", "flat", "compound"}]):
            # delete all of the corresponding children if needed
            while len(added_left)>0 and left_boundary < gold_left:
                c = added_left[-1]
                if labels[c] in del_set:
                    left_boundary = spans[c][1]
                    added_left.pop()
                else:
                    break
            while len(added_right)>0 and right_boundary > gold_right:
                c = added_right[-1]
                if labels[c] in del_set:
                    right_boundary = spans[c][0]
                    added_right.pop()
                else:
                    break
            if left_boundary==gold_left and right_boundary==gold_right:
                return fix_step
        # step 2: add new ones
        add_set = set()
        for fix_step_idx, new_add_set in enumerate([{"det"}, {"nmod"}, {"punct"}, {"conj"}], 3):
            add_set.update(new_add_set)
            fix_step = f"s{fix_step_idx}_add_{'-'.join(sorted(add_set))}"
            # add all of the corresponding children if needed
            for c in left_children:
                if left_boundary <= gold_left:
                    break
                if c<left_boundary and labels[c] in add_set:
                    left_boundary = spans[c][0]
            for c in right_children:
                if right_boundary >= gold_right:
                    break
                if c>=right_boundary and labels[c] in add_set:
                    right_boundary = spans[c][1]
            if left_boundary==gold_left and right_boundary==gold_right:
                return fix_step
        # others: verb phrase
        if cur_pos == "VERB":
            return "sz0_verb"
        # others: not valid part of constituent
        min_left, max_right = head_wid, head_wid+1
        for one_left, one_right in spans:
            if one_left>=gold_left and one_right<=gold_right:
                min_left = min(min_left, one_left)
                max_right = max(max_right, one_right)
        if min_left>gold_left or max_right<gold_right:
            return "sz1_frag"  # fragmented
        return "szz_unk"

# from outside tables output by the expander module
class SpanExpanderExternal:
    def __init__(self, expand_span_ext_file, **kwargs):
        self.tables = {}
        with open(expand_span_ext_file) as fd:
            for line in fd:
                data = json.loads(line)
                key, head, start, end = data["key"], data["head"], data["start"], data["end"]
                assert key not in self.tables
                self.tables[key] = (head, start, end)
        zlog(f"Read from {expand_span_ext_file}, {len(self.tables)} entries")

    # todo(note): head_wid includes the ROOT offset
    def expand_span(self, head_wid, sent):
        doc_id, sid = sent.doc.doc_id, sent.sid
        key = f"{doc_id}_{sid}_{head_wid}"
        if key not in self.tables:
            zwarn("Not covered from external-table!!")
            return head_wid, 1  # simply return singleton
        else:
            head, start, end = self.tables[key]
            assert head == head_wid-1
            return start+1, end+1-start
