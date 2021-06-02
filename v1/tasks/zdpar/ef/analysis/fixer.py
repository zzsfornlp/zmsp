#

# fixing: transform from pred to gold (multiple rounds)
# bottom up: heading & attaching & labeling
## heading: gold as the backbone
## attaching/labeling: individually for the remainings

from typing import List, Set, Tuple
from collections import namedtuple
from msp.utils import Conf

# =====
# types

ulabel_types = {
"Nivre17": {
    "FUN": ["aux", "case", "cc", "clf", "cop", "det", "mark"],
    "MWE": ["compound", "fixed", "flat", "goeswith"],
    "CORE": ["ccomp", "csubj", "iobj", "nsubj", "obj", "xcomp"],
    "NON-CORE": ["acl", "advcl", "advmod", "amod", "appos", "conj", "dep", "discourse", "dislocated", "expl", "list", "nmod", "nummod", "obl", "orphan", "parataxis", "reparandum", "root", "vocative"],
    "PUNCT": ["punct"],
},
"UDDOC": {
    # first put the others (some can be non-core, but too rare)
    "Others": ["vocative", "expl", "dislocated", "discourse", "list", "parataxis", "orphan", "reparandum", "dep"],
    # then special ones
    "Root": ["root"],
    "Punct": ["punct"],
    "Mwe": ["compound", "fixed", "flat", "goeswith"],
    "Conj": ["conj", "cc"],
    # function
    "Fun": ["aux", "case", "clf", "cop", "det", "mark"],
    # Nom/Clause/Mod(other) -> Core/Ncore/Nom, need more details to distinguish N/PP
    ("N", "Core"): ["nsubj", "obj", "iobj"],
    # ("N", "Ncore"): ["obl"],
    # ("N", "Nom"): ["nmod", "appos"],
    ("N", "Other"): ["obl", "nmod", "appos"],
    ("C", "Core"): ["csubj", "ccomp", "xcomp"],
    # ("C", "Ncore"): ["advcl"],
    # ("C", "Nom"): ["acl"],
    ("C", "Other"): ["advcl", "acl"],
    ("Mod", ): ["advmod", "amod", "nummod"],
},
# ['Others', 'Root', 'Punct', 'Mwe', 'Conj', 'Fun', 'N.Core', 'N.Other', 'C.Core', 'C.Other', 'Mod']
}

ulabel2type = {}  # type-system -> type-name -> List[edge-types]
for tsys, td in ulabel_types.items():
    ulabel2type[tsys] = {}
    for k, vs in td.items():
        if not isinstance(k, (List, Tuple)):
            k = [k]
        k = tuple(k)
        for v in vs:
            ulabel2type[tsys][v] = k

def get_coarse_type(v: str, tsys_name: str="UDDOC", layer=2):
    type_tuple = ulabel2type[tsys_name].get(v.split(":")[0], ("UNK",))
    return ".".join(type_tuple[:layer])
# =====

#
class FixerConf(Conf):
    def __init__(self):
        pass

# one edge
class DepEdge:
    def __init__(self, m, h, label, back_gap=-1):
        self.m = m
        self.h = h
        self.label = label
        self.back_gap = back_gap  # h is actually the decedent of m by how many gaps, <0 means no such relation

# one fix step for one edge
class FixChange:
    def __init__(self, old_edge: DepEdge, new_h, new_label=None):
        self.old_edge = old_edge
        self.new_h = new_h
        self.new_label = new_label

    @property
    def m(self): return self.old_edge.m

    @property
    def old_label(self): return self.old_edge.label

    @property
    def old_h(self): return self.old_edge.h

    def __repr__(self):
        return f"{self.m}<{self.old_label}>{self.old_h} => <{self.new_label}>{self.new_h}"

# group of fixing changes in one step
class FixOperation:
    def __init__(self, changes: List[FixChange], cur_type: str, cur_round: int, **kwargs):
        assert len(changes) > 0
        self.changes = changes
        self.main_change = self.changes[0]
        self.type = cur_type
        self.round = cur_round
        #
        self.cate = None
        self.category = None
        self.corrections: List = None  # list of m-idxes that are corrected (final fix) by this fix
        # set extra attribute
        for k, v in kwargs.items():
            assert not hasattr(self, k)
            setattr(self, k, v)

    def __str__(self):
        changes_str = ', '.join([str(z) for z in self.changes])
        x = f"{self.type}-R{self.round}, {self.category}, [{changes_str}], <{len(self.corrections)}>{self.corrections}"
        return x

    # category for this fixing according to template
    def set_category(self, gold_tree: 'DepTree') -> Tuple:
        m, old_h, old_label, new_h = self.main_change.m, self.main_change.old_h, \
                                     self.main_change.old_label, self.main_change.new_h
        gold_h, gold_label = gold_tree.heads[m], gold_tree.labels[m]
        if self.type == "attaching":
            assert gold_h == new_h
            cate = (gold_label, old_label)
        # elif self.type == "heading1":
        #     cate = (old_label, gold_label, gold_tree.labels[old_h])
        elif self.type == "heading":
            assert self.back_gap == 1, "No implementation for larger gap"
            cate = (gold_tree.labels[old_h], old_label)
        elif self.type == "labeling":
            assert old_h == new_h and gold_h == new_h
            cate = (gold_label, old_label)
        else:
            raise NotImplementedError(f"No implementation of checking category for {self.type}")
        self.cate = cate
        self.category = self.match_cetegory(cate)

    def match_cetegory(self, cate: Tuple):
        return tuple(get_coarse_type(z) for z in cate)

# general data structure
class DepTree:
    # todo(note): assume already padding 0 for artificial root
    def __init__(self, heads: List[int], labels: List[str], is_gold):
        # =====
        # these fields will be changed
        self.length = len(heads)
        self.heads = heads
        self.labels = labels
        # build the children set
        self.children = [set() for _ in range(self.length)]
        for m, h in enumerate(heads):
            if m>0:
                self.children[h].add(m)
        # =====
        # only used for gold tree
        # build the antecedent info, might waste some space/time, but which is ok
        self.is_gold = is_gold
        if is_gold:
            self.antecedent_maps = self.get_antecedent_maps()  # antecedent -> gap
            self.depths = self.get_depths()
            self.fixes = None
            self.fixing_sources = None
        else:
            self.antecedent_maps = None
            self.depths = None
            self.fixes = []  # list of FixOperations
            # list[modifier] of related fixing operations
            self.fixing_sources: List[List[FixOperation]] = [[] for _ in range(self.length)]

    def get_antecedent_maps(self):
        ret = [{} for _ in range(self.length)]
        def _get_ante_map(cur_idx: int, cur_path: List):
            assert len(ret[cur_idx]) == 0
            ret[cur_idx] = {x: i+1 for i,x in enumerate(reversed(cur_path))}  # gap along the spine
            cur_path.append(cur_idx)
            for new_idx in self.children[cur_idx]:
                _get_ante_map(new_idx, cur_path)
            cur_path.pop()
        tmp_path = []
        _get_ante_map(0, tmp_path)
        return ret

    def get_depths(self):
        ret = [0] * self.length
        def _get_depths(cur_idx: int, cur_depth: int):
            ret[cur_idx] = cur_depth
            for new_idx in self.children[cur_idx]:
                _get_depths(new_idx, cur_depth+1)
        _get_depths(0, 0)
        return ret

    # for example: get_nodes("depths", True) for bottom up ones
    def get_nodes(self, sort_by, reverse):
        sorting_keys = getattr(self, sort_by)
        # todo(note): here, there is an extra backoff idx sorting criterion at the last
        sorting_items = [(k, idx) for idx, k in enumerate(sorting_keys)]
        sorting_items.sort(reverse=reverse)
        return [z[-1] for z in sorting_items]

    # apply a fix operation
    def apply_fix(self, fix: FixOperation):
        assert not self.is_gold
        for one_change in fix.changes:
            m, orig_h, new_h, new_lab = one_change.m, one_change.old_h, one_change.new_h, one_change.new_label
            assert self.heads[m] == orig_h
            self.heads[m] = one_change.new_h
            if new_lab is not None:
                self.labels[m] = new_lab
            self.children[orig_h].remove(m)
            self.children[new_h].add(m)
            self.fixing_sources[m].append(fix)
        self.fixes.append(fix)

    # finish the fixing, group each error edges to the final fix
    def finish_fix(self, gold_tree: 'DepTree'):
        # classify each fixes
        assert not self.is_gold
        for one_fix in self.fixes:
            one_fix.set_category(gold_tree)
            one_fix.corrections = []  # open this one!
        # group edge-errors into their final fixes
        assert len(self.fixing_sources[0]) == 0
        for m_idx, one_fix_list in enumerate(self.fixing_sources):
            assert self.heads[m_idx] == gold_tree.heads[m_idx]
            if len(one_fix_list) > 0:
                one_fix_list[-1].corrections.append(m_idx)
        pass

    # get edge with back_edge characters
    def characterize_edges(self, gold_tree: 'DepTree') -> List[DepEdge]:
        assert self.length == gold_tree.length
        ret = [None] * self.length
        for idx in range(0, gold_tree.length):
            gold_ante_map = gold_tree.antecedent_maps[idx]
            pred_children_set: Set[int] = self.children[idx]  # set of children in current pred
            for one in pred_children_set:
                one_gap = gold_ante_map.get(one)
                one_pred_label = self.labels[one]
                if one_gap is not None:
                    one_edge = DepEdge(one, idx, one_pred_label, one_gap)
                else:
                    one_edge = DepEdge(one, idx, one_pred_label)
                assert ret[one] is None
                ret[one] = one_edge
        return ret

# =====
# error fixer using the gold as the reference
class GoldRefFixer:
    def __init__(self, conf: FixerConf):
        pass

    def fix_heading(self, gold_tree: DepTree, pred_tree: DepTree, cur_round: int, applying: bool, max_gap: int):
        # =====
        # pass: fix heading errors with bottom-up on gold (fix from the aspect of h->m)
        # actually this is fixing from head to mod, instead of mod attaching as in the attaching error
        # todo(+N): should we use pred-based, should it be dfs-post_order, or alternative strategies?
        all_fos = []
        gold_depths = gold_tree.depths
        for to_fix_m in gold_tree.get_nodes("depths", True):
            gold_ante_map = gold_tree.antecedent_maps[to_fix_m]
            pred_children_set: Set[int] = pred_tree.children[to_fix_m]  # set of children in current pred
            # check if there are gold-antecedents
            pred_children_antecedents: List[int] = []
            pred_children_others: List[int] = []
            for h in pred_children_set:
                if h in gold_ante_map:
                    pred_children_antecedents.append(h)
                    # # todo(+N): do we filter by max-gap here or later?
                    # if max_gap<=0 or gold_ante_map[h]<=max_gap:  # max_gap not activate if <=0
                    #     pred_children_antecedents.append(h)
                else:
                    pred_children_others.append(h)
            # sort by depth and only fix the lowest one
            pred_children_antecedents.sort(key=lambda x: -gold_depths[x])
            pred_children_antecedents_gaps = [gold_depths[to_fix_m]-gold_depths[x] for x in pred_children_antecedents]
            # prune by max_gap (allowing continous gaps with each one <=max_gap)
            if max_gap > 0:
                prev_gap = 0
                cur_idx = 0
                while cur_idx<len(pred_children_antecedents_gaps) and pred_children_antecedents_gaps[cur_idx]<=max_gap+prev_gap:
                    prev_gap = pred_children_antecedents_gaps[cur_idx]
                    cur_idx += 1
                pred_children_antecedents = pred_children_antecedents[:cur_idx]
                pred_children_antecedents_gaps = pred_children_antecedents_gaps[:cur_idx]
            # if no remaining, then skip this one
            if len(pred_children_antecedents) == 0:
                continue  # not this one
            # start the specific fixing:
            lowest_child_antecedent = pred_children_antecedents[0]
            # find the first incorrect pred-antecedent (to_fix_m_upper_head -> to_fix_m_upper)
            to_fix_m_upper = to_fix_m
            while True:
                # cannot be always true, otherwise there will be loop to lowest_child_antecedent
                assert to_fix_m_upper > 0
                to_fix_m_upper_head = pred_tree.heads[to_fix_m_upper]
                if to_fix_m_upper_head == gold_tree.heads[to_fix_m_upper]:
                    to_fix_m_upper = to_fix_m_upper_head
                else:
                    break
            all_changes = []
            plabs = pred_tree.labels
            # todo(note): back edge as the first fix!
            # 1) the lowest antecedent is attached to the first incorrect pred-antecedent (also change label here)
            all_changes.append(FixChange(DepEdge(lowest_child_antecedent, to_fix_m,
                                                 plabs[lowest_child_antecedent], pred_children_antecedents_gaps[0]),
                                         to_fix_m_upper_head, pred_tree.labels[to_fix_m_upper]))
            # 2) link to_fix_m's first incorrect pred-antecedent's child to lowest antecedent
            all_changes.append(FixChange(DepEdge(to_fix_m_upper, to_fix_m_upper_head, plabs[to_fix_m_upper]),
                                         lowest_child_antecedent))
            # 3) make all upper antecedents to the lower one (still back-edge to be fixed later)
            all_changes.extend([FixChange(DepEdge(z, to_fix_m, g), lowest_child_antecedent)
                                for z, g in zip(pred_children_antecedents[1:], pred_children_antecedents_gaps[1:])])
            # 4) the all-antecedents' children are given to the lowest antecedent
            all_antecedents_children = set()
            for ante_idx in pred_children_antecedents:
                all_antecedents_children.update(gold_tree.children[ante_idx])
            all_changes.extend([FixChange(DepEdge(z, to_fix_m, plabs[z]), lowest_child_antecedent)
                                for z in pred_children_others if z in all_antecedents_children])
            fo = FixOperation(all_changes, "heading", cur_round, back_gap=pred_children_antecedents_gaps[0])
            all_fos.append(fo)
            if applying:
                pred_tree.apply_fix(fo)  # directly apply the fix
        return all_fos

    def fix_attaching(self, gold_tree: DepTree, pred_tree: DepTree, cur_round: int, applying: bool):
        # attach each errors independently (fix from the aspect of m) and also fix label here!
        all_fos = []
        for to_fix_m in gold_tree.get_nodes("depths", True):
            if to_fix_m>0:
                pred_h, pred_lab = pred_tree.heads[to_fix_m], pred_tree.labels[to_fix_m]
                gold_h, gold_lab = gold_tree.heads[to_fix_m], gold_tree.labels[to_fix_m]
                if pred_h != gold_h:
                    plabs = pred_tree.labels
                    fo = FixOperation([FixChange(DepEdge(to_fix_m, pred_h, plabs[to_fix_m]), gold_h,
                                                 new_label=(None if (pred_lab==gold_lab) else gold_lab))], "attaching", cur_round)
                    all_fos.append(fo)
                    if applying:
                        pred_tree.apply_fix(fo)  # directly apply the fix
        return all_fos

    def fix_labeling(self, gold_tree: DepTree, pred_tree: DepTree, cur_round: int, applying: bool):
        # only fix labeling errors for correct attachments
        all_fos = []
        for to_fix_m in range(gold_tree.length):
            pred_h, pred_lab = pred_tree.heads[to_fix_m], pred_tree.labels[to_fix_m]
            gold_h, gold_lab = gold_tree.heads[to_fix_m], gold_tree.labels[to_fix_m]
            if pred_h==gold_h and pred_lab!=gold_lab:
                fo = FixOperation([FixChange(DepEdge(to_fix_m, pred_h, pred_lab), gold_h, gold_lab)], "labeling", cur_round)
                all_fos.append(fo)
                if applying:
                    pred_tree.apply_fix(fo)  # directly apply the fix
        return all_fos

    # this is actually fix_heading(gap=1)
    def fix_direct_reverse(self, gold_tree: DepTree, pred_tree: DepTree, cur_round: int, applying: bool):
        # fix with reversed edges (fix from the aspect of m)
        all_fos = []
        for to_fix_m in gold_tree.get_nodes("depths", True):
            if to_fix_m > 0:
                # can be iterative
                while True:
                    pred_h = pred_tree.heads[to_fix_m]
                    if gold_tree.heads[pred_h] == to_fix_m:
                        assert pred_h != gold_tree.heads[to_fix_m]
                        plabs = pred_tree.labels
                        all_changes = []
                        # 1) m is attached to pred_h's pred_h
                        upper_h = pred_tree.heads[pred_h]  # pred_h's pred_h
                        upper_label = pred_tree.labels[pred_h]
                        all_changes.append(FixChange(DepEdge(to_fix_m, pred_h, plabs[to_fix_m], 1), upper_h, upper_label))
                        # 2) pred_h is attached to m
                        all_changes.append(FixChange(DepEdge(pred_h, upper_h, plabs[pred_h]), to_fix_m))
                        # 3) pull some children to m
                        gold_children = gold_tree.children[to_fix_m]
                        all_changes.extend([FixChange(DepEdge(z, pred_h, plabs[z]), to_fix_m)
                                            for z in pred_tree.children[pred_h] if z!=to_fix_m and z in gold_children])
                        fo = FixOperation(all_changes, "heading", cur_round, back_gap=1)
                        all_fos.append(fo)
                        if applying:
                            pred_tree.apply_fix(fo)  # directly apply the fix
                        else:
                            break
                    else:
                        break
        return all_fos

    # =====
    # specific fixing schemes

    # todo(WARN): there will be some labeling errors counting missing: thus we majorly focus on unlabeled errors
    def fix(self, gold_tree: DepTree, pred_tree: DepTree):
        self.fix_labeling(gold_tree, pred_tree, 0, True)
        for i in range(0, gold_tree.length):
            fos = self.fix_direct_reverse(gold_tree, pred_tree, i, True)
            # fos = self.fix_heading(gold_tree, pred_tree, i, True, 1)  # still max-gap==1
            if len(fos) == 0:
                break
        self.fix_attaching(gold_tree, pred_tree, 0, True)
        pred_tree.finish_fix(gold_tree)
