#

# for dependency parsing
# -- mostly similar to zext; todo(+N): might refactor to share more??

__all__ = [
    "ALZdparConf", "ALZdparHelper",
]

import os
from collections import Counter, defaultdict
from mspx.data.inst import Doc, yield_sents, yield_frames
from mspx.utils import zlog, zwarn, ZHelper, ZObject, mkdir_p
from .base import *

class ALZdparConf(ALTaskConf):
    def __init__(self):
        super().__init__()
        # --
        from mspx.tasks.zdpar.mod import ZTaskDparConf
        self.tconf = ZTaskDparConf().direct_update(name='dpar0')

@ALZdparConf.conf_rd()
class ALZdparHelper(ALTaskHelper):
    def __init__(self, conf: ALZdparConf, **kwargs):
        super().__init__(conf, **kwargs)
        # --
        conf: ALZdparConf = self.conf
        self.name = conf.tconf.name
        assert len(self.name) > 0, "Must have a name!"
        # note: from ZModDpar
        self.KEY_STRG = f"{self.name}_strg"  # soft target
        self.KEY_HID = f"repr_hid"  # hidden layer
        self.KEY_CALI = f"{self.name}_cali"  # for calibration!
        self.IDX_PA = -1  # special one for UNK head!
        self.LAB_PA = '_UNK_'  # special one for UNK label!
        self.IDX_QUERY = 0  # just other than IDX_PA!
        self.LAB_QUERY = '_Q_'  # special one!
        self.PAST_PREFIX = "**"  # prefix for past ann
        self.selector = conf.selector.make_node()
        self._init_specs()
        # --

    # yield items from doc
    def _yield_items(self, doc, cc, yield_tok=False, yield_sent=False, skip_ann=False):
        conf: ALZdparConf = self.conf
        _label_level = conf.tconf.lab_level
        # --
        assert isinstance(doc, Doc)
        cc['doc'] += 1
        for sent in yield_sents(doc):
            _len = len(sent)
            cc['sent'] += 1
            cc['tok'] += _len
            # check the tree
            if sent.tree_dep is None:  # simply assign an empty tree if no tree
                sent.build_dep_tree([self.IDX_PA] * _len, [self.LAB_PA] * _len)
            _tree = sent.tree_dep
            _vH, _vL = _tree.seq_head.vals, _tree.get_labels(_label_level)
            flag_ann = [0] * _len  # token annotated?
            for _widx, (_hidx1, _lab) in enumerate(zip(_vH, _vL)):
                tok_ann = (_lab != self.LAB_PA)  # just check lab!
                flag_ann[_widx] = int(tok_ann)
                if tok_ann:
                    cc['tokA'] += 1  # actual annotated
                    assert _hidx1 >= 0 and len(_lab) > 0
                    if _lab == self.LAB_QUERY:
                        cc['tokAQ'] += 1  # special query one!
                else:
                    cc['tokU'] += 1
                if yield_tok:
                    if not (skip_ann and tok_ann):  # get one candidate of tok
                        tok_cand = ZObject(type='tok', gid=id(sent), sent=sent, widx=_widx, budget=1)
                        yield tok_cand
            # sent
            _num_ann = sum(flag_ann)
            cc['sentU'] += int(_num_ann == 0)  # empty Unannotated
            cc['sentA'] += int(_num_ann == _len)  # full Annotated
            cc['sentP'] += int(_num_ann > 0 and _num_ann < _len)  # partial
            if yield_sent and (not (skip_ann and _num_ann == _len)):
                sent_cand = ZObject(type='sent', gid=id(doc), sent=sent, budget=_len)
                yield sent_cand
        # --

    @property
    def main_entry(self):
        return "mspx.tasks.zdpar.main"

    # --
    # 0. setup

    # process inst at setup (inplace!)
    def setup_inst(self, inst: Doc, mark_unn: bool, mark_past=False):
        conf: ALZdparConf = self.conf
        cc = Counter()
        _special_labs = [self.LAB_PA, self.LAB_QUERY]
        for obj in self._yield_items(inst, cc, yield_sent=True):
            if mark_past:
                if obj.type == 'sent':  # mark the existing ordinary labels!
                    _labels = obj.sent.tree_dep.seq_label.vals
                    _labels = [(z if z in _special_labs else self.PAST_PREFIX+z) for z in _labels]
                    obj.sent.tree_dep.seq_label.vals = _labels
                else:
                    raise RuntimeError()
            if mark_unn:  # extra operations!
                if obj.type == 'sent':  # put an empty tree!
                    sent = obj.sent
                    _len = len(sent)
                    sent.build_dep_tree([self.IDX_PA] * _len, [self.LAB_PA] * _len)
                else:
                    raise RuntimeError()
        return cc

    # --
    # 1. query

    # yielding and add arrs!
    def _yield_cands(self, doc, cc, partial: bool):
        for one_cand in self._yield_items(doc, cc, yield_tok=partial, yield_sent=(not partial), skip_ann=True):
            sent = one_cand.sent
            # [L, L*V], [1+L, D]
            arr_strg, arr_hid = sent.arrs.get(self.KEY_STRG), sent.arrs.get(self.KEY_HID)
            one_cand.score_cand = None
            if arr_strg is not None:
                v0 = arr_strg[1:]  # [Lm, Lh, V]
                v1 = v0[one_cand.widx].copy() if partial else v0
                vv = v1.reshape(list(v1.shape)[:-2] + [-1])
                one_cand.arr_strg = vv
            if arr_hid is not None:  # CLS or 1+widx
                one_cand.arr_hid = arr_hid[(1+one_cand.widx) if partial else 0].copy()
            yield one_cand

    # obtain all cands
    def _obtain_all_cands(self, data_stream, partial: bool):
        cc = Counter()
        all_docs = []
        cands = []  # List of unann candidates
        for doc in data_stream:
            all_docs.append(doc)  # store it!
            for one_cand in self._yield_cands(doc, cc, partial):
                cands.append(one_cand)
        return all_docs, cands, cc

    # actual querying
    def _do_query(self, data_stream, dev_stream, ref_stream=None, refD_stream=None, no_strg=False, repr_helper=None):
        conf: ALZdparConf = self.conf
        cc = Counter()
        _query_partial = conf.curr_is_partial(self.curr_iter)
        _score_randomly = (no_strg or (not conf.query_use_unc))
        # --
        if not conf.query_selv2:
            all_docs, cands, cc0 = self._obtain_all_cands(data_stream, _query_partial)
            cc += cc0
            query_cands = self.selector.select(cands, self.curr_budget, conf.budget_group, score_randomly=_score_randomly)
        else:
            all_docs, cands, cc0 = self._obtain_all_cands(data_stream, True)  # first get all tokens
            cc += cc0
            all_sents = yield_sents(all_docs)
            if conf.selv2_only_empty:
                all_s2_sents = [z for z in all_sents if all(z2==self.LAB_PA for z2 in z.tree_dep.seq_label.vals)]
            else:
                all_s2_sents = all_sents
            cand_items = [cands]
            cand_sc = [conf.selv2_sc0]
            cand_ratios = [self._get_curr_val(conf.selv2_ratios0)]
            cand_threshs = [self.curr_selv2_ths[0]]
            comb_params = [1.]
            dev_items = None if dev_stream is None else [self._yield_dev_cands(dev_stream, self.KEY_CALI)]
            _final_cands = self.selector.select_v2(all_s2_sents, [self.curr_budget, conf.budget_group], None, cand_items, dev_items, cand_sc, cand_ratios, cand_threshs, _score_randomly, _query_partial, conf.selv2_ratio_sentwise, 'score', comb_params, ref_helper=self._get_ref_helper(ref_stream), repr_helper=repr_helper)
            query_cands = _final_cands[0] if _query_partial else _final_cands  # fit here!
        # --
        # then prepare the query insts
        for doc in all_docs:  # whether clear all?
            self.setup_inst(doc, mark_unn=(not conf.qann_with_past), mark_past=conf.qann_with_past)
        cc['q_candA'] += len(cands)
        cc['q_candQ'] += len(query_cands)
        hit_sents = {}  # (doc.id, sid) -> sent
        hit_toks = set()
        for one_cand in query_cands:
            cc['q_budget'] += one_cand.budget
            hit_sents[one_cand.sent.dsids] = one_cand.sent
            # --
            _sent = one_cand.sent
            _tree_heads = _sent.tree_dep.seq_head.vals
            _tree_labs = _sent.tree_dep.seq_label.vals
            if one_cand.type == 'tok':  # each tok individual!
                _span = (one_cand.widx, 1)
            elif one_cand.type == 'span':  # a span
                _span = one_cand.span
            else:  # query the full sent!
                assert one_cand.type == 'sent'
                _span = (0, len(_sent))
            for ii in range(_span[0], _span[0]+_span[1]):
                _key = (id(_sent), ii)
                assert _key not in hit_toks, "Query for repeated tokens?"
                hit_toks.add(_key)
                _tree_heads[ii] = self.IDX_QUERY
                _tree_labs[ii] = self.LAB_QUERY
        hit_doc_ids = set(z[0] for z in hit_sents.keys())
        if conf.qann_with_hit:
            ret_docs = [d for d in all_docs if d.id in hit_doc_ids]  # if there are queries
        else:
            ret_docs = all_docs
        cc['q_doc'] += len(hit_doc_ids)
        cc['q_sent'] += len(hit_sents)
        cc['q_tok'] += len(hit_toks)
        assert cc['q_tok'] == cc['q_budget']  # no overlapping!
        return ret_docs, hit_sents, cc

    # --
    # 2./3. ann & comb

    # simulated ann according to ref
    def do_simul_ann(self, query_insts, ref_map, last_model=None):
        conf: ALZdparConf = self.conf
        cc = Counter()
        # --
        # note: simply label them all
        all_budget = self.curr_budget
        remaining_budget = all_budget
        hit_sents = set()
        for q_sent in yield_sents(query_insts):
            cc['q_sent'] += 1
            q_heads = q_sent.tree_dep.seq_head.vals
            q_labs = q_sent.tree_dep.seq_label.vals
            r_sent = ref_map[q_sent.doc.id].sents[q_sent.sid]  # must be there
            r_heads = r_sent.tree_dep.seq_head.vals
            r_labs = r_sent.tree_dep.seq_label.vals
            for ii in range(len(q_sent)):
                if q_labs[ii] == self.LAB_QUERY:  # simply judged by this!
                    cc['q_tok'] += 1
                    hit_sents.add(id(q_sent))
                    remaining_budget -= 1
                    q_heads[ii] = r_heads[ii]
                    q_labs[ii] = r_labs[ii]
        # --
        cc['q_sent_hit'] = len(hit_sents)
        cc['budgetC'] = all_budget - remaining_budget  # cost
        cc['budgetR'] = remaining_budget  # could be negative but could be neglected!
        return cc

    # combine new insts into trg
    def _do_comb(self, ann_inst, trg_map, cc):
        conf: ALZdparConf = self.conf
        # --
        _pp = self.PAST_PREFIX
        for a_sent in yield_sents(ann_inst):
            cc['a_sent'] += 1
            has_new_ann = False
            a_heads = a_sent.tree_dep.seq_head.vals
            a_labs = a_sent.tree_dep.seq_label.vals
            r_sent = trg_map[a_sent.doc.id].sents[a_sent.sid]  # must be there
            r_heads = r_sent.tree_dep.seq_head.vals
            r_labs = r_sent.tree_dep.seq_label.vals
            for ii in range(len(a_sent)):
                a_head, a_lab = a_heads[ii], a_labs[ii]
                if a_lab == self.LAB_QUERY:
                    cc['combD_unn'] += 1
                elif a_lab == self.LAB_PA:
                    pass
                elif a_lab.startswith(_pp):  # check equal?
                    if r_heads[ii] != a_head or r_labs[ii] != a_lab[len(_pp):]:
                        zwarn(f"Correction against past annotation: {r_heads[ii],a_head} {r_labs[ii],a_lab}")
                        r_heads[ii] = a_head
                        r_labs[ii] = a_lab[len(_pp):]  # remove mark!
                        cc['combD_aaC'] += 1  # correction
                        has_new_ann = True
                    else:
                        cc['combD_aa'] += 1  # already annotated
                else:
                    has_new_ann = True
                    assert r_heads[ii] == self.IDX_PA and r_labs[ii] == self.LAB_PA
                    cc['combD_new'] += 1
                    r_heads[ii] = a_head
                    r_labs[ii] = a_lab
                    if 1:  # add edge length!
                        cc['combD_new_elen'] += abs(a_head - 1 - ii)
        # --

    # --
    # 5. training
    # ...

# --
# b mspx/tools/al/tasks/zdpar:174
