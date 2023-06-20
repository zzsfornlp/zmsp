#

# for the extraction of alink(relation/argument) ...
# note: some parts from "ALZextHelper" but with changes

__all__ = [
    "ALZrelConf", "ALZrelHelper",
]

import os
import math
from collections import Counter, defaultdict
import numpy as np
from mspx.data.inst import Doc, yield_sents, yield_sent_pairs, yield_frames, Frame
from mspx.utils import zlog, zwarn, ZHelper, ZObject, mkdir_p, Random
from .base import *

# --
# stat with ace-train:
# en: Counter({'c_word': 272855, 'c_Fef': 47554, 'c_doc': 19204, 'c_sent': 19204, 'c_Aevt': 6607, 'c_Fevt': 4419})
# zh: Counter({'c_word': 162392, 'c_Fef': 29681, 'c_doc': 6305, 'c_sent': 6305, 'c_Aevt': 5581, 'c_Fevt': 2926})
# ar: Counter({'c_word': 89236, 'c_Fef': 16143, 'c_doc': 3218, 'c_sent': 3218, 'c_Aevt': 2506, 'c_Fevt': 1743})
# ef*evt: en=23174, zh=24162, ar=21733
# --
# ace_evt: Counter({'c_word': 243190, 'c_Fef': 41303, 'cp_cands': 21705, 'c_doc': 13306, 'c_sent': 13306, 'c_Aef': 6695, 'c_Aevt': 6189, 'cp_valid': 6189, 'c_Fevt': 4135})
# ace_rel: Counter({'c_word': 243190, 'cp_cands': 187744, 'c_Fef': 41303, 'c_doc': 13306, 'c_sent': 13306, 'c_Aef': 6695, 'cp_valid': 6695, 'c_Aevt': 6189, 'c_Fevt': 4135})
# --
"""
# note: status for each data file 
# note: The data file speaks for itself with KEY_PA; for simul, simply mark all as KEY_PA and add explicit NEG!
-- data.init.json: all gold || data.init.unc.json  # mix gold & pred || data.query.json  # mix gold & pred & query
-- data.ann.json  # ideally all gold (might contain pred or query) || data.comb.json  # all gold
"""
# --

class ALZrelConf(ALTaskConf):
    def __init__(self):
        super().__init__()
        # --
        from mspx.tasks.zrel.mod import ZTaskRelConf
        self.rconf = ZTaskRelConf().direct_update(cateHs=['evt'], cateTs=['ef'], name='rel0')
        self.ext_extra = {}  # extra confs for ext
        self.use_gold_mention = False  # special mode of using gold (original) mentions! Should be used together with special training specifications!
        # --
        # special ones
        self.selector.strg_bias = 1.  # by default make it [0, 1]
        self.no_self = True  # no self-link!
        self.query_no_repeatq = False  # no repeat Qs of tok & alink's mentions
        self.query_mentionq = False  # simply put every mention as Q
        self.query_keep_preds = False  # keep predictions in the query file (but not NIL)
        self.query_keep_alink_preds = False  # keep alink preds in partial mode (if not NIL)
        self.query_entity_budget = -1.  # how many links for each entity? (not using this if <0)
        # --
        # alink further reduce
        self.query_spath_spec = [1., 'count', 1., -1]  # reducing target, spath sig, reducing ratio alpha, reducing truncate
        self.query_reselect_spec = [0, 'count', 1.]  # reselect count, spath sig, selecting alpha
        self.query_reselect_spec2 = [0, 'raw', 10]  # reselect count, sig, repeat alpha
        # --
        self.budgetA = [100000.]  # special alink budget for partial annotating
        self.budget_full_qextra = 0.  # give some extra budget for full annotation to ensure enough budgetA?
        self.sg_df = 0.  # default group (sent) score if no items exist
        self.sg_ratio = [0.5]  # group (sent) score ratio for alink: sga*score(alink)+(1.-sga)*score(tok)
        self.sg_comb_method = 'score'  # group (sent) comb-score method: score, alter, rank
        # self.sent_alpha_thr = -1.  # ignore sents if alpha% of a sent is <this (NOPE, simply filter at outside!)
        self.sent_filter_specs = [0., 0.]  # min-len, min-alpha-ratio
        self.ann_fix_frames = True  # try to fix frames for query alink (in simulated ann)
        self.ann_nil_frames = True  # try to add NIL frames for query alink (in simulated ann)
        self.ann_nf_alink = False  # annotate explicit NF alink (in simulated ann)
        self.ann_frame_cost = [1., 1.]  # cost-weight for adding predicted frames: P/F
        self.setup_del_all_frames = False  # for setup-unn, simply delete all frames of sents!
        # special ann_phase2
        self.ann_phase2 = ""  # phase 2 for partial! r: ratio, t: thresh, a: all, '': nope
        self.ann_phase2_onlyann = False  # only use the mentions annotated in the first phase
        self.ann_phase2_onlynew = True  # only use new/corrected mentions
        self.ann_phase2_ratio_specs = [1, 0]  # [sent-wise?, overall?]
        # special doc selection
        self.sdoc_budget = 0  # active if >0, number of docs to select each time
        self.sdoc_seed = 0  # seed for random selection

@ALZrelConf.conf_rd()
class ALZrelHelper(ALTaskHelper):
    def __init__(self, conf: ALZrelConf, **kwargs):
        super().__init__(conf, **kwargs)
        # --
        from .zext import ALZextConf
        conf: ALZrelConf = self.conf
        self.name = conf.rconf.name
        assert len(self.name) > 0, "Must have a name!"
        _cate_farmes = []
        for one_cates in [conf.rconf.cateHs, conf.rconf.cateTs]:
            for one_cate in one_cates:
                if one_cate not in _cate_farmes:
                    _cate_farmes.append(one_cate)
        _ext_conf = ALZextConf().direct_update(tconf__frame_cate=_cate_farmes, tconf__name='extH', **conf.ext_extra)
        self.ext_helper = _ext_conf.make_node()
        # --
        # note: from ZTaskRelMod
        self.KEY_PA = f"{self.name}_ispart"  # whether partial
        self.KEY_STRG = f"{self.name}_strg"  # soft target
        self.KEY_CALI = f"{self.name}_cali"  # for calibration!
        self.KEY_HID = f"repr_hid"  # hidden layer
        self.LAB_PA = '_UNK_'  # special one for UNK label!
        self.LAB_QUERY = '_Q_'
        self.LAB_NIL = '_NIL_'
        self.PAST_PREFIX = "**"  # prefix for past ann
        self.selector = conf.selector.make_node()
        # todo(+N): special additions for the base task: required decoding to get predicted frames!
        # note: no_strg1 to save space, also remember to mark frame's rel0-partial!
        _base_name = self.ext_helper.name
        conf.specs_infQ = f"{_base_name}.pred_do_dec:1 {_base_name}.pred_no_strg1:1 {_base_name}.pred_score_topk:1 {_base_name}.pred_mark_partial:{self.KEY_PA} " + conf.specs_infQ
        conf.specs_infT = f"{_base_name}.pred_do_dec:1 {_base_name}.pred_no_strg1:1 {_base_name}.pred_score_topk:1 {_base_name}.pred_mark_partial:{self.KEY_PA} " + conf.specs_infT
        self._init_specs()
        # --
        self.doc_cache = {}  # cache for doc
        # --

    @property
    def tasks_out(self):
        conf: ALZrelConf = self.conf
        return [self.ext_helper.name, self.name] if not conf.use_gold_mention else [self.name]  # all output tasks

    @property
    def tasks_all(self):
        conf: ALZrelConf = self.conf
        return ['enc0', self.ext_helper.name, self.name] if not conf.use_gold_mention else [self.name]  # all output tasks

    @property
    def curr_budgetA(self):
        return self._get_curr_val(self.conf.budgetA)

    @property
    def curr_sg_ratio(self):
        return self._get_curr_val(self.conf.sg_ratio)

    def setup_iter(self, curr_iter: int):
        super().setup_iter(curr_iter)
        self.ext_helper.setup_iter(curr_iter)

    # --
    # helper

    # yield items from doc
    def _yield_items(self, doc, cc, yield_tok=False, yield_sent=False, skip_ann=False,
                     yield_frame=False, yield_alink=False, add_df_alink=False):
        conf: ALZrelConf = self.conf
        ext_helper = self.ext_helper
        cateHs, cateTs = conf.rconf.cateHs, conf.rconf.cateTs
        # --
        _ignore_flabs = [ext_helper.LAB_NIL, ext_helper.LAB_QUERY, ext_helper.PAST_PREFIX+ext_helper.LAB_NIL]
        _alink_nils = [self.LAB_NIL, self.PAST_PREFIX + self.LAB_NIL]
        for item in ext_helper._yield_items(doc, cc, yield_tok=yield_tok, yield_sent=True, skip_ann=False, yield_frame=yield_frame):
            if item.type == 'sent':  # special handling
                frame_hs, frame_ts = [z for z in item.sent.get_frames(cates=cateHs) if z.label not in _ignore_flabs],\
                                     [z for z in item.sent.get_frames(cates=cateTs) if z.label not in _ignore_flabs]
                cc['rfH'] += len(frame_hs)
                cc['rfT'] += len(frame_ts)
                for one_h in frame_hs:
                    if one_h.info.get('is_pred', False):
                        one_h.info[self.KEY_PA] = True  # note: simply view pred as partial!
                    one_h_partial = one_h.info.get(self.KEY_PA, False)  # whether partial by default?
                    arg_map = {id(z.arg): z for z in one_h.args}  # existing alink
                    for one_t in frame_ts:
                        if conf.no_self and one_h is one_t: continue  # no self-link
                        # --
                        cc['alink'] += 1  # possible alink
                        _alink = arg_map.get(id(one_t))
                        if _alink is None:   # note: always make new ones for holding things
                            _alink_label = (self.LAB_PA if one_h_partial else self.LAB_NIL)
                            if add_df_alink:  # adding df ones
                                _alink = one_h.add_arg(one_t, _alink_label)
                                if one_h_partial:
                                    _alink.info['is_pred'] = True  # mark it as pred!
                                cc[f'alink_zadd_{_alink_label}'] += 1
                            else:  # missing df ones
                                cc[f'alink_zmissdf_{_alink_label}'] += 1
                        else:
                            _alink_label = _alink.label
                        cc[f'alink_NIL'] += int(_alink_label in _alink_nils)  # whether alink is NIL
                        cc[f'alink_Q'] += int(_alink_label == self.LAB_QUERY)  # whether alink is Query
                        _ann = (_alink_label != self.LAB_PA) and (not (_alink is not None and _alink.info.get('is_pred', False)))  # annotated if not PA and not predicted
                        cc['alink_ann'] += int(_ann)
                        cc['alink_unn'] += 1 - int(_ann)
                        if yield_alink and ((not _ann) or (not skip_ann)):
                            alink_cand = ZObject(type='alink', gid=id(item.sent), sent=item.sent,
                                                 alink=_alink, budget=1, budgetR=(1-int(_ann)))
                            yield alink_cand
                        # --
            # yield the item
            if item.type != 'sent' or yield_sent:
                if (not skip_ann) or (item.get('budgetR', 1) > 0):
                    yield item
        # --

    @property
    def main_entry(self):
        return "mspx.tasks.zrel.main"

    @property
    def always_do_pred_prep_query(self):
        return True  # need to obtain predicted mentions!

    # process inst at setup (inplace!)
    def setup_inst(self, inst: Doc, mark_unn: bool, mark_past=False):
        conf: ALZrelConf = self.conf
        ext_helper = self.ext_helper
        cc = Counter()
        assert not mark_past, "Not implemented here!"
        all_objs = list(self._yield_items(inst, cc, yield_frame=True, yield_sent=True, yield_alink=True))
        for obj in all_objs:
            if mark_unn:  # extra operations!
                if conf.use_gold_mention:  # only deleting alinks in this mode!
                    if obj.type == 'sent':   # simply delete all the links here!
                        for frame in obj.sent.get_frames():
                            for arg in frame.get_args():
                                arg.del_self()
                    elif obj.type == 'frame':
                        obj.frame.info[self.KEY_PA] = True
                    elif obj.type == 'alink':
                        pass
                        # if obj.alink is not None:
                        #     obj.alink.del_self()
                    else:
                        raise RuntimeError()
                else:
                    if obj.type == 'sent':
                        obj.sent.info[ext_helper.KEY_PA] = True  # mark partial!
                        if conf.setup_del_all_frames:  # note: clear them all!
                            for frame in obj.sent.get_frames():
                                frame.del_self()
                    elif obj.type == 'frame':
                        if not conf.setup_del_all_frames:
                            obj.frame.del_self()  # delete existing ones!
                    elif obj.type == 'alink':
                        pass  # nothing to do here, since they will be removed along with frames!
                    else:
                        raise RuntimeError()
        return cc

    # yielding and add arrs!
    def _yield_cands(self, doc, cc, partial: bool, invalid_toks):
        conf: ALZrelConf = self.conf
        for one_cand in self._yield_items(doc, cc, skip_ann=True, yield_alink=True, add_df_alink=True):
            all_frame_toks = [(id(_ff.mention.sent), _ii) for _ff in [one_cand.alink.main, one_cand.alink.arg]
                              for _ii in range(_ff.mention.widx, _ff.mention.wridx)]
            # cc['alink_C0'] += 1
            if invalid_toks is None or not any(t in invalid_toks for t in all_frame_toks):
                # cc['alink_C1'] += 1
                one_cand.arr_strg = one_cand.alink.arrs.get(self.KEY_STRG)
                if one_cand.arr_strg is None:  # sometimes may happen, for e.g. due to truncating
                    cc['alink_nostrg'] += 1
                    if not conf.use_gold_mention:
                        continue  # let it in for random mode!
                # todo(+N): currently skip HID related stuffs
                one_cand.score_cand = None
                yield one_cand
        # --

    # obtain all cands
    def _obtain_all_cands(self, data_stream, partial: bool, invalid_toks=None):
        cc = Counter()
        all_docs = []
        cands = []  # List of unann candidates
        for doc in data_stream:
            all_docs.append(doc)  # store it!
            for one_cand in self._yield_cands(doc, cc, partial, invalid_toks):
                cands.append(one_cand)
        return all_docs, cands, cc

    # special random doc selection
    def _select_sents_by_doc(self, sents):
        conf: ALZrelConf = self.conf
        _budget, _seed = conf.sdoc_budget, conf.sdoc_seed
        frame_cate = self.ext_helper.conf.tconf.frame_cate
        # --
        # first group sents by docs
        docs0 = {}
        for sent in sents:
            doc_key = sent.doc.id
            if doc_key not in docs0:
                docs0[doc_key] = []
            docs0[doc_key].append(sent)
        doc_keys = sorted(docs0.keys())
        doc_scores = []
        _gen = Random.get_np_generator(_seed)
        for kk in doc_keys:  # note: cache for "fixed" random order (since in iter0, every one will be scored)!
            ss = self.doc_cache.get(kk)
            if ss is None:
                ss = _gen.random()
                self.doc_cache[kk] = ss
            doc_scores.append(ss)
        # selection
        sel_doc_keys = []
        for ss, kk in sorted(zip(doc_scores, doc_keys)):
            if len(sel_doc_keys) >= _budget:
                break  # out of budget
            if any(not z2.info.get('is_pred', False) for z in docs0[kk] for z2 in z.get_frames(cates=frame_cate)):
                continue  # skip already annotated ones
            sel_doc_keys.append(kk)
        # --
        sel_doc_keys.sort()  # for pretty printing
        ret = sum([docs0[z] for z in sel_doc_keys], [])
        zlog(f"Finished doc-selection: D({len(docs0)}->{len(sel_doc_keys)}),S({len(sents)}->{len(ret)}): {sel_doc_keys}")
        return ret

    # special "selector.select"
    def _my_select(self, all_sents, cands0_tok, cands0_alink, dev_items, score_randomly: bool, partial: bool, ref_stream=None, repr_helper=None):
        conf: ALZrelConf = self.conf
        cc = Counter()
        selector = self.selector
        frame_cate = self.ext_helper.conf.tconf.frame_cate
        _budget_group, _budget, _budgetA, _budget_full_qextra = \
            conf.budget_group, self.curr_budget, self.curr_budgetA, conf.budget_full_qextra
        _sg_df = conf.sg_df
        _filter_specs = conf.sent_filter_specs
        # --
        # get utility score for both
        cc['cand0_sent'] = len(all_sents)
        cc['cand0_tok'] = len(cands0_tok)
        cc['cand0_alink'] = len(cands0_alink)
        if not conf.query_selv2:
            raise NotImplementedError("Only support selv2 now!!")
        else:  # with "empty sents"!
            all_s2_sents = all_sents  # start with all
            if conf.sdoc_budget > 0:
                all_s2_sents = self._select_sents_by_doc(all_s2_sents)
            if conf.selv2_only_empty:
                if conf.use_gold_mention:
                    all_s2_sents = [z for z in all_s2_sents if not any(not z3.info.get('is_pred', False) for z2 in z.get_frames(cates=frame_cate) for z3 in z2.args)]
                else:
                    all_s2_sents = [z for z in all_s2_sents if not any(not z2.info.get('is_pred', False) for z2 in z.get_frames(cates=frame_cate))]
            if any(z>0 for z in _filter_specs):
                all_s2_sents = [z for z in all_s2_sents if len(z)>=_filter_specs[0] and sum((any(str.isalpha(c) for c in t)) for t in z.seq_word.vals) / len(z) >= _filter_specs[1]]
            # --
            cc['cand1_sent'] = len(all_s2_sents)
            cc['cand1_tok'] = len(cands0_tok)
            cc['cand1_alink'] = len(cands0_alink)
            zlog(f"Stat before selv2: {cc}")
            # --
            cand_items = [cands0_tok, cands0_alink]  # alink as trg1!
            cand_sc = [conf.selv2_sc0, conf.selv2_sc1]
            cand_ratios = [self._get_curr_val(conf.selv2_ratios0), self._get_curr_val(conf.selv2_ratios1)]
            cand_threshs = self.curr_selv2_ths
            comb_params = [1.-self.curr_sg_ratio, self.curr_sg_ratio]
            _final_cands = selector.select_v2(all_s2_sents, [_budget, _budget_group], _sg_df, cand_items, dev_items, cand_sc, cand_ratios, cand_threshs, score_randomly, partial, conf.selv2_ratio_sentwise, conf.sg_comb_method, comb_params, ref_helper=self._get_ref_helper(ref_stream), repr_helper=repr_helper)
            final_ret = _final_cands
        # --
        return final_ret

    def _yield_dev_rel_cands(self, dev, devR):
        helper = self._get_ref_helper(devR)
        alink_cands = list(self._obtain_all_cands(dev, True))[1]
        helper.eval_cands(alink_cands)
        yield from alink_cands
        # --

    # special preparing: use ref mentions but remember marking is_pred
    def prep_query_with_ref(self, input_data: str, output_data: str, ref_data: str):
        from mspx.data.rw import ReaderGetterConf, WriterGetterConf
        # --
        frame_cate = self.ext_helper.conf.tconf.frame_cate
        i_insts = list(ReaderGetterConf().get_reader(input_path=input_data))
        r_insts = list(ReaderGetterConf().get_reader(input_path=ref_data))
        cc = Counter()
        for i_sent, r_sent in yield_sent_pairs(i_insts, r_insts):
            cc['sent'] += 1
            assert i_sent.seq_word.vals == r_sent.seq_word.vals
            i_ann = [0] * len(i_sent)
            for ff in i_sent.get_frames(cates=frame_cate):
                cc['frameI'] += 1
                _widx, _wlen = ff.mention.get_span()
                i_ann[_widx:_wlen+_wlen] = [1] * _wlen
            for ff in r_sent.get_frames(cates=frame_cate):
                cc['frameR'] += 1
                _widx, _wlen = ff.mention.get_span()
                if any(i_ann[z] for z in range(_widx, _widx+_wlen)):
                    cc['frameR_old'] += 1
                else:
                    cc['frameR_new'] += 1
                    new_frame = i_sent.make_frame(_widx, _wlen, ff.label, ff.cate)
                    new_frame.info['is_pred'] = True  # note: mark pred
                    new_frame.info[self.KEY_PA] = True  # note: mark PA
        zlog(f"prep_query_with_ref: {ZHelper.resort_dict(cc)}")
        with WriterGetterConf().get_writer(output_path=output_data) as writer:
            writer.write_insts(i_insts)
        # --

    # actual querying
    def _do_query(self, data_stream, dev_stream, ref_stream=None, refD_stream=None, no_strg=False, repr_helper=None):
        conf: ALZrelConf = self.conf
        ext_helper = self.ext_helper
        cc = Counter()
        _query_partial = conf.curr_is_partial(self.curr_iter)
        _score_randomly = (no_strg or (not conf.query_use_unc))
        cateHs, cateTs = conf.rconf.cateHs, conf.rconf.cateTs
        frame_cate = ext_helper.conf.tconf.frame_cate
        _query_keep_preds = conf.query_keep_preds
        # --
        # collect both cands: note: go partial here!
        all_docs, cands0_tok, _ = ext_helper._obtain_all_cands(data_stream, partial=True)
        _, cands0_alink, cc0 = self._obtain_all_cands(all_docs, partial=True)
        if not _score_randomly:  # further ignore those without strg
            cands0_alink_valid = [z for z in cands0_alink if z.arr_strg is not None]
            if len(cands0_alink_valid) < len(cands0_alink):
                zwarn(f"Ignore {len(cands0_alink)}-{len(cands0_alink_valid)}"
                      f"={len(cands0_alink)-len(cands0_alink_valid)} alinks without strg.")
            cands0_alink = cands0_alink_valid
        cc += cc0
        # --
        # select
        all_sents = sum([d.sents for d in all_docs], [])
        dev_items = None
        if dev_stream is not None:
            if conf.use_gold_mention:  # no need of ent scores
                dev_items = [[]]
            else:
                dev_items = [self._yield_dev_cands(dev_stream, self.ext_helper.KEY_CALI)]
            # note: special eval for dev-rel with predicted mentions!
            dev_items.append(self._yield_dev_rel_cands(dev_stream, refD_stream))
        query_ret = self._my_select(all_sents, cands0_tok, cands0_alink, dev_items, _score_randomly, _query_partial, ref_stream=ref_stream, repr_helper=repr_helper)
        if _query_partial:  # has alink-query
            q_hit_sents = {_c.sent.dsids: _c.sent for _c in query_ret[0]+query_ret[1]}
        else:
            q_hit_sents = {_c.sent.dsids: _c.sent for _c in query_ret}
        # --
        # further reduce alinks by diversity?
        _query_spath_spec = conf.query_spath_spec
        if _query_partial and _query_spath_spec[0] < 1.:
            _reduce_trg, _group_sig, _reduce_alpha, _reduce_t = conf.query_spath_spec
            # --
            _s0_cands = [z for z in cands0_alink if z.sent.dsids in q_hit_sents]  # all selected-sent cands
            _s1_cands = list(query_ret[1])  # original selected cands
            _s0_groups = self.group_by_spath(_s0_cands, _group_sig)
            _s1_groups = self.group_by_spath(_s1_cands, _group_sig)
            for _kk in _s0_groups.keys():
                if _kk not in _s1_groups:
                    _s1_groups[_kk] = []  # for easier processing
            _sg_keys = sorted(_s1_groups.keys(), key=(lambda k: -len(_s1_groups[k])))  # sort by selected!
            assert all(k in _sg_keys for k in _s1_groups.keys())
            # count
            _counts0, _counts1 = np.asarray([len(_s0_groups[k]) for k in _sg_keys]), np.asarray([len(_s1_groups[k]) for k in _sg_keys])
            _reduce_weights = np.asarray([len(_s1_groups[k]) ** _reduce_alpha for k in _sg_keys])
            if _reduce_t > 0:
                _reduce_weights[_reduce_t:] = 0.  # no delete for fewer ones
            _reduce_weights = _reduce_weights / _reduce_weights.sum()  # [...]
            _reduce_counts0 = (1-_reduce_trg) * len(_s1_cands) * _reduce_weights  # remove counts [...]
            _reduce_counts = self.round_counts(_reduce_counts0)
            _counts2 = [0] * len(_counts0)
            # get final results
            _survive_cands = []
            for _ii, _kk in enumerate(_sg_keys):
                _k_cands = sorted(_s1_groups[_kk], key=(lambda z: -z.score_cand))
                _k_c = _reduce_counts[_ii]
                inc_cands = _k_cands[:-_k_c] if _k_c>0 else _k_cands
                _survive_cands.extend(inc_cands)
                _counts2[_ii] = len(inc_cands)
            zlog(f"Reduce by spath with {_query_spath_spec}: {len(_s0_cands)}/{len(_s1_cands)} -> {len(_survive_cands)} ;;; {[(k,a,b,c) for k,a,b,c in zip(_sg_keys, _counts0, _counts1, _counts2)]}")
            query_ret[1] = sorted(_survive_cands, key=(lambda z: -z.score_cand))
            # breakpoint()
        # reselect v2
        _query_reselect_spec2 = conf.query_reselect_spec2
        if _query_partial and _query_reselect_spec2[0] > 0:  # enabled
            _resel2_count, _resel2_sig, _resel2_alpha = _query_reselect_spec2
            if _resel2_alpha < 1.:
                _resel2_alpha = int(_resel2_count * _resel2_alpha)
            # --
            _s0_cands = sorted(cands0_alink, key=(lambda z: -z.score_cand))  # note: simply select from all
            _s0_groups = self.group_by_spath(_s0_cands, _resel2_sig)
            _sig_countO, _sig_countS, _sig_countA = Counter(), Counter(), Counter()
            _survive_cands = []
            for _ii, _ss in enumerate(_s0_cands):
                _sp = _ss.sp
                _sig_countA[_sp] += 1
                if _ii < _resel2_count:
                    _sig_countO[_sp] += 1
                if _sig_countS[_sp] < _resel2_alpha:
                    _survive_cands.append(_ss)
                    _sig_countS[_sp] += 1
                if len(_survive_cands) >= _resel2_count:
                    break
            _sig_keys = sorted(_sig_countA.keys(), key=(lambda x: -_sig_countA[x]))
            _counts = [sum(z.values()) for z in [_sig_countA, _sig_countO, _sig_countS]]
            zlog(f"After reselect2 [{_counts}]: {[(k, _sig_countA[k], _sig_countO[k], _sig_countS[k]) for k in _sig_keys]}")
            query_ret[1] = sorted(_survive_cands, key=(lambda z: -z.score_cand))
            # breakpoint()
            # zzpp = (lambda x: (print([x.sp, x.score_cand]), print(x.alink.str_auto())))
            # zzpp(_s0_cands[0])
        # simply re-selecting:
        _query_reselect_spec = conf.query_reselect_spec
        if _query_partial and _query_reselect_spec[0] > 0:  # enabled
            _resel_count, _resel_sig, _resel_alpha = _query_reselect_spec
            # --
            # _s0_cands = [z for z in cands0_alink if z.sent.dsids in q_hit_sents]  # all selected-sent cands
            _s0_cands = list(cands0_alink)  # note: simply select from all
            _s0_groups = self.group_by_spath(_s0_cands, _resel_sig)
            _sg_keys = sorted(_s0_groups.keys(), key=(lambda k: -len(_s0_groups[k])))  # sort by count!
            _counts0 = np.asarray([len(_s0_groups[k]) for k in _sg_keys])
            _trg_weights = np.asarray([len(_s0_groups[k]) ** _resel_alpha for k in _sg_keys])
            _trg_weights = _trg_weights / _trg_weights.sum()  # [...]
            _trg_counts = self.round_counts(_trg_weights * _resel_count)
            # get final results
            _survive_cands = []
            for _ii, _kk in enumerate(_sg_keys):
                _k_cands = sorted(_s0_groups[_kk], key=(lambda z: -z.score_cand))
                inc_cands = _k_cands[:_trg_counts[_ii]]
                _survive_cands.extend(inc_cands)
            _scc = Counter([z.sp for z in _survive_cands])
            # further counts
            # for _sig in ['raw', 'direct', 'count', 'nope']:
            for _sig in ['direct', 'count', 'nope']:
                _sg0 = self.group_by_spath(_s0_cands, _sig)  # assign sigs
                _sg1 = self.group_by_spath(_survive_cands, _sig)  # assign sigs
                _all_sigs = sorted(_sg0.keys(), key=(lambda k: -len(_sg0[k])))  # all keys
                _tmp_counts = [(k,len(_sg0[k]),len(_sg1.get(k,[]))) for k in _all_sigs]
                zlog(f"After reselect {len(_s0_cands)}->{len(_survive_cands)}: counts [{_sig}] are: {_tmp_counts}")
            zlog(f"After reselect [sp]: {_scc}")
            # --
            query_ret[1] = sorted(_survive_cands, key=(lambda z: -z.score_cand))
            # breakpoint()
        # --
        # then prepare the query insts
        kept_items = set()  # ids of the kept frames and alinks (could be predicted ones)
        if _query_partial:  # has alink-query
            for cand_alink in query_ret[1]:
                assert cand_alink.type == 'alink'
                _alink = cand_alink.alink
                kept_items.update([id(_alink), id(_alink.main), id(_alink.arg)])
        # --
        for doc in all_docs:
            for sent in doc.sents:
                _cur_keep_preds = _query_keep_preds and (sent.dsids in q_hit_sents)
                frame_hs, frame_ts = sent.get_frames(cates=cateHs), sent.get_frames(cates=cateTs)
                # --
                # storing predicted spans!
                if conf.ann_phase2:
                    _key2 = (lambda _f: (_f.mention.get_span() + (_f.cate,)))  # no label!
                    sent.info['ann2_ks'] = list({_key2(z) for z in (frame_hs + frame_ts)})
                # --
                processed_frames = set()
                for one_h in frame_hs:
                    if id(one_h) in processed_frames: continue
                    processed_frames.add(id(one_h))
                    h_is_past = not one_h.info.get('is_pred', False)  # past annotated
                    if id(one_h) in kept_items or (conf.qann_with_past and h_is_past) or (_cur_keep_preds and one_h.label != self.LAB_NIL):  # keep it if kept or past
                        if h_is_past:  # mark as past!
                            one_h.set_label(ext_helper.PAST_PREFIX + one_h.label)
                        for one_alink in one_h.get_args():
                            a_is_past = not one_alink.info.get('is_pred', False)
                            if id(one_alink) in kept_items or (conf.qann_with_past and a_is_past):
                                if a_is_past:  # mark as past!
                                    assert one_alink.label != self.LAB_PA
                                    one_alink.set_label(ext_helper.PAST_PREFIX + one_alink.label)
                            elif _cur_keep_preds and one_alink.label != self.LAB_NIL:
                                pass
                            else:  # otherwise delete!
                                one_alink.del_self()
                    else:  # otherwise delete!
                        one_h.del_self()
                for one_t in frame_ts:  # no processings of args for T!
                    if id(one_t) in processed_frames: continue
                    processed_frames.add(id(one_t))
                    t_is_past = not one_t.info.get('is_pred', False)  # past annotated
                    if id(one_t) in kept_items or (conf.qann_with_past and t_is_past) or (_cur_keep_preds and one_t.label != self.LAB_NIL):  # keep it if kept or past
                        if t_is_past:
                            one_t.set_label(self.PAST_PREFIX + one_t.label)
                    else:  # otherwise delete!
                        one_t.del_self()
        # --
        # mark queries!
        cc.update({'q_budget': 0, 'q_budgetA': 0})
        if not _query_partial:  # full sents
            for one_cand in query_ret:
                assert one_cand.type == 'sent'
                cc['q_budget'] += one_cand.budget
                frame = one_cand.sent.make_frame(0, len(one_cand.sent), ext_helper.LAB_QUERY, frame_cate)
                frame.info[self.KEY_PA] = True  # mark UNK args
                frame.score = one_cand.score_cand  # set utility score to frame.score!
                frame.info['q_full'] = True  # full-level query!
        else:  # tok & alink
            query_toks, query_alinks = query_ret
            tok_maps = {}
            if conf.query_no_repeatq:  # no repeat for tok and alink
                for one_cand in query_alinks:
                    for one_item in [one_cand.alink.main, one_cand.alink.arg]:
                        _sent = one_item.mention.sent
                        _widx, _wlen = one_item.mention.get_span()
                        _v = "GP"[int(one_item.info.get('is_pred', False))]
                        for _ii in range(_widx, _widx+_wlen):
                            tok_maps[(id(_sent), _ii)] = _v
            for one_cand in query_toks:
                assert one_cand.type == 'tok'
                # --
                _v = tok_maps.get((id(one_cand.sent), one_cand.widx))
                if _v is not None:
                    if _v == 'G':
                        zwarn(f"Query for gold tok?: {one_cand}")
                    cc['q_repeat'] += 1  # ignore repeat!
                    continue
                # --
                cc['q_budget'] += one_cand.budget
                frame = one_cand.sent.make_frame(one_cand.widx, 1, ext_helper.LAB_QUERY, frame_cate)
                frame.info[self.KEY_PA] = True  # mark UNK args
                frame.score = one_cand.score_cand  # set utility score to frame.score!
                # todo(+N): avoid overlapping with alink's pred?
            if conf.query_entity_budget > 0:
                final_q_alinks = []
                hit_ent = defaultdict(int)
                hit_pair = set()
                keep_cand_ids = set()
                for one_cand in sorted(query_alinks, key=lambda x: -x.score_cand):
                    _keys = tuple(sorted([id(one_cand.alink.main), id(one_cand.alink.arg)]))
                    _keep = True
                    if hit_ent[_keys[0]]>=conf.query_entity_budget or hit_ent[_keys[1]]>=conf.query_entity_budget:
                        _keep = False  # our of budget
                    if one_cand.alink.label not in [self.LAB_NIL, self.LAB_PA]:  # keep a predicted one
                        _keep = True
                    if _keys in hit_pair:  # the other direction already get it!
                        _keep = False
                    if _keep:
                        hit_ent[_keys[0]] += 1
                        hit_ent[_keys[1]] += 1
                        hit_pair.add(_keys)
                        keep_cand_ids.add(id(one_cand))
                # --
                for one_cand in query_alinks:
                    if id(one_cand) in keep_cand_ids:
                        final_q_alinks.append(one_cand)
                    else:
                        one_cand.alink.del_self()
            else:
                final_q_alinks = query_alinks
            for one_cand in final_q_alinks:
                assert one_cand.type == 'alink'
                cc['q_budgetA'] += one_cand.budget
                if (not conf.query_keep_alink_preds) or (one_cand.alink.label in [self.LAB_NIL, self.LAB_PA]):
                    one_cand.alink.set_label(self.LAB_QUERY)  # simply change the label
                one_cand.alink.score = one_cand.score_cand
                one_cand.alink.main.info[self.KEY_PA] = True  # note: since we are still querying it!
                if conf.query_mentionq:
                    one_cand.alink.main.set_label(ext_helper.LAB_QUERY)
                    one_cand.alink.arg.set_label(ext_helper.LAB_QUERY)
        # --
        hit_doc_ids = set(z[0] for z in q_hit_sents.keys())
        if conf.qann_with_hit:
            ret_docs = [d for d in all_docs if d.id in hit_doc_ids]  # if there are queries
        else:
            ret_docs = all_docs  # still put them all
        cc['q_doc'] += len(hit_doc_ids)
        cc['q_sent'] += len(q_hit_sents)
        return ret_docs, q_hit_sents, cc

    # helper
    def group_by_spath(self, alink_cands, sig):
        for one_cand in alink_cands:
            _spath = one_cand.sent.tree_dep.get_path_between_mentions(
                one_cand.alink.main.mention, one_cand.alink.arg.mention)
            _LINK = '..'
            _all_sp = []
            for _s in sig.split(_LINK):
                if _s == 'raw':
                    _sp = '_'.join(_spath)
                elif _s == 'direct':
                    _sp = ''.join(['^' if z.startswith("^") else '-' for z in _spath])
                elif _s.startswith("count"):
                    if len(_s) > len("count"):
                        _m = min(len(_spath), int(_s[len("count"):]))
                    else:
                        _m = len(_spath)
                    _sp = str(_m)
                elif _s.startswith("rtop"):
                    _k = int(_s[len("rtop"):])
                    import torch
                    _arr = torch.tensor(one_cand['arr_strg'])
                    _arr[0] -= 100.  # excluding NIL
                    _ii = sorted(_arr.topk(_k)[1].tolist())
                    _sp = ":".join([str(z) for z in _ii])
                elif _s == 'nope':
                    _sp = ''  # simply one group
                elif _s == 'evt':
                    _sp = one_cand.alink.main.type
                elif _s == 'ent':
                    _sp = one_cand.alink.arg.type
                else:
                    raise NotImplementedError()
                _all_sp.append(_sp)
            one_cand.sp = _LINK.join(_all_sp)
        _sp_groups = {}  # put into groups
        for one_cand in alink_cands:
            if one_cand.sp not in _sp_groups:
                _sp_groups[one_cand.sp] = []
            _sp_groups[one_cand.sp].append(one_cand)
        return _sp_groups

    def round_counts(self, counts):
        _tmp_gen = Random.stream(Random.get_np_generator(len(counts)).random_sample)
        _ret = []  # int
        for _ii, _cc in enumerate(counts):
            _k_c = int(counts[_ii])
            _k_c = _k_c + int(next(_tmp_gen) < (counts[_ii] - _k_c))  # consider the fraction!
            _ret.append(_k_c)
        return _ret

    # --
    # 2./3. ann & comb

    # make a new frame in q_sent according to r_frame
    def _make_frame(self, q_sent, r_frame, r_map):
        ret = r_map.get(id(r_frame))  # sometimes previously might hit this!
        if ret is None:
            _widx, _wlen = r_frame.mention.get_span()
            ret = q_sent.make_frame(_widx, _wlen, r_frame.label, r_frame.cate)
            ret.info.update(r_frame.info)
            ret.info[self.KEY_PA] = True  # note: mark partial!
            r_map[id(r_frame)] = ret  # note: record this to avoid repeating!!
        return ret

    # make a new alink
    def _make_alink(self, q_h, q_t, r_h, r_t):
        ext_helper = self.ext_helper
        if r_h is not None:
            assert ext_helper._score_match(q_h, r_h) == (1., 1.)
        if r_t is not None:
            assert ext_helper._score_match(q_t, r_t) == (1., 1.)
        if r_h is None or r_t is None:
            r_alinks = []  # allow explicit NIL-NIL link
        else:
            r_alinks = [z for z in r_h.args if z.arg is r_t]
        # --
        # remove existing ones
        q_alinks = [z for z in q_h.args if z.arg is q_t]
        if len(q_alinks) > 0:  # sometimes there can be repeated anns because of mention span wrong prediction
            if sorted([z.label for z in q_alinks]) != sorted([z.label for z in r_alinks]):
                zwarn(f"Remove existing alinks: {q_alinks} vs {r_alinks}")
            for q_arg in q_alinks:
                q_arg.del_self()
        # --
        if len(r_alinks) > 1:
            zwarn(f"Find multiple alinks: {r_alinks}")
        ret = None
        if len(r_alinks) == 0:  # no alink
            ret = q_h.add_arg(q_t, self.LAB_NIL)
        else:
            for r_arg in r_alinks:  # add them all!
                ret = q_h.add_arg(q_t, r_arg.label)
                ret.info.update(r_arg.info)
        # --
        return ret

    # simul annotate alink; return cost
    def _simul_ann_alink(self, q_head, q_tail, ref_map, hit_ones, hit_sents, map_corr, map_ref):
        conf: ALZrelConf = self.conf
        ext_helper = self.ext_helper
        cateHs, cateTs = conf.rconf.cateHs, conf.rconf.cateTs
        frame_cate = ext_helper.conf.tconf.frame_cate
        _ann_fix, _ann_nil, _ann_nf_alink = conf.ann_fix_frames, conf.ann_nil_frames, conf.ann_nf_alink
        _wP, _wF = conf.ann_frame_cost
        # --
        # first check frames
        cost_fix_frame = 0.
        check_results = []
        for one_q_frame, one_cate in zip([q_head, q_tail], [cateHs, cateTs]):
            # find ref frames
            q_widx, q_wlen = one_q_frame.mention.get_span()
            q_sent = one_q_frame.sent
            hit_sents.add(id(q_sent))
            r_sent = ref_map[q_sent.doc.id].sents[q_sent.sid]  # must be there
            best_ref, best_ref_score = ext_helper._match_frame(one_q_frame, r_sent, one_cate)
            # process
            if not one_q_frame.info.get('is_pred', False):  # already past gold ones!
                assert best_ref is not None and best_ref_score == (1., 1.)  # note: must be gold!
                res_q_frame, res_code = one_q_frame, "G"  # gold
            else:  # handle predicted ones
                if id(one_q_frame) in map_corr:  # already fixed before (no repeated removing!)
                    res_q_frame, res_code = map_corr[id(one_q_frame)]  # directly use the previous one!
                else:  # no fixed before
                    if best_ref_score == (1., 1.):  # correct!
                        assert best_ref is not None
                        res_q_frame = self._make_frame(q_sent, best_ref, map_ref)
                        hit_ones.add(id(best_ref))
                        res_code = "P"  # perfect prediction
                        cost_fix_frame += best_ref.mention.wlen * _wP  # note: extra cost for adding!
                    else:  # fix? (with extra cost for fixing!)
                        if not _ann_fix:  # None if no fix!
                            res_q_frame, res_code = None, "D"  # delete
                        else:
                            if best_ref is None:  # removed: no fixing or no ref
                                best_ref2, _ = ext_helper._match_frame(one_q_frame, r_sent, frame_cate)
                                if (not _ann_nil) or best_ref2 is not None:
                                    res_q_frame, res_code = None, "D"  # avoid excluding other frames!
                                else:  # add explicit NIL
                                    res_q_frame = q_sent.make_frame(q_widx, q_wlen, ext_helper.LAB_NIL, one_cate)
                                    res_q_frame.info[self.KEY_PA] = True  # note: mark partial!
                                    for ii in range(q_widx, q_widx+q_wlen):
                                        hit_ones.add((id(q_sent), ii))
                                    res_code = 'N'
                            else:
                                res_q_frame = self._make_frame(q_sent, best_ref, map_ref)
                                hit_ones.add(id(best_ref))
                                res_code = "F"  # fixed
                        cost_fix_frame += (0 if res_q_frame is None else res_q_frame.mention.wlen) * _wF
                    map_corr[id(one_q_frame)] = (res_q_frame, res_code)
            check_results.append((best_ref, res_q_frame, res_code))
        # --
        # then for the alink!
        h_ref, h_q, h_code = check_results[0]
        t_ref, t_q, t_code = check_results[1]
        ret_code = h_code + t_code
        if h_q is not None and t_q is not None and (_ann_nf_alink or 'N' not in ret_code):  # require both to be aligned!
            self._make_alink(h_q, t_q, h_ref, t_ref)
        return cost_fix_frame, 1., ret_code  # always cost 1 for alink!

    # second phase of extra annotation (maybe mixed with 'budgetA:0' for no-alinks in phase1)
    # new: ann_phase2:a ann_phase2_onlynew:1
    # ratio: ann_phase2:r selv2_record_final_ratio:0 ann_phase2_ratio_specs:1,1
    # thresh: ann_phase2:t selv2_record_final_ratio:1
    def _ann_phase2(self, query_insts, ref_map, last_model):
        from mspx.data.rw import ReaderGetterConf, WriterGetterConf
        conf: ALZrelConf = self.conf
        ext_helper = self.ext_helper
        cateHs, cateTs = conf.rconf.cateHs, conf.rconf.cateTs
        _ignore_flabs = [ext_helper.LAB_NIL, ext_helper.LAB_QUERY, ext_helper.PAST_PREFIX+ext_helper.LAB_NIL]
        _pp = self.PAST_PREFIX
        # --
        # inference with annotated frames!
        _tmp_in, _tmp_out = "_phase2.in.json", "_phase2.out.json"  # todo(+N): simply temp file at curr root
        _mark = 'INC_ann2'
        with WriterGetterConf().get_writer(output_path=_tmp_in) as writer:
            for q_sent in yield_sents(query_insts):
                if any(not z.label.startswith(_pp) for z in q_sent.get_frames(cates=cateHs+cateTs)):
                    q_sent.info[_mark] = True
            writer.write_insts(query_insts)
        if conf.ann_phase2_onlyann:  # no more prediction for mentions
            _specs = self.specs_infQ + f" rel0.inc_nil_frames:0 test0.inst_f:sentK:{_mark} test0.tasks:enc0,{self.name}"
        else:
            _specs = self.specs_infQ + f" rel0.inc_nil_frames:0 test0.inst_f:sentK:{_mark}"
        last_model = self.prep_query_model(last_model)
        self._do_inf(last_model, _tmp_in, _tmp_out, _specs, '', '')
        inf_insts = list(ReaderGetterConf().get_reader(input_path=_tmp_out))
        # again read cands!
        cc = Counter()
        cc_code = Counter()
        all_cands = []
        all_orig_keys = set()
        for q_sent, i_sent in yield_sent_pairs(query_insts, inf_insts):
            all_orig_keys.update([(id(i_sent),)+tuple(z) for z in q_sent.info['ann2_ks']])
            cc['sent0'] += 1
            if i_sent.info.get(_mark):
                cc['sentQ'] += 1  # queried sent!
                hs = [z for z in i_sent.get_frames(cates=cateHs) if z.label not in _ignore_flabs]
                ts = [z for z in i_sent.get_frames(cates=cateTs) if z.label not in _ignore_flabs]
                for one_h in hs:
                    _maph = {id(z.arg): z for z in one_h.args}
                    for one_t in ts:
                        if conf.no_self and one_h is one_t: continue  # no self-link
                        if id(one_t) not in _maph:  # probably truncated ...
                            zwarn(f"Cannot find {one_h} -> {one_t}")
                            continue
                        cc['alink0'] += 1
                        _alink = _maph[id(one_t)]  # must be there!
                        if not _alink.info.get('is_pred'):  # already annotated!
                            cc['alink0_aa'] += 1
                            continue
                        _arr_strg = _alink.arrs.get(self.KEY_STRG)
                        if _arr_strg is None:  # sometimes may happen, for e.g. due to truncating
                            cc['alink0_nostrg'] += 1
                            continue
                        cc['alink0_zcand'] += 1
                        alink_cand = ZObject(type='alink', gid=id(i_sent), sent=i_sent,
                                             alink=_alink, budget=1, budgetR=1, arr_strg=_arr_strg)
                        all_cands.append(alink_cand)
        # --
        # check cands!
        _sel = self.selector
        _sel._select_score(all_cands, False)
        all_cands.sort(key=lambda x: -x.score_cand)
        # --
        _key2 = (lambda _f: (id(_f.mention.sent),) + _f.mention.get_span() + (_f.cate,))  # no label!
        all_check_cands = [z for z in all_cands if any(_key2(zz) not in all_orig_keys for zz in [z.alink.main, z.alink.arg])] if conf.ann_phase2_onlynew else all_cands
        # --
        _p2_mode = conf.ann_phase2
        _p2_thr = _p2_ratio = None
        if _p2_mode == 'a':  # all the ones with new frames
            final_cands = all_check_cands
        elif _p2_mode == 't':  # thresh
            _p2_thr = _sel.rthr_history[1][-1]  # note: specific!!
            final_cands = [z for z in all_check_cands if z.score_cand >= _p2_thr]
        elif _p2_mode == 'r':  # ratio
            _p2_ratio = _sel.ratio_history[1][-1]  # note: specific!!
            _ratio_specs = conf.ann_phase2_ratio_specs
            _inc_ids = set()
            if _ratio_specs[0]:  # apply the ratio to each sent
                _groups = defaultdict(list)
                for _one_cand in all_check_cands:
                    _groups[_one_cand.gid].append(_one_cand)
                for _one_cands in _groups.values():
                    _ceil = int(math.ceil(_p2_ratio * len(_one_cands)))
                    _inc_ids.update([id(z) for z in _one_cands[:_ceil]])
            if _ratio_specs[1]:  # apply the ratio overall
                _ceil = int(math.ceil(_p2_ratio * len(all_check_cands)))
                _inc_ids.update([id(z) for z in all_check_cands[:_ceil]])
            # --
            final_cands = [z for z in all_check_cands if id(z) in _inc_ids]
        else:
            raise NotImplementedError(f"UNK phase2 mode of {_p2_mode}")
        cc.update({'al0_all': len(all_cands), 'al0_check': len(all_check_cands), 'al0_zfinal': len(final_cands)})
        # --
        # process phase2 query
        kept_items = set(sum(([id(z.alink), id(z.alink.main), id(z.alink.arg)] for z in final_cands), []))
        for sent in yield_sents(inf_insts):
            frame_hs, frame_ts = sent.get_frames(cates=cateHs), sent.get_frames(cates=cateTs)
            processed_frames = set()
            for one_fs in [frame_hs , frame_ts]:
                for one_f in one_fs:
                    if id(one_f) in processed_frames: continue
                    processed_frames.add(id(one_f))
                    not_pred = not one_f.info.get('is_pred', False)
                    if id(one_f) in kept_items or not_pred:  # keep
                        for one_alink in one_f.get_args():
                            a_not_pred = not one_alink.info.get('is_pred', False)
                            if id(one_alink) in kept_items or a_not_pred:
                                pass
                            else:
                                one_alink.del_self()
                    else:  # otherwise delete!
                        one_f.del_self()
        for one_cand in final_cands:
            assert one_cand.type == 'alink'
            one_cand.alink.set_label(self.LAB_QUERY)  # simply change the label
            one_cand.alink.score = one_cand.score_cand
            one_cand.alink.main.info[self.KEY_PA] = True  # note: since we are still querying it!
        # --
        # annotate again!
        hit_ones = set()  # already hit toks or frames
        hit_sents = set()
        map_corr = {}  # id(pred) -> list[correct frames]
        map_ref = {}  # id(ref) -> frame(corr-pred)
        for cand in final_cands:
            q_alink = cand.alink
            q_alink.del_self()  # no matter what delete self!
            q_cost, q_costA, _code = self._simul_ann_alink(q_alink.main, q_alink.arg, ref_map, hit_ones, hit_sents, map_corr, map_ref)
            cc['cost_tok'] += q_cost  # further potential token cost!
            cc['cost_alink'] += q_costA
            if "D" not in _code:  # not deleted
                cc[f'p2ann_AannA'] += 1  # add alink
                if "N" not in _code:  # not N-N
                    cc['p2ann_AannV'] += 1  # add valid-actual pair
            cc_code[_code] += 1
        zlog(f"Simul_ann(phase2) alink code: {cc_code}")
        zlog(f"Perform phase2-ann with {_p2_mode}({_p2_ratio}/{_p2_thr}): {ZHelper.resort_dict(cc)}")
        # breakpoint()
        # --
        return inf_insts

    # simulated ann according to ref
    def do_simul_ann(self, query_insts, ref_map, last_model=None):
        conf: ALZrelConf = self.conf
        ext_helper = self.ext_helper
        cc = Counter()
        cc_code = Counter()
        cateHs, cateTs = conf.rconf.cateHs, conf.rconf.cateTs
        frame_cate = ext_helper.conf.tconf.frame_cate
        # --
        # all frames
        all_frames = [z for z in yield_frames(query_insts, cates=frame_cate) if z.label==self.LAB_QUERY]
        all_frames.sort(key=(lambda f: f.score), reverse=True)
        all_full_frames, all_partial_frames = [z for z in all_frames if z.info.get('q_full')], \
                                              [z for z in all_frames if not z.info.get('q_full')]
        if len(all_full_frames) > 0 and len(all_partial_frames) > 0:
            zwarn(f"Mixed querying?? full={len(all_full_frames)} and partial={len(all_partial_frames)}")
        # --
        # annotate!
        all_budget, all_budgetA = self.curr_budget, self.curr_budgetA
        remaining_budget, remaining_budgetA = all_budget, all_budgetA
        hit_ones = set()  # already hit toks or frames
        hit_sents = set()
        map_corr = {}  # id(pred) -> list[correct frames]
        map_ref = {}  # id(ref) -> frame(corr-pred)
        # first handle full ann frames
        for q_frame in all_full_frames:
            cc['aF_Fall'] += 1
            q_frame.del_self()
            if remaining_budget <= 0:  # skip if tok-budget run out!
                cc['aF_Fdel'] += 1  # note: with "full_qextra", there could be some extra toks
                continue
            cc['aF_Fann'] += 1
            q_sent = q_frame.sent
            if not conf.use_gold_mention:
                assert len(q_sent.get_frames(cates=frame_cate)) == 0, "Bad mixing of frames!"
                q_cost = ext_helper._simul_ann_frame(q_frame, ref_map, hit_ones, hit_sents, frame_marks=[self.KEY_PA])
                remaining_budget -= q_cost  # note: always annotate it regardless of budget!
            # --
            # newly added ones
            frame_hs, frame_ts = q_sent.get_frames(cates=cateHs), q_sent.get_frames(cates=cateTs)
            for one_h in frame_hs:
                if one_h.label == ext_helper.LAB_NIL: continue  # ignore NIL
                for one_t in frame_ts:
                    if one_t.label == ext_helper.LAB_NIL: continue  # ignore NIL
                    if conf.no_self and one_h is one_t: continue  # no self-link
                    cc['aF_Aall'] += 1
                    if remaining_budgetA > 0:
                        cc['aF_Aann'] += 1
                        q_cost, q_costA, _ = self._simul_ann_alink(
                            one_h, one_t, ref_map, hit_ones, hit_sents, None, None)
                        assert q_cost == 0, "Should be NO tok-cost!"
                        remaining_budgetA -= q_costA  # note: always annotate it regardless of budget!
                    else:  # skip out-of-budget ones even in full mode!
                        cc['aF_Adel'] += 1
        # then handle alinks!
        all_alinks = [z2 for z in yield_frames(query_insts, cates=cateHs) for z2 in z.args if z2.label==self.LAB_QUERY]
        all_alinks.sort(key=(lambda a: a.score), reverse=True)
        for q_alink in all_alinks:
            cc['aP_Aall'] += 1
            q_alink.del_self()  # no matter what delete self!
            if remaining_budget <= 0 or remaining_budgetA <= 0:  # skip since no budgets
                cc['aP_Adel'] += 1
            else:
                q_cost, q_costA, _code = self._simul_ann_alink(
                    q_alink.main, q_alink.arg, ref_map, hit_ones, hit_sents, map_corr, map_ref)
                cc['aP_Aann'] += 1
                cc_code[_code] += 1
                if "D" not in _code:  # not deleted
                    cc[f'aP_AannA'] += 1  # add alink
                    if "N" not in _code:  # not N-N
                        cc['aP_AannV'] += 1  # add valid-actual pair
                cc['aP_Atcost'] += q_cost  # tok-cost when ann alink
                remaining_budget -= q_cost
                remaining_budgetA -= q_costA
        # note: there might be extra pred frames, simply ignore them when combine!
        # then handle the remaining frames
        for q_frame in all_partial_frames:
            cc['aP_Fall'] += 1
            q_frame.del_self()  # no matter what delete self!
            if remaining_budget <= 0:  # skip since no budgets
                cc['aP_Fdel'] += 1
            else:
                cc['aP_Fann'] += 1
                q_cost = ext_helper._simul_ann_frame(q_frame, ref_map, hit_ones, hit_sents, frame_marks=[self.KEY_PA])
                remaining_budget -= q_cost
        # finally cleanup and remove all predicted frames!
        final_all_frames = list(yield_frames(query_insts, cates=frame_cate))
        for one_frame in final_all_frames:
            assert one_frame.label != self.LAB_QUERY, "Should have handled all queries!"
            if one_frame.info.get('is_pred', False):
                one_frame.del_self()  # deleting pred ones!
                cc['q_pred_del'] += 1
        # --
        cc['q_sent_hit'] = len(hit_sents)
        cc['q_predcorrL'] = len(map_corr)  # should be equal to "q_pred_del"
        cc.update({"budgetTC": all_budget-remaining_budget, "budgetTR": remaining_budget,
                   "budgetAC": all_budgetA-remaining_budgetA, "budgetAR": remaining_budgetA})
        zlog(f"Simul_ann with partial's alink code: {cc_code}")
        # --
        if conf.ann_phase2 and len(all_alinks) > 0:  # only in partial mode!
            new_insts = self._ann_phase2(query_insts, ref_map, last_model)
            return new_insts, cc
        # --
        return cc

    # combine new insts into trg
    def _do_comb(self, ann_inst, trg_map, cc):
        conf: ALZrelConf = self.conf
        ext_helper = self.ext_helper
        cateHs, cateTs = conf.rconf.cateHs, conf.rconf.cateTs
        frame_cate = ext_helper.conf.tconf.frame_cate
        # --
        # use allowed pair (h->t) to constrict candidates
        from mspx.tasks.zrel.cons_table import CONS_SET
        _allowed_pairs = set()
        for _kk in ['evt', 'rel']:
            for h, r, t in CONS_SET[_kk]:
                _allowed_pairs.add((h, t))
        # --
        # todo(+N): for full mode, need to explicitly add NILs from the outside
        # --
        # first with frames
        frame_map = {}
        for a_frame in yield_frames(ann_inst, cates=frame_cate):
            a_sent = a_frame.sent
            t_sent = trg_map[a_sent.doc.id].sents[a_sent.sid]  # must be there
            t_frame = ext_helper._comb_frame(a_frame, t_sent, cc)
            frame_map[id(a_frame)] = (a_frame, t_frame)
        # --
        # then with alinks
        _pp = self.PAST_PREFIX
        for a_frame, t_frame in frame_map.values():
            if a_frame.cate not in cateHs: continue
            for a_arg in a_frame.args:
                if a_arg.arg.cate not in cateTs: continue
                _, t_tail = frame_map[id(a_arg.arg)]  # must be there
                # --
                if t_frame is None or t_tail is None:
                    zwarn(f"Failed to add alink because no valid mentions for: {a_frame}")
                    cc['combA_fail'] += 1
                    continue
                # --
                _label = a_arg.label
                if _label.startswith(_pp):  # check previous existing
                    _trg_alinks2 = [z for z in t_frame.args if z.arg is t_tail]
                    if len(_trg_alinks2) == 0:
                        zwarn(f"Ignoring since no past annotation for: {a_arg}")
                        cc['combA_aaN'] += 1
                    elif any(z.label == _label[len(_pp):] for z in _trg_alinks2):
                        cc['combA_aa'] += 1  # already annotated
                    else:  # simply correct the first one!
                        zwarn(f"Correction against past annotation: {a_arg} vs {_trg_alinks2[0]}")
                        _trg_alinks2[0].set_label(_label[len(_pp):])
                        _trg_alinks2[0].clear_cached_vals()
                        cc['combA_aaC'] += 1
                elif _label == self.LAB_QUERY:  # ignore unannotated ones
                    cc['combA_unn'] += 1
                else:
                    new_alink = self._make_alink(t_frame, t_tail, a_arg.main, a_arg.arg)
                    cc['combA_new'] += 1
                    cc['combA_newV'] += int(all([z.label != ext_helper.LAB_NIL for z in [a_arg.main, a_arg.arg]]))  # valid new alink!
                    cc['combA_nnew'] += int(new_alink is not None and new_alink.label != self.LAB_NIL)
                    cc['combA_pnew'] += int((t_frame.label, t_tail.label) in _allowed_pairs)  # allowed pair
        # --

    # --
    # 5. training
    # ...

# --
# b mspx/tools/al/tasks/zrel:
