#

__all__ = [
    "SelectorConf", "Selector",
]

import os
from typing import List
import numpy as np
import math
from collections import Counter
from itertools import chain
from mspx.utils import Conf, Configurable, zwarn, zlog, ZHelper, Random, Constants, ZObject

# --
# selector

class SelectorConf(Conf):
    def __init__(self):
        self.sel_seed = 0  # for random selecting
        self.weight_rand = 0.  # weight for random
        self.strg_use_log = False  # log-prob as score
        self.strg_f = "margin"  # how to calculate utility score based on strg
        self.strg_bias = 0.  # additional bias for scores
        self.aggr_spec = [0., 1.]  # (k, alpha) mean of topk ones for aggregation (0 means mean-all)
        self.aggrH = -1  # two-layer aggr for alink (-1 means nope, 0 means avg)
        self.span_range = [0, 0]  # positive if applied
        self.cluster_k = 0  # number of clusters (using if >1)
        self.cluster_beta = 10.  # take first beta*budget for clustering
        self.score_filter = "True"  # "lambda s: ..."
        # --
        # extra ones for selv2
        self.selv2_arM = -1.  # momentum for auto-ratios, valid if >=0
        self.selv2_ar_thr = 0.5  # count as correct only if score<this!
        self.selv2_ar_thr2 = -1.  # count as correct if argmax(prob)>=this!
        self.selv2_ar_method = 'logr'  # estimate method: logr, local, point
        self.selv2_ar_gmin = 1.  # minimum group keep-rate
        self.selv2_ar_reb = 0  # re-balance for logr?
        self.selv2_ar_winA = 0.5  # for accD: local window-ratio (of low-utility side)
        self.selv2_ar_winB = 0.1  # for accD: local window-size (percentage of data) for estimation
        self.selv2_ar_adjust = [1., 0.]  # ax+b
        self.selv2_ar_logtransform = True  # use log(x) as feature rather than raw ones
        self.selv2_ratio_specs = [1, 1]  # [sent-wise?, overall?]
        self.selv2_record_final_ratio = False  # use final ratio or the specified/auto one?
        self.selv2_random_sent = False  # first random sentence!
        self.selv2_doc_sel = False  # select full docs
        self.selv2_doc_sel_ratio = 1.  # select full docs ratio
        # check query's accuracies?
        self.selv2_check_query = True  # use ref!
        self.selv2_oracle_ratio = False  # for checking/debugging purpose!
        # auto comb params
        self.selv2_comb_load = []  # loading for previous states?
        self.selv2_autocomb_specs = [1., 0.1, -1.]  # val0, val1, switch-thr(eval-t1)
        # special for wrong-frame
        self.selv2_nf_specs = [0, 1, 0., 0.]  # feature_num(0 means nope!), ignore_non_frame?, tok-ext, ar-ext
        self.selv2_nf_winext = [0,0]  # extending to surrounding tokens
        self.selv2_nf_amode = 'pnf'  # pnf or pacc for alink's nf
        # self.selv2_wf_specs = [0, 0., 0.]  # switch, pacc<=invalid_thr, ar-ext

@SelectorConf.conf_rd()
class Selector(Configurable):
    def __init__(self, conf: SelectorConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: SelectorConf = self.conf
        # --
        self.strg_f = self.get_strg_f('bald' if conf.strg_f.startswith('bald') else conf.strg_f)
        if conf.strg_f.startswith('bald'):
            np.seterr(under='ignore')  # ignore underflow
        self.gen = Random.get_np_generator(conf.sel_seed)
        self.score_filter = eval("lambda s: " + conf.score_filter)
        # histories
        _MAXC = 10  # should be enough!
        self.ratio_history = [[] for _ in range(_MAXC)]  # for auto-ratio
        self.rthr_history = [[] for _ in range(_MAXC)]  # corresponding thresh for auto-ratio
        if conf.selv2_comb_load:  # with loading
            _load = [float(z) for z in conf.selv2_comb_load]
            self.comb_stat_history = _load
            zlog(f"Load for auto_comb {self.comb_stat_history}")
        else:  # start empty!
            self.comb_stat_history = []  # for auto-comb
        # --

    # --
    # scoring functions
    @staticmethod
    def _score_margin(s):  # -(top1 - top2)
        _ap = np.argpartition(s, -2, axis=-1)[..., -2:]  # [*, 2]
        _ss0 = np.take_along_axis(s, _ap, -1)  # [*, 2]
        ret = - np.abs(_ss0[..., 0] - _ss0[..., 1])
        return ret

    @staticmethod
    def _score_entropy(s):
        from scipy.stats import entropy
        return entropy(s+1e-8, axis=-1)

    @staticmethod
    def _score_prob(s):  # - max_prob
        return -np.max(s, axis=-1)

    @staticmethod
    def _score_bald(s):  # entropy(avg) - avg(entropy)
        from scipy.stats import entropy
        return entropy(s.mean(-2)+1e-8, axis=-1) - entropy(s+1e-8, axis=-1).mean(-1)

    def get_strg_f(self, key: str):
        if hasattr(self, f"_score_{key}"):
            return getattr(self, f"_score_{key}")
        return ZHelper.eval_ff(key, default_args='s')

    # get one argmax
    @staticmethod
    def get_strg_argmax(arr_strg, return_val=False):
        if len(arr_strg.shape) == 1:
            best_idx = arr_strg.argmax()
            return arr_strg[best_idx] if return_val else best_idx
        else:  # multiple runs for bald
            best_idx = arr_strg.mean(0).argmax()
            return arr_strg.mean(0)[best_idx] if return_val else best_idx

    # mean of topk
    def get_mean_topk(self, s, spec):
        k, alpha = int(spec[0]), float(spec[1])
        _len = s.shape[-1]
        if k > 0 and k < _len:  # use topk
            _ap = np.argpartition(s, -k, axis=-1)[..., -k:]  # [*, K]
            scores = np.take_along_axis(s, _ap, -1)  # [*, K]
        else:  # use all
            scores = s
        return np.sum(scores, -1) / (scores.shape[-1] ** alpha)  # [*]
    # --

    def _prep_score(self, arr_strg, trg_dim: int):
        conf: SelectorConf = self.conf
        if conf.strg_f == 'margin':
            _penalty = np.asarray((arr_strg.sum(-1) <= 0), dtype=arr_strg.dtype)
        else:
            _penalty = 0.
        if conf.strg_use_log:  # using log-prob!
            arr_strg = arr_strg + 1e-8
            arr_strg = np.log(arr_strg / arr_strg.sum(-1, keepdims=True))
        arr_score = self.strg_f(arr_strg)  # [*]
        # note: down-weight strange ones, maybe by truncation in dpar
        arr_score -= _penalty
        if len(arr_score.shape) > trg_dim:  # aggregate!
            arr_score = self.get_mean_topk(arr_score, conf.aggr_spec)
        if conf.strg_bias != 0.:
            arr_score += conf.strg_bias
        return arr_score

    def _select_score(self, cands, score_randomly: bool):
        conf: SelectorConf = self.conf
        if len(cands) == 0:
            return  # no items to score!
        if score_randomly:  # only random: typically in the first iter!
            zlog("Simply select randomly!")
            arr_score = self.gen.random(len(cands))
        else:
            if any(z.arr_strg.shape != cands[0].arr_strg.shape for z in cands):
                zlog("Select with strg(A0)!")
                scores = []
                for one_cand in cands:  # loop!
                    _score = self._prep_score(one_cand.arr_strg, 0)  # []
                    scores.append(_score)
                arr_score = np.asarray(scores)
            else:
                zlog("Select with strg(A1)!")
                arr_strg = np.stack([z.arr_strg for z in cands], axis=0)  # [*, V]
                arr_score = self._prep_score(arr_strg, 1)  # [*]
            if conf.weight_rand > 0:
                arr_score += self.gen.random(len(arr_score)) * conf.weight_rand
        # --
        for i, c in enumerate(cands):
            c.score_cand = arr_score[i].item()
        # --

    def _select_group(self, cands, score_randomly: bool, ret_map=False):
        conf: SelectorConf = self.conf
        # put cands into groups by gid
        groups = {}  # gid -> [[group-cands], [arr_score], budget, score]
        for one_cidx, one_cand in enumerate(cands):
            gid = one_cand.gid
            if gid not in groups:
                groups[gid] = [[], [], 0., None]
            groups[gid][0].append(one_cand)  # note: candidates will still be sorted within group!
            groups[gid][1].append(one_cand.score_cand)
            groups[gid][2] += one_cand.budget
        if len(groups) == len(cands):  # no need to group
            for one_group in groups.values():
                assert len(one_group[1]) == 1
                one_group[3] = one_group[1][0]
        else:
            if score_randomly:
                arr_score = self.gen.random(len(groups))
                for one_ii, one_group in enumerate(groups.values()):
                    one_group[3] = arr_score[one_ii].item()
            else:
                for one_group in groups.values():
                    if conf.aggrH >= 0 and any(z.type=='alink' for z in one_group[0]):
                        ah_cands = {}  # id(Frame) -> []
                        for ah_cand in one_group[0]:
                            for _key in [id(ah_cand.alink.main), id(ah_cand.alink.arg)]:  # add for both ends!
                                if _key not in ah_cands:
                                    ah_cands[_key] = []
                                ah_cands[_key].append(ah_cand)
                        ah_scores = [self.get_mean_topk(np.asarray([z.score_cand for z in zz]), [conf.aggrH, 1.])
                                     for zz in ah_cands.values()]  # frame-level aggr
                        one_group[3] = self.get_mean_topk(np.asarray(ah_scores), conf.aggr_spec)  # sent-level
                        # breakpoint()
                    else:
                        one_group[3] = self.get_mean_topk(np.asarray(one_group[1]), conf.aggr_spec)
        if ret_map:
            return groups
        else:
            sorted_groups = sorted(groups.values(), key=(lambda x: x[-1]), reverse=True)
            return sorted_groups

    # special mode of span(subsequence)
    def _select_span(self, cands, r0: int, r1: int, score_randomly: bool):
        conf: SelectorConf = self.conf
        # looping for continuous spans
        last_gid, last_widx = None, 0
        span_cands = []  # new span cands
        curr_conti = []  # current continuous ones
        for one_cand in chain(cands, [None]):  # note: inputs must be ordered by widx!
            if one_cand is None or one_cand.gid != last_gid or one_cand.widx != last_widx + 1:
                if len(curr_conti) > 0:  # make new ones
                    _len = len(curr_conti)  # use all if small _len
                    for _r in range(min(r0, _len), min(r1, _len)+1):  # todo(+N): could be more efficient!
                        for _i in range(0, _len-_r+1):
                            _toks = curr_conti[_i:_i+_r]
                            assert len(_toks) == _r
                            new_score = self.get_mean_topk(np.asarray([z.score_cand for z in _toks]), conf.aggr_spec)
                            new_span_cand = ZObject(type='span', gid=_toks[0].gid, sent=_toks[0].sent,
                                                    span=(_toks[0].widx, _r), budget=_r,
                                                    score_cand=new_score, toks=_toks)
                            span_cands.append(new_span_cand)
                            if getattr(_toks[0], 'arr_hid', None) is not None:  # simply average!
                                new_span_cand.arr_hid = np.stack([z.arr_hid for z in _toks], 0).mean(0)
                    # clear!
                    curr_conti = []
                    last_gid, last_widx = None, 0
            if one_cand is not None:
                assert one_cand.type == 'tok'
                last_gid, last_widx = one_cand.gid, one_cand.widx
                curr_conti.append(one_cand)
        # --
        if score_randomly:
            arr_score = self.gen.random(len(span_cands))
            for one_ii, one_cand in enumerate(span_cands):
                one_cand.score_cand = arr_score[one_ii].item()
        return span_cands

    def _select_cluster(self, all_cands, cluster_budget):
        conf: SelectorConf = self.conf
        cluster_cands = []  # cands to cluster
        for one_cand in all_cands:
            if cluster_budget <= 0:
                break
            cluster_cands.append(one_cand)
            cluster_budget -= one_cand.budget
        # --
        clu_k = min(conf.cluster_k, len(cluster_cands))
        from sklearn.cluster import KMeans
        _input = np.asarray([z.arr_hid for z in cluster_cands])
        res = KMeans(n_clusters=clu_k, random_state=conf.sel_seed).fit(_input)  # reuse seed!
        final_cands = [[] for _ in range(clu_k)]
        for ii, ll in enumerate(res.labels_):  # put cands
            final_cands[ll].append(cluster_cands[ii])
        ret_cc = Counter({'clusterK': len(final_cands), 'clusterC0': len(all_cands),
                          'clusterC': len(cluster_cands)})
        return final_cands, ret_cc

    # final picking
    def _select_pick(self, final_sorted_cands, budget, score_randomly: bool):
        # sweeping them horizontally
        remaining_budget = budget
        ret0, curr_round = [], 0
        hit_toks = set()  # check non-overlapping spans!
        all_types = set(z2.type for z1 in final_sorted_cands for z2 in z1)
        assert len(all_types) <= 1, "Should not mix types!"
        while True:
            curr_cands = [z[curr_round] for z in final_sorted_cands if len(z)>curr_round]  # one round
            if len(curr_cands) == 0:  # no more cands
                break
            curr_cands.sort(key=(lambda x: -x.score_cand))
            for one_cand in curr_cands:
                if one_cand.type == 'span':
                    _keys = [(id(z.sent), z.widx) for z in one_cand.toks]
                    if any(k in hit_toks for k in _keys):
                        continue  # overlapping!
                    ret0.append(one_cand)
                    remaining_budget -= one_cand.budget
                    hit_toks.update(_keys)  # add the toks!
                else:  # note: other ones will not overlap!
                    ret0.append(one_cand)
                    remaining_budget -= one_cand.budget
                if remaining_budget <= 0:
                    break
            if remaining_budget <= 0:
                break
            curr_round += 1
        # --
        # final final filtering
        if score_randomly:  # no filtering for random!
            ret = ret0
        else:
            ret = [z for z in ret0 if self.score_filter(z.score_cand)]
        # --
        return ret, Counter({'input_cand': sum(len(z) for z in final_sorted_cands),
                             'final_cand0': len(ret0), 'final_cand': len(ret), 'final_round': curr_round})

    def select(self, cands, budget: int, budget_group: int, score_randomly: bool):
        conf: SelectorConf = self.conf
        cc = Counter()
        # --
        # get utility score
        self._select_score(cands, score_randomly)
        cc['cand_orig'] = len(cands)
        # --
        # constraining by groups
        if budget_group > 0:
            sorted_groups = self._select_group(cands, score_randomly)
            new_cands = []
            remaining_budget, remaining_budget_group = budget, budget_group
            for one_group in sorted_groups:
                one_cands, one_budget = one_group[0], one_group[2]
                remaining_budget -= one_budget
                remaining_budget_group -= 1
                new_cands.extend(one_cands)
                if remaining_budget <= 0 and remaining_budget_group <= 0:
                    break  # note: satisfying both!
            cc.update({'groupNum': len(sorted_groups), 'group0': budget_group,
                       'groupRB': remaining_budget, 'groupRBG': remaining_budget_group})
            cands = new_cands  # replace it!
        # --
        # special span-selection mode (inputs should be tokens!)
        span_r0, span_r1 = [max(1,z) for z in conf.span_range]
        use_span = (span_r1 > 1)
        cand_types = Counter(z.type for z in cands)
        if use_span:  # switch on
            if list(cand_types.keys()) == ["tok"]:
                span_cands = self._select_span(cands, span_r0, span_r1, score_randomly)
                cc['cand_span'] = len(span_cands)
                cands = span_cands
            else:
                zwarn(f"There are non-tok cands, skip span selection: {cand_types}")
        # --
        # clustering
        cands.sort(key=(lambda x: -x.score_cand))
        if conf.cluster_k > 1:
            cluster_budget = budget * conf.cluster_beta
            final_cands, clu_cc = self._select_cluster(cands, cluster_budget)  # should be sorted!
            cc += clu_cc
        else:
            final_cands = [cands]
        # --
        # final picking!
        ret, final_cc = self._select_pick(final_cands, budget, score_randomly)
        cc += final_cc
        zlog(f"Finishing selecting with {ZHelper.resort_dict(cc)}")
        return ret

    # --
    # select v2: first sent then items!
    def select_v2(self, all_sents, sent_budget: List, sent_aggr_df: float,
                  cand_items: List, dev_items: List, cand_sc: List, cand_ratios: List, cand_threshs: List,
                  score_randomly: bool, partial: bool, ratio_sentwise: bool,
                  comb_method: str, comb_params: List[float], ref_helper, repr_helper):
        conf: SelectorConf = self.conf
        # --
        if dev_items is not None and not score_randomly:
            dev_items = [list(z) for z in dev_items]
            for _dev_items in dev_items:
                self._select_score(_dev_items, False)
        # --
        # prepare
        item_ntype = len(cand_items)  # number of item types
        union_groups = {id(s): {'sent': s, 'cands': [[] for _ in range(item_ntype)],
                                'scores': [sent_aggr_df for _ in range(item_ntype)], 'fs': None} for s in all_sents}
        # score the cands
        for c_ii, c_items in enumerate(cand_items):
            self._select_score(c_items, score_randomly)
        # gather the cands
        for c_ii, c_items in enumerate(cand_items):
            c_groups = self._select_group(c_items, score_randomly, ret_map=True)
            for one_gid, one_group in c_groups.items():
                if one_gid in union_groups:
                    assert len(union_groups[one_gid]['cands'][c_ii]) == 0
                    union_groups[one_gid]['cands'][c_ii].extend(one_group[0])  # group's cands
                    union_groups[one_gid]['scores'][c_ii] = one_group[3]  # group score
        for kk in list(union_groups.keys()):
            if all(len(z)==0 for z in union_groups[kk]['cands']):
                del union_groups[kk]  # no cands!
        # aggregate final group score and sort
        if score_randomly or conf.selv2_random_sent:  # simply score randomly!
            zlog("This iter selects random-sent!!")
            _rands = self.gen.random(len(union_groups)).tolist()
            for ii, vv in enumerate(union_groups.values()):
                vv['fs'] = _rands[ii]
        else:  # comb score: add "fs"!
            _comb_params = self._get_auto_comb(cand_items, dev_items, df=comb_params)
            self.comb_group_scores(union_groups, comb_method, _comb_params)
        sorted_groups = sorted(union_groups.values(), key=(lambda z: -z['fs']))
        # --
        cc = Counter()
        cc.update({"sent_orig": len(sorted_groups)})
        for c_ii in range(item_ntype):
            _items = sum([z['cands'][c_ii] for z in sorted_groups], [])
            cc.update({f"item{c_ii}_orig": len(_items)})
            if not score_randomly:
                self.check_cand_utility(_items, f"item{c_ii}-ORIG")
        # first select sent
        selected_groups = []
        remaining_sent_budget, sent_group = sent_budget
        if repr_helper is not None:
            assert sent_group == 0
            sel_idxes = repr_helper.repr_selection([z['sent'] for z in sorted_groups], [z['fs'] for z in sorted_groups], [len(z['sent']) for z in sorted_groups], remaining_sent_budget)
            selected_groups = [sorted_groups[z] for z in sel_idxes]
        else:
            if conf.selv2_doc_sel:
                # further group sent groups into doc-group
                doc_groups = {}
                for sg in sorted_groups:  # note: already sorted!
                    doc_key = sg['sent'].doc.id
                    assert doc_key is not None
                    if doc_key not in doc_groups:
                        doc_groups[doc_key] = []
                    doc_groups[doc_key].append(sg)
                for one_doc_group in sorted(doc_groups.values(), key=(lambda vs: -np.mean([z['fs'] for z in vs]))):
                    _trg_sents = one_doc_group[:int(math.ceil(conf.selv2_doc_sel_ratio * len(one_doc_group)))]
                    for sg in _trg_sents:
                        selected_groups.append(sg)
                        remaining_sent_budget -= len(sg['sent'])  # note: simply judge by full-length!
                    if remaining_sent_budget <= 0:
                        break
            else:
                sent_group = None if sent_group<=0 else sent_group
                for one_group in sorted_groups[:sent_group]:
                    if remaining_sent_budget <= 0:
                        break
                    selected_groups.append(one_group)
                    remaining_sent_budget -= len(one_group['sent'])  # note: simply judge by full-length!
        cc.update({"sent_ssel": len(selected_groups)})
        for c_ii in range(item_ntype):
            _items = sum([z['cands'][c_ii] for z in selected_groups], [])
            cc.update({f"item{c_ii}_ssel": len(_items)})
            if not score_randomly:
                self.check_cand_utility(_items, f"item{c_ii}-SSEL")
        # --
        if not partial:  # already ok!
            ret = [ZObject(type='sent', gid=id(z['sent'].doc), sent=z['sent'], budget=len(z['sent']),
                           score_cand=z['fs']) for z in selected_groups]
        else:  # further select items
            _selv2_arM = conf.selv2_arM
            _ratio_specs = conf.selv2_ratio_specs  # note: deprecating "ratio_sentwise"!
            # note: essentially no-query for items <= _thresh
            zlog(f"Selecting partially with {cand_sc} {cand_ratios} {cand_threshs}")
            ret = []
            _tok_map = None  # ((id(sent), widx) -> cand_item
            _invalid_toks = None
            for c_ii in range(item_ntype):
                _sc, _ratio_clip, _thresh = cand_sc[c_ii], cand_ratios[c_ii], cand_threshs[c_ii]
                _all_items = sorted(sum([z['cands'][c_ii] for z in selected_groups], []), key=lambda x: -x.score_cand)
                c_type_name = self.get_unique_type_name(_all_items)
                # --
                _div_all_items = max(len(_all_items), 1)  # for later div purpose
                if not score_randomly:  # auto adjusting ratios
                    if len(dev_items[c_ii]) > 0:
                        _auto_ratio0 = self._get_auto_ratio(_all_items, dev_items[c_ii], ref_helper)
                    else:
                        _auto_ratio0 = 1.  # no data to learn!
                else:
                    _auto_ratio0 = None
                # --
                _auto_ratio_alpha = None  # simply increase ratio?
                _auto_ratio_alpha2 = None  # div by 1-this
                if c_type_name == 'alink':
                    if _tok_map is not None:
                        _counter0 = Counter()
                        _all_pnf = []
                        for _one_item in _all_items:
                            _tkeys = self.get_alink_toks(_one_item, conf.selv2_nf_winext)
                            _has_invalid = False
                            if _invalid_toks is not None:
                                _has_invalid = any(z in _invalid_toks for z in _tkeys)  # has invalid ones
                            # has nil-frame
                            if conf.selv2_nf_amode == 'pnf':
                                _has_nf = any(_tok_map[z].gold_idx == 0 for z in _tkeys if z in _tok_map)
                                _pnfs = [_tok_map[z].pnf for z in _tkeys if z in _tok_map and hasattr(_tok_map[z], 'pnf')]
                            elif conf.selv2_nf_amode == 'perr':
                                _has_nf = any(not _tok_map[z].corr for z in _tkeys if z in _tok_map)
                                _pnfs = [(1.-_tok_map[z].pacc) for z in _tkeys if z in _tok_map and hasattr(_tok_map[z], 'pacc')]
                            else:
                                raise NotImplementedError()
                            _pred_nf = 0. if len(_pnfs) == 0 else (
                                        1 - np.prod([1 - z for z in _pnfs]).item())  # any-one is nf?
                            _all_pnf.append(_pred_nf)
                            _counter0['all'] += 1
                            _counter0['invalid'] += int(_has_invalid)
                            _counter0['invalid_nf'] += int(_has_invalid and _has_nf)
                            _counter0['valid'] += int(not _has_invalid)
                            _counter0['valid_nf'] += int(_has_nf and not _has_invalid)
                        zlog(f"Pred_nf: sumP={sum(z > 0.5 for z in _all_pnf)}, sum={sum(_all_pnf)}")
                        zlog(f"Pred_nf: Filter with _invalid_toks: {ZHelper.resort_dict(_counter0)}")
                        _ext = conf.selv2_nf_specs[3]
                        if _ext > 0:
                            if conf.selv2_nf_amode == 'pnf':
                                _auto_ratio_alpha = sum(_all_pnf) * _ext / _div_all_items
                                zlog(f"Pred_nf: Set auto_ratio_alpha[alink] as {_auto_ratio_alpha}")
                            elif conf.selv2_nf_amode == 'perr':
                                _auto_ratio_alpha2 = sum(_all_pnf) * _ext / _div_all_items
                                zlog(f"Pred_nf: Set auto_ratio_alpha2[alink] as {_auto_ratio_alpha2}")
                # --
                # for estimating nil-frame
                if c_type_name == 'tok' and conf.selv2_nf_specs[0] > 0:  # on!
                    qnf = self._get_pred_nilframe(_all_items, dev_items[c_ii], ref_helper)  # note: simply use pacc!
                    assert _tok_map is None  # TODO(+N): assuming only one tok task!!
                    _tok_map = {(id(z.sent), z.widx): z for z in _all_items}
                    _ext = conf.selv2_nf_specs[2]
                    if _ext > 0:
                        _auto_ratio_alpha = qnf
                        zlog(f"Pred_nf: Set auto_ratio_alpha[tok] as {_auto_ratio_alpha}")
                # --
                # decide the ratio
                _ratio_clip = [float(z) for z in str(_ratio_clip).split(":")]  # [low,high] or [high]
                if len(_ratio_clip) == 1:
                    _ratioL, _ratioR = 0., _ratio_clip[0]
                else:
                    _ratioL, _ratioR = _ratio_clip[:2]
                _ratio = _ratioR  # by default take the high-one
                if not score_randomly:  # auto adjusting ratios
                    _a, _b = conf.selv2_ar_adjust
                    _auto_ratio1 = min(max(_ratioL, _auto_ratio0 * _a + _b), _ratioR)
                    if _auto_ratio_alpha is not None:
                        _auto_ratio1a = _auto_ratio_alpha + (1-_auto_ratio_alpha) * _auto_ratio1
                    else:
                        _auto_ratio1a = _auto_ratio1
                    if _auto_ratio_alpha2 is not None:
                        _auto_ratio1a = _auto_ratio1a / (1 - _auto_ratio_alpha2)
                    if len(self.ratio_history[c_ii]) > 0:  # momentum rolling
                        _auto_ratio2 = _selv2_arM * (self.ratio_history[c_ii][-1]) + (1.-_selv2_arM) * _auto_ratio1a
                    else:
                        _auto_ratio2 = _auto_ratio1a
                    if _selv2_arM >= 0.:
                        _ratio = _auto_ratio2  # actually change!
                        zlog(f"Adjusting for auto-ratio[{c_ii}]={_auto_ratio0}->{_auto_ratio1}->[{_auto_ratio1a}]->{_auto_ratio2}")
                # --
                _queried_ids = set()
                if _invalid_toks is None:  # original mode!
                    if _sc > 0:
                        for one_group in selected_groups:
                            one_items = sorted(one_group['cands'][c_ii], key=lambda x: -x.score_cand)
                            _queried_ids.update([id(z) for z in one_items[:_sc]])  # add least query these!
                    if _ratio_specs[0]:  # apply the ratio to each sent
                        for one_group in selected_groups:
                            one_items = sorted(one_group['cands'][c_ii], key=lambda x: -x.score_cand)
                            _ceil = int(math.ceil(_ratio*len(one_items)))
                            _queried_ids.update([id(z) for z in [z for z in one_items[:_ceil] if z.score_cand>_thresh]])
                    if _ratio_specs[1]:  # apply the ratio overall
                        _ceil = int(math.ceil(_ratio*_div_all_items))
                        _queried_ids.update([id(z) for z in _all_items[:_ceil] if z.score_cand>_thresh])
                else:
                    _invalid_ids = set(id(z) for z in _all_items
                                       if any(zk in _invalid_toks for zk in self.get_alink_toks(z)))
                    if _sc > 0:
                        for one_group in selected_groups:
                            one_items = sorted(one_group['cands'][c_ii], key=lambda x: -x.score_cand)
                            _queried_ids.update(self.get_until_budget(one_items, _invalid_ids, _sc))
                    if _ratio_specs[0]:  # apply the ratio to each sent
                        for one_group in selected_groups:
                            one_items = sorted(one_group['cands'][c_ii], key=lambda x: -x.score_cand)
                            _ceil = int(math.ceil(_ratio*len(one_items)))
                            _tmp_items = [z for z in one_items if z.score_cand>_thresh]
                            _queried_ids.update(self.get_until_budget(one_items, _invalid_ids, _ceil))
                    if _ratio_specs[1]:  # apply the ratio overall
                        _ceil = int(math.ceil(_ratio*_div_all_items))
                        _tmp_items = [z for z in _all_items if z.score_cand>_thresh]
                        _queried_ids.update(self.get_until_budget(_tmp_items, _invalid_ids, _ceil))
                # --
                _final_items = [z for z in _all_items if id(z) in _queried_ids]
                ret.append(_final_items)
                cc.update({f"item{c_ii}_final": len(_final_items), f"item{c_ii}R": round(_ratio, 4),
                           f"item{c_ii}P": round(len(_final_items)/_div_all_items, 4)})
                # --
                if not score_randomly:
                    # check selected acc
                    _noquery_items = [z for z in _all_items if id(z) not in _queried_ids]
                    _acc_all = np.mean([z.corr for z in _all_items]) if len(_all_items)>0 else -1.
                    _acc_final = np.mean([z.corr for z in _final_items]) if len(_final_items)>0 else -1.
                    _acc_noquery = np.mean([z.corr for z in _noquery_items]) if len(_noquery_items)>0 else -1.
                    zlog(f"Query-real-acc for item{c_ii}: [Ratio={_ratio:.4f},realPerc={len(_final_items)/_div_all_items:.4f}] ALL={_acc_all:.4f}[{1-_acc_all:.4f}],Q={_acc_final:.4f},NQ={_acc_noquery:.4f}")
                # --
                # record history (potentially for phase2 usage!)
                record_ratio = len(_final_items)/_div_all_items if conf.selv2_record_final_ratio else _ratio
                self.ratio_history[c_ii].append(record_ratio)
                _rthr = _all_items[min(_div_all_items-1, int(record_ratio*_div_all_items))].score_cand if len(_all_items)>0 else 0.  # todo(+N): probably ok if no items?
                self.rthr_history[c_ii].append(_rthr)
                zlog(f"ZRATIO-T[{c_ii}]: R={record_ratio}, T={_rthr}")
                if not score_randomly:
                    self.check_cand_utility(_final_items, f"item{c_ii}-FINAL")
        # --
        zlog(f"Finishing selectV2 with {ZHelper.resort_dict(cc)}")
        # if os.environ.get("ZZDEBUG"):
        #     self.debug_nil_frame(selected_groups)
        # --
        return ret
    # --

    @staticmethod
    def debug_nil_frame(selected_groups):  # analyzing!
        # --
        # helpers
        from scipy.stats import pearsonr
        def _pf(_cc, _ff):
            _cc1 = sum(_ff(z) for z in _cc)
            return (len(_cc), _cc1, round(_cc1/len(_cc), 4))
        # --
        cands0 = sorted(sum([z['cands'][0] for z in selected_groups], []), key=lambda x: -x.score_cand)
        cands1 = sorted(sum([z['cands'][1] for z in selected_groups], []), key=lambda x: -x.score_cand)
        tmap = {(id(z.sent), z.widx): z for z in cands0}
        # alinkH = (lambda tt, cc: [tt[id(z.alink.main.sent), z.alink.main.mention.widx] for z in cc])(tmap, cands1)
        # alinkT = (lambda tt, cc: [tt[id(z.alink.arg.sent), z.alink.arg.mention.widx] for z in cc])(tmap, cands1)
        for one_cand1 in cands1:  # simply assign!
            one_cand1.tH = tmap[(id(one_cand1.alink.main.sent), one_cand1.alink.main.mention.widx)]
            one_cand1.tT = tmap[(id(one_cand1.alink.arg.sent), one_cand1.alink.arg.mention.widx)]
            one_cand1.nfY = (one_cand1.tH.gold_idx == 0 or one_cand1.tT.gold_idx == 0)
            one_cand1.ms = max(one_cand1.tH.score_cand, one_cand1.tT.score_cand)
            one_cand1.ms0 = max(one_cand1.tH.arr_strg[0].item(), one_cand1.tT.arr_strg[0].item())
        breakpoint()
        # --
        # for kk in range(100,1400,100): print(_pf(cands1[:kk], (lambda x: x.nfY)))
        # for kk in range(100,1400,100): print(_pf(sorted(cands1, key=(lambda z: z.ms))[:kk], (lambda x: x.nfY)))
        # for thr in [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,1]: print(thr, _pf((lambda k0,c0:[z for z in c0 if z.ms<k0])(thr,cands1), (lambda x: x.nfY)))
        # pearsonr([float(z.nfY) for z in cands1], [z.ms for z in cands1])
        # --
        # pp [[z['scores'], len(z['cands'][0]), len(z['cands'][1])] for z in sorted_groups[:100]]
        # --

    @staticmethod
    def get_unique_type_name(items):
        names = set([z.type for z in items])
        assert len(names) <= 1
        return None if len(names)==0 else list(names)[0]

    @staticmethod
    def get_alink_toks(item, winexts=(0,0), tok_map=None):  # whether ignore this one?
        ret = []
        if item.type == 'alink':
            for frame, winext in zip([item.alink.main, item.alink.arg], winexts):
                widx, wlen = frame.mention.get_span()
                _left = max(0, widx-winext)
                _right = min(len(frame.sent), widx+wlen+winext)
                for ii in range(_left, _right):
                    _key = (id(frame.sent), ii)
                    if tok_map is not None:
                        if _key in tok_map:
                            ret.append(tok_map[_key])
                    else:
                        ret.append(_key)
        return ret

    @staticmethod
    def get_until_budget(items, invalid_ids, budget):
        ret = []
        for one_item in items:
            if budget <= 0:
                break
            _id = id(one_item)
            if invalid_ids is not None and _id in invalid_ids:
                continue
            ret.append(id(one_item))
            budget -= 1
        return ret

    @staticmethod
    def check_cand_utility(cands, headline: str):
        if len(cands) == 0:
            zwarn(f"Cand utility for {headline}: NO cands!")
            return
        bins = Counter()
        bins.update([int(z.score_cand*10) for z in cands])  # simply bins for *10!
        all_keys = sorted(bins.keys(), reverse=True)
        avg_score = np.mean([z.score_cand for z in cands]).item()
        zlog(f"Cand utility for {headline}: {len(cands)} {avg_score}")
        zlog(f"{ZHelper.get_counts_info_table(bins, keys=all_keys).to_string()}")

    @staticmethod
    def comb_group_scores(union_groups, comb_method: str, comb_params: List[float]):
        _fs_key = 'fs'
        # --
        if comb_method.startswith('score'):  # simply weighted sum of scores
            _zparams = [(0.,1.) for _ in comb_params]
            if comb_method == 'scoreZ':
                for ii in range(len(_zparams)):
                    _one_scores = np.asarray([g['scores'][ii] for g in union_groups.values()])
                    _one_scores2 = _one_scores[_one_scores>0]  # filter >0!
                    _zparams[ii] = (_one_scores2.mean().item(), _one_scores2.std().item())
                zlog(f"Adjust scores by: {_zparams}")
            for g in union_groups.values():
                fs = 0.
                for ii, pp in enumerate(comb_params):
                    _a, _b = _zparams[ii]
                    fs += (g['scores'][ii] - _a) / _b * pp
                g[_fs_key] = fs
        else:  # rank combination or alternative picking
            _ug = union_groups
            _all_ids = list(_ug.keys())  # all keys
            sorted_lists = []  # sort according to each ntype
            # assign ranks
            for k, g in union_groups.items():
                g['ranks'] = [None] * len(comb_params)
            for ii, pp in enumerate(comb_params):
                l0 = sorted(_all_ids, key=(lambda x: [_ug[x]['scores'][ii]] + _ug[x]['scores']), reverse=True)
                sorted_lists.append(l0)
                for rr, kk in enumerate(l0, 1):
                    union_groups[kk]['ranks'][ii] = rr
            # --
            assert comb_method in ['alter', 'rank']
            is_alter = (comb_method == 'alter')
            for k, g in union_groups.items():
                if is_alter:
                    comb_rr = min(a/max(1e-10,b) for a,b in zip(g['ranks'], comb_params))  # alter
                else:
                    comb_rr = sum(a*b for a,b in zip(g['ranks'], comb_params))  # weighted rank
                g[_fs_key] = - comb_rr  # higher as utility score!
            # --
            # gs1 = sorted(union_groups.values(), key=lambda x: x['ranks'][1])
            # gsR = sorted(union_groups.values(), key=lambda x: -x['fs'])
            # sum(len(z['cands'][1]) for z in gs1[800:1000])
            # pp [(len(z['cands'][0]), len(z['cands'][1]), z['ranks'], z['scores']) for z in gs1[:100]]
            # --
        # --
        return

    # fit model for acc prediction: X=score=1-margin, Y=corr(or acc), trainXY are sorted
    def _ar_fit_acc(self, trainX, trainY, testX, ext_thr=None):
        conf: SelectorConf = self.conf
        _thr, _method = conf.selv2_ar_thr, conf.selv2_ar_method
        _selv2_ar_logtransform = conf.selv2_ar_logtransform
        if ext_thr is not None:
            _thr = ext_thr
        _eps = 1e-10
        # --
        # utility <-> logit
        def _transform1(_x):
            _x2 = _x.clip(min=_eps, max=1.-_eps)
            # return np.log(_x2 / (1-_x2))
            return np.log(_x2)
        # --
        _transform_f = _transform1 if _selv2_ar_logtransform else (lambda x: x)
        # weighting?
        train_weight = np.ones(len(trainX))
        if _method == 'logr':  # logistic regression
            from sklearn.linear_model import LogisticRegression
            class_weight = None
            for _ in range(conf.selv2_ar_reb + 1):
                clf = LogisticRegression(random_state=0, class_weight=class_weight).fit(
                    _transform_f(trainX), trainY.astype(int), sample_weight=train_weight)
                p_trainY = clf.predict_proba(_transform_f(trainX))[:,1]
                p_testY = clf.predict_proba(_transform_f(testX))[:,1]
                # zlog(f"Fit LogR of: {clf.coef_}*margin.log()+{clf.intercept_}")
                _c0, _c1 = 1.-trainY.mean(), trainY.mean()
                _p0, _p1 = 1.-p_testY.mean(), p_testY.mean()
                class_weight = {0: _p0/_c0, 1: _p1/_c1}
                zlog(f"Fit LogR: {_c1:.4f} {_p1:.4f} {class_weight}")
                if _p1 >= _c1:  # no need!
                    break
        elif _method == 'point':  # simply use the average
            _res = (trainY * train_weight).sum() / train_weight.sum()
            p_trainY = np.full(len(trainX), _res)
            p_testY = np.full(len(testX), _res)
        elif _method == 'local':  # my local estimate
            _alpha, _beta = conf.selv2_ar_winA, conf.selv2_ar_winB
            _betaR = len(trainX) * _beta
            _arange = np.arange(len(trainX))
            _diff = _arange[:, None] - _arange  # [N, N]
            _mask = ((_diff >= -_betaR*(1-_alpha)) & (_diff <= _betaR*_alpha)).astype(float)
            _accD = (trainY * _mask).sum(-1) / _mask.sum(-1).clip(min=1)
            # estimate for query
            _idxesQ = (np.searchsorted(trainX[:,0], testX[:,0])).clip(min=0, max=len(trainX)-1)
            _accQ = _accD[_idxesQ]  # estimated acc
            p_trainY, p_testY = _accD, _accQ
        else:
            raise NotImplemented(f"UNK method if {_method}")
        # view it as 0 if ambiguous
        p_trainY = p_trainY * (trainX[:,0] < _thr).astype(p_trainY.dtype)
        p_testY = p_testY * (testX[:,0] < _thr).astype(p_testY.dtype)
        return p_trainY, p_testY

    # auto decide ratio based on estimated accuracy
    def _get_auto_ratio(self, query_items, dev_items, ref_helper):
        import pandas as pd
        # --
        conf: SelectorConf = self.conf
        _thr, _thr2, _method = conf.selv2_ar_thr, conf.selv2_ar_thr2, conf.selv2_ar_method
        real_query_acc = -1
        if conf.selv2_check_query:
            ref_helper.eval_cands(query_items)
        else:
            for one_cand in query_items:
                one_cand.gold_idx = -100  # put a dummy one!
        for _cands in [dev_items, query_items]:
            for one_cand in _cands:
                one_cand.corr = float(one_cand.score_cand < _thr and self.get_strg_argmax(one_cand.arr_strg, True) >= _thr2 and one_cand.gold_idx == self.get_strg_argmax(one_cand.arr_strg))
        # --
        _selv2_ar_gmin = conf.selv2_ar_gmin
        if _selv2_ar_gmin < 1:  # select harder dev ones!
            dev_groups = self._select_group(dev_items, False)
            query_groups = self._select_group(query_items, False)
            _g_keep = len(dev_items) * _selv2_ar_gmin
            _g_thr = query_groups[-1][-1]  # query min group-utility
            # --
            new_dev_gn, new_dev_items = 0, []
            for one_dev_group in dev_groups:  # already sorted
                if len(new_dev_items) < _g_keep or one_dev_group[-1] >= _g_thr:  # add in!
                    new_dev_gn += 1
                    new_dev_items += one_dev_group[0]
                else:
                    break
            zlog(f"Select groups by K={_g_keep},Gt={_g_thr}: {new_dev_gn}/{len(dev_groups)} || {len(new_dev_items)}/{len(dev_items)}")
            dev_items = new_dev_items  # simply change it!
            # breakpoint()
        # --
        sort_dev = sorted([(z.score_cand, z.corr) for z in dev_items])  # ascending
        dS, dA = np.asarray([z[:1] for z in sort_dev]), np.asarray([z[-1] for z in sort_dev])
        qS = np.asarray([(z.score_cand, ) for z in query_items])
        _accD, _accQ = self._ar_fit_acc(dS, dA, qS)
        if _thr2 > 0:  # further thresh for different types of strg_f!
            _accQ = _accQ * np.asarray([(self.get_strg_argmax(z.arr_strg, True) >= _thr2) for z in query_items]).astype(_accQ.dtype)
        for _cands, _acc in zip([dev_items, query_items], [_accD, _accQ]):
            for one_cand, one_acc in zip(_cands, _acc):
                one_cand.pacc = float(one_acc)  # predicted acc
        _ratioD, _ratioQ = 1. - _accD.mean().item(), 1. - _accQ.mean().item()
        if conf.selv2_check_query:
            qA = np.asarray([z.corr for z in query_items])
            real_query_acc = np.mean(qA).item()
        # --
        # printing
        _records = []
        _BINS = 10
        for _ii in range(_BINS+1):
            _a = _ii / _BINS
            _records.append({'A': _a, 'Dc': (_accD<=_a).sum(), 'Dp': (_accD<=_a).mean(),
                             'Qc': (_accQ<=_a).sum(), 'Qp': (_accQ<=_a).mean()})
        df = pd.DataFrame.from_records(_records)
        zlog(f"{df.to_string()}")
        zlog(f"Calculate auto_ratio={_ratioQ} // Q={1-_ratioQ:.4f}, Qreal={real_query_acc:.4f}, D={1-_ratioD:.4f}, Dreal={dA.mean():.4f}")
        _q_eval_res = self._my_evaluate([z.gold_idx for z in query_items], [self.get_strg_argmax(z.arr_strg) if (z.score_cand<_thr and self.get_strg_argmax(z.arr_strg, True)>=_thr2) else -1 for z in query_items])
        zlog(f"Eval query: {_q_eval_res}")
        # --
        # breakpoint()
        if conf.selv2_oracle_ratio:
            zwarn("Use oracle ratio (only for checking/debugging)!!")
            _ratioQ = 1. - real_query_acc
        return _ratioQ

    # evaluate F1
    def _my_evaluate(self, t_gold, t_pred, t_bad=None, t_mask=None, nil=0, adjust_RC=None):
        import torch
        t_gold = torch.as_tensor(t_gold)
        t_pred = torch.as_tensor(t_pred)
        # --
        if t_mask is not None:
            t_mask = torch.as_tensor(t_mask)
            t_gold = t_gold[t_mask]
            t_pred = t_pred[t_mask]
        t_corr = (t_gold == t_pred)  # correct
        if t_bad is not None:
            t_bad = torch.as_tensor(t_bad)
            t_corr = t_corr & (~t_bad)
        t_gold_v = (t_gold != nil)  # not nil
        t_pred_v = (t_pred != nil)  # not nil
        A = t_corr.sum() / torch.ones_like(t_gold).sum().clamp(min=1)
        P = (t_corr & t_pred_v).sum() / t_pred_v.sum().clamp(min=1)
        R = (t_corr & t_gold_v).sum() / t_gold_v.sum().clamp(min=(1 if adjust_RC is None else adjust_RC))
        F = 2 * P * R / (P + R).clamp(min=1e-5)
        M = torch.tensor(1.) if t_mask is None else t_mask.float().mean()
        ret = {a: round(b.item(), 4) for a, b in zip("APRFM", [A, P, R, F, M])}
        return ret

    # auto decide comb
    def _get_auto_comb(self, query_items, dev_items, df):
        conf: SelectorConf = self.conf
        _thr = conf.selv2_ar_thr
        _v0, _v1, _switch_thr = conf.selv2_autocomb_specs
        # --
        if _switch_thr<=0:  # no auto!
            zlog(f"Auto-comb-df with {df}")
            return df
        assert len(query_items) == len(dev_items)
        len_c = len(query_items)
        _history = self.comb_stat_history
        for c_ii, (c_qs, c_ds) in enumerate(zip(query_items, dev_items)):
            # adjust recall
            adjust_recall_counts = {}
            for z in c_ds:
                if hasattr(z, 'gold_ninfo'):
                    k, v = z.gold_ninfo
                    adjust_recall_counts[k] = v
            adjust_recall_count = max(1,sum(adjust_recall_counts.values())) if len(adjust_recall_counts)>0 else None
            # --
            _res = self._my_evaluate([z.gold_idx for z in c_ds], [self.get_strg_argmax(z.arr_strg) for z in c_ds],
                                     t_bad=[z.score_cand >= _thr for z in c_ds], adjust_RC=adjust_recall_count)
            zlog(f"Eval-for-ac[{c_ii}]: {_res}")
            _history.append(_res['F'])  # using F-score
        # --
        _arr = np.asarray(_history).reshape([-1, len_c])  # [*, C]
        # todo(+N): currently simply two stages based on task0!
        switched = any((_arr[ii+1,0]-_arr[ii,0]).item()<=_switch_thr for ii in range(len(_arr)-1))
        r0 = _v1 if switched else _v0
        ret = [r0] + ([(1-r0)/(len_c-1) for _ in range(len_c-1)] if len_c>1 else [])
        zlog(f"Auto-comb with switched={switched}: {ret} // {_history} ")
        return ret

    # predict the nil frames: very similar to the process of "_get_auto_ratio"
    def _get_pred_nilframe(self, query_items, dev_items, ref_helper):
        conf: SelectorConf = self.conf
        _feature_num, _ignore_non_frame = conf.selv2_nf_specs[:2]
        # filter frame toks
        if bool(int(_ignore_non_frame)):
            lenQ0, lenD0 = len(query_items), len(dev_items)
            query_items = [z for z in query_items if self.get_strg_argmax(z.arr_strg)!=0]
            dev_items = [z for z in dev_items if self.get_strg_argmax(z.arr_strg)!=0]
            zlog(f"First filter non-frame: Q:{lenQ0}->{len(query_items)} D:{lenD0}->{len(dev_items)}")
        else:
            zlog(f"No filter of: Q:{len(query_items)} D:{len(dev_items)}")
        # check_query?
        real_nf_rate = -1
        if conf.selv2_check_query:
            ref_helper.eval_cands(query_items)
        else:
            for one_cand in query_items:
                one_cand.gold_idx = -100  # put a dummy one!
        for _cands in [dev_items, query_items]:
            for one_cand in _cands:
                one_cand.is_nf = (one_cand.gold_idx == 0)  # simply check whether 0?
        # fit
        sort_dev = sorted([(z.arr_strg[0].item(), z.score_cand, z.is_nf) for z in dev_items])  # ascending
        dS, dA = np.asarray([z[:int(_feature_num)] for z in sort_dev]), np.asarray([z[-1] for z in sort_dev])
        qS = np.asarray([(z.arr_strg[0].item(), z.score_cand)[:int(_feature_num)] for z in query_items])
        _nfD, _nfQ = self._ar_fit_acc(dS, dA, qS, ext_thr=1.1)  # basically no thr!
        for _cands, _nf in zip([dev_items, query_items], [_nfD, _nfQ]):
            for one_cand, one_nf in zip(_cands, _nf):
                one_cand.pnf = float(one_nf)  # predicted nil
        if conf.selv2_check_query:
            qA = np.asarray([z.is_nf for z in query_items])
            real_nf_rate = np.mean(qA).item()
        ret = _nfQ.mean().item()
        zlog(f"Calculate Pred_nf: Q={ret:.4f}, Qreal={real_nf_rate:.4f}, D={_nfD.mean():.4f}, Dreal={dA.mean():.4f}")
        # --
        return ret

# --
# b mspx/tools/al/tasks/selector:130
