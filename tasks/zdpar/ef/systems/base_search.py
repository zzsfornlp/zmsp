#

# base system

from typing import List

from msp.utils import Helper, Constants, Conf, zlog, Random
from msp.nn import BK
from msp.search.lsg import BfsAgenda, BfsSearcher, BfsExpander, BfsLocalSelector, BfsGlobalArranger, BfsEnder

from ..scorer import Scorer, SL0Layer
from .base_system import EfState, EfAction, EfOracler, EfCoster, EfSignaturer, ParseInstance, Graph
from .systems import StateBuilder

# =====
# searching
# todo(warn): all the caches are modified inplaced to save memory,
#  thus the nn-graph should be reconstructed later for backward,
#  therefore, does not support local normalization here!
# Two choices: 1. Three runs for training: search+forward+backward, 2. using Cache for batched management

#
class ScorerHelper:
    # get features from action and state
    class HmFeatureGetter:
        def __init__(self, ignore_chs_label_mask, fdrop_chs: float, fdrop_par: float):
            self.ignore_chs_label_mask = ignore_chs_label_mask
            if fdrop_chs>0.:
                self.chs_getter_f = self.get_fchs_dropped
                self.chs_rand_gen = Random.stream_bool(fdrop_chs)
            else:
                self.chs_getter_f = self.get_fchs_base
            if fdrop_par>0.:
                self.par_getter_f = self.get_fpar_dropped
                self.par_rand_gen = Random.stream_bool(fdrop_par)
            else:
                self.par_getter_f = self.get_fpar_base

        def get_hm_features(self, actions: List[EfAction], states: List[EfState]):
            hm_features = ([[], [], [], [], []], [[], [], [], [], []])  # head, mod
            chs_getter_f, par_getter_f = self.chs_getter_f, self.par_getter_f
            for action, state in zip(actions, states):
                hm_idxes = action.head, action.mod
                for idx, features_group in zip(hm_idxes, hm_features):
                    features_group[0].append(idx)
                    par_idx, par_label = par_getter_f(state.list_arc[idx], state.list_label[idx])
                    features_group[1].append(par_idx)
                    features_group[2].append(par_label)
                    chs_idxes, chs_labels = chs_getter_f(state.idxes_chs[idx], state.labels_chs[idx])
                    features_group[3].append(chs_idxes)
                    features_group[4].append(chs_labels)
            return hm_features

        # =====
        def get_fpar_base(self, par_idx, par_label):
            # todo(+2): currently no label filtering here!
            return par_idx, par_label

        def get_fchs_base(self, chs_idxes, chs_labels):
            ignore_chs_label_mask = self.ignore_chs_label_mask
            ret_idxes, ret_labels = [], []
            for a, b in zip(chs_idxes, chs_labels):
                if not ignore_chs_label_mask[b]:  # excluded if True in ignore_mask
                    ret_idxes.append(a)
                    ret_labels.append(b)
            return ret_idxes, ret_labels

        def get_fpar_dropped(self, par_idx, par_label):
            if next(self.par_rand_gen):
                return -1, -1  # todo(note): -1 means None for par
            else:
                return par_idx, par_label

        def get_fchs_dropped(self, chs_idxes, chs_labels):
            ignore_chs_label_mask = self.ignore_chs_label_mask
            ret_idxes, ret_labels = [], []
            for a,b in zip(chs_idxes, chs_labels):
                if (b not in ignore_chs_label_mask) and (not next(self.chs_rand_gen)):
                    ret_idxes.append(a)
                    ret_labels.append(b)
            return ret_idxes, ret_labels

    # from features -> srepr [*, D]
    @staticmethod
    def calc_repr(s_enc: SL0Layer, features_group, enc_expr, bidxes_expr):
        cur_idxes, par_idxes, labels, chs_idxes, chs_labels = features_group
        # get padded idxes: [*] or [*, ?]
        cur_idxes_t = BK.input_idx(cur_idxes)
        par_idxes_t, label_t, par_mask_t = s_enc.pad_par(par_idxes, labels)
        chs_idxes_t, chs_label_t, chs_mask_t, chs_valid_mask_t = s_enc.pad_chs(chs_idxes, chs_labels)
        # gather enc-expr: [*, D], [*, D], [*, max-chs, D]
        dim1_range_t = bidxes_expr
        dim2_range_t = dim1_range_t.unsqueeze(-1)
        cur_t = enc_expr[dim1_range_t, cur_idxes_t]
        par_t = enc_expr[dim1_range_t, par_idxes_t]
        chs_t = None if chs_idxes_t is None else enc_expr[dim2_range_t, chs_idxes_t]
        # update reprs: [*, D]
        new_srepr = s_enc.calculate_repr(cur_t, par_t, label_t, par_mask_t, chs_t, chs_label_t, chs_mask_t, chs_valid_mask_t)
        return cur_idxes_t, new_srepr

# scoring cache
class EfScoringCacheArc:
    def __init__(self, scorer: Scorer, slayer: SL0Layer, system_labeled: bool):
        self.scorer = scorer
        self.slayer = slayer
        self.system_labeled = system_labeled
        self.num_label = scorer.num_label
        # scoring caches
        self.head_arc_cache = None  # tmp repr-cache for decoder for as-head
        self.head_label_cache = None
        self.mod_arc_cache = None  # tmp repr-cache for decoder for as-mod
        self.mod_label_cache = None
        # scores
        self.arc_scores = None  # paired arc scores (m, h): [*, len-m, len-h]
        # extra scores (from g1)
        self.g1_arc_scores = self.g1_lab_scores = None

    def init_cache(self, enc_repr, g1_pack):
        # dec cache, *[orig_bsize, max_slen, ?]
        # todo(note): since at the start there are no structured features
        enc_s_repr = self.slayer.forward_repr(enc_repr)
        self.head_arc_cache, self.mod_arc_cache = self.scorer.transform_space_arc(enc_s_repr)
        if self.system_labeled:
            self.head_label_cache, self.mod_label_cache = self.scorer.transform_space_label(enc_s_repr)
        # arc score (no mask applied here)
        head_inputs = self.head_arc_cache.unsqueeze(-3)  # [*, 1, len-h, D]
        mod_inputs = [z.unsqueeze(-2) for z in self.mod_arc_cache]  # [*, len-m, 1, ?]
        self.arc_scores = self.scorer.score_arc(mod_inputs, head_inputs).squeeze(-1)  # [*, len-m, len-h]
        if g1_pack is not None:
            self.g1_arc_scores, self.g1_lab_scores = g1_pack  # [*, lem-m, len-h], [*, len-m, len-h, L]

    def arange_cache(self, bidxes_device):
        self.head_arc_cache = self.head_arc_cache.index_select(0, bidxes_device)
        self.mod_arc_cache = [z.index_select(0, bidxes_device) for z in self.mod_arc_cache]
        if self.system_labeled:
            self.head_label_cache = self.head_label_cache.index_select(0, bidxes_device)
            self.mod_label_cache = [z.index_select(0, bidxes_device) for z in self.mod_label_cache]
        self.arc_scores = self.arc_scores.index_select(0, bidxes_device)
        if self.g1_arc_scores is not None:
            self.g1_arc_scores = self.g1_arc_scores.index_select(0, bidxes_device)
        if self.g1_lab_scores is not None:
            self.g1_lab_scores = self.g1_lab_scores.index_select(0, bidxes_device)

    # update caches and scores; [*], [*, D]
    def update_cache_and_score(self, node_idxes_t, bsize_range_t, node_srepr, update_as_head: bool, update_as_mod: bool):
        system_labeled = self.system_labeled
        # get caches of [*, D]
        node_ah_expr, node_am_pack = self.scorer.transform_space_arc(node_srepr, update_as_head, update_as_mod)
        if system_labeled:
            node_lh_expr, node_lm_pack = self.scorer.transform_space_label(node_srepr, update_as_head, update_as_mod)
        # todo(note): inplaced update, which means not backward on this (only forward for search graph)
        dim1_range_t = bsize_range_t
        if update_as_head:
            self.head_arc_cache[dim1_range_t, node_idxes_t] = node_ah_expr
            if system_labeled:
                self.head_label_cache[dim1_range_t, node_idxes_t] = node_lh_expr
            # [*, L]
            scores_as_head = self.scorer.score_arc(self.mod_arc_cache, node_ah_expr.unsqueeze(-2)).squeeze(-1)
            self.arc_scores[dim1_range_t, :, node_idxes_t] = scores_as_head
        if update_as_mod:
            for v_to_be_filled, v_to_fill in zip(self.mod_arc_cache, node_am_pack):
                v_to_be_filled[dim1_range_t, node_idxes_t] = v_to_fill
            if system_labeled:
                for v_to_be_filled, v_to_fill in zip(self.mod_label_cache, node_lm_pack):
                    v_to_be_filled[dim1_range_t, node_idxes_t] = v_to_fill
            # [*, L]
            scores_as_mod = self.scorer.score_arc([x.unsqueeze(-2) for x in node_am_pack], self.head_arc_cache).squeeze(-1)
            self.arc_scores[dim1_range_t, node_idxes_t] = scores_as_mod

    # label scores: [*, k]
    def get_selected_label_scores(self, idxes_m_t, idxes_h_t, bsize_range_t, oracle_mask_t, oracle_label_t,
                                  arc_margin: float, label_margin: float):
        # todo(note): in this mode, no repeated arc_margin
        dim1_range_t = bsize_range_t
        dim2_range_t = dim1_range_t.unsqueeze(-1)
        if self.system_labeled:
            selected_m_cache = [z[dim2_range_t, idxes_m_t] for z in self.mod_label_cache]
            selected_h_repr = self.head_label_cache[dim2_range_t, idxes_h_t]
            ret = self.scorer.score_label(selected_m_cache, selected_h_repr)  # [*, k, labels]
            if label_margin > 0.:
                oracle_label_idxes = oracle_label_t[dim2_range_t, idxes_m_t, idxes_h_t].unsqueeze(-1)  # [*, k, 1] of int
                ret.scatter_add_(-1, oracle_label_idxes, BK.constants(oracle_label_idxes.shape, -label_margin))
        else:
            # todo(note): otherwise, simply put zeros (with idx=0 as the slightly best to be consistent)
            ret = BK.zeros(BK.get_shape(idxes_m_t) + [self.num_label])
            ret[:, :, 0] += 0.01
        if self.g1_lab_scores is not None:
            ret += self.g1_lab_scores[dim2_range_t, idxes_m_t, idxes_h_t]
        return ret

    # full arc scores (with margin): [*, m, n]
    def get_arc_scores(self, oracle_mask_t, margin: float):
        if margin <= 0.:
            ret = self.arc_scores
        else:
            ret = self.arc_scores - margin * oracle_mask_t
        if self.g1_arc_scores is not None:
            # todo(warn): no adding inplaced to avoid change self scores
            ret = ret + self.g1_arc_scores
        return ret

# (batched) running cache (main for repr and scoring)
class EfRunningCache:
    def __init__(self, scorer: Scorer, slayer: SL0Layer, hm_feature_getter, max_slen, orig_bsize,
                 enc_repr, enc_mask_arr, g1_pack, insts, system_labeled):
        self.scorer = scorer
        self.slayer = slayer
        self.hm_feature_getter = hm_feature_getter
        # repr
        self.enc_repr = None  # repr output from encoder (not s_enc) for the current running
        # scoring, todo(+2): choose by the flag here, not elegant though!
        self.scoring_cache = EfScoringCacheArc(scorer, slayer, system_labeled)
        # masks
        self.scoring_fixed_mask_ct = None  # fixed mask (0 for self_loop, root_mod and sent_mask)
        self.scoring_mask_ct = None  # mask before scoring (cpu_tensor)
        # oracles (as masks)
        # todo(note): for the current systems, the oracle masks can be fixed
        self.oracle_mask_t = None  # mask for oracle
        self.oracle_mask_ct = None  # same as previous, used for selection-validation
        self.oracle_label_t = None  # idxes for labels (this can be calculated at init)
        # others
        self.step = 0
        self.max_slen = max_slen
        self.orig_bsize = orig_bsize
        self.cur_bsize = -1
        self.bsize_range_t = None
        self.update_bsize(orig_bsize)
        # -----
        self.init_cache(enc_repr, enc_mask_arr, insts, g1_pack)

    def update_step(self):
        self.step += 1

    def update_bsize(self, new_bsize):
        if new_bsize != self.cur_bsize:
            self.cur_bsize = new_bsize
            self.bsize_range_t = BK.arange_idx(new_bsize)

    # re-arrange the caches to make the first-dim batch corresponded
    def arange_cache(self, bidxes):
        new_bsize = len(bidxes)
        # if the idxes are already fine, then no need to select
        if not Helper.check_is_range(bidxes, self.cur_bsize):
            # mask is on CPU to make assigning easier
            bidxes_ct = BK.input_idx(bidxes, BK.CPU_DEVICE)
            self.scoring_fixed_mask_ct = self.scoring_fixed_mask_ct.index_select(0, bidxes_ct)
            self.scoring_mask_ct = self.scoring_mask_ct.index_select(0, bidxes_ct)
            self.oracle_mask_ct = self.oracle_mask_ct.index_select(0, bidxes_ct)
            # other things are all on target-device (possibly GPU)
            bidxes_device = BK.to_device(bidxes_ct)
            self.enc_repr = self.enc_repr.index_select(0, bidxes_device)
            self.scoring_cache.arange_cache(bidxes_device)
            # oracles
            self.oracle_mask_t = self.oracle_mask_t.index_select(0, bidxes_device)
            self.oracle_label_t = self.oracle_label_t.index_select(0, bidxes_device)
            # update bsize
            self.update_bsize(new_bsize)

    # get init fixed masks
    def _init_fixed_mask(self, enc_mask_arr):
        tmp_device = BK.CPU_DEVICE
        # by token mask
        mask_ct = BK.input_real(enc_mask_arr, device=tmp_device)  # [*, len]
        full_mask_ct = mask_ct.unsqueeze(-1) * mask_ct.unsqueeze(-2)  # [*, len-mod, len-head]
        # no self loop
        full_mask_ct *= (1.-BK.eye(self.max_slen, device=tmp_device))
        # no root as mod; todo(warn): assume it is 3D
        full_mask_ct[:, 0, :] = 0.
        return full_mask_ct

    # init reprs and first-order scores
    def init_cache(self, enc_repr, enc_mask_arr, insts, g1_pack):
        # init caches and scores, [orig_bsize, max_slen, D]
        self.enc_repr = enc_repr
        self.scoring_fixed_mask_ct = self._init_fixed_mask(enc_mask_arr)
        # init other masks
        self.scoring_mask_ct = BK.copy(self.scoring_fixed_mask_ct)
        full_shape = BK.get_shape(self.scoring_mask_ct)
        # init oracle masks
        oracle_mask_ct = BK.constants(full_shape, value=0., device=BK.CPU_DEVICE)
        # label=0 means nothing, but still need it to avoid index error (dummy oracle for wrong/no-oracle states)
        oracle_label_ct = BK.constants(full_shape, value=0, dtype=BK.int64, device=BK.CPU_DEVICE)
        for i, inst in enumerate(insts):
            EfOracler.init_oracle_mask(inst, oracle_mask_ct[i], oracle_label_ct[i])
        self.oracle_mask_t = BK.to_device(oracle_mask_ct)
        self.oracle_mask_ct = oracle_mask_ct
        self.oracle_label_t = BK.to_device(oracle_label_ct)
        # scoring cache
        self.scoring_cache.init_cache(enc_repr, g1_pack)

    # update reprs and scores incrementally
    def update_cache(self, flattened_states: List[EfState]):
        assert len(flattened_states) == self.cur_bsize, "Err: Mismatched batch size!"
        # after adding an edge, head obtain an additional child and mod obtains an additional parent
        # todo(+3): can these be simplified?
        # 1. collect (batched) features; todo(note): use current state for updating scoring
        hm_features = self.hm_feature_getter.get_hm_features([x.action for x in flattened_states], flattened_states)
        # 2. get new sreprs
        # todo(note): no recurrence or recursive here, therefore using the original enc_repr
        s_enc = self.slayer
        node_h_idxes_t, node_h_srepr = ScorerHelper.calc_repr(s_enc, hm_features[0], self.enc_repr, self.bsize_range_t)
        node_m_idxes_t, node_m_srepr = ScorerHelper.calc_repr(s_enc, hm_features[1], self.enc_repr, self.bsize_range_t)
        # 3. update cache and score
        # todo(note): does not matter for the interactions, since inter-influenced parts are to-be-masked-out
        # todo(+3): for some specific mode, can further reduce some calculations
        self.scoring_cache.update_cache_and_score(node_h_idxes_t, self.bsize_range_t, node_h_srepr, True, True)
        # mod cannot be mod again, thus no need to update (scores will be masked out later)
        self.scoring_cache.update_cache_and_score(node_m_idxes_t, self.bsize_range_t, node_m_srepr, True, False)

    # =====
    # todo(warn): here margin is utilized to let oracle ones have less scores

    # label scores: [*, k]
    def get_selected_label_scores(self, idxes_m_t, idxes_h_t, arc_margin: float, label_margin: float):
        return self.scoring_cache.get_selected_label_scores(idxes_m_t, idxes_h_t, self.bsize_range_t, self.oracle_mask_t, self.oracle_label_t, arc_margin, label_margin)

    # full arc scores (with margin): [*, m, n]
    def get_arc_scores(self, arc_margin: float):
        return self.scoring_cache.get_arc_scores(self.oracle_mask_t, arc_margin)

# handling repr/arc-score update
class EfExpander(BfsExpander):
    def __init__(self, scorer: Scorer):
        # self.scorer = scorer
        self.cache: EfRunningCache = None
        # self.margin: float = None
        # self.cost_weight_arc: float = None
        self.mw_arc: float = None

    def refresh(self, cache, margin, cost_weight_arc, cost_weight_label):
        self.cache = cache
        # self.margin = margin
        # self.cost_weight_arc = cost_weight_arc
        self.mw_arc = margin * cost_weight_arc

    def expand(self, ags: List[BfsAgenda]):
        # flatten things out
        flattened_states = []
        for ag in ags:
            flattened_states.extend(ag.beam)
            flattened_states.extend(ag.gbeam)
        # read/write cache to change status
        cur_cache = self.cache
        # update reprs and scores
        EfState.set_running_bidxes(flattened_states)
        if cur_cache.step > 0:
            bidxes = [s.prev.running_bidx for s in flattened_states]
            # extend previous cache to the current bsize (dimension 0 at batch-dim)
            cur_cache.arange_cache(bidxes)
            # update cahces and scores
            # todo(+N): for certain modes, some calculations are not needed
            cur_cache.update_cache(flattened_states)
        cur_cache.update_step()
        # get new masks and final scores
        scoring_mask_ct = cur_cache.scoring_mask_ct
        for sidx, state in enumerate(flattened_states):
            state.update_cands_mask(scoring_mask_ct[sidx])  # inplace mask update
        scoring_mask_ct *= cur_cache.scoring_fixed_mask_ct  # apply the fixed masks
        scoring_mask_device = BK.to_device(scoring_mask_ct)
        cur_arc_scores = cur_cache.get_arc_scores(self.mw_arc) + Constants.REAL_PRAC_MIN*(1.-scoring_mask_device)
        # todo(+N): possible normalization for the scores
        return flattened_states, cur_arc_scores, scoring_mask_ct

# handling selecting and label-scoring
class EfLocalSelector(BfsLocalSelector):
    def __init__(self, scorer, plain_mode, oracle_mode, plain_k_arc, plain_k_label, oracle_k_arc, oracle_k_label,
                 oracler: EfOracler, system_labeled):
        super().__init__(plain_mode, oracle_mode, plain_k_arc, plain_k_label, oracle_k_arc, oracle_k_label, oracler)
        # self.scorer = scorer
        self.cache: EfRunningCache = None
        #
        # self.margin = 0.
        self.system_labeled = system_labeled
        self.mw_arc: float = None
        self.mw_label: float = None

    def refresh(self, cache, margin, cost_weight_arc, cost_weight_label):
        self.cache = cache
        # self.margin = margin
        self.mw_arc = margin * cost_weight_arc
        self.mw_label = margin * cost_weight_label

    # get new state from the selected edge
    # todo(+N): currently not recording the ScoreSlice here!
    def _new_states(self, flattened_states: List[EfState], scoring_mask_ct,
                    topk_arc_scores, topk_m, topk_h, topk_label_scores, topk_label_idxes):
        topk_arc_scores, topk_m, topk_h, topk_label_scores, topk_label_idxes = \
            (BK.get_value(z) for z in (topk_arc_scores, topk_m, topk_h, topk_label_scores, topk_label_idxes))
        new_states = []
        # for each batch element
        for one_state, one_mask, one_arc_scores, one_ms, one_hs, one_label_scores, one_labels in \
                zip(flattened_states, scoring_mask_ct, topk_arc_scores, topk_m, topk_h, topk_label_scores, topk_label_idxes):
            one_new_states = []
            # for each of the k arc selection
            for cur_arc_score, cur_m, cur_h, cur_label_scores, cur_labels in \
                    zip(one_arc_scores, one_ms, one_hs, one_label_scores, one_labels):
                # first need that selection to be valid
                cur_arc_score, cur_m, cur_h = cur_arc_score.item(), cur_m.item(), cur_h.item()
                if one_mask[cur_m, cur_h].item() > 0.:
                    # for each of the label
                    for this_label_score, this_label in zip(cur_label_scores, cur_labels):
                        this_label_score, this_label = this_label_score.item(), this_label.item()
                        # todo(note): actually add new state; do not include label score if label does not come from ef
                        cur_all_score = (cur_arc_score+this_label_score) if self.system_labeled else cur_arc_score
                        this_new_state = one_state.build_next(action=EfAction(cur_h, cur_m, this_label), score=cur_all_score)
                        one_new_states.append(this_new_state)
            new_states.append(one_new_states)
        return new_states

    # get oracle mask from OracleManager
    def _get_oracle_mask(self, flattened_states):
        # todo(note): fixed oracle masks
        cur_cache = self.cache
        return cur_cache.oracle_mask_t, cur_cache.oracle_label_t

    # =====
    # these two only return flattened results

    def select_plain(self, ags: List[BfsAgenda], candidates, mode, k_arc, k_label) -> List[List]:
        flattened_states, cur_arc_scores, scoring_mask_ct = candidates
        cur_cache = self.cache
        cur_bsize = len(flattened_states)
        cur_slen = cur_cache.max_slen
        cur_arc_scores_flattend = cur_arc_scores.view([cur_bsize, -1])  # [bs, Lm*Lh]
        if mode == "topk":
            # arcs [*, k]
            topk_arc_scores, topk_arc_idxes = BK.topk(
                cur_arc_scores_flattend, min(k_arc, BK.get_shape(cur_arc_scores_flattend, -1)), dim=-1, sorted=False)
            topk_m, topk_h = topk_arc_idxes / cur_slen, topk_arc_idxes % cur_slen  # [m, h]
            # labels [*, k, k']
            cur_label_scores = cur_cache.get_selected_label_scores(topk_m, topk_h, self.mw_arc, self.mw_label)
            topk_label_scores, topk_label_idxes = BK.topk(
                cur_label_scores, min(k_label, BK.get_shape(cur_label_scores, -1)), dim=-1, sorted=False)
            return self._new_states(flattened_states, scoring_mask_ct, topk_arc_scores, topk_m, topk_h,
                                    topk_label_scores, topk_label_idxes)
        elif mode == "":
            return [[]] * cur_bsize
        # todo(+N): other modes like sampling to be implemented: sample, topk-sample
        else:
            raise NotImplementedError(mode)

    def select_oracle(self, ags: List[BfsAgenda], candidates, mode, k_arc, k_label) -> List[List]:
        flattened_states, cur_arc_scores, scoring_mask_ct = candidates
        cur_cache = self.cache
        cur_bsize = len(flattened_states)
        cur_slen = cur_cache.max_slen
        if mode == "topk":
            # todo(note): there can be multiple oracles, select topk(usually top1) in this mode.
            # get and apply oracle mask
            cur_oracle_mask_t, cur_oracle_label_t = self._get_oracle_mask(flattened_states)
            # [bs, Lm*Lh]
            cur_oracle_arc_scores = (cur_arc_scores + Constants.REAL_PRAC_MIN*(1.-cur_oracle_mask_t)).view([cur_bsize, -1])
            # arcs [*, k]
            topk_arc_scores, topk_arc_idxes = BK.topk(cur_oracle_arc_scores, k_arc, dim=-1, sorted=False)
            topk_m, topk_h = topk_arc_idxes / cur_slen, topk_arc_idxes % cur_slen  # [m, h]
            # labels [*, k, 1]
            # todo(note): here we gather labels since one arc can only have one oracle label
            cur_label_scores = cur_cache.get_selected_label_scores(topk_m, topk_h, 0., 0.)  # [*, k, labels]
            topk_label_idxes = cur_oracle_label_t[cur_cache.bsize_range_t.unsqueeze(-1), topk_m, topk_h].unsqueeze(-1)  # [*, k, 1]
            # todo(+N): here is the trick to avoid repeated calculations, maybe not correct when using full dynamic oracle
            topk_label_scores = BK.gather(cur_label_scores, topk_label_idxes, -1) - self.mw_label
            # todo(+N): here use both masks, which may lead to no oracles! Can we simply drop the oracle_mask?
            return self._new_states(flattened_states, scoring_mask_ct*cur_cache.oracle_mask_ct, topk_arc_scores,
                                    topk_m, topk_h, topk_label_scores, topk_label_idxes)
        elif mode == "":
            return [[]] * cur_bsize
        # todo(+N): other modes like sampling to be implemented: sample, topk-sample, gather
        else:
            raise NotImplementedError(mode)

# the same as the general one
EfGlobalArranger = BfsGlobalArranger

# the most basic ender
class EfEnder(BfsEnder):
    def __init__(self, ending_mode):
        super().__init__(ending_mode)

    def is_end(self, state: EfState):
        return state.num_rest <= 0

#
class EfSearchConf(Conf):
    def __init__(self):
        # general system / expander
        self.ef_mode = "free"  # which ef system
        self.nf_dist = 2  # distance for n2f mode
        self._system_labeled = True  # set by outside!!
        # local selector
        self.plain_mode = "topk"  # topk/sample/... (empty "" means Nope)
        self.plain_k_arc = 1
        self.plain_k_label = 1
        self.oracle_mode = "topk"
        self.oracle_k_arc = 1
        self.oracle_k_label = 1
        # global selector
        self.plain_beam_size = 1
        self.gold_beam_size = 1
        self.cost_type = "full"  # "" means no coster
        self.cost_weight_arc = 1.  # arc-cost for margin
        self.cost_weight_label = 1.  # label-cost for margin
        self.sig_type = "labeled"  # labeled/unlabeled/""
        # ender
        self.ending_mode = "plain"  # plain/eu/bso/maxv
        pass

# todo(warn): currently can only search from scratch
# build(searcher); foreach-run { start; go; }
class EfSearcher(BfsSearcher):
    def __init__(self, sconf: EfSearchConf, scorer: Scorer, slayer: SL0Layer):
        super().__init__()
        self.scorer: Scorer = scorer
        self.slayer: SL0Layer = slayer
        self.state_builder: StateBuilder = None
        #
        self.system_labeled = sconf._system_labeled
        self.cost_weight_arc, self.cost_weight_label = sconf.cost_weight_arc, sconf.cost_weight_label

    def __repr__(self):
        return f"<EfSearcher: {' '.join([str(x) for x in [self.state_builder, self.expander, self.local_selector, self.global_arranger, self.ender]])}>"

    # todo(+3): ugly!! here set both local/global ones (to the same value of arc)
    def set_plain_arc_beam_size(self, k):
        assert k>=1
        self.local_selector.plain_k_arc = k
        self.global_arranger.plain_beam_size = k

    # init the states and agendas + build cache + refresh components
    def start(self, insts: List[ParseInstance], hm_feature_getter, enc_repr, enc_mask_arr, g1_pack, margin=0., require_sg=False):
        # build cache and refresh
        max_slen = enc_mask_arr.shape[-1]  # padded max sent length
        cache = EfRunningCache(self.scorer, self.slayer, hm_feature_getter, max_slen, len(insts),
                               enc_repr, enc_mask_arr, g1_pack, insts, self.system_labeled)
        self.expander.refresh(cache, margin, self.cost_weight_arc, self.cost_weight_label)
        self.local_selector.refresh(cache, margin, self.cost_weight_arc, self.cost_weight_label)
        self.global_arranger.refresh(margin)
        self.ender.refresh(margin)
        # build init states and agendas
        agendas = []
        for orig_bidx, inst in enumerate(insts):
            sg = Graph() if require_sg else None
            init_state = self.state_builder.build(sg=sg, inst=inst, orig_bidx=orig_bidx, max_slen=max_slen)
            agendas.append(BfsAgenda(inst, init_beam=[init_state]))
        return agendas

    # =====
    # the building/creating of the Searcher
    @staticmethod
    def build(sconf: EfSearchConf, scorer: Scorer, slayer: SL0Layer):
        s = EfSearcher(sconf, scorer, slayer)
        oracler = EfOracler()
        if sconf.cost_type == "":
            coster = None
        else:
            coster = EfCoster(sconf.cost_weight_arc, sconf.cost_weight_label)
        if sconf.sig_type == "":
            signaturer = None
        else:
            signaturer = EfSignaturer(sconf.sig_type == "labeled")
        s.expander = EfExpander(scorer)
        s.local_selector = EfLocalSelector(scorer, sconf.plain_mode, sconf.oracle_mode, sconf.plain_k_arc, sconf.plain_k_label, sconf.oracle_k_arc, sconf.oracle_k_label, oracler, sconf._system_labeled)
        s.global_arranger = EfGlobalArranger(sconf.plain_beam_size, sconf.gold_beam_size, coster, signaturer)
        s.ender = EfEnder(sconf.ending_mode)
        s.state_builder = StateBuilder(sconf.ef_mode, sconf.nf_dist)
        #
        zlog("Finish building the searcher: " + str(s))
        return s

# tasks/zdpar/ef/systems/base_search:397
