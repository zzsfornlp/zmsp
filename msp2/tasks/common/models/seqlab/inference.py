#

# some simple inference helpers

__all__ = [
    "BigramInferenceHelper", "SimpleInferencer",
]

from typing import List
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.utils import Constants

# =====
# helper for CRF-styled losser and searcher

class BigramInferenceHelper:
    # forward inference, return logsumexp potentials (partition)
    # [*, slen, L], [L, L], [*, slen], beam_size -> [*]
    @staticmethod
    def inference_forward(scores_t: BK.Expr, mat_t: BK.Expr, mask_t: BK.Expr, beam_k: int = 0):
        scores_shape = BK.get_shape(scores_t)  # [*, slen, L]
        need_topk = (beam_k > 0) and (beam_k < scores_shape[-1])  # whether we need topk
        # --
        score_slices = split_at_dim(scores_t, -2, True)  # List[*, 1, L]
        mask_slices = split_at_dim(mask_t, -1, True)  # List[*, 1]
        # the loop on slen
        start_shape = scores_shape[:-2] + [1]  # [*, 1]
        last_labs_t = BK.constants_idx(start_shape, 0)  # [*, K], todo(note): start with 0!
        last_accu_scores = BK.zeros(start_shape)  # accumulated scores: [*, K]
        last_potential = BK.zeros(start_shape)  # accumulated potentials: [*, K]
        full_labs_t = BK.arange_idx(scores_shape[-1]).view([1]*(len(scores_shape)-2) + [-1])  # [*, L]
        cur_step = 0
        for one_score_slice, one_mask_slice in zip(score_slices, mask_slices):  # [*,L],[*,1]
            one_mask_slice_neg = 1. - one_mask_slice  # [*,1]
            # get current scores
            if cur_step == 0:  # no transition at start!
                one_cur_scores = one_score_slice  # [*, 1, L]
            else:
                one_cur_scores = one_score_slice + mat_t[last_labs_t]  # [*, K, L]
            # first for potentials
            expanded_potentials = last_potential.unsqueeze(-1) + one_cur_scores  # [*, K, L]
            merged_potentials = log_sum_exp(expanded_potentials, -2)  # [*, L]
            # optional for topk with merging; note: not really topk!!
            if need_topk:
                # todo(+W): another option is to directly select with potentials rather than accu_scores
                expanded_scores = last_accu_scores.unsqueeze(-1) + one_cur_scores  # [*, K, L]
                # max at -2, merge same current label
                max_scores, max_idxes = expanded_scores.max(-2)  # [*, L]
                # topk at current step, no need to sort!
                new_accu_scores, new_labs_t = max_scores.topk(beam_k, -1, sorted=False)  # [*, K]
                new_potential = merged_potentials.gather(-1, new_labs_t)  # [*, K]
                # mask and update
                last_potential = last_potential * one_mask_slice_neg + new_potential * one_mask_slice  # [*, K]
                last_accu_scores = last_accu_scores * one_mask_slice_neg + new_accu_scores * one_mask_slice  # [*, K]
                last_labs_t = last_labs_t * one_mask_slice_neg.long() + new_labs_t * one_mask_slice.long()  # [*, K]
            else:
                # mask and update
                last_potential = last_potential * one_mask_slice_neg + merged_potentials * one_mask_slice  # [*, L(K)]
                # note: still need to mask this!
                last_labs_t = last_labs_t * one_mask_slice_neg.long() + full_labs_t * one_mask_slice.long()
            cur_step += 1
        # finally sum all
        ret_potential = log_sum_exp(last_potential, -1)  # [*]
        return ret_potential

    # viterbi or with approximation like beam search
    # [*, slen, L], [L, L], [*, slen], beam_size -> idxes [*, slen], scores [*]
    @staticmethod
    def inference_search(scores_t: BK.Expr, mat_t: BK.Expr, mask_t: BK.Expr, beam_k: int = 0):
        scores_shape = BK.get_shape(scores_t)  # [*, slen, L]
        need_topk = (beam_k > 0) and (beam_k < scores_shape[-1])  # whether we need topk
        # --
        score_slices = split_at_dim(scores_t, -2, True)  # List[*, 1, L]
        mask_slices = split_at_dim(mask_t, -1, True)  # List[*, 1]
        # the loop on slen
        all_sel_labs = []  # List of [*, K]
        all_sel_scores = []  # List of [*, K]
        all_tracebacks = []  # List of [*, K]
        start_vals_shape = scores_shape[:-2] + [1]  # [*, 1]
        full_idxes_shape = scores_shape[:-2] + [-1]  # [*, ?]
        last_labs_t = BK.constants_idx(start_vals_shape, 0)  # [*, K], todo(note): start with 0!
        last_accu_scores = BK.zeros(start_vals_shape)  # accumulated scores: [*, K]
        full_labs_t = BK.arange_idx(scores_shape[-1]).expand(full_idxes_shape)  # [*, L]
        cur_step = 0
        for one_score_slice, one_mask_slice in zip(score_slices, mask_slices):  # [*,L],[*,1]
            one_mask_slice_neg = 1. - one_mask_slice  # [*,1]
            # get current scores
            if cur_step == 0:  # no transition at start!
                one_cur_scores = one_score_slice  # [*, 1, L]
            else:  # len(all_sel_labs) must >0
                one_cur_scores = one_score_slice + mat_t[last_labs_t]  # [*, K, L]
            # expand scores
            expanded_scores = last_accu_scores.unsqueeze(-1) + one_cur_scores  # [*, K, L]
            # max at -2, merge same current label
            max_scores, max_idxes = expanded_scores.max(-2)  # [*, L]
            # need topk?
            if need_topk:
                # topk at current step, no need to sort!
                new_accu_scores, new_labs_t = max_scores.topk(beam_k, -1, sorted=False)  # [*, K]
                new_traceback = max_idxes.gather(-1, new_labs_t)  # [*, K]
                last_labs_t = last_labs_t * one_mask_slice_neg.long() + new_labs_t * one_mask_slice.long()  # [*, K]
            else:
                new_accu_scores = max_scores  # [*, L(K)]
                new_traceback = max_idxes
                # note: still need to mask this!
                last_labs_t = last_labs_t * one_mask_slice_neg.long() + full_labs_t * one_mask_slice.long()
            # mask and update
            last_accu_scores = last_accu_scores * one_mask_slice_neg + new_accu_scores * one_mask_slice  # [*, K]
            default_traceback = BK.arange_idx(BK.get_shape(expanded_scores, -2))\
                .view([1]*(len(scores_shape)-2) + [-1])  # [*, K(arange)]
            last_traceback_t = default_traceback * one_mask_slice_neg.long() + new_traceback * one_mask_slice.long()  # [*, K]
            all_sel_labs.append(last_labs_t)
            all_tracebacks.append(last_traceback_t)
            one_new_scores = one_cur_scores[BK.arange_idx(scores_shape[0]).unsqueeze(-1), last_traceback_t, last_labs_t]  # [*, K]
            one_new_scores *= one_mask_slice
            all_sel_scores.append(one_new_scores)
            cur_step += 1
        # traceback
        _, last_idxes = last_accu_scores.max(-1)  # [*]
        last_idxes = last_idxes.unsqueeze(-1)  # [*, 1]
        all_preds, all_scores = [], []
        for cur_step in range(len(all_tracebacks)-1, -1, -1):
            all_preds.append(all_sel_labs[cur_step].gather(-1, last_idxes).squeeze(-1))  # [*]
            all_scores.append(all_sel_scores[cur_step].gather(-1, last_idxes).squeeze(-1))  # [*]
            last_idxes = all_tracebacks[cur_step].gather(-1, last_idxes)  # [*, 1]
        # remember to reverse!!
        all_preds.reverse()
        all_scores.reverse()
        best_labs = BK.stack(all_preds, -1)  # [*, slen]
        best_scores = BK.stack(all_scores, -1)  # [*, slen]
        return best_labs, best_scores  # [*, slen]

# =====
# simple searcher (template method)
# -- for those not complicated to use State and FullSearcher with merging (mainly s2s styled)

class SimpleInferencer:
    def beam_search(self, batch_size: int, beam_k: int, ret_best: bool = True):
        _NEG_INF = Constants.REAL_PRAC_MIN
        # --
        cur_step = 0
        cache: DecCache = None
        # init: keep the seq of scores rather than traceback!
        start_vals_shape = [batch_size, 1]  # [bs, 1]
        all_preds_t = BK.constants_idx(start_vals_shape, 0).unsqueeze(-1)  # [bs, K, step], todo(note): start with 0!
        all_scores_t = BK.zeros(start_vals_shape).unsqueeze(-1)  # [bs, K, step]
        accu_scores_t = BK.zeros(start_vals_shape)  # [bs, K]
        arange_t = BK.arange_idx(batch_size).unsqueeze(-1)  # [bs, 1]
        # while loop
        prev_k = 1  # start with single one
        while not self.is_end(cur_step):
            # expand and score
            cache, scores_t, masks_t = self.step_score(cur_step, prev_k, cache)  # ..., [bs*pK, L], [bs*pK]
            scores_t_shape = BK.get_shape(scores_t)
            last_dim = scores_t_shape[-1]  # L
            # modify score to handle mask: keep previous pred for the masked items!
            sel_scores_t = BK.constants([batch_size, prev_k, last_dim], 1.)  # [bs, pk, L]
            sel_scores_t.scatter_(-1, all_preds_t[:,:,-1:], -1)  # [bs, pk, L]
            sel_scores_t = scores_t + _NEG_INF * (sel_scores_t.view(scores_t_shape) * (1.-masks_t).unsqueeze(-1))  # [bs*pK, L]
            # first select topk locally, note: here no need to sort!
            local_k = min(last_dim, beam_k)
            l_topk_scores, l_topk_idxes = sel_scores_t.topk(local_k, -1, sorted=False)  # [bs*pK, lK]
            # then topk globally on full pK*K
            add_score_shape = [batch_size, prev_k, local_k]
            to_sel_shape = [batch_size, prev_k*local_k]
            global_k = min(to_sel_shape[-1], beam_k)  # new k
            to_sel_scores, to_sel_idxes = \
                (l_topk_scores.view(add_score_shape) + accu_scores_t.unsqueeze(-1)).view(to_sel_shape), \
                l_topk_idxes.view(to_sel_shape)  # [bs, pK*lK]
            _, g_topk_idxes = to_sel_scores.topk(global_k, -1, sorted=True)  # [bs, gK]
            # get to know the idxes
            new_preds_t = to_sel_idxes.gather(-1, g_topk_idxes)  # [bs, gK]
            new_pk_idxes = (g_topk_idxes // local_k)  # which previous idx (in beam) are selected? [bs, gK]
            # get current pred and scores (handling mask)
            scores_t3 = scores_t.view([batch_size, -1, last_dim])  # [bs, pK, L]
            masks_t2 = masks_t.view([batch_size, -1])  # [bs, pK]
            new_masks_t = masks_t2[arange_t, new_pk_idxes]  # [bs, gK]
            # -- one-step score for new selections: [bs, gK], note: zero scores for masked ones
            new_scores_t = scores_t3[arange_t, new_pk_idxes, new_preds_t] * new_masks_t  # [bs, gK]
            # ending
            new_arrange_idxes = (arange_t * prev_k + new_pk_idxes).view(-1)  # [bs*gK]
            cache.arrange_idxes(new_arrange_idxes)
            self.step_end(cur_step, global_k, cache, new_preds_t.view(-1))  # modify in cache
            # prepare next & judge ending
            all_preds_t = BK.concat([all_preds_t[arange_t, new_pk_idxes], new_preds_t.unsqueeze(-1)], -1)  # [bs, gK, step]
            all_scores_t = BK.concat([all_scores_t[arange_t, new_pk_idxes], new_scores_t.unsqueeze(-1)], -1)  # [bs, gK, step]
            accu_scores_t = accu_scores_t[arange_t, new_pk_idxes] + new_scores_t  # [bs, gK]
            prev_k = global_k  # for next step
            cur_step += 1
        # --
        # sort and ret at a final step
        _, final_idxes = accu_scores_t.topk(prev_k, -1, sorted=True)  # [bs, K]
        ret_preds = all_preds_t[arange_t, final_idxes][:,:,1:]  # [bs, K, steps], exclude dummy start!
        ret_scores = all_scores_t[arange_t, final_idxes][:,:,1:]  # [bs, K, steps]
        if ret_best:
            return ret_preds[:,0], ret_scores[:,0]  # [bs, slen]
        else:
            return ret_preds, ret_scores  # [bs, topk, slen]

    # =====
    # to be implemented
    def step_score(self, cur_step: int, cur_repeat: int, cache: DecCache): raise NotImplementedError()
    def step_end(self, cur_step: int, cur_repeat: int, cache: DecCache, preds: BK.Expr): raise NotImplementedError()
    def is_end(self, cur_step: int): raise NotImplementedError()
