#

# algorithms for seqlab

__all__ = [
    "SeqlabHelper", "SeqlabHelperConf",
]

from mspx.utils import Conf, Configurable, ZHelper
from mspx.nn import BK, split_at_dim, log_sum_exp

class SeqlabHelperConf(Conf):
    def __init__(self):
        super().__init__()
        self.mode = 'crf'  # crf or greedy
        self.crf_beam_k = 0  # 0 means include all!
        # --

# note: if in-middle mask=0., simply force it as 0 and the L/R pieces are separate!
@SeqlabHelperConf.conf_rd()
class SeqlabHelper(Configurable):
    def __init__(self, conf: SeqlabHelperConf = None, **kwargs):
        super().__init__(conf, **kwargs)
        # --
        conf: SeqlabHelperConf = self.conf
        self.is_crf, self.is_greedy = ZHelper.check_hit_one(conf.mode, ['crf', 'greedy'])
        self.crf_beam_k = conf.crf_beam_k

    @property
    def requires_binary(self):
        return self.is_crf

    # forward inference, return logsumexp potentials (partition)
    # [*, slen, L], [L, L], [*, slen], [*, slen, B], beam_size -> [*]
    def _crf_forward(self, scores_t: BK.Expr, mat_t: BK.Expr, mask_t: BK.Expr,
                     beam_cons_t: BK.Expr = None, beam_k: int = 0):
        scores_shape = BK.get_shape(scores_t)  # [*, slen, L]
        need_topk = ((beam_k > 0) and (beam_k < scores_shape[-1])) or (beam_cons_t is not None)  # whether we need topk
        # --
        score_slices = split_at_dim(scores_t, -2, True)  # List[*, 1, L]
        mask_slices = split_at_dim(mask_t, -1, True)  # List[*, 1]
        bc_slices = [None] * len(score_slices) if beam_cons_t is None else split_at_dim(beam_cons_t, -2, True)  # [*, 1, B]
        # the loop on slen
        start_shape = scores_shape[:-2] + [1]  # [*, 1]
        last_labs_t = BK.constants_idx(start_shape, 0)  # [*, K?], todo(note): start with 0!
        last_mask_t = last_labs_t.to(BK.DEFAULT_FLOAT)  # [*, 1]
        seq_valid_t = BK.zeros(start_shape)  # whether start the seq (meet non-mask)? [*, 1]
        last_accu_scores = BK.zeros(start_shape)  # accumulated scores: [*, K]
        last_potential = BK.zeros(start_shape)  # accumulated potentials(logsumexp): [*, K]
        full_labs_t = BK.arange_idx(scores_shape[-1]).view([1]*(len(scores_shape)-2) + [-1])  # [*, L]
        cur_step = 0
        # store all info
        all_labs, all_potentials = [], []  # finally: List[*, K?]
        for one_score_slice, one_mask_slice, one_bc_slice in zip(score_slices, mask_slices, bc_slices):
            one_mask_slice_neg = 1. - one_mask_slice  # [*,1]
            # get current scores (0. if last is invalid!)
            one_cur_scores = one_score_slice + mat_t[last_labs_t] * last_mask_t.unsqueeze(-1)  # [*, K, L]
            # first for potentials
            expanded_potentials = last_potential.unsqueeze(-1) + one_cur_scores  # [*, K, L]
            merged_potentials = log_sum_exp(expanded_potentials, -2) * seq_valid_t \
                                + one_score_slice.squeeze(-2) * (1.-seq_valid_t)  # full [*, L]
            # optional for topk with merging; note: not really topk!!
            if need_topk:
                # todo(+W): another option is to directly select with potentials rather than accu_scores
                expanded_scores = last_accu_scores.unsqueeze(-1) + one_cur_scores  # [*, K, L]
                # max at -2, merge same current label
                max_scores, max_idxes = expanded_scores.max(-2)  # note: take maximum [*, L]
                if one_bc_slice is not None:
                    new_labs_t = one_bc_slice.squeeze(-2)  # [*, K]
                    new_accu_scores = max_scores.gather(-1, new_labs_t)  # [*, K]
                else:
                    # topk at current step, no need to sort!
                    new_accu_scores, new_labs_t = max_scores.topk(beam_k, -1, sorted=False)  # [*, K]
                new_potential = merged_potentials.gather(-1, new_labs_t)  # [*, K]
                # mask and update
                last_potential = last_potential * one_mask_slice_neg + new_potential * one_mask_slice  # [*, K]
                last_accu_scores = last_accu_scores * one_mask_slice_neg + new_accu_scores * one_mask_slice  # [*, K]
                last_labs_t = new_labs_t * one_mask_slice.long()  # [*, K]
            else:
                # mask and update
                last_potential = last_potential * one_mask_slice_neg + merged_potentials * one_mask_slice  # [*, L(K)]
                # note: still need to mask this!
                last_labs_t = full_labs_t * one_mask_slice.long()
            # --
            last_mask_t = one_mask_slice
            seq_valid_t = (seq_valid_t + one_mask_slice).clamp(max=1.)
            cur_step += 1
            all_labs.append(last_labs_t)
            all_potentials.append(last_potential)
        # finally sum all
        ret_potential = log_sum_exp(last_potential, -1)  # [*]
        all_potentials_t, all_labs_t = BK.stack(all_potentials, -2), BK.stack(all_labs, -2)  # [*, L, K]
        return ret_potential, all_potentials_t, all_labs_t

    # backward (similarly)
    def _crf_backward(self, scores_t: BK.Expr, mat_t: BK.Expr, mask_t: BK.Expr,
                      beam_cons_t: BK.Expr = None, beam_k: int = 0):
        # note: simply reverse and use forward!
        r_scores_t, r_mat_t, r_mask_t = scores_t.flip(-2), mat_t.T, mask_t.flip(-1)
        r_beam_cons_t = None if beam_cons_t is None else beam_cons_t.flip(-2)
        ret_potential, r_all_potentials_t, r_all_labs_t = \
            self._crf_forward(r_scores_t, r_mat_t, r_mask_t, r_beam_cons_t, beam_k)
        all_potentials_t, all_labs_t = r_all_potentials_t.flip(-2), r_all_labs_t.flip(-2)
        return ret_potential, all_potentials_t, all_labs_t  # [..., L]

    # calculate marginal (forward and backward)
    def _crf_marginal(self, scores_t: BK.Expr, mat_t: BK.Expr, mask_t: BK.Expr,
                      beam_cons_t: BK.Expr = None, beam_k: int = 0):
        scores_shape = BK.get_shape(scores_t)  # [*, slen, L]
        need_topk = ((beam_k > 0) and (beam_k < scores_shape[-1])) or (beam_cons_t is not None)
        # do forward
        ret_potential, forw_potentials_t, forw_labs_t = \
            self._crf_forward(scores_t, mat_t, mask_t, beam_cons_t, beam_k)
        # do backward
        if beam_cons_t is None and need_topk:
            beam_cons_t = forw_labs_t  # constrained by forward beam search
        _, back_potentials_t, back_labs_t = self._crf_backward(scores_t, mat_t, mask_t, beam_cons_t)
        # extend
        if need_topk:
            _tmp = BK.zeros(scores_shape)  # [*, L, V]
            forw_t = _tmp.scatter(-1, forw_labs_t, forw_potentials_t)
            back_t = _tmp.scatter(-1, back_labs_t, back_potentials_t)
            unary_mask = BK.zeros(scores_shape)  # [*, L, V]
            unary_mask.scatter_(-1, forw_labs_t, 1.)  # assign valid ones!
            unary_mask = unary_mask * mask_t.unsqueeze(-1)
        else:
            forw_t, back_t = forw_potentials_t, back_potentials_t
            unary_mask = mask_t.unsqueeze(-1)
        # combine
        raw_unary_t = forw_t + back_t - scores_t  # [*, L, V]
        raw_binary_t = forw_t[..., :-1, :].unsqueeze(-1) + mat_t + back_t[..., 1:, :].unsqueeze(-2)  # [*, L-1, V, V]
        # exp
        m_unary_t, m_binary_t = (raw_unary_t - ret_potential.unsqueeze(-1).unsqueeze(-1)).exp(), \
                                (raw_binary_t - ret_potential.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).exp()
        # apply mask
        m_unary_t = m_unary_t * unary_mask
        m_binary_t = m_binary_t * (unary_mask[..., :-1, :].unsqueeze(-1) * unary_mask[..., 1:, :].unsqueeze(-2))
        # --
        return m_unary_t, m_binary_t

    # viterbi or with approximation like beam search
    # [*, slen, L], [L, L], [*, slen], beam_size -> idxes [*, slen], scores [*]
    def _crf_argmax(self, scores_t: BK.Expr, mat_t: BK.Expr, mask_t: BK.Expr, beam_k: int = 0):
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
        last_mask_t = last_labs_t.to(BK.DEFAULT_FLOAT)  # [*, 1]
        last_accu_scores = BK.zeros(start_vals_shape)  # accumulated scores: [*, K]
        full_labs_t = BK.arange_idx(scores_shape[-1]).expand(full_idxes_shape)  # [*, L]
        cur_step = 0
        for one_score_slice, one_mask_slice in zip(score_slices, mask_slices):  # [*,L],[*,1]
            one_mask_slice_neg = 1. - one_mask_slice  # [*,1]
            # get current scores (0. if last is invalid!)
            one_cur_scores = one_score_slice + mat_t[last_labs_t] * last_mask_t.unsqueeze(-1)  # [*, K, L]
            # expand scores
            expanded_scores = last_accu_scores.unsqueeze(-1) + one_cur_scores  # [*, K, L]
            # max at -2, merge same current label
            max_scores, max_idxes = expanded_scores.max(-2)  # [*, L]
            # need topk?
            if need_topk:
                # topk at current step, no need to sort!
                new_accu_scores, new_labs_t = max_scores.topk(beam_k, -1, sorted=False)  # [*, K]
                new_traceback = max_idxes.gather(-1, new_labs_t)  # [*, K]
                last_labs_t = new_labs_t * one_mask_slice.long()  # [*, K]
            else:
                new_accu_scores = max_scores  # [*, L(K)]
                new_traceback = max_idxes
                # note: still need to mask this!
                last_labs_t = full_labs_t * one_mask_slice.long()
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
            last_mask_t = one_mask_slice
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

    # --
    # APIs: E[*, L, V], T[V, V], mask[*, L]

    # -> [*]
    def get_partition(self, scores_t: BK.Expr, mat_t: BK.Expr, mask_t: BK.Expr):
        if self.is_crf:
            return self._crf_forward(scores_t, mat_t, mask_t, beam_k=self.crf_beam_k)[0]
        elif self.is_greedy:  # note: ignore mat_t in this mode!
            return (log_sum_exp(scores_t, -1) * mask_t).sum(-1)

    # -> [*, L, V], [*, L-1, V, V]
    def get_marginals(self, scores_t: BK.Expr, mat_t: BK.Expr, mask_t: BK.Expr):
        if self.is_crf:
            return self._crf_marginal(scores_t, mat_t, mask_t, beam_k=self.crf_beam_k)
        elif self.is_greedy:
            if mat_t is not None:  # still use fb
                return self._crf_marginal(scores_t.log_softmax(-1), mat_t, mask_t, beam_k=self.crf_beam_k)
            else:  # real greedy, simple do softmax!
                m0 = scores_t.softmax(-1) * mask_t.unsqueeze(-1)  # [*, L, V]
                m1 = m0[..., :-1, :].unsqueeze(-1) * m0[..., 1:, :].unsqueeze(-2)  # [*, L-1, V, V]
                return (m0, m1)  # form an M1 for simplicity

    # -> [*, L]
    def get_argmax(self, scores_t: BK.Expr, mat_t: BK.Expr, mask_t: BK.Expr):
        if self.is_crf:
            return self._crf_argmax(scores_t, mat_t, mask_t, beam_k=self.crf_beam_k)[0]
        elif self.is_greedy:
            if mat_t is not None:  # still use viterbi!
                return self._crf_argmax(scores_t.log_softmax(-1), mat_t, mask_t, beam_k=self.crf_beam_k)[0]
            else:  # real greedy, simply do argmax!
                ret = scores_t.argmax(-1)
                ret = ret * mask_t.to(ret)
                return ret
