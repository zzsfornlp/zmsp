#

# various helper functions

__all__ = [
    "split_at_dim", "prepare_step_inputs", "mask2posi", "mask2posi_padded", "expand_ranged_idxes",
    "apply_piece_pooling", "log_sum_exp", "select_topk", "select_topk_non_overlapping", "EntropyHelper",
    "sum_or_none", "topk_avg",
]

from typing import Union, Callable
import numpy as np
from msp2.nn import BK
from msp2.utils import Constants
from .base import ActivationHelper

# helper function on split to 1
def split_at_dim(expr: BK.Expr, dim: int, keep_dim: bool):
    if keep_dim:
        return BK.split(expr, 1, dim=dim)
    else:
        # todo(+N): bug??
        # return [z.squeeze(1) for z in BK.split(expr, 1, dim=dim)]
        return [z.squeeze(dim) for z in BK.split(expr, 1, dim=dim)]

# prepare step inputs
def prepare_step_inputs(input_expr: BK.Expr, mask_expr: BK.Expr, step_dim: int, keep_dim: bool):
    step_inputs = split_at_dim(input_expr, step_dim, keep_dim)  # Iter[(bsize, D)]
    step_size = len(step_inputs)
    if mask_expr is None:
        step_masks = [None] * step_size
    else:
        step_masks = split_at_dim(mask_expr, step_dim, keep_dim)
        assert len(step_masks) == step_size, "input and mask size mismatch"
    return step_inputs, step_masks

# mask2position: [bsize, ssize]
# eg: {-1,0} => [0,0,1,1,0,1] -> [0(-1),0(-1),0,1,1,2]
def mask2posi(mask: BK.Expr, offset: int, cmin: int):
    positions = BK.cumsum(mask, 1).long() + offset  # [*, step]
    # todo(note): if all invalid, then first one!
    positions.clamp_(min=cmin)  # inplace clip!
    return positions

# special mask2position: padded version [bsize, ssize]
# eg: {0,-1} => [0,1,1,0,0,1,0] -> [-1,1,2,2,2,5,5]
def mask2posi_padded(mask: BK.Expr, offset: int, cmin: int):
    with BK.no_grad_env():
        bsize, ssize = BK.get_shape(mask)
        ret = BK.arange_idx(ssize).repeat(bsize, 1)  # [1, ssize]
        rmask_long_t = (mask==0.).long()  # reverse-mask [bsize, ssize]
        conti_zeros = BK.constants_idx([bsize], 0)  # [bsize], number of continous zeros
        for sidx in range(ssize):
            slice = rmask_long_t[:, sidx]  # [bsize]
            conti_zeros = (conti_zeros + slice) * slice  # [bsize], *slice to reset
            ret[:, sidx] -= conti_zeros
        # --
        ret += offset
        ret.clamp_(min=cmin)
        return ret

# --
# expand ranged idxes: add another axis at the end
# widx[*], wlen[*], PAD -> [*, max_width]
def expand_ranged_idxes(widx_t: BK.Expr, wlen_t: BK.Expr, pad: int = 0, max_width: int = None):
    if max_width is None:  # if not provided
        if BK.is_zero_shape(wlen_t):
            max_width = 1  # at least one
        else:
            max_width = wlen_t.max().item()  # overall max width
    # --
    input_shape = BK.get_shape(widx_t)  # [*]
    mw_range_t = BK.arange_idx(max_width).view([1]*len(input_shape)+[-1])  # [*, MW]
    expanded_idxes = widx_t.unsqueeze(-1) + mw_range_t  # [*, MW]
    expanded_masks_bool = (mw_range_t < wlen_t.unsqueeze(-1))  # [*, MW]
    expanded_idxes.masked_fill_(~expanded_masks_bool, pad)  # [*, MW]
    return expanded_idxes, expanded_masks_bool.float()

# --
# apply piece and pooling
def apply_piece_pooling(t: BK.Expr, piece: int,
                        f: Union[Callable, str] = ActivationHelper.get_pool('max'), dim: int = -1):
    # first do things like chunk by piece
    if piece == 1:
        return t  # nothing to do
    # reshape
    orig_shape = BK.get_shape(t)
    if dim<0:  # should do this!
        dim = len(orig_shape) + dim
    orig_shape[dim] = piece  # replace it with piece
    new_shape = orig_shape[:dim] + [-1] + orig_shape[dim:]  # put before it
    reshaped_t = t.view(new_shape)  # [..., -1, piece, ...]
    if isinstance(f, str):
        f = ActivationHelper.get_pool(f)
    return f(reshaped_t, dim+1)  # +1 since we make a new dim

# log sum exp
def log_sum_exp(t: BK.Expr, dim: int, t_max: BK.Expr = None):
    if t_max is None:
        t_max, _ = t.max(dim)  # get maximum value; [*, *]
    ret = t_max + (t - t_max.unsqueeze(dim)).exp().sum(dim).log()  # [*, *]
    return ret

# --
# several select functions

# [*, D, *], [*, 1, *], [*, D, *]
def select_topk(score_t: BK.Expr, topk_t: Union[int,BK.Expr], mask_t: BK.Expr=None, dim=-1):
    # prepare K
    if isinstance(topk_t, int):
        K = topk_t
        tmp_shape = BK.get_shape(score_t)
        tmp_shape[dim] = 1  # set it as 1
        topk_t = BK.constants_idx(tmp_shape, K)
    else:
        K = topk_t.max().item()
    exact_rank_t = topk_t - 1  # [bsize, 1]
    exact_rank_t.clamp_(min=0, max=K-1)  # make it in valid range!
    # mask values
    if mask_t is not None:
        score_t = score_t + Constants.REAL_PRAC_MIN * (1. - mask_t)
    # topk
    topk_vals, _ = score_t.topk(K, dim, largest=True, sorted=True)  # [*, K, *]
    # gather score
    sel_thresh = topk_vals.gather(dim, exact_rank_t)  # [*, 1, *]
    # get topk_mask
    topk_mask = (score_t >= sel_thresh).float()  # [*, D, *]
    if mask_t is not None:
        topk_mask *= mask_t
    return topk_mask

# [*, D, *], [*, 1, *], [*, D, *] ;; [*, D, *], [*, L, *]
# select non overlapping ones
def select_topk_non_overlapping(score_t: BK.Expr, topk_t: Union[int,BK.Expr],
                                widx_t: BK.Expr, wlen_t: BK.Expr, input_mask_t: BK.Expr,
                                mask_t: BK.Expr=None, dim=-1):
    score_shape = BK.get_shape(score_t)
    assert dim==-1 or dim==len(score_shape-1), "Currently only support last-dim!!"  # todo(+2): we can permute to allow any dim!
    # --
    # prepare K
    if isinstance(topk_t, int):
        tmp_shape = score_shape.copy()
        tmp_shape[dim] = 1  # set it as 1
        topk_t = BK.constants_idx(tmp_shape, topk_t)
    # --
    reshape_trg = [np.prod(score_shape[:-1]).item(), -1]  # [*, ?]
    _, sorted_idxes_t = score_t.sort(dim, descending=True)
    # --
    # put it as CPU and use loop; todo(+N): more efficient ways?
    arr_sorted_idxes_t, arr_topk_t, arr_widx_t, arr_wlen_t, arr_input_mask_t, arr_mask_t = \
        [BK.get_value(z.reshape(reshape_trg)) for z in [sorted_idxes_t, topk_t, widx_t, wlen_t, input_mask_t, mask_t]]
    _bsize, _cnum = BK.get_shape(arr_sorted_idxes_t)  # [bsize, NUM]
    arr_topk_mask = np.full([_bsize, _cnum], 0.)  # [bsize, NUM]
    _bidx = 0
    for aslice_sorted_idxes_t, aslice_topk_t, aslice_widx_t, aslice_wlen_t, aslice_input_mask_t, aslice_mask_t \
            in zip(arr_sorted_idxes_t, arr_topk_t, arr_widx_t, arr_wlen_t, arr_input_mask_t, arr_mask_t):
        aslice_topk_mask = arr_topk_mask[_bidx]
        # --
        cur_ok_mask = np.copy(aslice_input_mask_t)
        cur_budget = aslice_topk_t.item()
        for _cidx in aslice_sorted_idxes_t:
            _cidx = _cidx.item()
            if cur_budget<=0: break  # no budget left
            if not aslice_mask_t[_cidx].item(): continue  # non-valid candidate
            one_widx, one_wlen = aslice_widx_t[_cidx].item(), aslice_wlen_t[_cidx].item()
            if np.prod(cur_ok_mask[one_widx:one_widx+one_wlen]).item() == 0.:  # any hit one?
                continue
            # ok! add it!
            cur_budget -= 1
            cur_ok_mask[one_widx:one_widx + one_wlen] = 0.
            aslice_topk_mask[_cidx] = 1.
        _bidx += 1
    # note: no need to *=mask_t again since already check in the loop
    return BK.input_real(arr_topk_mask).reshape(score_shape)


# =====
# various entropy

class EntropyHelper:
    @staticmethod
    def _entropy(p, q, qlog_ival, dim=-1):
        # - sum p*(q.log() if q>0 else q_zero_val)
        q_valid = (q>0).float()
        q_invalid = 1. - q_valid
        q2 = q_valid * q + q_invalid * 1.  # make it valid to log
        q_log = q_valid * q2.log() + q_invalid * qlog_ival
        ret = - (p*q_log).sum(dim)
        return ret

    @staticmethod
    def self_entropy(p, dim=-1):
        return EntropyHelper._entropy(p, p, 0., dim=dim)

    @staticmethod
    def cross_entropy(p, q, dim=-1):
        return EntropyHelper._entropy(p, q, Constants.REAL_PRAC_MIN, dim=dim)

# --
# sum a list of tensors
def sum_or_none(ts):
    ts = [t for t in ts if t is not None]
    return None if len(ts)==0 else BK.stack(ts, dim=0).sum(dim=0)

# topk and average (note: special semantics, k==0 means no-avg!!)
def topk_avg(val_t, mask_t, k, dim=-1, largest=True):
    if k<=0 or k>BK.get_shape(val_t, dim):
        # no need to topk, simply sum all!
        topk_val, topk_mask = val_t, mask_t
    else:
        topk_val, topk_idx = val_t.topk(k, dim=dim, largest=largest, sorted=False)
        topk_mask = mask_t.expand_as(val_t).gather(dim, topk_idx)
    ret = topk_val.sum(dim)
    if k != 0:  # note: k==0 means sum
        ret /= (topk_mask.sum(dim) + 1e-12)
    return ret
