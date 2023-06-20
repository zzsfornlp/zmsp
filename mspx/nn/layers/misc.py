#

__all__ = [
    "down_neg", "unsqueeze_expand", "split_at_dim", "mask2posi", "mask2posi_keeporig",
    "expand_ranged_idxes", "piece_pooling", "log_sum_exp", "EntropyHelper", "sum_or_none", "topk_avg",
    "make_causual_mask", "extend_idxes", "flatten_dims", "fulfill_idx_ranges", "label_smoothing",
]

import numpy as np
from typing import Union, Callable
from collections import OrderedDict
from mspx.utils import Constants
from ..backends import BK

# --
# down-weight for negatives, [???], [???], float
def down_neg(mask_t, pos_t, ratio: float, batch_ndims=0, do_sample=False, max_ratio=1.0):
    # set up shapes
    _shape = BK.get_shape(mask_t)  # [???]
    _reshape = _shape[:batch_ndims] + [np.prod(_shape[batch_ndims:]).item()]  # [..., *]
    m_t, p_t = mask_t.view(_reshape), (pos_t * mask_t).view(_reshape)
    n_t = m_t * (1. - p_t)  # negative ones
    # get ratios
    p_count = p_t.sum(-1, keepdims=True)  # [..., 1]
    n_count = n_t.sum(-1, keepdims=True)  # [..., 1]
    _ratio = ((p_count * ratio) / n_count.clamp(min=1.)).clamp(max=max_ratio)  # [..., 1]
    # prepare return
    if do_sample:  # actual sampling
        n_mask = (BK.rand(n_t.shape) < _ratio).to(BK.DEFAULT_FLOAT)  # [..., *]
        ret0 = m_t * (1.-n_t) + n_mask * n_t  # [..., *]
    else:  # simply change weight
        ret0 = m_t * (1.-n_t) + _ratio * n_t  # [..., *]
    ret = ret0.view(_shape)  # [???]
    return ret

# --
# unsqueeze and expand
def unsqueeze_expand(t, dim: int, size: int):
    t2 = t.unsqueeze(dim)  # [..., 1, ...]
    sizes = [-1] * len(t2.shape)
    sizes[dim] = size
    ret = t2.expand(*sizes)
    return ret

# helper function on split to 1
def split_at_dim(expr: BK.Expr, dim: int, keep_dim: bool):
    if keep_dim:
        return BK.split(expr, 1, dim=dim)
    else:
        return [z.squeeze(dim) for z in BK.split(expr, 1, dim=dim)]

# mask2position
# eg: offset=-1,cmin=0 => [0,0,1,1,0,1] -> [0(-1),0(-1),0,1,1,2]
def mask2posi(mask: BK.Expr, offset: int, cmin: int, dim=-1):
    positions = BK.cumsum(mask, dim).long() + offset  # [*, step]
    positions.clamp_(min=cmin)  # inplace clip!
    return positions

# special mask2position: keep the original arange ones, make 0s the same as previous non-0!
# eg: offset=0,cmin=-1 => [0,1,1,0,0,1,0] -> [-1,1,2,2,2,5,5]
def mask2posi_keeporig(mask: BK.Expr, offset: int, cmin: int, dim=-1):
    assert dim == -1
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
    return expanded_idxes, expanded_masks_bool.to(BK.DEFAULT_FLOAT)

# --
# apply piece and pooling
def piece_pooling(t: BK.Expr, piece_size: int = None, piece_num: int = None,
                  f: Union[Callable, str] = 'max', dim: int = -1):
    orig_shape = BK.get_shape(t)
    # first do things like chunk by piece
    if piece_size is None:
        piece_size = orig_shape[dim] // piece_num
    if BK.is_zero_shape(t):
        orig_shape[dim] = orig_shape[dim] // piece_size  # fold
        return t.view(orig_shape)
    if piece_size == 1:
        return t  # nothing to do
    # reshape
    if dim<0:  # should do this!
        dim = len(orig_shape) + dim
    new_shape = orig_shape[:dim] + [orig_shape[dim]//piece_size, piece_size] + orig_shape[dim+1:]  # put before it
    reshaped_t = t.view(new_shape)  # [..., -1, piece, ...]
    if isinstance(f, str):
        from .base import ActivationHelper
        f = ActivationHelper.get_pool(f)
    return f(reshaped_t, dim+1)  # +1 since we make a new dim

# log sum exp
def log_sum_exp(t: BK.Expr, dim: int, t_max: BK.Expr = None):
    if t_max is None:
        t_max, _ = t.max(dim)  # get maximum value; [*, *]
    ret = t_max + (t - t_max.unsqueeze(dim)).exp().sum(dim).log()  # [*, *]
    return ret

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

# sum a list of tensors
def sum_or_none(ts):
    ts = [t for t in ts if t is not None]
    return None if len(ts)==0 else BK.stack(ts, dim=0).sum(dim=0)

# topk and average (note: special semantics, k==0 means no-avg!!)
def topk_avg(val_t, mask_t, k: int, dim=-1, largest=True):
    if BK.is_zero_shape(val_t):
        return val_t.mean(dim)
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

# make causual mask
def make_causual_mask(t_or_len: Union[BK.Expr, int]):
    if BK.is_tensor(t_or_len):
        t_or_len = BK.get_shape(t_or_len, -1)
    tmp = BK.arange_idx(t_or_len)  # [L]
    ret = (tmp.unsqueeze(-1) >= tmp.unsqueeze(-2)).to(BK.DEFAULT_FLOAT)  # [Q, K]
    return ret

# extend idxes
def extend_idxes(t_idxes: BK.Expr, v: int, dim=-1):
    t_idxes = t_idxes.unsqueeze(dim)
    _shape = BK.get_shape(t_idxes)  # [..., 1, ...]
    _shape[dim] = v
    ret = BK.zeros(_shape)
    ret.scatter_(dim, t_idxes, 1.)
    return ret

# flatten certain dims
def flatten_dims(t: BK.Expr, start: int, end: int):
    shape = BK.get_shape(t)
    start = 0 if start is None else start
    end = len(shape) if end is None else end
    new_shape = shape[:start] + [np.prod(shape[start:end]).item()] + shape[end:]
    ret = t.view(new_shape)
    return ret

# prep idxes ranges: t_wlen==0 means nope, by default 0
def fulfill_idx_ranges(t_widxes, t_wlen, t_val, length: int, dim=-1):
    if not BK.is_expr(t_val):
        t_val = t_wlen * 0 + t_val  # same shape!
    idx_shape = BK.get_shape(t_widxes)  # [..., K, ...]
    dimP1 = (dim if dim>=0 else (dim + len(idx_shape))) + 1
    t2_start = t_widxes.unsqueeze(dimP1)  # [..., K, 1, ...]
    t2_end = (t_widxes + t_wlen.clamp(min=0)).unsqueeze(dimP1)  # [..., K, 1, ...]
    t_arange = BK.arange_idx(length).view([length] + [1] * (len(idx_shape) - dimP1))  # [L, ...]
    t_hit = ((t_arange >= t2_start) & (t_arange < t2_end)).to(BK.DEFAULT_INT)  # [..., K, L, ...]
    t_hit = (t_hit.cumsum(dimP1-1) <= 1).to(t_hit) * t_hit  # note: only keep the first one!
    t_res = (t_hit * t_val.unsqueeze(dimP1)).sum(dimP1-1)  # [..., L, ...]
    return t_res

# for label smoothing
def label_smoothing(t_prob, alpha, kldim=1, t_mask=None):
    _shape = BK.get_shape(t_prob)  # [prev, last]
    _last_shape = _shape[-kldim:]  # k-last-dim
    _v = BK.constants(_last_shape, value=1).view(_last_shape)  # [last]
    # apply mask
    if t_mask is not None:
        _shape2 = BK.get_shape(t_mask)
        if len(_shape2) < len(_shape):
            t_mask = t_mask.view(_shape2 + [1] * (len(_shape) - len(_shape2)))
    else:
        t_mask = BK.input_real(1.).view([1]*len(_shape))  # all ones!
    _v = _v * t_mask
    # mean
    _vsum = _v
    for ii in range(kldim):
        _vsum = _vsum.sum(-ii-1, keepdims=True)
    _vv = _v / _vsum.clamp(min=1e-5)
    # --
    ret0 = alpha * _vv + (1.-alpha) * t_prob
    _valid = (_vsum > 0).to(ret0)
    ret = _valid * ret0 + (1.-_valid) * t_prob
    return ret
