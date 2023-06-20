#

# some misc helpers

import torch

# special routines
# note: for mask->idx: 1) argsort, 2) pad 1s + nonzero, 3) loop; => v2 is the fastest!
# the inputs should be 1. or 0. (float); [*, L, *] -> [*, max-count, *]
def mask2idx(mask_f, dim=-1, pad=0):
    DEFAULT_INT, DEFAULT_FLOAT = torch.int64, torch.float32
    # --
    mask_shape = mask_f.shape  # [*, L, *]
    # judge zero-shape
    if any(z==0 for z in mask_shape):
        _shape = list(mask_shape)
        _shape[dim] = 1
        zz = torch.zeros(_shape).to(mask_f)  # [*, 1, *], put an one here!
        return zz.to(DEFAULT_INT), zz.to(DEFAULT_FLOAT)
    # --
    mask_f = mask_f.to(DEFAULT_FLOAT)  # [*, L, *]
    # get max counts
    counts = mask_f.sum(dim=dim, keepdims=True)  # [*, 1, *]
    max_count = max(1, int(counts.max().item()))  # M
    padding_counts = max_count - counts  # [*, 1, *]
    max_padding_count = int(padding_counts.max().item())  # int, the max count of padding
    # pad and concat
    _arange_idx = torch.arange(max_padding_count).to(device=mask_f.device)  # [mp]
    to_expand_dim = (-dim-1) if dim<0 else (len(mask_shape)-1-dim)
    pad_t = (_arange_idx.view([max_padding_count]+[1]*to_expand_dim) < padding_counts).to(DEFAULT_FLOAT)  # [*, mp, *]
    concat_t = torch.cat([mask_f, pad_t], dim)  # [*, L+mp, *]
    # nonzero and extract
    final_shape = list(mask_shape)
    final_shape[dim] = max_count
    if dim != -1 or dim != len(mask_shape) - 1:
        final_shape = final_shape[:dim] + final_shape[dim:][1:] + [max_count]
        _p0 = list(range(len(mask_shape)))  # [0, N)
        _p1 = _p0[:dim] + _p0[dim:][1:] + [dim]
        _p2 = _p0[:dim] + [-1] + [z-1 for z in _p0[dim:][1:]]
        ret_idxes = concat_t.permute(_p1).nonzero(as_tuple=False)[:, -1].view(final_shape).permute(_p2)
    else:
        ret_idxes = concat_t.nonzero(as_tuple=False)[:, dim].view(final_shape)  # [*, M, *]
    # get valid mask and set pad for invalid ones
    max_len = mask_shape[dim]  # L
    valid_mask = (ret_idxes < max_len).to(DEFAULT_FLOAT)  # [*, M, *]
    ret_idxes[valid_mask<=0.] = pad
    return ret_idxes, valid_mask

def idx2mask(idxes_t, mask_t, full_len: int, dim=-1):
    DEFAULT_INT, DEFAULT_FLOAT = torch.int64, torch.float32
    # --
    input_shape = idxes_t.shape  # [*, N, *]
    # judge zero-shape
    if any(z == 0 for z in input_shape):
        _shape = list(input_shape)
        _shape[dim] = 1
        zz = torch.zeros(_shape).to(mask_t)  # [*, 1, *], put an one here!
        return zz.to(DEFAULT_FLOAT)
    # --
    if full_len is None:  # infer it!
        full_len = idxes_t.max().item()  # overall max!
    output_shape = list(input_shape)
    output_shape[dim] = 1+full_len  # [*, 1+L, *]
    ret0 = torch.zeros(output_shape).to(mask_t)
    idxes_t = idxes_t + (idxes_t<0).long() * full_len  # prepare proper index
    idxes_t = idxes_t.clamp(min=0, max=full_len-1)  # clamp!
    ret0.scatter_(dim, (1+idxes_t) * mask_t.long(), 1.)  # +1 to put idx0 as NIL
    ret = ret0.narrow(dim, 1, full_len)
    return ret  # [*, L, *]

# put arange into bins: basically, larger-equal than how many ones?
def idx2bin(idxes_t, mask_t, seq_mask_t, dim=-1):
    DEFAULT_INT, DEFAULT_FLOAT = torch.int64, torch.float32
    # --
    assert dim == -1  # todo(+N): for simplicity
    _L = seq_mask_t.shape[-1]
    t_arange = torch.arange(_L).to(idxes_t)  # [L]
    if mask_t is not None:
        idxes_t = idxes_t.clone()  # [..., K]
        idxes_t[mask_t] = _L + 100  # cannot be larger than this!
    t_res = (t_arange >= idxes_t.unsqueeze(-2)).sum(-1)  # [..., L]
    t_res = t_res * seq_mask_t.to(DEFAULT_INT)
    return t_res

# extend idxes
def extend_idxes(t_idxes, v: int, dim=-1):
    t_idxes = t_idxes.unsqueeze(dim)
    _shape = list(t_idxes.shape)  # [..., 1, ...]
    _shape[dim] = v
    ret = torch.zeros(_shape).to(t_idxes)
    ret.scatter_(dim, t_idxes, 1.)
    return ret

# - log softmax(margin(score_expr))[idx]
# [..., C, ...], [..., ...] -> [..., ...]
def loss_nll(score_expr, gold_idxes, dim=-1, label_smoothing=0., margin=0.):
    gold_idxes_t1 = gold_idxes.unsqueeze(dim)  # [..., 1, ...]
    if margin > 0:
        _m = torch.full_like(score_expr, fill_value=margin)
        _m.scatter_(dim, gold_idxes_t1, 0.)
        score_expr = score_expr + _m
    # note: keep it simple!
    nll_t = - score_expr.log_softmax(dim=dim)  # [..., C, ...]
    ret_t = nll_t.gather(dim, gold_idxes_t1).squeeze(dim)  # [..., ...]
    if label_smoothing > 0.:
        ret_t = (1.-label_smoothing) * ret_t + label_smoothing * nll_t.mean(dim)  # [..., ...]
    return ret_t
