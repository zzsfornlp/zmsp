#

__all__ = [
    "down_neg", "ZRunCache", "ZCachedValue", "unsqueeze_expand",
]

import numpy as np
from collections import OrderedDict
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
        n_mask = (BK.rand(n_t.shape) < _ratio).float()  # [..., *]
        ret0 = m_t * (1.-n_t) + n_mask * n_t  # [..., *]
    else:  # simply change weight
        ret0 = m_t * (1.-n_t) + _ratio * n_t  # [..., *]
    ret = ret0.view(_shape)  # [???]
    return ret

# --
# cache: to store current values
class ZRunCache:
    def __init__(self, ibatch):
        self.ibatch = ibatch  # input batch
        self._cc = {}  # cached values

    def set_cache(self, k, v, app=False, app_info=None):
        _cc = self._cc
        # --
        if app:  # appending mode
            zv = _cc.get(k)
            if zv is None:
                zv = ZCachedValue()
                _cc[k] = zv
            zv.append(v, app_info)
        else:  # adding like a dict
            assert k not in _cc
            _cc[k] = v
        # --

    def get_cache(self, k, df=None):
        return self._cc.get(k, df)

    def get_cache_val(self, k, **kwargs):
        val = self.get_cache(k)
        return val.get_val(**kwargs)

# (layered/multiple) value container: can store hids, attns, scores, ...
# -> note: basically a list, the values should have the same shape!!
class ZCachedValue:
    def __init__(self):
        self.vals = []  # List[val]
        self.infos = []  # List[info]
        self.vmap = OrderedDict()  # info->val
        # --
        self._cache = OrderedDict()  # cached value, for example, selected ones!

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, item):
        return self.vals[item]

    def append(self, v, info=None):
        if v is not None:  # note: ignore none!!
            self.vals.append(v)
            self.infos.append(info)
            if info is not None:  # extra store!
                assert info not in self.vmap
                self.vmap[info] = v
            # clear the cache whenever we add new things!
            self._cache.clear()

    # get val (if idx is None, then stack all!!)
    def get_val(self, idx=-1, stack_dim=-2, signature=None, function=None, no_cache=False):
        _k = (idx, stack_dim, signature)  # key for cache
        ret = None
        if not no_cache:
            ret = self._cache.get(_k)
        if ret is None:  # calculate!!
            if idx is None:
                v0 = BK.stack(self.vals, dim=stack_dim)  # [*, ilen, ND, *]
            else:
                v0 = self.vals[idx]  # [*, ilen, *]
            ret = function(v0) if function is not None else v0  # post-processing!
            if not no_cache:
                self._cache[_k] = ret   # store cache
        # --
        return ret

    # get cached: by default the last one
    def get_cached_value(self, idx=-1, assert_unique=False):
        all_cached_vals = list(self._cache.values())
        if assert_unique:
            assert len(all_cached_vals)==1
        return all_cached_vals[idx] if (len(all_cached_vals)>0) else None

# --
# unsqueeze and expand
def unsqueeze_expand(t, dim: int, size: int):
    t2 = t.unsqueeze(dim)  # [..., 1, ...]
    sizes = [-1] * len(t2.shape)
    sizes[dim] = size
    ret = t2.expand(*sizes)
    return ret
