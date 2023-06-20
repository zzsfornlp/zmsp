#

# a unified wrapper module for base models (emb+enc+dec+lmhead)

__all__ = [
    "ZBmodBaseConf", "ZBmodBaseMod", "ZBmodHelper",
]

from typing import List
from mspx.nn import BK
from ...layers import NnConf, NnLayer

# note: no reg!
class ZBmodBaseConf(NnConf):
    def __init__(self):
        super().__init__()
        self.b_cache_dir = ""  # dir for downloading
        self.b_extra_tokens = []  # add extra tokens to vocab?
        self.b_extra_tokens_trg = []  # add extra tokens to (trg) vocab?
        self.b_inc_emb = True  # whether include emb?
        self.b_inc_enc = True  # whether include enc if there is?
        self.b_inc_dec = True  # whether include dec if there is?
        self.b_inc_lmhead = False  # whether include lmhead if there is?

    @property
    def cache_dir_or_none(self):
        return self.b_cache_dir if self.b_cache_dir else None

class ZBmodBaseMod(NnLayer):
    def forward_enc(self, t_ids, t_mask=None, t_emb=None, t_ihid=None): raise NotImplementedError()
    def forward_dec(self, t_ids, t_mask=None, t_emb=None, t_ihid=None, t_cross=None, t_cross_mask=None, cache=None): raise NotImplementedError()
    def forward_emb(self, t_ids, mixes=None, forw_full=False): raise NotImplementedError()
    def forward_emb_trg(self, t_ids, mixes=None, forw_full=False, cache=None): raise NotImplementedError()
    def get_mdim(self) -> int: raise NotImplementedError()
    def get_head_num(self) -> int: raise NotImplementedError()

    def has_enc(self): return self.enc is not None
    def has_dec(self): return self.dec is not None
    def has_lmhead(self): return self.lmhead is not None
    def forward_lmhead(self, t_hid): return self.lmhead(t_hid)
    def dec_reorder_cache(self, cache, t_idxes): return ZBmodHelper.dec_reorder_cache(cache, t_idxes)

# --
class ZBmodHelper:
    @staticmethod
    def dec_reorder_cache(cache, t_idxes):
        _ff = ZBmodHelper.dec_reorder_cache
        if isinstance(cache, (list, tuple)):
            return type(cache)([_ff(z, t_idxes) for z in cache])
        elif isinstance(cache, dict):
            return {k: _ff(v, t_idxes) for k,v in cache.items()}
        else:
            return cache.index_select(0, t_idxes.to(cache.device))
        # --

    @staticmethod
    def mix_embs(t_base, mixes):
        ret = t_base
        if mixes is not None:
            for t_mix_w, t_mix_emb in mixes:  # note: order matters!
                _w = BK.input_real(t_mix_w).unsqueeze(-1)  # [*, L, 1]
                ret = _w * t_mix_emb + (1. - _w) * ret
        return ret
