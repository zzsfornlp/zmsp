#

# mediators

__all__ = [
    "ZMediatorConf", "ZMediator", "ZCache", "ZValue",
]

from typing import List, Dict
from collections import OrderedDict, Counter
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.utils import Conf, Constants

# mediate between encoder and decoders
class ZMediatorConf(Conf):
    def __init__(self):
        self.satisfy_layer = Constants.INT_PRAC_MAX  # at least for how many layers, by default go through all

class ZMediator:
    def __init__(self, encoder, decoders: List, conf: ZMediatorConf = None):
        self.conf = ZMediatorConf.direct_conf(conf)
        conf: ZMediatorConf = self.conf
        # --
        # static
        self.encoder = encoder
        self.enc_name = encoder.name
        self.decoders = decoders
        # state
        self.lidx = 0
        self.cache: ZCache = None

    # new batch
    def restart(self, ibatch):
        self.lidx = 0
        self.cache = ZCache(self, ibatch)

    # end of one layer, typical values are: emb[*, slen, D], attn[*, H, lenq, lenk]; return whether early exit?
    def layer_end(self, enc_vals: Dict):
        conf: ZMediatorConf = self.conf
        # --
        # first set encoder's vals
        cc = self.cache
        lidx = self.lidx
        _enc_name = self.enc_name
        for k, v in enc_vals.items():  # put lidx as info!
            cc.set_cache((_enc_name, k), v, app=True, app_info=lidx)
        # then go to decoders: use self to communicate!
        all_feeds = []  # feeds back to the encoder!
        all_satisfied = (lidx>=conf.satisfy_layer)
        for d in self.decoders:
            # note: scores are set inside the decoders!!
            one_feeds, one_satisfied = d.layer_end(self)
            all_feeds.extend(one_feeds)
            all_satisfied = all_satisfied and one_satisfied
        # next & ret
        ret_feeds = sum_or_none(all_feeds)  # note: simply add them together!
        self.lidx += 1
        return ret_feeds, all_satisfied

    # --
    # useful shortcuts

    def set_cache(self, k, v, **kwargs):
        return self.cache.set_cache(k, v, **kwargs)

    def get_cache(self, k):
        return self.cache.get_cache(k)

    def get_enc_cache(self, k):
        return self.cache.get_cache((self.enc_name, k))

    def get_cache_val(self, k, **kwargs):
        return self.cache.get_cache_val(k, **kwargs)

    def get_enc_cache_val(self, k, **kwargs):
        return self.cache.get_cache_val((self.enc_name, k), **kwargs)

    def get_mask(self, use_dsel: bool):
        seq_info = self.ibatch.seq_info
        return seq_info.dec_sel_masks if use_dsel else seq_info.enc_input_masks

    @property
    def ibatch(self):
        return self.cache.ibatch

    # =====
    def do_prep_enc(self, *args, **kwargs):
        for dec in self.decoders:
            if dec.activate_output:
                dec.prep_enc(self, *args, **kwargs)
        # --

    def do_losses(self, *args, **kwargs):
        all_losses = []
        info = Counter()
        for dec in self.decoders:
            if dec.activate_output:
                one_loss, one_info = dec.loss(self, *args, **kwargs)
                all_losses.append(one_loss)
                info += Counter(one_info)
        return all_losses, info

    def do_preds(self, *args, **kwargs):
        info = Counter()
        for dec in self.decoders:
            if dec.activate_output:
                one_info = dec.predict(self, *args, **kwargs)
                info += Counter(one_info)
        return info

    def do_scores(self, *args, **kwargs):
        info = Counter()
        for dec in self.decoders:
            if dec.activate_output:
                one_info = dec.score(self, *args, **kwargs)
                info += Counter(one_info)
        return info

# overall value cache center (also recording current status)
class ZCache:
    def __init__(self, med: ZMediator, ibatch):
        self.med = med  # store the link!
        self.ibatch = ibatch  # store the ibatch
        # --
        self._cc = {}  # actual values

    # --
    def set_cache(self, k, v, app=False, app_info=None):
        _cc = self._cc
        # --
        if app:  # appending mode
            zv = _cc.get(k)
            if zv is None:
                zv = ZValue()
                _cc[k] = zv
            zv.append(v, app_info)
        else:  # adding like a dict
            assert k not in _cc
            _cc[k] = v
        # --

    def get_cache(self, k, df=None):
        return self._cc.get(k, df)
    # --

    def get_cache_val(self, k, **kwargs):
        val = self.get_cache(k)
        return val.get_val(**kwargs)

# (layered/multiple) value container: can store hids, attns, scores, ...
# -> note: basically a list, the values should have the same shape!!
class ZValue:
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
# b msp2/tasks/zmtl2/mods/common/med:?
