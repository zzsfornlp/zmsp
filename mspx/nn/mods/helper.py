#

__all__ = [
    "ParamRegConf", "ParamRegHelper", "LossHelper",
    "ZRunCache", "ZCachedValue",
]

from typing import List, Dict
from collections import OrderedDict, defaultdict
from mspx.utils import Conf, zlog, zwarn
from ..backends import BK

# --
# loss helper

# common format of loss_info is: {'name1': {'sum', 'count', ...}, 'name2': ...}
class LossHelper:
    # compile one loss for one component
    @staticmethod
    def compile_leaf_loss(name: str, loss_sum: BK.Expr, loss_count: BK.Expr, loss_lambda=1., **other_values):
        if loss_lambda <= 0.:
            return OrderedDict()
        local_dict = {"sum0": loss_sum, "sum": loss_sum*loss_lambda, "count": loss_count, "run": 1}
        local_dict.update(other_values)
        return OrderedDict({name: local_dict})

    # collect all losses for (possibly) multiple runs
    @staticmethod
    def combine_multiple_losses(inputs: List[Dict], prefix="", extra_lambda=1.):
        # each input is a flattened loss Dict
        ret = OrderedDict()
        for one_input in inputs:
            if one_input is None:  # skip None
                continue
            for name, leaf_info in one_input.items():
                name = prefix + name
                if name in ret:
                    # adding together
                    target_info = ret[name]
                    for k, v in leaf_info.items():
                        target_info[k] = target_info.get(k, 0.) + v
                else:
                    # direct set
                    ret[name] = leaf_info
        if extra_lambda != 1.:
            for v in ret.values():
                v["sum"] = v["sum"] * extra_lambda
        return ret

# --
# param-reg

class ParamRegConf(Conf):
    def __init__(self):
        super().__init__()
        # --
        self.mod_trg = ""  # target module
        self.mod_ref = ""  # ref module, same as trg if empty
        # specifications (some of these are forced by certain modes!)
        self.l2_reg = 0.  # l2 reg weight
        self.init_as_ref = False  # whether init as ref?
        self.detach_ref = False  # whether detach ref
        self.extract_ref = False  # extract init values and freeze those
        self.ignore_names = []  # ignore param names that contain these
        # reg_method: 'loss': putting inside loss, 'update': specifically updating, 'hard': hard sharing
        self.reg_method = "loss"

class ParamRegHelper:
    def __init__(self, conf: ParamRegConf, root: 'Zmodel'):
        self.conf = conf
        # --
        _modes = [conf.reg_method == z for z in ["loss", "update", "hard"]]
        assert sum(_modes) == 1
        self.reg_method_loss, self.reg_method_update, self.reg_method_hard = _modes
        # --
        mod_trg0, mod_name, mod_trg = self._get_mod(root, conf.mod_trg)
        if conf.mod_ref == "":  # self
            mod_ref = mod_trg
            _do_init, _do_extract = False, True  # must extract self, otherwise no meaning!
        else:
            _, _, mod_ref = self._get_mod(root, conf.mod_ref)
            _do_init, _do_extract = conf.init_as_ref, conf.extract_ref
        # --
        # special mode:
        _param_tuples = []
        _stat = defaultdict(list)
        if self.reg_method_hard:
            assert mod_ref is not mod_trg, "No meaning to tie self!"
            setattr(mod_trg0, mod_name, mod_ref)
            zlog(f"Hard sharing by tying: {conf.mod_trg} <- {conf.mod_ref}")
        else:
            ref_dict = {n: p for n, p in mod_ref.named_parameters()}
            for name, p in mod_trg.named_parameters():
                if any(n in name for n in conf.ignore_names):
                    _stat["ignore"].append(name)
                    continue  # ignored by conf's ignore_names
                ref_p = ref_dict.get(name)
                if ref_p is None:
                    _stat["notfound"].append(name)
                    continue  # not found
                if ref_p is p and not _do_extract:  # unless extracting (freezing)
                    _stat["alreadyshared"].append(name)
                    continue  # already shared!
                if BK.get_shape(p) != BK.get_shape(ref_p):
                    _stat["diffshape"].append(name)
                    zwarn(f"Diff-shape of {name}: {BK.get_shape(p)} vs {BK.get_shape(ref_p)}")
                    continue  # skip since different shapes!
                # --
                if _do_init:  # if init, first set value
                    BK.set_value(p, ref_p)
                if _do_extract:  # if extract, make another fixed tensor
                    ref_p = ref_p.detach().clone()
                _param_tuples.append((name, p, ref_p))
                _stat["ok"].append(name)
                # --
            # --
            assert len(_param_tuples) > 0, "No params to reg?"
            zlog(f"Setup param-reg {conf.mod_trg} <- {conf.mod_ref}: { {k: len(v) for k,v in _stat.items()} }")
        # --
        self.stat, self.param_tuples = _stat, _param_tuples
        # --

    def _get_mod(self, root, name: str):
        last, cur = None, root
        field_name = None
        for f in name.split("."):  # hierarchically get attr
            last = cur
            field_name = f
            cur = getattr(cur, f)
        return last, field_name, cur

    def compute_loss(self):
        conf: ParamRegConf = self.conf
        assert self.reg_method_loss
        _l2_reg, _detach = conf.l2_reg, conf.detach_ref
        all_l2 = []
        for n, p, p_ref in self.param_tuples:
            one_l2 = ((p - (p_ref.detach() if _detach else p_ref)) ** 2).sum()
            all_l2.append(one_l2)
        ret_loss = BK.stack(all_l2, 0).sum() if len(all_l2)>0 else BK.input_real(0.)
        return ret_loss, _l2_reg

    def perform_update_after(self, lrate: float):
        conf: ParamRegConf = self.conf
        if not self.reg_method_update:
            return
        _l2_reg, _detach = conf.l2_reg, conf.detach_ref
        _alpha = -(lrate * _l2_reg)
        with BK.no_grad_env():
            for n, p, p_ref in self.param_tuples:
                p.add_(p-p_ref, alpha=_alpha)
                if (not _detach) and p_ref.requires_grad:  # update the other direction
                    p_ref.add_(p_ref-p, alpha=_alpha)
        # --

    @staticmethod
    def loss_regs(regs: List['ParamRegHelper']):
        loss_items = []
        for ii, reg in enumerate(regs):
            if reg.reg_method_loss:
                _loss, _loss_lambda = reg.compute_loss()
                _loss_item = LossHelper.compile_leaf_loss(f'reg_{ii}', _loss, BK.input_real(1.), loss_lambda=_loss_lambda)
                loss_items.append(_loss_item)
        ret_loss = LossHelper.combine_multiple_losses(loss_items)
        return ret_loss

# --
# cache: to store current values
class ZRunCache:
    def __init__(self, ibatch):
        self.ibatch = ibatch  # input batch
        self._cc = {}  # cached values

    def set_cache(self, k, v, app=False, app_info=None):
        if isinstance(k, str):
            k = [z for z in k.split(":") if z]  # ignore empty!
        _cc = self.get_cache(k[:-1], add_dict=True)  # add the path!
        _key = k[-1]
        # --
        if app:  # appending mode
            zv = _cc.get(_key)
            if zv is None:
                zv = ZCachedValue()
                _cc[_key] = zv
            zv.append(v, app_info)
        else:  # adding like a dict
            assert _key not in _cc
            _cc[_key] = v
        # --

    def get_cache(self, k, df=None, add_dict=False):
        if isinstance(k, str):
            k = [z for z in k.split(":") if z]  # ignore empty!
        # special handling!
        cur_item = self._cc
        for ii, one_f in enumerate(k):
            if isinstance(cur_item, (list, tuple)):
                one_f = int(one_f)  # as index
                cur_item = cur_item[one_f]
            elif isinstance(cur_item, dict):
                if add_dict and (one_f not in cur_item):
                    cur_item[one_f] = {}  # add new dict for non-last ones!
                cur_item = cur_item.get(one_f, df)  # as key
            elif hasattr(cur_item, one_f):
                cur_item = getattr(cur_item, one_f)  # as attribute
            else:
                cur_item = df  # not found
            if cur_item is None:
                break
        return cur_item

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
