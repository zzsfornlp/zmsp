#

# Model

__all__ = [
    "LossHelper", "ZmodelConf", "Zmodel",
]

# --
from typing import List, Dict
from collections import OrderedDict, Counter, defaultdict
from msp2.utils import zlog, Conf, zwarn
from ..backends import BK
from ..modules import LossHelper
from .base import *
from .misc import *

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
        mod_trg = self._get_mod(root, conf.mod_trg)
        if conf.mod_ref == "":  # self
            mod_ref = mod_trg
            _do_init, _do_extract = False, True  # must extract self, otherwise no meaning!
        else:
            mod_ref = self._get_mod(root, conf.mod_ref)
            _do_init, _do_extract = conf.init_as_ref, conf.extract_ref
        if self.reg_method_hard:
            assert mod_ref is not mod_trg
            _do_init, _do_extract = True, False
            # todo(+N): need to do more for hard??
        # --
        _param_tuples = []
        _stat = defaultdict(list)
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
        self.stat, self.param_tuples = _stat, _param_tuples
        assert len(self.param_tuples) > 0, "No params to reg?"
        zlog(f"Setup param-reg {conf.mod_trg} <- {conf.mod_ref}: { {k: len(v) for k,v in _stat.items()} }")
        # --

    def _get_mod(self, root, name: str):
        cur = root
        for f in name.split("."):  # hierarchically get attr
            cur = getattr(cur, f)
        return cur

    def compute_loss(self):
        conf: ParamRegConf = self.conf
        assert self.reg_method_loss
        _l2_reg, _detach = conf.l2_reg, conf.detach_ref
        all_l2 = []
        for n, p, p_ref in self.param_tuples:
            one_l2 = ((p - (p_ref.detach() if _detach else p_ref)) ** 2).sum()
            all_l2.append(one_l2)
        ret_loss = BK.stack(all_l2, 0).sum()
        return ret_loss, _l2_reg

    def perform_update_before(self, lrate: float):
        if not self.reg_method_hard:
            return
        # --
        # simply add grads to make them tied (plus they are init as the same!)
        with BK.no_grad_env():
            for n, p, p_ref in self.param_tuples:
                g = None
                if p.grad is not None:
                    g = p.grad if g is None else g+p.grad
                if p_ref.grad is not None:
                    g = p_ref.grad if g is None else g+p_ref.grad
                if g is not None:  # assign them!
                    p.grad = g
                    p_ref.grad = g
        # --

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
# model: parameter collections

_SR_MAXN = 10  # max entries for param share or reg!
class ZmodelConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        # --
        for ii in range(_SR_MAXN):
            setattr(self, f"reg{ii}", ParamRegConf())
        # --

    def get_all_regs(self):
        return [getattr(self, f"reg{ii}") for ii in range(_SR_MAXN)]

@node_reg(ZmodelConf)
class Zmodel(Zlayer):
    def __init__(self, conf: ZmodelConf):
        super().__init__(conf)
        conf: ZmodelConf = self.conf
        # --
        self.mods = OrderedDict()
        self.regs: List[ParamRegHelper] = []  # later
        # --

    def add_mod(self, mod, name=None):
        if name is None:
            _name = mod.name
        else:
            _name = name
        assert _name not in self.mods
        self.mods[_name] = mod
        self.add_module(f"M{_name}", mod)
        # --

    def get_mod(self, mname: str, df=None):
        return self.mods.get(mname, df)

    # finish building mods
    def finish_mods(self):
        # now we can move to target device
        self.to(BK.DEFAULT_DEVICE)

    # note: final finishing of the building, also this should be after pre-loading!
    def finish_build(self):
        conf: ZmodelConf = self.conf
        # --
        for rconf in conf.get_all_regs():
            if rconf.mod_trg != "":
                one_reg = ParamRegHelper(rconf, self)
                self.regs.append(one_reg)
        # --

    def update_regs(self, is_before: bool, lrate: float):
        for r in self.regs:
            if is_before:
                r.perform_update_before(lrate)
            else:
                r.perform_update_after(lrate)
        # --

    # =====
    # collect all loss
    def collect_loss(self, losses: List[Dict], ret_dict=False):
        final_loss_dict = LossHelper.combine_multiple_losses(losses)  # loss_name -> {}
        if len(final_loss_dict) <= 0:
            return BK.zeros([]), {}  # no loss!
        final_losses = []
        final_losses_dict = OrderedDict()
        ret_info = OrderedDict()
        for loss_name, loss_info in final_loss_dict.items():
            one_real_loss = loss_info['sum']/(loss_info['count']+1e-5)
            # --
            final_losses.append(one_real_loss)  # add scalar-tensor
            final_losses_dict[loss_name] = one_real_loss
            # --
            for k, v in loss_info.items():
                ret_info[f"loss:{loss_name}_{k}"] = float(v.item()) if hasattr(v, "item") else float(v)
            ret_info[f"loss:{loss_name}_div"] = float(one_real_loss.item()) \
                if hasattr(one_real_loss, "item") else float(one_real_loss)
        if ret_dict:
            return final_losses_dict, ret_info
        else:
            return BK.stack(final_losses).sum() if len(final_losses)>0 else BK.zeros([]), ret_info

    # == load and save models
    # todo(+n): load and save optim states to allow continue training?
    def load(self, path, **kwargs):
        BK.load_model(self, path, **kwargs)
        zlog(f"Load {self} from {path}.", func="io")

    def save(self, path, **kwargs):
        BK.save_model(self, path, **kwargs)
        zlog(f"Save {self} to {path}.", func="io")

    def __repr__(self):
        return f"{self.__class__}(NumParam={self.count_param_number()})"

    def str_details(self):
        return super().__repr__()

    def forward(self, ibatch, do_loss=False, do_pred=False, **kwargs):
        cur_mods = [self.mods[t] for t in ibatch.dataset.tasks]
        rc = ZRunCache(ibatch)
        # --
        for m in cur_mods:
            m.do_prep(rc, **kwargs)
        if do_loss:
            all_losses = []
            info = Counter()
            for m in cur_mods:
                one_loss, one_info = m.do_loss(rc, **kwargs)
                all_losses.append(one_loss)
                info += Counter(one_info)
            # extra ref loss
            all_losses.append(ParamRegHelper.loss_regs(self.regs))
            # --
            ret_info = {"inst": len(ibatch), "fb": 1, "fb0": 0}
            ret_info.update(info)
            final_loss, loss_info = self.collect_loss(all_losses)
            ret_info.update(loss_info)
            return final_loss, ret_info
        if do_pred:
            with BK.no_grad_env():
                info = Counter()
                for m in cur_mods:
                    one_info = m.do_predict(rc, **kwargs)
                    info += Counter(one_info)
                ret_info = {"inst": len(ibatch), "ff": 1}
                ret_info.update(info)
            return ret_info
        # --

# --
# b msp2/nn/l3/model:127
