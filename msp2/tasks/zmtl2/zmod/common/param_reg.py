#

# param reg (also including soft param sharing)

__all__ = [
    "ParamRegConf", "ParamRegHelper",
]

from typing import List
from collections import Counter
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.nn.modules import LossHelper
from msp2.utils import zlog

# --

class ParamRegConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self.mod_trg = ""  # target module
        self.mod_ref = ""  # ref module, same as trg if ""
        # specifications
        self.l2_reg = 0.  # l2 reg weight
        self.init_as_ref = False  # whether init as ref?
        self.detach_ref = False  # whether detach ref
        self.extract_ref = False  # extract init values and freeze those
        self.ignore_names = []  # ignore param names that contain these
        # reg_method: 'loss': putting inside loss, 'update': specifically updating before real-update
        self.reg_method = "loss"

@node_reg(ParamRegConf)
class ParamRegHelper(BasicNode):
    def __init__(self, conf: ParamRegConf, root: BasicNode, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ParamRegConf = self.conf
        # --
        self.reg_method_loss, self.reg_method_update = [conf.reg_method==z for z in ["loss", "update"]]
        # --
        mod_trg = self._get_mod(root, conf.mod_trg)
        if conf.mod_ref == "":  # self
            mod_ref = mod_trg
            _do_init, _do_extract = False, True  # must extract self, otherwise no meaning!
        else:
            mod_ref = self._get_mod(root, conf.mod_ref)
            _do_init, _do_extract = conf.init_as_ref, conf.extract_ref
        # --
        _param_tuples = []
        _stat = Counter()
        ref_dict = {n:p for n,p in mod_ref.named_parameters()}
        for name, p in mod_trg.named_parameters():
            if any(n in name for n in conf.ignore_names):
                _stat["ignore"] += 1
                continue  # ignored by conf's ignore_names
            ref_p = ref_dict.get(name)
            if ref_p is None:
                _stat["notfound"] += 1
                continue  # not found
            if BK.get_shape(p) != BK.get_shape(ref_p):
                _stat["diffshape"] += 1
                continue  # skip since different shapes!
            # --
            if _do_init:  # if init, first set value
                BK.set_value(p, ref_p)
            if _do_extract:  # if extract, make another fixed tensor
                ref_p = ref_p.detach().clone()
            _param_tuples.append((name, p, ref_p))
            _stat["ok"] += 1
        self.stat, self.param_tuples = _stat, _param_tuples
        assert len(self.param_tuples) > 0, "No params to reg?"
        zlog(f"Setup param-reg {conf.mod_trg} <- {conf.mod_ref}: {self.stat}")
        # --

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

    def perform_update(self, lrate: float):
        conf: ParamRegConf = self.conf
        assert self.reg_method_update
        _l2_reg, _detach = conf.l2_reg, conf.detach_ref
        _alpha = -(lrate * _l2_reg)
        with BK.no_grad_env():
            for n, p, p_ref in self.param_tuples:
                p.add_(p-p_ref, alpha=_alpha)
                if (not _detach) and p_ref.requires_grad:  # update the other direction
                    p_ref.add_(p_ref-p, alpha=_alpha)
        # --

    def _get_mod(self, root: BasicNode, name: str):
        cur = root
        for f in name.split("."):  # hierarchically get attr
            cur = getattr(cur, f)
        return cur

    # --
    # overall helpers

    @staticmethod
    def perform_hard_sharing(m: BasicNode, code: str):
        # note: simply use eval(code) to do that, for example:
        #  "m.Mpb1.idec_arg.nodes[-1].core.setattr_borrow('satt',m.Mudep.idec_udep.nodes[-1].core.satt,assert_nonexist=False)"
        zlog(f"Running for hard sharing with m={m}: ``{code}''")
        exec(code)
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

    @staticmethod
    def perform_updates(regs: List['ParamRegHelper'], lrate: float):
        for reg in regs:
            if reg.reg_method_update:
                reg.perform_update(lrate)
        # --
