#

# mtl model (in some way a collector)

__all__ = [
    "ZModelConf", "ZModel",
]

from typing import List
from collections import OrderedDict
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.nn.modules import BaseModelConf, BaseModel
from msp2.utils import zlog
from ..core import ZMod, InputBatch
from .common import ZMediator, ZMediatorConf, ParamRegConf, ParamRegHelper
from .enc import ZEncoder
from .dec import ZDecoder

# --

_SR_MAXN = 10  # max entries for param share or reg!
class ZModelConf(BaseModelConf):
    def __init__(self):
        super().__init__()
        # --
        self.med_conf = ZMediatorConf()
        self.no_pre_build_optims = False
        # --
        for ii in range(_SR_MAXN):
            setattr(self, f"reg{ii}", ParamRegConf())
            setattr(self, f"hs{ii}", "")  # hard sharing code str
        # --

    def get_all_regs(self):
        return [getattr(self, f"reg{ii}") for ii in range(_SR_MAXN)]

    def get_all_hs(self):
        return [getattr(self, f"hs{ii}") for ii in range(_SR_MAXN)]

@node_reg(ZModelConf)
class ZModel(BaseModel):
    def __init__(self, conf: ZModelConf):
        super().__init__(conf)
        # --
        # nothing added at this time, to add task/mod later on!!
        self.mods = OrderedDict()
        self.encoder: ZEncoder = None  # later
        self.decoders: List = []  # later
        self.med: ZMediator = None
        self.regs: List[ParamRegHelper] = []  # later
        # for ddp: init when finishing the model!!
        self.ddp = None

    def add_mod(self, mod: ZMod):
        _name = mod.name
        assert _name not in self.mods
        self.mods[_name] = mod
        self.add_module(f"M{_name}", mod)
        # --
        if isinstance(mod, ZEncoder):
            assert self.encoder is None
            self.setattr_borrow("encoder", mod, assert_nonexist=False)  # add name
        else:  # currently, otherwise should be decoder!!
            assert isinstance(mod, ZDecoder)
            self.decoders.append(mod)
        # --

    def finish_mods(self):
        conf: ZModelConf = self.conf
        # --
        self.med = ZMediator(self.encoder, self.decoders, conf.med_conf)
        if not conf.no_pre_build_optims:
            zzz = self.optims  # finally build optim!
        if BK.use_ddp():  # wrap self!!
            self.setattr_borrow('ddp', BK.wrap_ddp_model(self), assert_nonexist=False)
            zlog(f"Build DDP: {self.ddp}")
        # --

    def finish_sr(self):
        conf: ZModelConf = self.conf
        # --
        # todo(+N+W): although RegHelper is BasicNode, currently not added to the model since no its own params!
        for rconf in conf.get_all_regs():
            if rconf.mod_trg != "":
                one_reg = ParamRegHelper(rconf, self)
                self.regs.append(one_reg)
        for hs in conf.get_all_hs():
            if hs != "":
                ParamRegHelper.perform_hard_sharing(self, hs)
        # --

    # --
    # running

    def _mark_active(self, ibatch: InputBatch):
        for mod in self.decoders:
            mod.set_activate_output(False)  # first de-activate all!
        for tname in ibatch.dataset.dec_tasks:
            self.mods[tname].set_activate_output(True)
        # todo(+N): do we want to make p.grad=None for inactive params to fully disallow updates?
        # --

    def update(self, lrate: float, grad_factor: float):
        ParamRegHelper.perform_updates(self.regs, lrate)  # first to see if need to reg param
        super().update(lrate, grad_factor)

    # forward in training: define this for DDP
    def forward(self, ibatch: InputBatch):
        # --
        # restart
        self.encoder.restart(ibatch, self.med)
        # prepare enc
        self.med.do_prep_enc()
        # enc forward
        self.encoder.forward(self.med)
        # get all losses
        all_losses, dec_info = self.med.do_losses()
        # extra reg loss
        all_losses.append(ParamRegHelper.loss_regs(self.regs))
        # --
        # final loss and backward
        info = {"inst": len(ibatch), "fb": 1, "fb0": 0}
        info.update(dec_info)
        final_loss, loss_info = self.collect_loss(all_losses)
        info.update(loss_info)
        self.med.restart(None)  # clean
        return final_loss, info

    def loss_on_batch(self, ibatch: InputBatch, loss_factor=1., training=True, **kwargs):
        self.refresh_batch(training)
        self._mark_active(ibatch)
        # --
        if self.ddp is None:
            final_loss, info = self.forward(ibatch)
        else:
            final_loss, info = self.ddp.forward(ibatch)
        # --
        if training:
            # if BK.get_value(final_loss).item() >= 0:  # note: loss should be >=0 usually!!
            if final_loss.requires_grad:  # note: if requires_grad
                BK.backward(final_loss, loss_factor)
            else:  # no need to backward if no loss
                assert self.ddp is None, "Cannot bypass backward in DDP mode!"
                info["fb0"] = 1
        return info

    def predict_on_batch(self, ibatch: InputBatch, **kwargs):
        with BK.no_grad_env():
            self.refresh_batch(False)
            self._mark_active(ibatch)
            # --
            # restart
            self.encoder.restart(ibatch, self.med)
            # prepare
            self.med.do_prep_enc()
            # enc forward
            self.encoder.forward(self.med)
            # get all losses
            pred_info = self.med.do_preds()
            # --
            info = {"inst": len(ibatch), "ff": 1}
            info.update(pred_info)
            self.med.restart(None)  # clean
            return info

# --
# b msp2/tasks/zmtl2/zmod/model:?
