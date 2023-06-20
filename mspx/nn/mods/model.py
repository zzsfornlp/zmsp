#

# model and task-center

__all__ = [
    "ZmodelConf", "Zmodel",
]

from typing import List, Dict
from collections import OrderedDict, Counter, defaultdict
from ..backends import BK
from ..layers import *
from .helper import *

# --
# model: parameter collections (ModCenter)

@NnConf.rd('ZM')
class ZmodelConf(NnConf):
    _SR_MAXN = 10  # max entries for param share or reg!

    def __init__(self):
        super().__init__()
        # reg
        for ii in range(ZmodelConf._SR_MAXN):
            setattr(self, f"reg{ii}", ParamRegConf())
        # --

    def get_all_regs(self):
        return [getattr(self, f"reg{ii}") for ii in range(ZmodelConf._SR_MAXN)]

@ZmodelConf.conf_rd()
class Zmodel(NnLayer):
    def __init__(self, conf: ZmodelConf):
        super().__init__(conf)
        conf: ZmodelConf = self.conf
        # --
        self.mod_names = OrderedDict()  # str -> Mname
        self.regs: List[ParamRegHelper] = []  # later
        # --

    def add_mod(self, mod, name=None):
        if name is None:
            _name = mod.name
        else:
            _name = name
        assert _name not in self.mod_names
        self.mod_names[_name] = f"M{_name}"
        self.add_module(f"M{_name}", mod)
        # --

    def get_mod(self, mname: str, df=None):
        if mname in self.mod_names:
            return getattr(self, self.mod_names[mname])
        else:
            return df

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

    def update_regs(self, lrate: float):
        for r in self.regs:
            r.perform_update_after(lrate)
        # --

    # =====
    # collect all loss
    def collect_loss(self, losses: List[Dict], ret_dict=False):
        final_loss_dict = LossHelper.combine_multiple_losses(losses)  # loss_name -> {}
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
            return BK.stack(final_losses).sum() if len(final_losses)>0 else BK.input_real(0.), ret_info

    def forward(self, ibatch, tasks=None, do_loss=False, do_pred=False, **kwargs):
        from mspx.proc.run import InputBatch
        if not isinstance(ibatch, InputBatch):
            ibatch = InputBatch(ibatch)  # input should be a list of insts!
        if tasks is None:
            tasks = ibatch.dataset.tasks  # must be there!
        cur_mods = [self.get_mod(t) for t in tasks]
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
