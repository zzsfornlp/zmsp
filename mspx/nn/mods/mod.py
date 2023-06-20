#

# basic ztask/zmod

__all__ = [
    "ZTaskConf", "ZTask", "ZModConf", "ZMod",
]

import os
from typing import List, Dict
from mspx.utils import Registrable, Configurable, Conf, zlog, default_pickle_serializer, zglob1, ZHelper, \
    ConfEntryChoices, ZResult
from ..layers import *
from .helper import *

# --
# ztask

@Registrable.rd('T')
class ZTaskConf(Conf):
    def __init__(self):
        self.name = ""  # note: should also be name outside!!
        self.mod: ZModConf = None
        self.eval = None  # evaler conf
        self.eval_weight = 1.  # weight for final eval

    @classmethod
    def get_base_conf_type(cls): return ZTaskConf
    @classmethod
    def get_base_node_type(cls): return ZTask

@Registrable.rd('_T')
class ZTask(Configurable):
    def __init__(self, conf: ZTaskConf, tc=None, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZTaskConf = self.conf
        # --
        self.tc = tc  # task-center (for accessing to other tasks)
        self.vpack = None  # vocabs to be built or load
        self.mod: ZMod = None  # Mod to be built
        self.eval = None  # evaler
        if conf.eval is not None:
            self.eval = conf.eval.make_node()
        # --

    @property
    def name(self):
        return self.conf.name

    @property
    def vocab0(self):
        return self.vpack[0]  # usually the first one as the main one!

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    # --
    # part 1: data related

    # -- part 1.1: vocab
    # build vocabs with the datasets
    def build_vocab(self, datasets: List):
        pass  # by default nothing to do!

    def save_vocab(self, v_dir: str):
        vp_file = os.path.join(v_dir, f"v_{self.name}.pkl")
        if self.vpack is not None:
            default_pickle_serializer.to_file(self.vpack, vp_file)
            zlog(f"Save vocabs ``{self.vpack}'' for {self} to {vp_file}")

    def load_vocab(self, v_dir: str):  # return whether succeed!
        if v_dir:
            v_dir = zglob1(v_dir)  # allow flexible search!
        vp_file = os.path.join(v_dir, f"v_{self.name}.pkl")
        if os.path.exists(vp_file):
            self.vpack = default_pickle_serializer.from_file(vp_file)
            zlog(f"Load vocabs ``{self.vpack}'' for {self} from {vp_file}")
            return True
        else:
            self.vpack = None  # not_found
            return False
        # --

    # -- part 1.2: data
    # note: now we make it all lazy preparation!

    # -- part 1.3: test/eval
    # eval and return metric (by default nope)
    def eval_insts_with_evaler(self, evaler, pred_insts: List, gold_insts: List, quite=False):
        evaler.reset_er()
        res0 = evaler.eval(pred_insts, gold_insts)
        if not quite:
            res_detailed_str0 = res0.get_str(brief=False)
            res_detailed_str = ZHelper.split_prefix_join(res_detailed_str0, '\t', sep='\n')
            zlog(f"{self.name} detailed results:\n{res_detailed_str}", func="result")
        ret = res0.get_res()
        return ret

    def eval_insts(self, pred_insts: List, gold_insts: List, quite=False):
        evaler = self.eval
        if evaler is None:
            return None  # no eval result!
        if isinstance(evaler, (list, tuple)):
            rets = [self.eval_insts_with_evaler(z, pred_insts, gold_insts, quite=quite) for z in evaler]
            ret = ZResult.stack(rets)
        else:
            ret = self.eval_insts_with_evaler(evaler, pred_insts, gold_insts, quite=quite)
        return ret

    # --
    # part 2: model related
    def build_mod(self, model):
        return self.conf.mod.make_node(self, model)  # {**Zmod}(conf, ztask, zmod, ...)

# --
# zmod

@Registrable.rd('M')
class ZModConf(NnConf):
    def __init__(self):
        super().__init__()
        self.loss_weight = 1.

    @classmethod
    def get_base_conf_type(cls): return ZModConf
    @classmethod
    def get_base_node_type(cls): return ZMod

@Registrable.rd('_M')
class ZMod(NnLayer):
    def __init__(self, conf: ZModConf, ztask: ZTask, zmodel, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZModConf = self.conf
        # --
        self.ztask = ztask
        self.setattr_borrow('zmodel', zmodel)
        # --

    @property
    def name(self):
        return self.ztask.name

    def process_kwargs(self, d):
        prefix = f"{self.name}__"
        ret = {}
        for k, v in d.items():
            if "__" in k:
                if k.startswith(prefix):
                    ret[k[len(prefix):]] = v
            else:
                ret[k] = v
        return ret

    # shortcut!
    def compile_leaf_loss(self, *args, **kwargs):
        return LossHelper.compile_leaf_loss(*args, **kwargs)

    def compile_losses(self, losses: List[Dict]):
        return LossHelper.combine_multiple_losses(losses, self.name+".", self.conf.loss_weight)

    # ==
    # to be implemented

    def do_prep(self, rc: ZRunCache, *args, **kwargs):  # sth to prepare before enc?
        pass  # by default nothing!

    def do_loss(self, rc: ZRunCache, *args, **kwargs):
        raise NotImplementedError()

    def do_predict(self, rc: ZRunCache, *args, **kwargs):
        raise NotImplementedError()
