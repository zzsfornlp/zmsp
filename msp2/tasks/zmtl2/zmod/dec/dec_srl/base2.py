#

# srl decoder2, mainly for allowing stacking architecture
# todo(+N): for convenience, implemented in a non-elegant way ...

__all__ = [
    "ZTaskSrl2Conf", "ZTaskSrl2",
]

from typing import List
from msp2.proc import MyPBEvalConf
from msp2.nn import BK
from msp2.nn.layers import *
from .base import *
from ...enc import ZEncoder
from ...common import ZMediator, ZLabelConf, ZlabelNode

# --

class ZTaskSrl2Conf(ZTaskSrlConf):
    def __init__(self):
        super().__init__()
        # --
        # overwrite
        self.name = "srl2"
        self.srl_eval = MyPBEvalConf()
        self.srl_conf = ZDecoderSrl2Conf()

    def build_task(self):
        return ZTaskSrl2(self)

    @staticmethod
    def make_conf(name: str):
        ret = ZTaskSrl2Conf()
        ret.name = name  # assign name
        if name.startswith("pb"):
            ret.srl_eval = MyPBEvalConf()
        else:
            raise NotImplementedError()
        return ret

class ZTaskSrl2(ZTaskSrl):
    def __init__(self, conf: ZTaskSrl2Conf):
        super().__init__(conf)
        # --

    def build_vocab(self, datasets: List):
        # split by subname!
        data = {}
        for d in datasets:
            sub_name = d.dec_tasks[self.name]
            if sub_name not in data:
                data[sub_name] = [d]
            else:
                data[sub_name].append(d)
        # --
        res = {k: ZTaskSrl.build_vocab(self, ds) for k,ds in data.items()}
        return res

    def prep_inst(self, inst, dataset):
        sub_name = dataset.dec_tasks[self.name]
        # note: ugly reuse!
        _orig_vpack = self.vpack
        self.vpack = _orig_vpack[sub_name]
        ZTaskSrl.prep_inst(self, inst, dataset)
        self.vpack = _orig_vpack

    def build_mod(self, model):
        return ZDecoderSrl2(self.conf.srl_conf, self, model.encoder)

# --
# special stacking labeler
class ZStackingLabelConf(ZLabelConf):
    def __init__(self):
        super().__init__()
        # --
        self.use_input_emb = True  # use input emb, "assert base.emb_size>0"
        self.use_input_base = True  # use input base scores
        self.input_base_softmax = True  # further softmax weights (on base) to learn make it more interpretable?
        # --

@node_reg(ZStackingLabelConf)
class ZStackingLabelNode(ZlabelNode):
    def __init__(self, conf: ZStackingLabelConf, base_node: ZlabelNode, **kwargs):
        # --
        conf.__dict__.update(base_node.conf.__dict__)  # update from the base one!
        # --
        conf.emb_size = -1  # no use of basic ones!
        super().__init__(conf, **kwargs)
        conf: ZStackingLabelConf = self.conf
        # --
        self.setattr_borrow("_base", base_node)
        #
        assert conf.use_input_emb or conf.use_input_base
        dim_emb = base_node.conf.emb_size
        dim_base = base_node.conf._csize
        if conf.use_input_emb:
            self.aff_final = AffineNode(None, isize=dim_emb, osize=conf._csize, no_drop=True)
        else:
            self.aff_final = None
        if conf.use_input_base:
            self.W_base = BK.new_param([dim_base, conf._csize])
            BK.init_param(self.W_base, "glorot")
        else:
            self.W_base = None
        # --

    def _get_core_score(self, expr_t: BK.Expr, nil_add_score: float = None):
        conf: ZStackingLabelConf = self.conf
        # aff?
        act_t = self.input_act(expr_t)
        if self.aff_final is not None:
            score_t0 = self.aff_final(act_t)  # [*, ..., L]
        else:
            score_t0 = 0  # note: here we no longer directly use input, but put a zero!
        if self.W_base is not None:
            base_score, _ = self._base._get_score(expr_t, nil_add_score)  # [..., Lbase]
            W = self.W_base  # [Lbase, L]
            if conf.input_base_softmax:
                W = BK.softmax(W, 0)  # [Lbase, L]
            score_t1 = BK.matmul(base_score, W)
            score_t = score_t0 + score_t1
        else:
            score_t = score_t0
        return score_t
# --

class ZDecoderSrl2Conf(ZDecoderSrlConf):
    def __init__(self):
        super().__init__()
        # --
        # use mid-size as csize!
        self.idec_ef.node.core.hid_dim = 0
        self.lab_ef.emb_size = 300
        self.lab_ef.input_act = 'elu'
        self.idec_evt.node.core.hid_dim = 0
        self.lab_evt.emb_size = 300
        self.lab_evt.input_act = 'elu'
        _idec_arg = self.idec_arg.choices['idec1']
        _idec_arg.node.core.no_scorer = True
        _idec_arg.node.core.satt.nh_qk = 200
        self.lab_arg.emb_size = 200
        self.lab_arg.input_act = 'elu'
        # --
        self.labS_ef = ZStackingLabelConf()
        self.labS_evt = ZStackingLabelConf()
        self.labS_arg = ZStackingLabelConf()
        # --

@node_reg(ZDecoderSrlConf)
class ZDecoderSrl2(ZDecoderSrl):
    def __init__(self, conf: ZDecoderSrl2Conf, ztask, main_enc: ZEncoder, **kwargs):
        # note: ugly reuse
        _orig_vpack = ztask.vpack
        ztask.vpack = _orig_vpack[""]  # use the default one to construct the base!
        super().__init__(conf, ztask, main_enc, **kwargs)
        ztask.vpack = _orig_vpack
        # --
        # further construct the extra labelers!
        self.labS = {}
        self.labS[""] = (self.lab_ef, self.lab_evt, self.lab_arg)
        for name, vocs in _orig_vpack.items():
            if name != "":
                voc_ef, voc_evt, voc_arg = vocs
                lab_ef = ZStackingLabelNode(conf.labS_ef, self.lab_ef, _csize=len(voc_ef))
                lab_evt = ZStackingLabelNode(conf.labS_evt, self.lab_evt, _csize=(2 if self.conf.binary_evt else len(voc_evt)))
                lab_arg = ZStackingLabelNode(conf.labS_arg, self.lab_arg, _csize=len(voc_arg))
                self.add_module(f"_M{name}ef", lab_ef)
                self.add_module(f"_M{name}evt", lab_evt)
                self.add_module(f"_M{name}arg", lab_arg)
                self.labS[name] = (lab_ef, lab_evt, lab_arg)
        # --

    def loss(self, med: ZMediator, *args, **kwargs):
        sub_name = med.ibatch.dataset.dec_tasks[self.ztask.name]
        _orig_labs = (self.lab_ef, self.lab_evt, self.lab_arg)
        _orig_vocs = (self.voc_ef, self.voc_evt, self.voc_arg)
        self.voc_ef, self.voc_evt, self.voc_arg = self.ztask.vpack[sub_name]
        self.lab_ef, self.lab_evt, self.lab_arg = self.labS[sub_name]
        ret = super().loss(med, *args, **kwargs)
        self.voc_ef, self.voc_evt, self.voc_arg = _orig_vocs
        self.lab_ef, self.lab_evt, self.lab_arg = _orig_labs
        return ret

    def predict(self, med: ZMediator, *args, **kwargs):
        sub_name = med.ibatch.dataset.dec_tasks[self.ztask.name]
        _orig_labs = (self.lab_ef, self.lab_evt, self.lab_arg)
        _orig_vocs = (self.voc_ef, self.voc_evt, self.voc_arg)
        self.voc_ef, self.voc_evt, self.voc_arg = self.ztask.vpack[sub_name]
        self.lab_ef, self.lab_evt, self.lab_arg = self.labS[sub_name]
        ret = super().predict(med, *args, **kwargs)
        self.voc_ef, self.voc_evt, self.voc_arg = _orig_vocs
        self.lab_ef, self.lab_evt, self.lab_arg = _orig_labs
        return ret

# --
# b msp2/tasks/zmtl2/zmod/dec/dec_srl/base2:?
