#

# sequence labeling module
# -- simple individual classifiers

from typing import List, Dict, Tuple
import numpy as np

from msp.utils import Conf, Random, zlog, JsonRW, zfatal, zwarn
from msp.nn import BK
from msp.nn.layers import Affine, NoDropRop

from ..base import BaseModuleConf, BaseModule, LossHelper
from .embedder import Inputter
from ...data.insts import GeneralSentence, SeqField

# --
class SeqLabNodeConf(BaseModuleConf):
    def __init__(self):
        super().__init__()
        self.loss_lambda.val = 0.  # by default no such loss
        # --
        # hidden layer
        self.hid_dim = 0
        self.hid_act = "elu"

class SeqLabNode(BaseModule):
    def __init__(self, pc: BK.ParamCollection, pname: str, input_dim: int, conf: SeqLabNodeConf, inputter: Inputter):
        super().__init__(pc, conf, name="SLB")
        self.conf = conf
        self.inputter = inputter
        self.input_dim = input_dim
        # this step is performed at the embedder, thus still does not influence the inputter
        self.add_root_token = self.inputter.embedder.add_root_token
        # --
        self.pname = pname
        self.attr_name = pname + "_seq"  # attribute name in Instance
        self.vocab = inputter.vpack.get_voc(pname)
        # todo(note): we must make sure that 0 means NAN
        assert self.vocab.non == 0
        # models
        if conf.hid_dim <= 0:  # no hidden layer
            self.hid_layer = None
            self.pred_input_dim = input_dim
        else:
            self.hid_layer = self.add_sub_node("hid", Affine(pc, input_dim, conf.hid_dim, act=conf.hid_act))
            self.pred_input_dim = conf.hid_dim
        self.pred_out_dim = self.vocab.unk  # todo(note): UNK is the prediction boundary
        self.pred_layer = self.add_sub_node("pr", Affine(pc, self.pred_input_dim, self.pred_out_dim, init_rop=NoDropRop()))

    # score
    def _score(self, repr_t):
        if self.hid_layer is None:
            hid_t = repr_t
        else:
            hid_t = self.hid_layer(repr_t)
        out_t = self.pred_layer(hid_t)
        return out_t

    # loss
    def loss(self, insts: List[GeneralSentence], repr_t, mask_t, **kwargs):
        conf = self.conf
        # score
        scores_t = self._score(repr_t)  # [bs, ?+rlen, D]
        # get gold
        gold_pidxes = np.zeros(BK.get_shape(mask_t), dtype=np.long)  # [bs, ?+rlen]
        for bidx, inst in enumerate(insts):
            cur_seq_idxes = getattr(inst, self.attr_name).idxes
            if self.add_root_token:
                gold_pidxes[bidx, 1:1+len(cur_seq_idxes)] = cur_seq_idxes
            else:
                gold_pidxes[bidx, :len(cur_seq_idxes)] = cur_seq_idxes
        # get loss
        margin = self.margin.value
        gold_pidxes_t = BK.input_idx(gold_pidxes)
        gold_pidxes_t *= (gold_pidxes_t<self.pred_out_dim).long()  # 0 means invalid ones!!
        loss_mask_t = (gold_pidxes_t>0).float() * mask_t  # [bs, ?+rlen]
        lab_losses_t = BK.loss_nll(scores_t, gold_pidxes_t, margin=margin)  # [bs, ?+rlen]
        # argmax
        _, argmax_idxes = scores_t.max(-1)
        pred_corrs = (argmax_idxes == gold_pidxes_t).float() * loss_mask_t
        # compile loss
        lab_loss = LossHelper.compile_leaf_info("slab", lab_losses_t.sum(), loss_mask_t.sum(), corr=pred_corrs.sum())
        return self._compile_component_loss(self.pname, [lab_loss])

    # predict
    def predict(self, insts: List[GeneralSentence], repr_t, mask_t, **kwargs):
        conf = self.conf
        # score
        scores_t = self._score(repr_t)  # [bs, ?+rlen, D]
        _, argmax_idxes = scores_t.max(-1)  # [bs, ?+rlen]
        argmax_idxes_arr = BK.get_value(argmax_idxes)  # [bs, ?+rlen]
        # assign; todo(+2): record scores?
        one_offset = int(self.add_root_token)
        for one_bidx, one_inst in enumerate(insts):
            one_pidxes = argmax_idxes_arr[one_bidx, one_offset:one_offset+len(one_inst)].tolist()
            one_pseq = SeqField(None)
            one_pseq.build_vals(one_pidxes, self.vocab)
            one_inst.add_item("pred_"+self.attr_name, one_pseq, assert_non_exist=False)
        return
