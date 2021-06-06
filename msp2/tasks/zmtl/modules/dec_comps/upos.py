#

# UPOS

__all__ = [
    "ZDecoderUPOSConf", "ZDecoderUPOSNode", "ZDecoderUPOSHelper",
]

from typing import List
import numpy as np
import time
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.nn.modules import LossHelper
from msp2.data.inst import Sent, Frame, Mention, DataPadder
from msp2.data.vocab import SimpleVocab, SeqVocab
from msp2.utils import Constants, zlog, ZObject, zwarn, ConfEntryChoices
from ..common import *
from ..enc import *
from ..dec import *

# =====

class ZDecoderUPOSConf(ZDecoderConf):
    def __init__(self):
        super().__init__()
        # --
        self.upos_conf = IdecSingleConf()  # simple classifier
        self.loss_upos = 1.
        self.upos_label_smoothing = 0.

@node_reg(ZDecoderUPOSConf)
class ZDecoderUPOSNode(ZDecoder):
    def __init__(self, conf: ZDecoderUPOSConf, name: str, vocab_upos: SimpleVocab, ref_enc: ZEncoder, **kwargs):
        super().__init__(conf, name, **kwargs)
        conf: ZDecoderUPOSConf = self.conf
        self.vocab_upos = vocab_upos
        _enc_dim, _head_dim = ref_enc.get_enc_dim(), ref_enc.get_head_dim()
        # --
        self.helper = ZDecoderUPOSHelper(conf, self.vocab_upos)
        # nodes
        self.upos_node: IdecNode = conf.upos_conf.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=len(vocab_upos))
        # --
        raise RuntimeError("Deprecated after MED's collecting of scores!!")

    def get_idec_nodes(self):
        return [self.upos_node]

    # return List[tensor], bool
    def layer_end(self, med: ZMediator):
        lidx = med.lidx
        rets = []
        if self.upos_node.need_app_layer(lidx):
            rets.append(self.upos_node.forward(med))
        return rets, (lidx >= self.max_app_lidx)

    # --
    def _loss_upos(self, mask_expr, expr_upos_labels):
        conf: ZDecoderUPOSConf = self.conf
        all_upos_scores = self.upos_node.buffer_scores.values()  # [*, slen, L]
        all_upos_losses = []
        for one_upos_scores in all_upos_scores:
            one_losses = BK.loss_nll(one_upos_scores, expr_upos_labels, label_smoothing=conf.upos_label_smoothing)
            all_upos_losses.append(one_losses)
        upos_loss_results = self.upos_node.helper.loss(all_losses=all_upos_losses)
        loss_items = []
        for loss_t, loss_alpha, loss_name in upos_loss_results:
            one_upos_item = LossHelper.compile_leaf_loss(
                "upos"+loss_name, (loss_t*mask_expr).sum(), mask_expr.sum(), loss_lambda=(loss_alpha*conf.loss_upos))
            loss_items.append(one_upos_item)
        return loss_items

    # get loss
    def loss(self, med: ZMediator):
        conf: ZDecoderUPOSConf = self.conf
        insts, mask_expr = med.insts, med.get_mask_t()
        # --
        # first prepare golds
        expr_upos_labels = self.helper.prepare(insts, True)
        loss_items = []
        if conf.loss_upos > 0.:
            loss_items.extend(self._loss_upos(mask_expr, expr_upos_labels))
        # =====
        # return loss
        ret_loss = LossHelper.combine_multiple_losses(loss_items)
        return ret_loss

    # pred for upos
    def _pred_upos(self):
        all_upos_raw_score = self.upos_node.buffer_scores.values()  # [*, slen, L]
        all_upos_logprobs = [z.log_softmax(-1) for z in all_upos_raw_score]
        final_upos_logprobs = self.upos_node.helper.pred(all_logprobs=all_upos_logprobs)  # [*, slen, L]
        pred_upos_scores, pred_upos_labels = final_upos_logprobs.max(-1)  # [*, slen]
        return pred_upos_labels, pred_upos_scores  # [*, slen]

    # predict
    def predict(self, med: ZMediator):
        # --
        pred_upos_labels, pred_upos_scores = self._pred_upos()
        all_arrs = [BK.get_value(z) for z in [pred_upos_labels, pred_upos_scores]]
        self.helper.put_results(med.insts, all_arrs)
        # --
        return {}

# helper
class ZDecoderUPOSHelper(ZDecoderHelper):
    def __init__(self, conf: ZDecoderUPOSConf, vocab_upos: SimpleVocab):
        self.conf = conf
        self.vocab_upos = vocab_upos
        # --

    # prepare inputs
    def prepare(self, insts: List, use_cache: bool = None):
        bsize, mlen = len(insts), max(len(z.sent) for z in insts) if len(insts)>0 else 1
        batched_shape = (bsize, mlen)
        arr_upos_labels = np.full(batched_shape, 0, dtype=np.int)
        for bidx, inst in enumerate(insts):
            zlen = len(inst.sent)
            arr_upos_labels[bidx, :zlen] = inst.sent.seq_upos.idxes
        expr_upos_labels = BK.input_idx(arr_upos_labels)
        return expr_upos_labels

    def put_results(self, insts: List, all_arrs):
        vocab_upos = self.vocab_upos
        pred_upos_labels, pred_upos_scores = all_arrs
        for bidx, inst in enumerate(insts):
            # todo(+W): if there are repeated sentences, may overwrite previous ones, but maybe does not matter!
            cur_len = len(inst.sent)
            cur_upos_labs, cur_upos_scores = pred_upos_labels[bidx][:cur_len], pred_upos_scores[bidx][:cur_len]
            # --
            upos_lab_idxes = cur_upos_labs.tolist()
            upos_lab_vals = vocab_upos.seq_idx2word(upos_lab_idxes)
            inst.sent.build_uposes(upos_lab_vals)
            inst.sent.seq_upos.set_idxes(upos_lab_idxes)
        # --
