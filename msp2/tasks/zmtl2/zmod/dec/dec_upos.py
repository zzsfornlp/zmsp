#

# UPOS

__all__ = [
    "ZTaskUposConf", "ZTaskUpos", "ZDecoderUposConf", "ZDecoderUpos",
]

from typing import List
import numpy as np
from msp2.data.inst import yield_sents, yield_sent_pairs
from msp2.data.vocab import SimpleVocab
from msp2.utils import AccEvalEntry, zlog
from msp2.proc import ResultRecord
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.nn.modules import LossHelper
from .base import *
from .base_idec import *
from ..common import ZMediator, ZLabelConf, ZlabelNode
from ..enc import ZEncoder

# --

class ZTaskUposConf(ZTaskDecConf):
    def __init__(self):
        super().__init__()
        self.name = "upos"  # a default name
        # --
        self.upos_conf = ZDecoderUposConf()
        self.upos_pred_clear = True  # clear exiting ones for all

    def build_task(self):
        return ZTaskUpos(self)

class ZTaskUpos(ZTaskDec):
    def __init__(self, conf: ZTaskUposConf):
        super().__init__(conf)
        # --

    # build vocab (simple gather all)
    def build_vocab(self, datasets: List):
        voc_upos = SimpleVocab.build_empty(self.name)
        for dataset in datasets:
            for sent in yield_sents(dataset.insts):
                voc_upos.feed_iter(sent.seq_upos.vals)
        # finnished
        voc_upos.build_sort()
        return (voc_upos, )

    # prepare one instance
    def prep_inst(self, inst, dataset):
        wset = dataset.wset
        if wset == "train":
            voc_upos, = self.vpack
            for sent in yield_sents(inst):
                seq_upos = sent.seq_upos
                seq_idxes = [voc_upos.get_else_unk(z) for z in seq_upos.vals]
                seq_upos.set_idxes(seq_idxes)
        elif self.conf.upos_pred_clear:  # clear if there are
            for sent in yield_sents(inst):
                sent.build_uposes(["UNK"]*len(sent))
        # --

    # prepare one input_item
    def prep_item(self, item, dataset):
        pass  # leave to the mod to handle!!

    # eval and return metric
    def eval_insts(self, gold_insts: List, pred_insts: List, quite=False):
        # simply calculate accuracy!
        acc = AccEvalEntry()
        for sent_gold, sent_pred in yield_sent_pairs(gold_insts, pred_insts):
            for a, b in zip(sent_gold.seq_upos.vals, sent_pred.seq_upos.vals):
                acc.record(int(a==b))
        res = ResultRecord(results={"details": acc.details}, description=str(acc), score=float(acc))
        if not quite:
            zlog(f"Upos detailed results:\n\tacc: {acc}", func="result")
        return res

    # build mod
    def build_mod(self, model):
        return ZDecoderUpos(self.conf.upos_conf, self, model.encoder)

# --

class ZDecoderUposConf(ZDecoderConf):
    def __init__(self):
        super().__init__()
        # --
        self.idec_upos = IdecConf.make_conf('score')  # decoder head conf
        self.loss_upos = 1.  # weights for the loss
        self.lab_upos = ZLabelConf().direct_update(fixed_nil_val=0.)  # label node
        # --

@node_reg(ZDecoderUposConf)
class ZDecoderUpos(ZDecoder):
    def __init__(self, conf: ZDecoderUposConf, ztask, main_enc: ZEncoder, **kwargs):
        super().__init__(conf, ztask, main_enc, **kwargs)
        conf: 'ZDecoderUposConf' = self.conf
        self.voc, = self.ztask.vpack
        # --
        _enc_dim, _head_dim = main_enc.get_enc_dim(), main_enc.get_head_dim()
        self.lab_upos = ZlabelNode(conf.lab_upos, _csize=len(self.voc))
        self.idec_upos = conf.idec_upos.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=self.lab_upos.get_core_csize())
        self.reg_idec('upos', self.idec_upos)
        # --

    def loss(self, med: ZMediator, *args, **kwargs):
        conf: ZDecoderUposConf = self.conf
        # --
        # prepare info
        ibatch = med.ibatch
        expr_upos_labels = self.prepare(ibatch)
        mast_t = self.get_dec_mask(ibatch, conf.msent_loss_center)
        # get losses
        loss_items = []
        _loss_upos = conf.loss_upos
        if _loss_upos > 0.:
            loss_items.extend(self.loss_from_lab(self.lab_upos, 'upos', med, expr_upos_labels, mast_t, _loss_upos))
        # --
        ret_loss = LossHelper.combine_multiple_losses(loss_items)
        return ret_loss, {}

    def predict(self, med: ZMediator, *args, **kwargs):
        self._pred_upos(med)
        return {}

    def _pred_upos(self, med: ZMediator):
        upos_score_cache = med.get_cache((self.name, 'upos'))
        # note: just gather the last one!!
        upos_scores_t = self.lab_upos.score_labels(upos_score_cache.vals, None)  # [*, dlen, L]
        upos_logprobs_t = upos_scores_t.log_softmax(-1)  # [*, dlen, L]
        self.decode_upos(med.ibatch, upos_logprobs_t)
        # --

    # --
    # helpers

    # prepare gold labels
    def prepare(self, ibatch):
        b_seq_info = ibatch.seq_info
        arr_upos_labels = np.full(BK.get_shape(b_seq_info.dec_sel_masks), 0, dtype=np.int)  # by default 0
        for bidx, item in enumerate(ibatch.items):  # for each item in the batch
            _dec_offsets = item.seq_info.dec_offsets
            for sidx, sent in enumerate(item.sents):  # for each sent in the msent item
                _start = _dec_offsets[sidx]
                arr_upos_labels[bidx, _start:_start+len(sent)] = sent.seq_upos.idxes
        expr_upos_labels = BK.input_idx(arr_upos_labels)  # [bs, dlen]
        return expr_upos_labels

    # decode with scores
    def decode_upos(self, ibatch, logprobs_t: BK.Expr):
        conf: ZDecoderUposConf = self.conf
        # get argmax label!
        pred_upos_scores, pred_upos_labels = logprobs_t.max(-1)  # [*, dlen]
        # arr_upos_scores, arr_upos_labels = BK.get_value(pred_upos_scores), BK.get_value(pred_upos_labels)
        arr_upos_labels = BK.get_value(pred_upos_labels)
        # put results
        voc = self.voc
        for bidx, item in enumerate(ibatch.items):  # for each item in the batch
            _dec_offsets = item.seq_info.dec_offsets
            for sidx, sent in enumerate(item.sents):
                if conf.msent_pred_center and (sidx != item.center_sidx):
                    continue  # skip non-center sent in this mode!
                _start = _dec_offsets[sidx]
                _len = len(sent)
                _upos_idxes = arr_upos_labels[bidx][_start:_start+_len].tolist()
                _upos_labels = voc.seq_idx2word(_upos_idxes)
                sent.build_uposes(_upos_labels)
        # --

# --
# b tasks/zmtl2/zmod/dec/dec_upos:?
