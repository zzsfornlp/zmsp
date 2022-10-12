#

# simply put tags to each ef/evt

__all__ = [
    "ZTaskTagConf", "ZTaskTag", "ZDecoderTagConf", "ZDecoderTag",
]

from typing import List
import numpy as np
from collections import Counter
from msp2.data.inst import yield_sents, yield_sent_pairs, set_ee_heads
from msp2.data.vocab import SimpleVocab
from msp2.utils import AccEvalEntry, zlog, StrHelper, zwarn
from msp2.proc import ResultRecord, FrameEvaler, FrameEvalConf, MatchedPair
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.nn.modules import LossHelper
from .base import *
from .base_idec import *
from ..common import ZMediator, ZLabelConf, ZlabelNode
from ..enc import ZEncoder

# --
TAG_GETTER_SETTERS = {
    "default": [(lambda x: x.label), (lambda x, _l, _s: (x.set_label(_l) and False) or x.set_score(_s))],
    "ner": [(lambda x: x.info['ner']), (lambda x, _l, _s: x.info.update({'ner':_l,'ner_score':_s}))],
    "fact": [(lambda x: x.info['fact']), (lambda x, _l, _s: x.info.update({'fact':_l,'fact_score':_s}))],
    "realis": [(lambda x: x.info['realis']), (lambda x, _l, _s: x.info.update({'realis':_l,'realis_score':_s}))],
}

class ZTaskTagConf(ZTaskDecConf):
    def __init__(self):
        super().__init__()
        self.name = "tagger"  # a default name
        # --
        self.tag_trg = 'ef'
        self.tag_conf = ZDecoderTagConf()
        self.tag_scheme = "default"

    def build_task(self):
        return ZTaskTag(self)

class ZTaskTag(ZTaskDec):
    def __init__(self, conf: ZTaskTagConf):
        super().__init__(conf)
        # --
        conf: ZTaskTagConf = self.conf
        self.trg_f = (lambda s: s.get_frames(conf.tag_trg))
        self.getter, self.setter = TAG_GETTER_SETTERS[conf.tag_scheme]
        self.evaler = FrameEvaler(FrameEvalConf.direct_conf(frame_getter=conf.tag_trg, weight_arg=0.))
        # --

    # build vocab (simple gather all)
    def build_vocab(self, datasets: List):
        # breakpoint()
        voc_tag = SimpleVocab.build_empty(self.name)
        for dataset in datasets:
            for sent in yield_sents(dataset.insts):
                for trg in self.trg_f(sent):
                    voc_tag.feed_one(self.getter(trg))
        # finished
        voc_tag.build_sort()
        return (voc_tag, )

    # prepare one instance
    def prep_inst(self, inst, dataset):
        wset = dataset.wset
        _key = "_idx_" + self.name
        if wset == "train":
            voc_tag, = self.vpack
            for sent in yield_sents(inst):
                for trg in self.trg_f(sent):
                    _idx = voc_tag.get_else_unk(self.getter(trg))
                    setattr(trg, _key, _idx)
        # --
        set_ee_heads(inst)
        # --

    # prepare one input_item
    def prep_item(self, item, dataset):
        pass  # leave to the mod to handle!!

    # eval and return metric
    def eval_insts(self, gold_insts: List, pred_insts: List, quite=False):
        set_ee_heads(gold_insts)  # in case we want to eval heads!
        evaler = self.evaler
        evaler.reset()
        res0 = evaler.eval(gold_insts, pred_insts)
        res = ResultRecord(results=res0.get_summary(), description=res0.get_brief_str(), score=float(res0.get_result()))
        if not quite:
            res_detailed_str0 = res0.get_detailed_str()
            res_detailed_str = StrHelper.split_prefix_join(res_detailed_str0, '\t', sep='\n')
            zlog(f"{self.name} detailed results:\n{res_detailed_str}", func="result")
            df = MatchedPair.breakdown_eval(res0.frame_pairs)
            zlog(f"Further breakdowns: \n{df[:10].to_string()}")  # simply just print tops
        return res

    # build mod
    def build_mod(self, model):
        return ZDecoderTag(self.conf.tag_conf, self, model.encoder)

# --

class ZDecoderTagConf(ZDecoderConf):
    def __init__(self):
        super().__init__()
        # --
        self.idec_tag = IdecConf.make_conf('score')  # decoder head conf
        self.loss_tag = 1.  # weights for the loss
        self.lab_tag = ZLabelConf().direct_update(fixed_nil_val=0.)  # label node
        # --

@node_reg(ZDecoderTagConf)
class ZDecoderTag(ZDecoder):
    def __init__(self, conf: ZDecoderTagConf, ztask, main_enc: ZEncoder, **kwargs):
        super().__init__(conf, ztask, main_enc, **kwargs)
        conf: ZDecoderTagConf = self.conf
        self.voc, = self.ztask.vpack
        # --
        _enc_dim, _head_dim = main_enc.get_enc_dim(), main_enc.get_head_dim()
        self.lab_tag = ZlabelNode(conf.lab_tag, _csize=len(self.voc))
        self.idec_tag = conf.idec_tag.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=self.lab_tag.get_core_csize())
        self.reg_idec('tag', self.idec_tag)
        # --

    def loss(self, med: ZMediator, *args, **kwargs):
        conf: ZDecoderTagConf = self.conf
        # --
        # prepare info
        ibatch = med.ibatch
        _trg_label_t = self.prepare(ibatch)
        mast_t = self.get_dec_mask(ibatch, conf.msent_loss_center)
        # get losses
        loss_items = []
        _loss_tag = conf.loss_tag
        if _loss_tag > 0.:
            loss_items.extend(self.loss_from_lab(self.lab_tag, 'tag', med, _trg_label_t, mast_t, _loss_tag))
        # --
        ret_loss = LossHelper.combine_multiple_losses(loss_items)
        return ret_loss, {}

    def prepare(self, ibatch):
        _key = "_idx_" + self.ztask.name
        trg_f = self.ztask.trg_f
        b_seq_info = ibatch.seq_info
        arr_labels = np.full(BK.get_shape(b_seq_info.dec_sel_masks), 0, dtype=np.int)  # by default 0
        for bidx, item in enumerate(ibatch.items):  # for each item in the batch
            _dec_offsets = item.seq_info.dec_offsets
            for sidx, sent in enumerate(item.sents):  # for each sent in the msent item
                _start = _dec_offsets[sidx]
                for _item in trg_f(sent):
                    arr_labels[bidx, _start+_item.mention.shead_widx] = getattr(_item, _key)
        _label_t = BK.input_idx(arr_labels)  # [bs, dlen]
        return _label_t

    def predict(self, med: ZMediator, *args, **kwargs):
        # get scores
        tag_score_cache = med.get_cache((self.name, 'tag'))
        tag_scores_t = self.lab_tag.score_labels(tag_score_cache.vals, None)  # [*, dlen, L]
        tag_logprobs_t = tag_scores_t.log_softmax(-1)  # [*, dlen, L]
        # put results
        trg_f = self.ztask.trg_f
        setter_f = self.ztask.setter
        voc = self.voc
        pred_scores, pred_labels = tag_logprobs_t.max(-1)  # [*, dlen]
        arr_scores, arr_labels = BK.get_value(pred_scores), BK.get_value(pred_labels)
        cc = Counter()
        for bidx, item in enumerate(med.ibatch.items):  # for each item in the batch
            _dec_offsets = item.seq_info.dec_offsets
            for sidx, sent in enumerate(item.sents):
                if sidx != item.center_sidx:
                    continue  # skip non-center sent in this mode!
                _start = _dec_offsets[sidx]
                for _item in trg_f(sent):
                    _iidx = _start+_item.mention.shead_widx
                    if _iidx >= len(arr_labels[bidx]):
                        zwarn(f"Warn: Ignore OOR widx {_iidx} for {_item}, probably due to truncation.")
                    else:
                        _score, _label = arr_scores[bidx,_iidx].item(), voc.idx2word(arr_labels[bidx,_iidx])
                        setter_f(_item, _label, _score)
                        cc[f'_L={_label}'] += 1
        # --
        return dict(cc)

# b msp2/tasks/zmtl2/zmod/dec/dec_tag:
