#

# Sequence-based Extractor (actually combined with labeler)

__all__ = [
    "SeqExtractorConf", "SeqExtractor", "SeqExtractorHelper",
]

from typing import List, Union
from collections import defaultdict
import numpy as np
from msp2.nn import BK
from msp2.nn.modules import LossHelper
from msp2.nn.layers import BasicConf, BasicNode, node_reg
from msp2.data.inst import Sent, Frame, Mention, DataPadder
from msp2.data.vocab import SimpleVocab, SeqVocab, SeqVocabConf
from msp2.utils import ZObject, Constants, zwarn, zlog
from .base import *

# =====

class SeqExtractorConf(BaseExtractorConf):
    def __init__(self):
        super().__init__()
        # --
        self.vconf = SeqVocabConf()  # seq vocab conf

@node_reg(SeqExtractorConf)
class SeqExtractor(BaseExtractor):
    def __init__(self, conf: SeqExtractorConf, vocab: SimpleVocab, **kwargs):
        seqvocab = SeqVocab(vocab, conf.vconf)
        zlog(f"Build seqvocab: {seqvocab}")
        super().__init__(conf, seqvocab, **kwargs)
        conf: SeqExtractorConf = self.conf
        # --
        self.s_vocab = seqvocab
        self.helper = SeqExtractorHelper(conf, seqvocab)

    def _build_extract_node(self, conf: SeqExtractorConf):
        return None  # no extraction node

    # [*, slen, D], [*, slen], [*, D']
    def loss(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
             pair_expr: BK.Expr = None, lookup_flatten=False, external_extra_score: BK.Expr = None):
        conf: SeqExtractorConf = self.conf
        # step 0: prepare golds
        expr_gold_slabs, expr_loss_weight_non = self.helper.prepare(insts, mlen=BK.get_shape(mask_expr, -1), use_cache=True)
        final_loss_weights = BK.where(expr_gold_slabs==0, expr_loss_weight_non.unsqueeze(-1)*conf.loss_weight_non, mask_expr)
        # step 1: label; with special weights
        loss_lab, loss_count = self.lab_node.loss(
            input_expr, pair_expr, mask_expr, expr_gold_slabs,
            loss_weight_expr=final_loss_weights, extra_score=external_extra_score)
        loss_lab_item = LossHelper.compile_leaf_loss(f"lab", loss_lab, loss_count,
                                                     loss_lambda=conf.loss_lab, gold=(expr_gold_slabs>0).float().sum())
        # ==
        # return loss
        ret_loss = LossHelper.combine_multiple_losses([loss_lab_item])
        return self._finish_loss(ret_loss, insts, input_expr, mask_expr, pair_expr, lookup_flatten)

    # [*, slen, D], [*, slen], [*, D']
    def predict(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
                pair_expr: BK.Expr = None, lookup_flatten=False, external_extra_score: BK.Expr=None):
        # --
        # note: check empty
        if BK.is_zero_shape(mask_expr):
            for inst in insts:  # still need to clear things!!
                self.helper._clear_f(inst)
        else:
            # simply labeling!
            best_labs, best_scores = self.lab_node.predict(input_expr, pair_expr, mask_expr, extra_score=external_extra_score)
            # put results
            self.helper.put_results(insts, best_labs, best_scores)
        # --
        # finally
        return self._finish_pred(insts, input_expr, mask_expr, pair_expr, lookup_flatten)

    # [*, slen, D], [*, slen], [*, D']
    def lookup(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
               pair_expr: BK.Expr = None):
        raise NotImplementedError("To be implemented!!")

    # plus flatten items's dims
    def lookup_flatten(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
                       pair_expr: BK.Expr = None):
        raise NotImplementedError("To be implemented!!")

# --
class SeqExtractorHelper(BaseExtractorHelper):
    def __init__(self, conf: SeqExtractorConf, vocab: SeqVocab):
        super().__init__(conf, vocab)
        conf: SeqExtractorConf = self.conf
        # --
        if conf.ftag == "arg":
            self._prep_f = self._prep_args
        else:
            self._prep_f = self._prep_frames

    def _prep_items(self, items: List, par: object, seq_len: int):
        vocab: SeqVocab = self.vocab
        # --
        core_spans = [self.core_span_getter(f.mention) + (f.label_idx,) for f in items]
        _loss_weight_non = getattr(par, "_loss_weight_non", 1.)  # todo(+N): special name; loss_weight_non
        tag_layers = vocab.spans2tags_idx(core_spans, seq_len)
        if len(tag_layers) > 1:
            zwarn(f"Warning: '{self.conf.ftag}' only use layer0 but the full needs multiple layers with {core_spans}")
            # breakpoint()
        trg_tags = tag_layers[0][0]
        # trg_first_items = [(items[i] if i>=0 else None) for i in tag_layers[0][1]]  # note: put it at the start!
        # return ZObject(loss_weight_non=_loss_weight_non, first_items=trg_first_items, tags=trg_tags, len=len(trg_tags))
        return ZObject(loss_weight_non=_loss_weight_non, tags=trg_tags, len=len(trg_tags))

    def _prep_frames(self, s: Sent):
        return self._prep_items(self._get_frames(s), s, len(s))

    def _prep_args(self, f: Frame):
        return self._prep_items(self._get_args(f), f, len(f.sent))

    # prepare inputs
    def prepare(self, insts: Union[List[Sent], List[Frame]], mlen: int, use_cache: bool):
        conf: SeqExtractorConf = self.conf
        # get info
        if use_cache:
            zobjs = []
            attr_name = f"_scache_{conf.ftag}"  # should be unique
            for s in insts:
                one = getattr(s, attr_name, None)
                if one is None:
                    one = self._prep_f(s)
                    setattr(s, attr_name, one)  # set cache
                zobjs.append(one)
        else:
            zobjs = [self._prep_f(s) for s in insts]
        # batch things
        bsize = len(insts)
        # mlen = max(z.len for z in zobjs)  # note: fed by outside!!
        batched_shape = (bsize, mlen)
        # arr_first_items = np.full(batched_shape, None, dtype=object)
        arr_slabs = np.full(batched_shape, 0, dtype=np.int)
        for zidx, zobj in enumerate(zobjs):
            # arr_first_items[zidx, zobj.len] = zobj.first_items
            arr_slabs[zidx, :zobj.len] = zobj.tags
        # final setup things
        expr_slabs = BK.input_idx(arr_slabs)
        expr_loss_weight_non = BK.input_real([z.loss_weight_non for z in zobjs])  # [*]
        # return arr_first_items, expr_slabs, expr_loss_weight_non
        return expr_slabs, expr_loss_weight_non

    # =====
    # put outputs

    # *[*, slen]
    def put_results(self, insts, best_labs, best_scores):
        conf: SeqExtractorConf = self.conf
        vocab: SeqVocab = self.vocab
        # --
        base_vocab = vocab.base_vocab
        arr_slabs, arr_scores = [BK.get_value(z) for z in [best_labs, best_scores]]
        for bidx, inst in enumerate(insts):
            self._clear_f(inst)  # first clean things
            cur_len = len(inst) if isinstance(inst, Sent) else len(inst.sent)
            # --
            cur_slabs, cur_scores = arr_slabs[bidx][:cur_len], arr_scores[bidx][:cur_len]
            cur_results = vocab.tags2spans_idx(cur_slabs)
            inst.info["slab"] = [vocab.idx2word(z) for z in cur_slabs]  # put seq-lab
            for one_widx, one_wlen, one_lab in cur_results:
                one_lab = int(one_lab)
                assert one_lab > 0, "Error: should not extract 'O'!?"
                new_item = self._new_f(
                    inst, int(one_widx), int(one_wlen), one_lab, np.mean(cur_scores[one_widx:one_widx+one_wlen]).item(),
                    vocab=base_vocab,
                )
        # --
