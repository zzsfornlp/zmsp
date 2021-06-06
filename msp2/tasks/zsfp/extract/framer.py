#

# the overall frame extractor

__all__ = [
    "MyFramerConf", "MyFramer",
]

from typing import List
import numpy as np
from msp2.nn import BK
from msp2.nn.modules import LossHelper, PlainEncoderConf, PlainEncoder
from msp2.nn.layers import BasicConf, BasicNode, node_reg
from msp2.data.inst import Sent, Frame, Mention, DataPadder
from msp2.data.vocab import SimpleVocab
from msp2.utils import Constants, zlog
from .extractors import ExtractorGetter, ConstrainerNode
from .constrainer import LexConstrainer, FEConstrainer

# =====

class MyFramerConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize = -1  # input expr size
        # --
        # for evt, with specific settings
        self.evt_conf = ExtractorGetter.make_conf_entry("evt")
        self.loss_evt = 1.0
        self.evt_cons_lex_file = ""  # for evt's lex-cons
        # --
        # for arg
        self.arg_conf = ExtractorGetter.make_conf_entry("arg")
        self.arg_cons_fe_file = ""  # for arg's fe-cons
        self.loss_arg = 1.0
        self.arg_use_finput = False  # maybe no need if we have fenc_conf
        # special encoder before arg
        self.fenc_mix_frame = True  # mix trigger info for the inputs of fenc
        self.fenc_conf = PlainEncoderConf()
        # =====
        # whether pred: otherwise directly use inputs
        self.pred_evt = True
        self.pred_arg = True

@node_reg(MyFramerConf)
class MyFramer(BasicNode):
    def __init__(self, conf: MyFramerConf, vocab_evt: SimpleVocab, vocab_arg: SimpleVocab, **kwargs):
        super().__init__(conf, **kwargs)
        conf: MyFramerConf = self.conf
        # --
        # evt
        if conf.evt_cons_lex_file:
            cons_lex = LexConstrainer.load_from_file(conf.evt_cons_lex_file)
            # note: not adding here!! # add missing frames
            # all_frames = set(f for v in cons_lex.cmap.values() for f in v.keys())
            # for f in all_frames:
            #     vocab_evt.feed_one(f, c=0)
            # zlog(f"After adding all frames from cons_lex: {vocab_evt}")
        else:
            cons_lex = None
        self.evt_extractor = ExtractorGetter.make_extractor(conf.evt_conf, vocab_evt, cons_lex=cons_lex, isize=conf.isize)
        evt_psize = self.evt_extractor.get_output_dims()[0]
        # arg
        if conf.arg_cons_fe_file:
            cons_arg = FEConstrainer.load_from_file(conf.arg_cons_fe_file)
            self.cons_arg = ConstrainerNode(cons_arg, vocab_evt, vocab_arg, None)  # note: currently no other confs
        else:
            self.cons_arg = None
        self.fenc = PlainEncoder(conf.fenc_conf, input_dim=conf.isize)
        self.arg_extractor = ExtractorGetter.make_extractor(conf.arg_conf, vocab_arg, isize=conf.isize,
                                                            psize=evt_psize if conf.arg_use_finput else -1)

    # helper for cons_ef
    def _get_arg_external_extra_score(self, flt_items):
        if self.cons_arg is not None:
            evt_idxes = [(0 if z is None else z.label_idx) for z in flt_items]
            valid_masks = self.cons_arg.lookup(BK.input_idx(evt_idxes))  # [*, L]
            ret = Constants.REAL_PRAC_MIN * (1. - valid_masks)  # [*, L]
            return ret.unsqueeze(-2)  # [bs, 1, L], let later broadcast!
        else:
            return None

    # [**, slen, D], [**, slen, D']
    def _forward_fenc(self, flt_input_expr: BK.Expr, flt_full_expr: BK.Expr, flt_mask_expr: BK.Expr):
        if self.conf.fenc_mix_frame:
            mixed_input_t = flt_input_expr + flt_full_expr  # simply adding
        else:
            mixed_input_t = flt_input_expr
        fenc_output = self.fenc.forward(mixed_input_t, mask_expr=flt_mask_expr)
        return fenc_output

    # [*, slen, D], [*, slen]
    def loss(self, insts: List[Sent], input_expr: BK.Expr, mask_expr: BK.Expr):
        conf: MyFramerConf = self.conf
        # --
        all_losses = []
        # evt
        if conf.loss_evt > 0.:
            evt_loss, evt_res = self.evt_extractor.loss(insts, input_expr, mask_expr)
            one_loss = LossHelper.compile_component_loss("evt", [evt_loss], loss_lambda=conf.loss_evt)
            all_losses.append(one_loss)
        else:
            evt_res = None
        # arg
        if conf.loss_arg > 0.:
            if evt_res is None:
                evt_res = self.evt_extractor.lookup_flatten(insts, input_expr, mask_expr)
            flt_items, flt_sidx, flt_expr, flt_full_expr = evt_res  # flatten to make dim0 -> frames
            flt_input_expr, flt_mask_expr = input_expr[flt_sidx], mask_expr[flt_sidx]
            flt_fenc_expr = self._forward_fenc(flt_input_expr, flt_full_expr, flt_mask_expr)  # [**, slen, D]
            arg_loss, _ = self.arg_extractor.loss(
                flt_items, flt_fenc_expr, flt_mask_expr, pair_expr=(flt_expr if conf.arg_use_finput else None),
                external_extra_score=self._get_arg_external_extra_score(flt_items))
            one_loss = LossHelper.compile_component_loss("arg", [arg_loss], loss_lambda=conf.loss_arg)
            all_losses.append(one_loss)
        # --
        ret_loss = LossHelper.combine_multiple_losses(all_losses)
        return ret_loss

    # [*, slen, D], [*, slen]
    def predict(self, insts: List[Sent], input_expr: BK.Expr, mask_expr: BK.Expr):
        conf: MyFramerConf = self.conf
        if conf.pred_evt:  # predict evt
            evt_res = self.evt_extractor.predict(insts, input_expr, mask_expr)
        else:
            evt_res = None
        if conf.pred_arg:  # predict arg
            # --
            # todo(+N): special delete for old ones here!!
            for s in insts:
                s.delete_frames(self.arg_extractor.conf.arg_ftag)
            # --
            if evt_res is None:
                evt_res = self.evt_extractor.lookup_flatten(insts, input_expr, mask_expr)
            flt_items, flt_sidx, flt_expr, flt_full_expr = evt_res  # flatten to make dim0 -> frames
            if len(flt_items)>0:  # can be erroneous if zero
                flt_input_expr, flt_mask_expr = input_expr[flt_sidx], mask_expr[flt_sidx]
                flt_fenc_expr = self._forward_fenc(flt_input_expr, flt_full_expr, flt_mask_expr)  # [**, slen, D]
                self.arg_extractor.predict(
                    flt_items, flt_fenc_expr, flt_mask_expr, pair_expr=(flt_expr if conf.arg_use_finput else None),
                    external_extra_score=self._get_arg_external_extra_score(flt_items))
        # --
