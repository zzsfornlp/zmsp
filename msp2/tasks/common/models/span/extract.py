#

# Span Extractor:
# -- various mode: (lookup, topk, lookup+topk)

__all__ = [
    "SpanExtractorConf", "SpanExtractorNode", "SpanExtractorOutput",
]

from typing import Tuple, Callable
from collections import namedtuple
import numpy as np
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.utils import Constants, ZObject, zlog
from .base import *

class SpanExtractorConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize = -1  # input repr dim
        # --
        # span: output dim is in here!
        self.span = BaseSpanConf()
        # scorer
        self.sconf = PlainScorerConf().direct_update(hid_nlayer=1)  # by default one layer
        # modes
        self.min_width = 1  # minimum width (inclusive)
        self.max_width = 15  # maximum width (inclusive)
        # special single-head mode
        self.shead_mode = False

    def _do_validate(self):
        if self.shead_mode or (self.min_width==1 and self.max_width==1):  # special mode!!
            self.shead_mode = True
            self.min_width = self.max_width = 1
            # if in this mode, we only need to put the expr itself!
            # zlog("Hit 'shead_mode' for SpanExtractor!!")
            self.span.direct_update(use_starts=True, use_ends=False, use_softhead=False, use_width=False, use_proj=False)

# the tupled output
class SpanExtractorOutput(ZObject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def arrange(self, aranget_t: BK.Expr, idx_t: BK.Expr, final_mask_t: BK.Expr):
        for key in list(self.__dict__.keys()):
            orig_v = getattr(self, key)
            if orig_v is not None:
                res_v = orig_v[aranget_t, idx_t]
                setattr(self, key, res_v)
        # todo(note): remeber to replace the actual mask!!
        if final_mask_t is not None:
            self.mask_expr = final_mask_t

@node_reg(SpanExtractorConf)
class SpanExtractorNode(BasicNode):
    def __init__(self, conf: SpanExtractorConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: SpanExtractorConf = self.conf
        # --
        # span
        self.spanner = BaseSpanNode(conf.span, isize=conf.isize)
        self.output_dim = self.spanner.get_output_dims()[0]
        # scorer
        self.scorer = PlainScorerNode(conf.sconf, isize=self.output_dim, osize=1)

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    # ==
    # various modes
    # input: *[bsize, slen, *]; span: *[bsize, NUM, *];
    #   gold-addr: [bsize, NUM], -1 means nope; add_gold means force adding gold

    # direct lookup and return all
    def go_lookup(self, input_expr: BK.Expr, widx_expr: BK.Expr, wlen_expr: BK.Expr, span_mask: BK.Expr, gaddr_expr: BK.Expr=None):
        span_expr = self.spanner(input_expr, widx_expr, wlen_expr)  # [bsize, NUM, D]
        score_expr = self.scorer(span_expr).squeeze(-1)  # [bsize, NUM]
        # (no need to add here!!) score_expr += Constants.REAL_PRAC_MIN * (1. - span_mask)  # [bsize, NUM]
        # return span_expr, score_expr
        return SpanExtractorOutput(span_expr=span_expr, score_expr=score_expr, widx_expr=widx_expr, wlen_expr=wlen_expr,
                                   mask_expr=span_mask, gaddr_expr=gaddr_expr)

    # common go_*: [bsize, NUM] -> [bsize, SNUM]
    # note: inplaced modification
    def _go_common(self, res: SpanExtractorOutput, sel_mask: BK.Expr, add_gold_rate: float):
        gaddr_expr, span_mask = res.gaddr_expr, res.mask_expr
        bsize = BK.get_shape(span_mask, 0)
        # add gold?
        if add_gold_rate>0.:  # inplace
            gold_mask = ((gaddr_expr>=0) & (BK.rand(sel_mask.shape) < add_gold_rate)).float()  # note: gaddr==-1 means nope
            sel_mask += gold_mask
            sel_mask.clamp_(max=1.)  # OR
        sel_mask *= span_mask  # must be valid
        # select masked
        final_idx_t, final_mask_t = BK.mask2idx(sel_mask, padding_idx=0)  # [bsize, ??]
        _tmp_arange_t = BK.arange_idx(bsize).unsqueeze(1)  # [bsize, 1]
        res.arrange(_tmp_arange_t, final_idx_t, final_mask_t)
        if res.gaddr_expr is not None:
            res.gaddr_expr.masked_fill_(final_mask_t==0., -1)  # make invalid ones -1
        return res  # [bsize, SNUM, *]

    # helper function for determine size: [*, 1]
    def _determine_size(self, length: BK.Expr, rate: float, count: float):
        if rate is None:
            ret_size = BK.constants(length.shape, count)  # make it constant
        else:
            ret_size = (length * rate)
            if count is not None:
                ret_size.clamp_(max=count)
        ret_size.ceil_()
        return ret_size

    # random sample
    def go_sample(self, input_expr: BK.Expr, input_mask: BK.Expr,  # input
                  widx_expr: BK.Expr, wlen_expr: BK.Expr, span_mask: BK.Expr, rate: float=None, count: float=None,  # span
                  gaddr_expr: BK.Expr=None, add_gold_rate: float=0.):  # gold
        lookup_res = self.go_lookup(input_expr, widx_expr, wlen_expr, span_mask, gaddr_expr)  # [bsize, NUM, *]
        # --
        # rate is according to overall input length
        _tmp_len = (input_mask.sum(-1, keepdim=True) + 1e-5)
        sample_rate = self._determine_size(_tmp_len, rate, count) / _tmp_len  # [bsize, 1]
        sample_mask = (BK.rand(span_mask.shape) < sample_rate).float()  # [bsize, NUM]
        # select and add_gold
        return self._go_common(lookup_res, sample_mask, add_gold_rate)

    # select by topk score
    def go_topk(self, input_expr: BK.Expr, input_mask: BK.Expr,  # input
                widx_expr: BK.Expr, wlen_expr: BK.Expr, span_mask: BK.Expr, rate: float=None, count: float=None,  # span
                gaddr_expr: BK.Expr=None, add_gold_rate: float=0.,  # gold
                non_overlapping=False, score_prune: float=None):  # non-overlapping!
        lookup_res = self.go_lookup(input_expr, widx_expr, wlen_expr, span_mask, gaddr_expr)  # [bsize, NUM, *]
        # --
        with BK.no_grad_env():  # no need grad here!
            all_score_expr = lookup_res.score_expr
            # get topk score: again rate is to the original input length
            if BK.is_zero_shape(lookup_res.mask_expr):
                topk_mask = lookup_res.mask_expr.clone()  # no need to go topk since no elements
            else:
                topk_expr = self._determine_size(input_mask.sum(-1, keepdim=True), rate, count).long()  # [bsize, 1]
                if non_overlapping:
                    topk_mask = select_topk_non_overlapping(
                        all_score_expr, topk_expr, widx_expr, wlen_expr, input_mask, mask_t=span_mask, dim=-1)
                else:
                    topk_mask = select_topk(all_score_expr, topk_expr, mask_t=span_mask, dim=-1)
            # further score_prune?
            if score_prune is not None:
                topk_mask *= (all_score_expr>=score_prune).float()
        # select and add_gold
        return self._go_common(lookup_res, topk_mask, add_gold_rate)

    # ==
    # prepare the inputs: enumerate all possible spans and attach gold annotations (gold address)

    # def good_wlen(self, wlen: int):
    #     conf: SpanExtractorConf = self.conf
    #     return wlen>=conf.min_width and wlen<=conf.max_width

    # common routine: [bsize, mlen], Callable(widx,wlen,other), *[bsize, glen] -> [bsize, ??]
    def _common_prepare(self, input_shape: Tuple[int], _mask_f: Callable,
                        gold_widx_expr: BK.Expr, gold_wlen_expr: BK.Expr, gold_addr_expr: BK.Expr):
        conf: SpanExtractorConf = self.conf
        min_width, max_width = conf.min_width, conf.max_width
        diff_width = max_width - min_width + 1  # number of width to extract
        # --
        bsize, mlen = input_shape
        # --
        # [bsize, mlen*(max_width-min_width)], mlen first (dim=1)
        # note: the spans are always sorted by (widx, wlen)
        _tmp_arange_t = BK.arange_idx(mlen*diff_width)  # [mlen*dw]
        widx_t0 = (_tmp_arange_t // diff_width)  # [mlen*dw]
        wlen_t0 = (_tmp_arange_t % diff_width) + min_width  # [mlen*dw]
        mask_t0 = _mask_f(widx_t0, wlen_t0)  # [bsize, mlen*dw]
        # --
        # compacting (use mask2idx and gather)
        final_idx_t, final_mask_t = BK.mask2idx(mask_t0, padding_idx=0)  # [bsize, ??]
        _tmp2_arange_t = BK.arange_idx(bsize).unsqueeze(1)  # [bsize, 1]
        # no need to make valid for mask=0, since idx=0 means (0, min_width)
        # todo(+?): do we need to deal with empty ones here?
        ret_widx = widx_t0[final_idx_t]  # [bsize, ??]
        ret_wlen = wlen_t0[final_idx_t]  # [bsize, ??]
        # --
        # prepare gold (as pointer-like addresses)
        if gold_addr_expr is not None:
            gold_t0 = BK.constants_idx((bsize, mlen*diff_width), -1)  # [bsize, mlen*diff]
            # check valid of golds (flatten all)
            gold_valid_t = ((gold_addr_expr>=0) & (gold_wlen_expr>=min_width) & (gold_wlen_expr<=max_width))
            gold_valid_t = gold_valid_t.view(-1)  # [bsize*_glen]
            _glen = BK.get_shape(gold_addr_expr, 1)
            flattened_bsize_t = BK.arange_idx(bsize*_glen) // _glen  # [bsize*_glen]
            flattened_fidx_t = (gold_widx_expr*diff_width+gold_wlen_expr-min_width).view(-1)  # [bsize*_glen]
            flattened_gaddr_t = gold_addr_expr.view(-1)
            # mask and assign
            gold_t0[flattened_bsize_t[gold_valid_t], flattened_fidx_t[gold_valid_t]] = flattened_gaddr_t[gold_valid_t]
            ret_gaddr = gold_t0[_tmp2_arange_t, final_idx_t]  # [bsize, ??]
            ret_gaddr.masked_fill_((final_mask_t == 0), -1)  # make invalid ones -1
        else:
            ret_gaddr = None
        # --
        return ret_widx, ret_wlen, final_mask_t, ret_gaddr

    # assuming all okay within lengths: [bsize, mlen], [bsize], *[bsize, glen] -> [bsize, ??]
    def prepare_with_lengths(self, input_shape: Tuple[int], length_expr: BK.Expr,
                             gold_widx_expr: BK.Expr, gold_wlen_expr: BK.Expr, gold_addr_expr: BK.Expr):
        _f = (lambda _widx, _wlen: ((_widx+_wlen).unsqueeze(0) <= length_expr.unsqueeze(-1)).float())  # [bsize, mlen*dw]
        return self._common_prepare(input_shape, _f, gold_widx_expr, gold_wlen_expr, gold_addr_expr)

    # note: any positions can be masked: [bsize, mlen], [bsize, mlen], *[bsize, glen] -> [bsize, ??]
    def prepare_with_masks(self, input_shape: Tuple[int], input_mask: BK.Expr,
                           gold_widx_expr: BK.Expr, gold_wlen_expr: BK.Expr, gold_addr_expr: BK.Expr):
        # -- need to check all mask (allow mask/excluding-tokens in the middle)
        def _f(_widx, _wlen):  # [mlen*dw] -> [bsize, mlen*dw]
            _mw = self.conf.max_width  # todo(note): also rely on this
            _bsize = BK.get_shape(input_mask, 0)
            _padded_input_mask = BK.pad(input_mask, (0,_mw), value=0.)  # [bsize, mlen+_mw], make idxing valid!!
            _er_idxes, _er_masks = expand_ranged_idxes(_widx, _wlen, 0, _mw)  # [mlen*dw, MW]
            _arange_t = BK.arange_idx(_bsize).unsqueeze(-1).unsqueeze(-1)  # [bsize, 1, 1]
            _idx_valid = _padded_input_mask[_arange_t, _er_idxes.unsqueeze(0)]  # [bsize, mlen*dw, MW]
            _idx_valid.masked_fill_((_er_masks==0.).unsqueeze(0), 1.)  # make sure paddings get 1
            _ret = _idx_valid.prod(-1)  # [bsize, mlen*dw], require all non-pad ones to be valid!
            return _ret
        # --
        return self._common_prepare(input_shape, _f, gold_widx_expr, gold_wlen_expr, gold_addr_expr)
