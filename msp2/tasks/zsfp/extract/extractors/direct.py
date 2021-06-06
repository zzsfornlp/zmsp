#

# Direct Span-based Extractor

__all__ = [
    "DirectExtractorConf", "DirectExtractor", "DirectExtractorHelper",
]

from typing import List, Union
from collections import defaultdict
import numpy as np
from msp2.nn import BK
from msp2.nn.modules import LossHelper
from msp2.nn.layers import BasicConf, BasicNode, node_reg, PosiEmbeddingNode, LayerNormNode
from msp2.data.inst import Sent, Frame, Mention, DataPadder
from msp2.data.vocab import SimpleVocab
from msp2.tasks.common.models.span import SpanExtractorConf, SpanExtractorNode
from msp2.utils import ZObject, Constants
from .base import *

# =====

class DirectExtractorConf(BaseExtractorConf):
    def __init__(self):
        super().__init__()
        # --
        # model
        self.span_conf = SpanExtractorConf()
        self.span_topk_rate = 0.4  # for training(if not sample) and testing
        self.span_topk_count = 1000  # for training(if not sample) and testing
        self.span_train_sample = True  # use sampling in training?
        self.span_train_sample_rate = 0.33
        self.span_train_sample_count = 1000
        self.span_train_topk_add_gold_rate = 1.0  # add gold in topk-training
        # train/loss related
        self.loss_cand = 0.5  # candidate binary loss
        self.cand_label_smoothing = 0.  # for binary loss
        self.loss_use_posi = False  # directly use provided items (positions) rather than extract spans: training only classifier
        self.loss_use_cons = False  # whether use cons
        self.loss_use_lu = False  # try to use provided LU in training
        # others
        self.lab_add_extract_score = False  # add extract score to lab!=0 in laber (if possible)
        # pred related
        self.pred_ignore_non = True  # ignore non predictions (idx=0)
        self.pred_non_overlapping = False  # non overlapping spans in prediction
        self.pred_use_posi = False  # directly use provided items (positions) rather than extract spans
        self.pred_use_cons = False  # whether use cons
        self.pred_use_lu = False  # try to use provided LU in predicting
        self.pred_score_prune = Constants.REAL_PRAC_MIN
        # special one: if ftag==arg
        self.prepare_on_args = True  # when prepare args, directly prepare them rather than cands (like efs)
        # special for flatten_lookup
        self.flatten_lookup_init_scale = 5.
        self.flatten_lookup_use_dist = True
        self.flatten_lookup_use_norm = False
        # --

@node_reg(DirectExtractorConf)
class DirectExtractor(BaseExtractor):
    def __init__(self, conf: DirectExtractorConf, vocab: SimpleVocab, **kwargs):
        super().__init__(conf, vocab, **kwargs)
        conf: DirectExtractorConf = self.conf
        # --
        self.helper = DirectExtractorHelper(conf, vocab)
        self.indicator_embed = PosiEmbeddingNode(  # todo(note): make its scale larger
            None, osize=conf.isize, max_val=100, min_val=-100, init_sincos=False, freeze=False,
            init_scale=conf.flatten_lookup_init_scale)
        self.indicator_norm = LayerNormNode(None, osize=conf.isize) if conf.flatten_lookup_use_norm else (lambda x: x)

    def _build_extract_node(self, conf: DirectExtractorConf):
        span_node = SpanExtractorNode(conf.span_conf, isize=conf.isize, shead_mode=(conf.core_span_mode=="shead"))
        return span_node

    def _extend_cand_score(self, cand_score: BK.Expr):
        if self.conf.lab_add_extract_score and cand_score is not None:
            non0_mask = self.lab_node.laber.speical_mask_non0
            ret = non0_mask * cand_score.unsqueeze(-1)  # [*, slen, L]
        else:
            ret = None
        return ret

    # [*, slen, D], [*, slen], [*, D']
    def loss(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
             pair_expr: BK.Expr = None, lookup_flatten=False, external_extra_score: BK.Expr=None):
        conf: DirectExtractorConf = self.conf
        # step 0: prepare golds
        arr_gold_items, expr_gold_gaddr, expr_gold_widxes, expr_gold_wlens, expr_loss_weight_non = \
            self.helper.prepare(insts, use_cache=True)
        # step 1: extract cands
        if conf.loss_use_posi:
            cand_res = self.extract_node.go_lookup(
                input_expr, expr_gold_widxes, expr_gold_wlens, (expr_gold_gaddr>=0).float(), gaddr_expr=expr_gold_gaddr)
        else:  # todo(note): assume no in-middle mask!!
            cand_widx, cand_wlen, cand_mask, cand_gaddr = self.extract_node.prepare_with_lengths(
                BK.get_shape(mask_expr), mask_expr.sum(-1).long(), expr_gold_widxes, expr_gold_wlens, expr_gold_gaddr)
            if conf.span_train_sample:  # simply do sampling
                cand_res = self.extract_node.go_sample(
                    input_expr, mask_expr, cand_widx, cand_wlen, cand_mask,
                    rate=conf.span_train_sample_rate, count=conf.span_train_sample_count,
                    gaddr_expr=cand_gaddr, add_gold_rate=1.0)  # note: always fully add gold for sampling!!
            else:  # beam pruner using topk
                cand_res = self.extract_node.go_topk(
                    input_expr, mask_expr, cand_widx, cand_wlen, cand_mask,
                    rate=conf.span_topk_rate, count=conf.span_topk_count,
                    gaddr_expr=cand_gaddr, add_gold_rate=conf.span_train_topk_add_gold_rate)
        # step 1+: prepare for labeling
        cand_gold_mask = (cand_res.gaddr_expr>=0).float() * cand_res.mask_expr  # [*, cand_len]
        # todo(note): add a 0 as idx=-1 to make NEG ones as 0!!
        flatten_gold_label_idxes = BK.input_idx([(0 if z is None else z.label_idx) for z in arr_gold_items.flatten()] + [0])
        gold_label_idxes = flatten_gold_label_idxes[cand_res.gaddr_expr]
        cand_loss_weights = BK.where(gold_label_idxes==0, expr_loss_weight_non.unsqueeze(-1)*conf.loss_weight_non, cand_res.mask_expr)
        final_loss_weights = cand_loss_weights * cand_res.mask_expr
        # cand loss
        if conf.loss_cand > 0. and not conf.loss_use_posi:
            loss_cand0 = BK.loss_binary(cand_res.score_expr, cand_gold_mask, label_smoothing=conf.cand_label_smoothing)
            loss_cand = (loss_cand0 * final_loss_weights).sum()
            loss_cand_item = LossHelper.compile_leaf_loss(f"cand", loss_cand, final_loss_weights.sum(),
                                                          loss_lambda=conf.loss_cand)
        else:
            loss_cand_item = None
        # extra score
        cand_extra_score = self._get_extra_score(
            cand_res.score_expr, insts, cand_res, arr_gold_items, conf.loss_use_cons, conf.loss_use_lu)
        final_extra_score = self._sum_scores(external_extra_score, cand_extra_score)
        # step 2: label; with special weights
        loss_lab, loss_count = self.lab_node.loss(
            cand_res.span_expr, pair_expr, cand_res.mask_expr, gold_label_idxes,
            loss_weight_expr=final_loss_weights, extra_score=final_extra_score)
        loss_lab_item = LossHelper.compile_leaf_loss(f"lab", loss_lab, loss_count,
                                                     loss_lambda=conf.loss_lab, gold=cand_gold_mask.sum())
        # ==
        # return loss
        ret_loss = LossHelper.combine_multiple_losses([loss_cand_item, loss_lab_item])
        return self._finish_loss(ret_loss, insts, input_expr, mask_expr, pair_expr, lookup_flatten)

    # [*, slen, D], [*, slen], [*, D']
    def predict(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
                pair_expr: BK.Expr = None, lookup_flatten=False, external_extra_score: BK.Expr=None):
        conf: DirectExtractorConf = self.conf
        # step 1: prepare targets
        if conf.pred_use_posi:
            # step 1a: directly use provided positions
            arr_gold_items, expr_gold_gaddr, expr_gold_widxes, expr_gold_wlens, _ = self.helper.prepare(insts, use_cache=False)
            cand_res = self.extract_node.go_lookup(input_expr, expr_gold_widxes, expr_gold_wlens,
                                                   (expr_gold_gaddr>=0).float(), gaddr_expr=expr_gold_gaddr)
        else:
            arr_gold_items = None
            # step 1b: extract cands (topk); todo(note): assume no in-middle mask!!
            cand_widx, cand_wlen, cand_mask, _ = self.extract_node.prepare_with_lengths(
                BK.get_shape(mask_expr), mask_expr.sum(-1).long(), None, None, None)
            cand_res = self.extract_node.go_topk(
                input_expr, mask_expr, cand_widx, cand_wlen, cand_mask,
                rate=conf.span_topk_rate, count=conf.span_topk_count,
                non_overlapping=conf.pred_non_overlapping, score_prune=conf.pred_score_prune)
        # --
        # note: check empty
        if BK.is_zero_shape(cand_res.mask_expr):
            if not conf.pred_use_posi:
                for inst in insts:  # still need to clear things!!
                    self.helper._clear_f(inst)
        else:
            # step 2: labeling
            # extra score
            cand_extra_score = self._get_extra_score(
                cand_res.score_expr, insts, cand_res, arr_gold_items, conf.pred_use_cons, conf.pred_use_lu)
            final_extra_score = self._sum_scores(external_extra_score, cand_extra_score)
            best_labs, best_scores = self.lab_node.predict(
                cand_res.span_expr, pair_expr, cand_res.mask_expr, extra_score=final_extra_score)
            # step 3: put results
            if conf.pred_use_posi:  # reuse the old ones, but replace label
                self.helper.put_labels(arr_gold_items, best_labs, best_scores)
            else:  # make new frames
                self.helper.put_results(insts, best_labs, best_scores, cand_res.widx_expr, cand_res.wlen_expr, cand_res.mask_expr)
        # --
        # finally
        return self._finish_pred(insts, input_expr, mask_expr, pair_expr, lookup_flatten)

    # [*, slen, D], [*, slen], [*, D']
    def lookup(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
               pair_expr: BK.Expr = None):
        arr_items, expr_gaddr, expr_widxes, expr_wlens, _ = \
            self.helper.prepare(insts, use_cache=self.is_training())  # only for training instances!
        cand_res = self.extract_node.go_lookup(input_expr, expr_widxes, expr_wlens, (expr_gaddr>=0).float())
        expr_labs = BaseExtractorHelper.get_batched_features(arr_items, 0, 'label_idx', dtype=BK.long)
        mlp_expr = self.lookup_node.lookup(cand_res.span_expr, expr_labs, cand_res.mask_expr)
        # --
        return arr_items, mlp_expr, cand_res.mask_expr, (expr_widxes, expr_wlens)  # [bs, ?, D]

    # plus flatten items's dims
    def lookup_flatten(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
                       pair_expr: BK.Expr = None):
        arr_items, mlp_expr, zmask_expr, extra_info = self.lookup(insts, input_expr, mask_expr, pair_expr)
        expr_widxes, expr_wlens = extra_info
        # flatten
        ret_items, ret_sidx, ret_expr, fl_widxes, fl_wlens = LookupNode.flatten_results(
            arr_items, zmask_expr, mlp_expr, expr_widxes, expr_wlens)
        # --
        # also make full expr
        # full_masks = ((_arange_t>=fl_widxes.unsqueeze(-1)) & (_arange_t<(fl_widxes+fl_wlens).unsqueeze(-1))).float()  # [??, slen]
        # ret_full_expr = full_masks.unsqueeze(-1) * ret_expr.unsqueeze(-2)  # [??, slen, D]
        if self.conf.flatten_lookup_use_dist:  # use posi again: [...,-2,-1,0,0,0,1,2,...]
            left_widxes = fl_widxes.unsqueeze(-1)  # [??, 1]
            right_widxes = (fl_widxes+fl_wlens-1).unsqueeze(-1)  # [??, 1]
            _arange_t = BK.arange_idx(BK.get_shape(mask_expr, 1)).unsqueeze(0)  # [1, slen]
            dist0 = _arange_t - left_widxes  # [??, slen]
            dist1 = _arange_t - right_widxes  # [??, slen]
            full_dist = (_arange_t < left_widxes).long() * dist0 + (_arange_t > right_widxes).long() * dist1
            ret_full_expr = self.indicator_norm(self.indicator_embed(full_dist))  # [??, slen, D]
            # # ret_full_expr = self.indicator_embed(full_dist)  # [??, slen, D]
        else:  # otherwise 0/1
            _arange_t = BK.arange_idx(BK.get_shape(mask_expr, 1)).unsqueeze(0)  # [1, slen]
            full_ind = ((_arange_t>=fl_widxes.unsqueeze(-1)) & (_arange_t<(fl_widxes+fl_wlens).unsqueeze(-1))).long()  # [??, slen]
            ret_full_expr = self.indicator_norm(self.indicator_embed(full_ind))  # [??, slen, D]
        # --
        return ret_items, ret_sidx, ret_expr, ret_full_expr  # [??, D]

    # get extra score
    def _get_extra_score(self, cand_score, insts, cand_res, arr_gold_items, use_cons: bool, use_lu: bool):
        # conf: DirectExtractorConf = self.conf
        # --
        # first cand score
        cand_score = self._extend_cand_score(cand_score)
        # then cons_lex score
        cons_lex_node = self.cons_lex_node
        if use_cons and cons_lex_node is not None:
            cons_lex = cons_lex_node.cons
            flt_arr_gold_items = arr_gold_items.flatten()
            _shape = BK.get_shape(cand_res.mask_expr)
            if cand_res.gaddr_expr is None:
                gaddr_expr = BK.constants(_shape, -1, dtype=BK.long)
            else:
                gaddr_expr = cand_res.gaddr_expr
            all_arrs = [BK.get_value(z) for z in [cand_res.widx_expr, cand_res.wlen_expr, cand_res.mask_expr, gaddr_expr]]
            arr_feats = np.full(_shape, None, dtype=object)
            for bidx, inst in enumerate(insts):
                one_arr_feats = arr_feats[bidx]
                _ii = -1
                for one_widx, one_wlen, one_mask, one_gaddr in zip(*[z[bidx] for z in all_arrs]):
                    _ii += 1
                    if one_mask == 0.: continue  # skip invlaid ones
                    if use_lu and one_gaddr>=0:
                        one_feat = cons_lex.lu2feat(flt_arr_gold_items[one_gaddr].info["luName"])
                    else:
                        one_feat = cons_lex.span2feat(inst, one_widx, one_wlen)
                    one_arr_feats[_ii] = one_feat
            cons_valids = cons_lex_node.lookup_with_feats(arr_feats)
            cons_score = (1.-cons_valids) * Constants.REAL_PRAC_MIN
        else:
            cons_score = None
        # sum
        return self._sum_scores(cand_score, cons_score)

# --
class DirectExtractorHelper(BaseExtractorHelper):
    def __init__(self, conf: DirectExtractorConf, vocab: SimpleVocab):
        super().__init__(conf, vocab)
        conf: DirectExtractorConf = self.conf
        # --
        if conf.ftag == "arg":
            if conf.prepare_on_args:
                self._prep_f = self._prep_args
            else:
                self._prep_f = self._prep_args_from_cands
        else:
            self._prep_f = self._prep_frames

    def _prep_items(self, items: List, par: object):
        core_spans = [self.core_span_getter(f.mention) for f in items]
        _loss_weight_non = getattr(par, "_loss_weight_non", 1.)  # todo(+N): special name; loss_weight_non
        return ZObject(items=items, par=par, len=len(items), loss_weight_non=_loss_weight_non,
                       core_widxes=[z[0] for z in core_spans], core_wlens=[z[1] for z in core_spans])

    def _prep_frames(self, s: Sent):
        return self._prep_items(self._get_frames(s), s)

    def _prep_args(self, f: Frame):
        return self._prep_items(self._get_args(f), f)

    def _prep_args_from_cands(self, f: Frame):
        conf = self.conf
        # --
        args = self._get_args(f)  # arg list
        cands = f.sent.get_frames(conf.arg_ftag)  # from cand list: arg_ftag
        # --
        arg_id_map = defaultdict(list)
        for a in args:  # note: there can be repeated ones
            arg_id_map[id(a.arg)].append(a)
        # --
        cand_res = self._prep_items(cands, f)
        arg_items = []
        for ef_item in cand_res.items:
            alinks = arg_id_map.get(id(ef_item))
            arg_items.append(None if alinks is None else alinks[0])  # todo(+W): currently we only find one arg!!
        cand_res.items = arg_items  # directly replace!
        return cand_res

    # =====
    # put outputs

    # *[*, slen]
    def put_results(self, insts, best_labs, best_scores, widx_expr, wlen_expr, mask_expr):
        conf: DirectExtractorConf = self.conf
        conf_pred_ignore_non = conf.pred_ignore_non
        # --
        all_arrs = [BK.get_value(z) for z in [best_labs, best_scores, widx_expr, wlen_expr, mask_expr]]
        for bidx, inst in enumerate(insts):
            self._clear_f(inst)  # first clean things
            for one_lab, one_score, one_widx, one_wlen, one_mask in zip(*[z[bidx] for z in all_arrs]):
                one_lab = int(one_lab)
                # todo(+N): again, assuming NON-idx == 0
                if one_mask == 0. or (conf_pred_ignore_non and one_lab == 0): continue  # invalid one: unmask or NON
                new_item = self._new_f(inst, int(one_widx), int(one_wlen), one_lab, float(one_score))
        # --

    def put_labels(self, arr_items, best_labs, best_scores):
        assert BK.get_shape(best_labs) == BK.get_shape(arr_items)
        # --
        all_arrs = [BK.get_value(z) for z in [best_labs, best_scores]]
        for cur_items, cur_labs, cur_scores in zip(arr_items, *all_arrs):
            for one_item, one_lab, one_score in zip(cur_items, cur_labs, cur_scores):
                if one_item is None: continue
                one_lab, one_score = int(one_lab), float(one_score)
                one_item.score = one_score
                one_item.set_label_idx(one_lab)
                one_item.set_label(self.vocab.idx2word(one_lab))

    # =====
    # prepare inputs
    def prepare(self, insts: Union[List[Sent], List[Frame]], use_cache: bool):
        conf: DirectExtractorConf = self.conf
        # get info
        if use_cache:
            zobjs = []
            attr_name = f"_dcache_{conf.ftag}"  # should be unique
            for s in insts:
                one = getattr(s, attr_name, None)
                if one is None:
                    one = self._prep_f(s)
                    setattr(s, attr_name, one)  # set cache
                zobjs.append(one)
        else:
            zobjs = [self._prep_f(s) for s in insts]
        # batch things
        bsize, mlen = len(insts), max(z.len for z in zobjs) if len(zobjs)>0 else 1  # at least put one as padding
        batched_shape = (bsize, mlen)
        arr_items = np.full(batched_shape, None, dtype=object)
        arr_gaddrs = np.arange(bsize*mlen).reshape(batched_shape)  # gold address
        arr_core_widxes = np.full(batched_shape, 0, dtype=np.int)
        arr_core_wlens = np.full(batched_shape, 1, dtype=np.int)
        # arr_ext_widxes = np.full(batched_shape, 0, dtype=np.int)
        # arr_ext_wlens = np.full(batched_shape, 1, dtype=np.int)
        for zidx, zobj in enumerate(zobjs):
            zlen = zobj.len
            arr_items[zidx, :zlen] = zobj.items
            arr_core_widxes[zidx, :zlen] = zobj.core_widxes
            arr_core_wlens[zidx, :zlen] = zobj.core_wlens
            # arr_ext_widxes[zidx, :zlen] = zobj.ext_widxes
            # arr_ext_wlens[zidx, :zlen] = zobj.ext_wlens
        arr_gaddrs[arr_items==None] = -1  # set -1 as gaddr
        # final setup things
        expr_gaddr = BK.input_idx(arr_gaddrs)  # [*, GLEN]
        expr_core_widxes = BK.input_idx(arr_core_widxes)
        expr_core_wlens = BK.input_idx(arr_core_wlens)
        # expr_ext_widxes = BK.input_idx(arr_ext_widxes)
        # expr_ext_wlens = BK.input_idx(arr_ext_wlens)
        expr_loss_weight_non = BK.input_real([z.loss_weight_non for z in zobjs])  # [*]
        # return arr_flatten_items, expr_gaddr, expr_core_widxes, expr_core_wlens, \
        #        expr_ext_widxes, expr_ext_wlens, expr_loss_weight_non
        return arr_items, expr_gaddr, expr_core_widxes, expr_core_wlens, expr_loss_weight_non
