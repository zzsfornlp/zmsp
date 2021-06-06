#

# soft (extract-)split (head or multiple head) extractor
# step 1: candidate extract, step 2 (key!): boundary decision and split, then step 3+: lab, ...

__all__ = [
    "SoftExtractorConf", "SoftExtractor", "SoftExtractorHelper",
]

from typing import List, Union
from collections import defaultdict
import numpy as np
from msp2.nn import BK
from msp2.nn.modules import LossHelper
from msp2.nn.layers import *
from msp2.data.inst import Sent, Frame, Mention, DataPadder
from msp2.data.vocab import SimpleVocab, SeqVocab, SeqVocabConf
from msp2.utils import ZObject, Constants, zwarn, zlog
from .base import *

# =====
class SoftExtractorConf(BaseExtractorConf):
    def __init__(self):
        super().__init__()
        # --
        # step 1: cand
        self.cand_sconf = PlainScorerConf().direct_update(hid_nlayer=1)  # by default one layer
        # cand select
        self.cand_topk_rate = 1.0  # training & testing
        self.cand_topk_count = 1000  # training & testing
        self.cand_thresh = 0.  # training & testing
        # step 2: split
        self.split_sconf = PairScorerConf().direct_update(biaffine_div=0.)  # pairwise split point classifier
        self.split_topk = 1  # how many representative nodes?
        # step 3+:
        # special for flatten_lookup
        self.flatten_lookup_init_scale = 5.
        # --
        # for training
        # step 1: cand
        self.loss_cand = 1.0
        self.loss_cand_entropy = 0.  # minimize entropy inside one arg's bag
        self.cand_label_smoothing = 0.  # for binary loss
        self.cand_loss_weight_alpha = 1.0  # exp(score*alpha) / ...
        self.cand_loss_div_max = True  # div_max otherwise div_sum
        self.cand_loss_weight_thresh = 0.  # truncate loss weights if too small
        self.cand_loss_pos_poison = False  # dis-encourage all pos tokens
        self.cand_feed_sample_rate = 0.  # this->sample, (1-this)->pred
        self.cand_detach_weight = True  # whether detach weight?
        # step 2: split
        self.loss_split = 1.0
        self.split_label_smoothing = 0.  # for binary loss
        self.split_feed_force_rate = 1.0  # teacher-forcing rate: this->force, (1-this)->pred

@node_reg(SoftExtractorConf)
class SoftExtractor(BaseExtractor):
    def __init__(self, conf: SoftExtractorConf, vocab: SimpleVocab, **kwargs):
        super().__init__(conf, vocab, **kwargs)
        conf: SoftExtractorConf = self.conf
        # --
        self.helper = SoftExtractorHelper(conf, vocab)
        # --
        self.cand_scorer = PlainScorerNode(conf.cand_sconf, isize=conf.isize, osize=1)
        self.split_scorer = PairScorerNode(conf.split_sconf, isize0=conf.isize, isize1=conf.isize, osize=1)
        # simply 0/1 indicator embeddings
        self.indicator_embed = EmbeddingNode(None, osize=conf.isize, n_words=2, init_scale=conf.flatten_lookup_init_scale)

    def _build_extract_node(self, conf: SoftExtractorConf):
        return None  # extraction node is complex, thus managed by self!

    # [*, slen, D], [*, slen], [*, D']
    def loss(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
             pair_expr: BK.Expr = None, lookup_flatten=False, external_extra_score: BK.Expr = None):
        conf: SoftExtractorConf = self.conf
        assert not lookup_flatten
        bsize, slen = BK.get_shape(mask_expr)
        # --
        # step 0: prepare
        arr_items, expr_seq_gaddr, expr_group_widxes, expr_group_masks, expr_loss_weight_non = \
            self.helper.prepare(insts, mlen=BK.get_shape(mask_expr, -1), use_cache=True)
        arange2_t = BK.arange_idx(bsize).unsqueeze(-1)  # [*, 1]
        # --
        # step 1: cand
        cand_full_scores, pred_cand_decisions = self._cand_score_and_select(input_expr, mask_expr)  # [*, slen]
        loss_cand_items, cand_widxes, cand_masks = self._loss_feed_cand(
            mask_expr, cand_full_scores, pred_cand_decisions,
            expr_seq_gaddr, expr_group_widxes, expr_group_masks, expr_loss_weight_non,
        )  # ~, [*, clen]
        # --
        # step 2: split
        cand_expr, cand_scores = input_expr[arange2_t, cand_widxes], cand_full_scores[arange2_t, cand_widxes]  # [*, clen]
        split_scores, pred_split_decisions = self._split_score(cand_expr, cand_masks)  # [*, clen-1]
        loss_split_item, seg_masks, seg_ext_widxes, seg_ext_masks, seg_weighted_expr, oracle_gaddr = self._loss_feed_split(
            mask_expr, split_scores, pred_split_decisions,
            cand_widxes, cand_masks, cand_expr, cand_scores, expr_seq_gaddr,
        )  # ~, [*, seglen, *?]
        # --
        # step 3: lab
        # todo(note): add a 0 as idx=-1 to make NEG ones as 0!!
        flatten_gold_label_idxes = BK.input_idx([(0 if z is None else z.label_idx) for z in arr_items.flatten()] + [0])
        gold_label_idxes = flatten_gold_label_idxes[oracle_gaddr]  # [*, seglen]
        lab_loss_weights = BK.where(oracle_gaddr>=0, expr_loss_weight_non.unsqueeze(-1)*conf.loss_weight_non, seg_masks)  # [*, seglen]
        final_lab_loss_weights = lab_loss_weights * seg_masks   # [*, seglen]
        # go
        loss_lab, loss_count = self.lab_node.loss(
            seg_weighted_expr, pair_expr, seg_masks, gold_label_idxes,
            loss_weight_expr=final_lab_loss_weights, extra_score=external_extra_score)
        loss_lab_item = LossHelper.compile_leaf_loss(
            f"lab", loss_lab, loss_count, loss_lambda=conf.loss_lab, gold=(gold_label_idxes>0).float().sum())
        # step 4: extend
        flt_mask = ((gold_label_idxes > 0) & (seg_masks > 0.))  # [*, seglen]
        flt_sidx = BK.arange_idx(bsize).unsqueeze(-1).expand_as(flt_mask)[flt_mask]  # [?]
        flt_expr = seg_weighted_expr[flt_mask]  # [?, D]
        flt_full_expr = self._prepare_full_expr(seg_ext_widxes[flt_mask], seg_ext_masks[flt_mask], slen)  # [?, slen, D]
        flt_items = arr_items.flatten()[BK.get_value(oracle_gaddr[flt_mask])]  # [?]
        loss_ext_item = self.ext_node.loss(flt_items, input_expr[flt_sidx], flt_expr, flt_full_expr, mask_expr[flt_sidx])
        # --
        # return loss
        ret_loss = LossHelper.combine_multiple_losses(loss_cand_items + [loss_split_item, loss_lab_item, loss_ext_item])
        return ret_loss, None

    # [*, slen, D], [*, slen], [*, D']
    def predict(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
                pair_expr: BK.Expr = None, lookup_flatten=False, external_extra_score: BK.Expr=None):
        conf: SoftExtractorConf = self.conf
        assert not lookup_flatten
        bsize, slen = BK.get_shape(mask_expr)
        # --
        for inst in insts:  # first clear things
            self.helper._clear_f(inst)
        # --
        # step 1: cand score and select
        cand_full_scores, cand_decisions = self._cand_score_and_select(input_expr, mask_expr)  # [*, slen]
        cand_widxes, cand_masks = BK.mask2idx(cand_decisions)  # [*, clen]
        # step 2: split and seg
        arange2_t = BK.arange_idx(bsize).unsqueeze(-1)  # [*, 1]
        arange3_t = BK.arange_idx(bsize).unsqueeze(-1).unsqueeze(-1)  # [*, 1, 1]
        cand_expr, cand_scores = input_expr[arange2_t, cand_widxes], cand_full_scores[arange2_t, cand_widxes]  # [*, clen]
        split_scores, split_decisions = self._split_score(cand_expr, cand_masks)  # [*, clen-1]
        # *[*, seglen, MW], [*, seglen]
        seg_ext_cidxes, seg_ext_masks, seg_masks = self._split_extend(split_decisions, cand_masks)
        seg_ext_widxes0, seg_ext_masks0 = cand_widxes[arange3_t, seg_ext_cidxes], seg_ext_masks  # [*, seglen, ORIG-MW]
        seg_ext_scores, seg_ext_cidxes, seg_ext_widxes, seg_ext_masks, seg_weighted_expr = self._split_aggregate(
            cand_expr, cand_scores, cand_widxes, seg_ext_cidxes, seg_ext_masks, conf.split_topk)  # [*, seglen, ?]
        # --
        # step 3: lab
        flt_items = []
        if not BK.is_zero_shape(seg_masks):
            best_labs, best_scores = self.lab_node.predict(
                seg_weighted_expr, pair_expr, seg_masks, extra_score=external_extra_score)  # *[*, seglen]
            flt_items = self.helper.put_results(
                insts, best_labs, best_scores, seg_masks,
                seg_ext_widxes0, seg_ext_widxes, seg_ext_masks0, seg_ext_masks,
                cand_full_scores, cand_decisions, split_decisions)
        # --
        # step 4: final extend (in a flattened way)
        if len(flt_items) > 0 and conf.pred_ext:
            flt_mask = ((best_labs>0) & (seg_masks>0.))  # [*, seglen]
            flt_sidx = BK.arange_idx(bsize).unsqueeze(-1).expand_as(flt_mask)[flt_mask]  # [?]
            flt_expr = seg_weighted_expr[flt_mask]  # [?, D]
            flt_full_expr = self._prepare_full_expr(seg_ext_widxes[flt_mask], seg_ext_masks[flt_mask], slen)  # [?, slen, D]
            self.ext_node.predict(flt_items, input_expr[flt_sidx], flt_expr, flt_full_expr, mask_expr[flt_sidx])
        # --
        # extra:
        self.pp_node.prune(insts)
        return None

    # [*, slen, D], [*, slen], [*, D']
    def lookup(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
               pair_expr: BK.Expr = None):
        raise NotImplementedError("To be implemented!!")

    # plus flatten items's dims
    def lookup_flatten(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
                       pair_expr: BK.Expr = None):
        raise NotImplementedError("To be implemented!!")

    # =====
    # helpers

    # step 1: cand select
    # [*, slen, D], [*, slen]
    def _cand_score_and_select(self, input_expr: BK.Expr, mask_expr: BK.Expr):
        conf: SoftExtractorConf = self.conf
        # --
        cand_full_scores = self.cand_scorer(input_expr).squeeze(-1) + (1.-mask_expr) * Constants.REAL_PRAC_MIN  # [*, slen]
        # decide topk count
        len_t = mask_expr.sum(-1)  # [*]
        topk_t = (len_t * conf.cand_topk_rate).clamp(max=conf.cand_topk_count).ceil().long().unsqueeze(-1)  # [*, 1]
        # get topk mask
        if BK.is_zero_shape(mask_expr):
            topk_mask = mask_expr.clone()  # no need to go topk since no elements
        else:
            topk_mask = select_topk(cand_full_scores, topk_t, mask_t=mask_expr, dim=-1)  # [*, slen]
        # thresh
        cand_decisions = topk_mask * (cand_full_scores >= conf.cand_thresh).float()  # [*, slen]
        return cand_full_scores, cand_decisions  # [*, slen]

    # step 2: split
    # [*, clen, D], [*, clen]
    def _split_score(self, cand_expr: BK.Expr, cand_mask: BK.Expr):
        split_scores = self.split_scorer.plain_score(cand_expr[:,:-1], cand_expr[:,1:]).squeeze(-1) + \
                       (1.-cand_mask[:,1:]) * Constants.REAL_PRAC_MIN  # [*, clen-1]
        split_decisions = (split_scores >= 0.).float()  # [*, clen-1]
        return split_scores, split_decisions  # [*, clen-1]

    # [*, clen-1], [*, clen]
    def _split_extend(self, split_decisions: BK.Expr, cand_mask: BK.Expr):
        # first augment/pad split_decisions
        slice_ones = BK.constants([BK.get_shape(split_decisions, 0), 1], 1.)  # [*, 1]
        padded_split_decisions = BK.concat([slice_ones, split_decisions], -1)  # [*, clen]
        seg_cidxes, seg_masks = BK.mask2idx(padded_split_decisions)  # [*, seglen]
        # --
        cand_lens = cand_mask.sum(-1, keepdim=True).long()  # [*, 1]
        seg_masks *= (cand_lens>0).float()  # for the case of no cands
        # --
        seg_cidxes_special = seg_cidxes + (1.-seg_masks).long() * cand_lens  # [*, seglen], fill in for paddings
        seg_cidxes_special2 = BK.concat([seg_cidxes_special, cand_lens], -1)  # [*, seglen+1]
        seg_clens = seg_cidxes_special2[:, 1:] - seg_cidxes_special  # [*, seglen]
        # extend the idxes
        seg_ext_cidxes, seg_ext_masks = expand_ranged_idxes(seg_cidxes, seg_clens)  # [*, seglen, MW]
        seg_ext_masks *= seg_masks.unsqueeze(-1)
        return seg_ext_cidxes, seg_ext_masks, seg_masks  # 2x[*, seglen, MW], [*, seglen]

    # *[*, clen], *[*, seglen, MW], int
    def _split_aggregate(self, cand_expr, cand_scores, cand_widxes, seg_ext_cidxes, seg_ext_masks, topk: int):
        arange3_t = BK.arange_idx(seg_ext_cidxes.shape[0]).unsqueeze(-1).unsqueeze(-1)  # [*, 1, 1]
        seg_ext_scores = cand_scores[arange3_t, seg_ext_cidxes] + (1.-seg_ext_masks) * Constants.REAL_PRAC_MIN  # [*, seglen, MW]
        # if need further topk?
        if topk>0 and BK.get_shape(seg_ext_scores, -1)>topk:  # need to further topk?
            seg_ext_scores, _tmp_idxes = seg_ext_scores.topk(topk, dim=-1, sorted=False)  # [*, seglen, K]
            seg_ext_cidxes = seg_ext_cidxes.gather(-1, _tmp_idxes)  # [*, seglen, K]
            seg_ext_masks = seg_ext_masks.gather(-1, _tmp_idxes)  # [*, seglen, K]
        # get expr and extend to full
        seg_ext_prob = seg_ext_scores.softmax(-1)  # [*, seglen, K]
        _tmp_expr = cand_expr[arange3_t, seg_ext_cidxes]  # [*, seglen, K, D]
        seg_weighted_expr = (_tmp_expr * seg_ext_prob.unsqueeze(-1)).sum(-2)  # [*, seglen, D]
        seg_ext_widxes = cand_widxes[arange3_t, seg_ext_cidxes]  # [*, seglen, K]
        return seg_ext_scores, seg_ext_cidxes, seg_ext_widxes, seg_ext_masks, seg_weighted_expr  # [*, seglen, ?]

    # [?, MW], [?, MW], int
    def _prepare_full_expr(self, flt_ext_widxes, flt_ext_masks, slen: int):
        tmp_bsize = BK.get_shape(flt_ext_widxes, 0)
        tmp_idxes = BK.zeros([tmp_bsize, slen+1])  # [?, slen+1]
        # note: (once a bug) should get rid of paddings!!
        _mask_lt = flt_ext_masks.long()  # [?, N]
        tmp_idxes.scatter_(-1, flt_ext_widxes * _mask_lt + slen * (1-_mask_lt), 1)  # [?, slen]
        tmp_idxes = tmp_idxes[:,:-1].long()  # [?, slen]
        tmp_embs = self.indicator_embed(tmp_idxes)  # [?, slen, D]
        return tmp_embs

    # =====
    # for loss and feed

    def _loss_feed_cand(self, mask_expr, cand_full_scores, pred_cand_decisions,
                       expr_seq_gaddr, expr_group_widxes, expr_group_masks, expr_loss_weight_non):
        conf: SoftExtractorConf = self.conf
        bsize, slen = BK.get_shape(mask_expr)
        arange3_t = BK.arange_idx(bsize).unsqueeze(-1).unsqueeze(-1)  # [*, 1, 1]
        # --
        # step 1.1: bag loss
        cand_gold_mask = (expr_seq_gaddr>=0).float() * mask_expr  # [*, slen], whether is-arg
        raw_loss_cand = BK.loss_binary(cand_full_scores, cand_gold_mask, label_smoothing=conf.cand_label_smoothing)  # [*, slen]
        # how to weight?
        extended_scores_t = cand_full_scores[arange3_t, expr_group_widxes] + (1.-expr_group_masks)*Constants.REAL_PRAC_MIN  # [*, slen, MW]
        if BK.is_zero_shape(extended_scores_t):
            extended_scores_max_t = BK.zeros(mask_expr.shape)  # [*, slen]
        else:
            extended_scores_max_t, _ = extended_scores_t.max(-1)  # [*, slen]
        _w_alpha = conf.cand_loss_weight_alpha
        _weight = ((cand_full_scores - extended_scores_max_t) * _w_alpha).exp()  # [*, slen]
        if not conf.cand_loss_div_max:  # div sum-all, like doing softmax
            _weight = _weight / ((extended_scores_t-extended_scores_max_t.unsqueeze(-1)) * _w_alpha).exp().sum(-1)
        _weight = _weight * (_weight >= conf.cand_loss_weight_thresh).float()  # [*, slen]
        if conf.cand_detach_weight:
            _weight = _weight.detach()
        # pos poison (dis-encouragement)
        if conf.cand_loss_pos_poison:
            poison_loss = BK.loss_binary(cand_full_scores, 1.-cand_gold_mask, label_smoothing=conf.cand_label_smoothing)  # [*, slen]
            raw_loss_cand = raw_loss_cand * _weight + poison_loss * cand_gold_mask * (1.-_weight)  # [*, slen]
        else:
            raw_loss_cand = raw_loss_cand * _weight
        # final weight it
        cand_loss_weights = BK.where(cand_gold_mask==0., expr_loss_weight_non.unsqueeze(-1)*conf.loss_weight_non, mask_expr)  # [*, slen]
        final_cand_loss_weights = cand_loss_weights * mask_expr  # [*, slen]
        loss_cand_item = LossHelper.compile_leaf_loss(
            f"cand", (raw_loss_cand * final_cand_loss_weights).sum(), final_cand_loss_weights.sum(), loss_lambda=conf.loss_cand)
        # step 1.2: feed cand
        # todo(+N): currently only pred/sample, whether adding certain teacher-forcing?
        sample_decisions = (BK.sigmoid(cand_full_scores) >= BK.rand(cand_full_scores.shape)).float() * mask_expr  # [*, slen]
        _use_sample_mask = (BK.rand([bsize]) <= conf.cand_feed_sample_rate).float().unsqueeze(-1)  # [*, 1], seq-level
        feed_cand_decisions = (_use_sample_mask * sample_decisions + (1.-_use_sample_mask) * pred_cand_decisions)  # [*, slen]
        # next
        cand_widxes, cand_masks = BK.mask2idx(feed_cand_decisions)  # [*, clen]
        # --
        # extra: loss_cand_entropy
        rets = [loss_cand_item]
        _loss_cand_entropy = conf.loss_cand_entropy
        if _loss_cand_entropy > 0.:
            _prob = extended_scores_t.softmax(-1)  # [*, slen, MW]
            _ent = EntropyHelper.self_entropy(_prob)  # [*, slen]
            # [*, slen], only first one in bag
            _ent_mask = BK.concat([expr_seq_gaddr[:, :1]>=0, expr_seq_gaddr[:,1:]!=expr_seq_gaddr[:,:-1]], -1).float() * cand_gold_mask
            _loss_ent_item = LossHelper.compile_leaf_loss(
                f"cand_ent", (_ent * _ent_mask).sum(), _ent_mask.sum(), loss_lambda=_loss_cand_entropy)
            rets.append(_loss_ent_item)
        # --
        return rets, cand_widxes, cand_masks

    def _loss_feed_split(self, mask_expr, split_scores, pred_split_decisions,
                         cand_widxes, cand_masks, cand_expr, cand_scores, expr_seq_gaddr):
        conf: SoftExtractorConf = self.conf
        bsize, slen = BK.get_shape(mask_expr)
        arange2_t = BK.arange_idx(bsize).unsqueeze(-1)  # [*, 1, 1]
        # --
        # step 2.1: split loss (only on good points (excluding -1|-1 or paddings) with dynamic oracle)
        cand_gaddr = expr_seq_gaddr[arange2_t, cand_widxes]  # [*, clen]
        cand_gaddr0, cand_gaddr1 = cand_gaddr[:, :-1], cand_gaddr[:, 1:]  # [*, clen-1]
        split_oracle = (cand_gaddr0 != cand_gaddr1).float() * cand_masks[:, 1:]  # [*, clen-1]
        split_oracle_mask = ((cand_gaddr0 >= 0) | (cand_gaddr1 >= 0)).float() * cand_masks[:, 1:]  # [*, clen-1]
        raw_split_loss = BK.loss_binary(split_scores, split_oracle,
                                        label_smoothing=conf.split_label_smoothing)  # [*, slen]
        loss_split_item = LossHelper.compile_leaf_loss(
            f"split", (raw_split_loss * split_oracle_mask).sum(), split_oracle_mask.sum(), loss_lambda=conf.loss_split)
        # step 2.2: feed split
        # note: when teacher-forcing, only forcing good points, others still use pred
        force_split_decisions = split_oracle_mask * split_oracle + (1. - split_oracle_mask) * pred_split_decisions  # [*, clen-1]
        _use_force_mask = (BK.rand([bsize]) <= conf.split_feed_force_rate).float().unsqueeze(-1)  # [*, 1], seq-level
        feed_split_decisions = (_use_force_mask * force_split_decisions + (1. - _use_force_mask) * pred_split_decisions)  # [*, clen-1]
        # next
        # *[*, seglen, MW], [*, seglen]
        seg_ext_cidxes, seg_ext_masks, seg_masks = self._split_extend(feed_split_decisions, cand_masks)
        seg_ext_scores, seg_ext_cidxes, seg_ext_widxes, seg_ext_masks, seg_weighted_expr = self._split_aggregate(
            cand_expr, cand_scores, cand_widxes, seg_ext_cidxes, seg_ext_masks, conf.split_topk)  # [*, seglen, ?]
        # finally get oracles for next steps
        # todo(+N): simply select the highest scored one as oracle
        if BK.is_zero_shape(seg_ext_scores):  # simply make them all -1
            oracle_gaddr = BK.constants_idx(seg_masks.shape, -1)  # [*, seglen]
        else:
            _, _seg_max_t = seg_ext_scores.max(-1, keepdim=True)  # [*, seglen, 1]
            oracle_widxes = seg_ext_widxes.gather(-1, _seg_max_t).squeeze(-1)  # [*, seglen]
            oracle_gaddr = expr_seq_gaddr.gather(-1, oracle_widxes)  # [*, seglen]
        oracle_gaddr[seg_masks<=0] = -1  # (assign invalid ones) [*, seglen]
        return loss_split_item, seg_masks, seg_ext_widxes, seg_ext_masks, seg_weighted_expr, oracle_gaddr

# --
class SoftExtractorHelper(BaseExtractorHelper):
    def __init__(self, conf: SoftExtractorConf, vocab: SimpleVocab):
        super().__init__(conf, vocab)
        conf: SoftExtractorConf = self.conf
        # --
        if conf.ftag == "arg":
            self._prep_f = self._prep_args
        else:
            self._prep_f = self._prep_frames

    # =====
    # put outputs

    # [*], 3x[*, seglen], 2x[*, seglen, ?], [*, slen], [*, clen]
    def put_results(self, insts, best_labs, best_scores, seg_masks,
                    seg_ext_widxes0, seg_ext_widxes, seg_ext_masks0, seg_ext_masks,
                    cand_full_scores, cand_decisions, split_decisions):
        conf: SoftExtractorConf = self.conf
        # --
        all_arrs = [BK.get_value(z) for z in [
            best_labs, best_scores, seg_masks,
            seg_ext_widxes0, seg_ext_widxes, seg_ext_masks0, seg_ext_masks,
            cand_full_scores, cand_decisions, split_decisions]]
        flattened_items = []
        for bidx, inst in enumerate(insts):
            self._clear_f(inst)  # first clean things
            cur_len = len(inst) if isinstance(inst, Sent) else len(inst.sent)
            # first set general result
            res_cand_score = all_arrs[-3][bidx, :cur_len].tolist()
            res_cand = all_arrs[-2][bidx, :cur_len].tolist()
            res_split = [1.] + all_arrs[-1][bidx, :int(sum(res_cand))-1].tolist()  # note: actually B-?
            inst.info.update({"res_cand": res_cand, "res_cand_score": res_cand_score, "res_split": res_split})
            # then set them separately
            for one_lab, one_score, one_mask, one_widxes0, one_widxes, one_wmasks0, one_wmasks in \
                    zip(*[z[bidx] for z in all_arrs[:-3]]):
                one_lab = int(one_lab)
                # todo(+N): again, assuming NON-idx == 0
                if one_mask == 0. or one_lab == 0: continue  # invalid one: unmask or NON
                # get widxes
                cur_widxes0 = sorted(one_widxes0[one_wmasks0>0.].tolist())  # original selections from cand
                cur_widxes = sorted(one_widxes[one_wmasks>0.].tolist())  # further possible topk
                tmp_widx = cur_widxes[0]  # currently simply set a tmp one!
                tmp_wlen = cur_widxes[-1] + 1 - tmp_widx
                # set it
                new_item = self._new_f(inst, tmp_widx, tmp_wlen, one_lab, float(one_score))
                new_item.mention.info["widxes0"] = cur_widxes0  # idxes after cand
                new_item.mention.info["widxes1"] = cur_widxes  # idxes after split
                flattened_items.append(new_item)
            # --
        return flattened_items

    # =====
    # prepare for training

    def _prep_items(self, items: List, par: object, seq_len: int):
        # sort items by (wlen, widx): larger span first!
        aug_items = []
        for f in items:
            widx, wlen = self.core_span_getter(f.mention)
            aug_items.append(((-wlen, widx), f))  # key, item
        aug_items.sort(key=lambda x: x[0])
        # get them
        ret_items = [z[1] for z in aug_items]  # List[item]
        seq_iidxes = [-1] * seq_len  # List[idx-item]
        group_widxes = []  # List[List[idx-word]]
        for ii, pp in enumerate(aug_items):
            neg_wlen, widx = pp[0]
            wlen = -neg_wlen
            seq_iidxes[widx:widx+wlen] = [ii] * wlen  # assign iidx
        # --
        cur_ii, last_jj = [], -1
        for ii, jj in enumerate(seq_iidxes):  # note: need another loop since there can be overlap
            if jj != last_jj or jj < 0:  # break
                group_widxes.extend([cur_ii] * len(cur_ii))
                cur_ii = []
            cur_ii.append(ii)
            last_jj = jj
        group_widxes.extend([cur_ii] * len(cur_ii))
        assert len(group_widxes) == seq_len
        # --
        _loss_weight_non = getattr(par, "_loss_weight_non", 1.)  # todo(+N): special name; loss_weight_non
        return ZObject(items=ret_items, par=par, len=seq_len, loss_weight_non=_loss_weight_non,
                       seq_iidxes=seq_iidxes, group_widxes=group_widxes)

    def _prep_frames(self, s: Sent):
        return self._prep_items(self._get_frames(s), s, len(s))

    def _prep_args(self, f: Frame):
        return self._prep_items(self._get_args(f), f, len(f.sent))

    # prepare inputs
    def prepare(self, insts: Union[List[Sent], List[Frame]], mlen: int, use_cache: bool):
        conf: SoftExtractorConf = self.conf
        # get info
        if use_cache:
            zobjs = []
            attr_name = f"_socache_{conf.ftag}"  # should be unique
            for s in insts:
                one = getattr(s, attr_name, None)
                if one is None:
                    one = self._prep_f(s)
                    setattr(s, attr_name, one)  # set cache
                zobjs.append(one)
        else:
            zobjs = [self._prep_f(s) for s in insts]
        # batch things
        bsize, mlen2 = len(insts), max(len(z.items) for z in zobjs) if len(zobjs)>0 else 1
        mnum = max(len(g) for z in zobjs for g in z.group_widxes) if len(zobjs)>0 else 1
        arr_items = np.full((bsize, mlen2), None, dtype=object)  # [*, ?]
        arr_seq_iidxes = np.full((bsize, mlen), -1, dtype=np.int)
        arr_group_widxes = np.full((bsize, mlen, mnum), 0, dtype=np.int)
        arr_group_masks = np.full((bsize, mlen, mnum), 0., dtype=np.float)
        for zidx, zobj in enumerate(zobjs):
            arr_items[zidx, :len(zobj.items)] = zobj.items
            iidx_offset = zidx * mlen2  # note: offset for valid ones!
            arr_seq_iidxes[zidx, :len(zobj.seq_iidxes)] = [(iidx_offset+ii) if ii>=0 else ii for ii in zobj.seq_iidxes]
            for zidx2, zwidxes in enumerate(zobj.group_widxes):
                arr_group_widxes[zidx, zidx2, :len(zwidxes)] = zwidxes
                arr_group_masks[zidx, zidx2, :len(zwidxes)] = 1.
        # final setup things
        expr_seq_iidxes = BK.input_idx(arr_seq_iidxes)  # [*, slen]
        expr_group_widxes = BK.input_idx(arr_group_widxes)  # [*, slen, MW]
        expr_group_masks = BK.input_real(arr_group_masks)  # [*, slen, MW]
        expr_loss_weight_non = BK.input_real([z.loss_weight_non for z in zobjs])  # [*]
        return arr_items, expr_seq_iidxes, expr_group_widxes, expr_group_masks, expr_loss_weight_non

# --
# b msp2/tasks/zsfp/extract/extractors/soft.py:73
