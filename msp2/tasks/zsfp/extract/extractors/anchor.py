#

# anchor extractor (simpler version than soft)

__all__ = [
    "AnchorExtractorConf", "AnchorExtractor", "AnchorExtractorHelper",
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
class AnchorExtractorConf(BaseExtractorConf):
    def __init__(self):
        super().__init__()
        # --
        self.loss_cand_entropy = 0.  # extra weights for entropy loss
        self.cand_loss_weight_alpha = 1.
        self.ext_loss_topk = 3  # select how many to extend?
        self.ext_loss_thresh = 1e-3  # to save some calculations
        self.flatten_lookup_init_scale = 5.
        # --
        self.detach_weight_lab = True  # detach the weight in training?
        self.detach_weight_ext = True

@node_reg(AnchorExtractorConf)
class AnchorExtractor(BaseExtractor):
    def __init__(self, conf: AnchorExtractorConf, vocab: SimpleVocab, **kwargs):
        super().__init__(conf, vocab, **kwargs)
        conf: AnchorExtractorConf = self.conf
        # --
        self.helper = AnchorExtractorHelper(conf, vocab)
        # simply 0/1 indicator embeddings
        self.indicator_embed = EmbeddingNode(None, osize=conf.isize, n_words=2, init_scale=conf.flatten_lookup_init_scale)

    def _build_extract_node(self, conf: AnchorExtractorConf):
        return None  # no extra extracting node!!

    # [*, slen]
    def _prepare_full_expr(self, flt_mask: BK.Expr):
        bsize, slen = BK.get_shape(flt_mask)
        arange2_t = BK.arange_idx(slen).unsqueeze(0)  # [1, slen]
        all_widxes = arange2_t.expand_as(flt_mask)[flt_mask]  # [?]
        tmp_idxes = BK.zeros([len(all_widxes), slen]).long()  # [?, slen]
        tmp_idxes.scatter_(-1, all_widxes.unsqueeze(-1), 1)  # [?, slen]
        tmp_embs = self.indicator_embed(tmp_idxes)  # [?, slen, D]
        return tmp_embs

    # [*, slen, D], [*, slen], [*, D']
    def loss(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
             pair_expr: BK.Expr = None, lookup_flatten=False, external_extra_score: BK.Expr = None):
        conf: AnchorExtractorConf = self.conf
        assert not lookup_flatten
        bsize, slen = BK.get_shape(mask_expr)
        # --
        # step 0: prepare
        arr_items, expr_seq_gaddr, expr_seq_labs, expr_group_widxes, expr_group_masks, expr_loss_weight_non = \
            self.helper.prepare(insts, mlen=BK.get_shape(mask_expr, -1), use_cache=True)
        arange2_t = BK.arange_idx(bsize).unsqueeze(-1)  # [*, 1]
        arange3_t = arange2_t.unsqueeze(-1)  # [*, 1, 1]
        # --
        # step 1: label, simply scoring everything!
        _main_t, _pair_t = self.lab_node.transform_expr(input_expr, pair_expr)
        all_scores_t = self.lab_node.score_all(_main_t, _pair_t, mask_expr, None, local_normalize=False,
                                               extra_score=external_extra_score)  # unnormalized scores [*, slen, L]
        all_probs_t = all_scores_t.softmax(-1)  # [*, slen, L]
        all_gprob_t = all_probs_t.gather(-1, expr_seq_labs.unsqueeze(-1)).squeeze(-1)  # [*, slen]
        # how to weight
        extended_gprob_t = all_gprob_t[arange3_t, expr_group_widxes] * expr_group_masks  # [*, slen, MW]
        if BK.is_zero_shape(extended_gprob_t):
            extended_gprob_max_t = BK.zeros(mask_expr.shape)  # [*, slen]
        else:
            extended_gprob_max_t, _ = extended_gprob_t.max(-1)  # [*, slen]
        _w_alpha = conf.cand_loss_weight_alpha
        _weight = ((all_gprob_t * mask_expr) / (extended_gprob_max_t.clamp(min=1e-5))) ** _w_alpha  # [*, slen]
        _label_smoothing = conf.lab_conf.labeler_conf.label_smoothing
        _loss1 = BK.loss_nll(all_scores_t, expr_seq_labs, label_smoothing=_label_smoothing)  # [*, slen]
        _loss2 = BK.loss_nll(all_scores_t, BK.constants_idx([bsize, slen], 0), label_smoothing=_label_smoothing)  # [*, slen]
        _weight1 = _weight.detach() if conf.detach_weight_lab else _weight
        _raw_loss = _weight1 * _loss1 + (1.-_weight1) * _loss2  # [*, slen]
        # final weight it
        cand_loss_weights = BK.where(expr_seq_labs==0, expr_loss_weight_non.unsqueeze(-1)*conf.loss_weight_non, mask_expr)  # [*, slen]
        final_cand_loss_weights = cand_loss_weights * mask_expr  # [*, slen]
        loss_lab_item = LossHelper.compile_leaf_loss(
            f"lab", (_raw_loss*final_cand_loss_weights).sum(), final_cand_loss_weights.sum(),
            loss_lambda=conf.loss_lab, gold=(expr_seq_labs>0).float().sum())
        # --
        # step 1.5
        all_losses = [loss_lab_item]
        _loss_cand_entropy = conf.loss_cand_entropy
        if _loss_cand_entropy > 0.:
            _prob = extended_gprob_t  # [*, slen, MW]
            _ent = EntropyHelper.self_entropy(_prob)  # [*, slen]
            # [*, slen], only first one in bag
            _ent_mask = BK.concat([expr_seq_gaddr[:,:1]>=0, expr_seq_gaddr[:,1:]!=expr_seq_gaddr[:,:-1]],-1).float() \
                        * (expr_seq_labs>0).float()
            _loss_ent_item = LossHelper.compile_leaf_loss(
                f"cand_ent", (_ent * _ent_mask).sum(), _ent_mask.sum(), loss_lambda=_loss_cand_entropy)
            all_losses.append(_loss_ent_item)
        # --
        # step 4: extend (select topk)
        if conf.loss_ext > 0.:
            if BK.is_zero_shape(extended_gprob_t):
                flt_mask = (BK.zeros(mask_expr.shape)>0)
            else:
                _topk = min(conf.ext_loss_topk, BK.get_shape(extended_gprob_t, -1))  # number to extract
                _topk_grpob_t, _ = extended_gprob_t.topk(_topk, dim=-1)  # [*, slen, K]
                flt_mask = (expr_seq_labs>0) & (all_gprob_t >= _topk_grpob_t.min(-1)[0]) & (_weight > conf.ext_loss_thresh)  # [*, slen]
            flt_sidx = BK.arange_idx(bsize).unsqueeze(-1).expand_as(flt_mask)[flt_mask]  # [?]
            flt_expr = input_expr[flt_mask]  # [?, D]
            flt_full_expr = self._prepare_full_expr(flt_mask)  # [?, slen, D]
            flt_items = arr_items.flatten()[BK.get_value(expr_seq_gaddr[flt_mask])]  # [?]
            flt_weights = _weight.detach()[flt_mask] if conf.detach_weight_ext else _weight[flt_mask]  # [?]
            loss_ext_item = self.ext_node.loss(
                flt_items, input_expr[flt_sidx], flt_expr, flt_full_expr, mask_expr[flt_sidx], flt_extra_weights=flt_weights)
            all_losses.append(loss_ext_item)
        # --
        # return loss
        ret_loss = LossHelper.combine_multiple_losses(all_losses)
        return ret_loss, None

    # [*, slen, D], [*, slen], [*, D']
    def predict(self, insts: Union[List[Sent], List[Frame]], input_expr: BK.Expr, mask_expr: BK.Expr,
                pair_expr: BK.Expr = None, lookup_flatten=False, external_extra_score: BK.Expr=None):
        conf: AnchorExtractorConf = self.conf
        assert not lookup_flatten
        bsize, slen = BK.get_shape(mask_expr)
        # --
        for inst in insts:  # first clear things
            self.helper._clear_f(inst)
        # --
        # step 1: simply labeling!
        best_labs, best_scores = self.lab_node.predict(input_expr, pair_expr, mask_expr, extra_score=external_extra_score)
        flt_items = self.helper.put_results(insts, best_labs, best_scores)  # [?]
        # --
        # step 2: final extend (in a flattened way)
        if len(flt_items) > 0 and conf.pred_ext:
            flt_mask = ((best_labs>0) & (mask_expr>0))  # [*, slen]
            flt_sidx = BK.arange_idx(bsize).unsqueeze(-1).expand_as(flt_mask)[flt_mask]  # [?]
            flt_expr = input_expr[flt_mask]  # [?, D]
            flt_full_expr = self._prepare_full_expr(flt_mask)  # [?, slen, D]
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

# --
class AnchorExtractorHelper(BaseExtractorHelper):
    def __init__(self, conf: AnchorExtractorConf, vocab: SimpleVocab):
        super().__init__(conf, vocab)
        conf: AnchorExtractorConf = self.conf
        # --
        if conf.ftag == "arg":
            self._prep_f = self._prep_args
        else:
            self._prep_f = self._prep_frames

    # =====
    # put outputs

    # *[*, slen]
    def put_results(self, insts, best_labs, best_scores):
        conf: AnchorExtractorConf = self.conf
        # --
        arr_labs, arr_scores = [BK.get_value(z) for z in [best_labs, best_scores]]
        flattened_items = []
        for bidx, inst in enumerate(insts):
            self._clear_f(inst)  # first clean things
            cur_len = len(inst) if isinstance(inst, Sent) else len(inst.sent)
            cur_labs, cur_scores = arr_labs[bidx][:cur_len], arr_scores[bidx][:cur_len]
            # simply put them
            for one_widx in range(cur_len):
                one_lab, one_score = int(cur_labs[one_widx]), float(cur_scores[one_widx])
                # todo(+N): again, assuming NON-idx == 0
                if one_lab == 0: continue  # invalid one: unmask or NON
                # set it
                new_item = self._new_f(inst, one_widx, 1, one_lab, float(one_score))
                new_item.mention.info["widxes1"] = [one_widx]  # save it
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
            aug_items.append(((-wlen, widx, f.label_idx), f))  # key, item
        aug_items.sort(key=lambda x: x[0])
        # get them
        ret_items = [z[1] for z in aug_items]  # List[item]
        seq_iidxes = [-1] * seq_len  # List[idx-item]
        seq_labs = [0] * seq_len  # List[lab-idx]
        group_widxes = []  # List[List[idx-word]]
        for ii, pp in enumerate(aug_items):
            neg_wlen, widx, lab_idx = pp[0]
            wlen = -neg_wlen
            seq_iidxes[widx:widx+wlen] = [ii] * wlen  # assign iidx
            seq_labs[widx:widx+wlen] = [lab_idx] * wlen  # assign lab
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
                       seq_iidxes=seq_iidxes, seq_labs=seq_labs, group_widxes=group_widxes)

    def _prep_frames(self, s: Sent):
        return self._prep_items(self._get_frames(s), s, len(s))

    def _prep_args(self, f: Frame):
        return self._prep_items(self._get_args(f), f, len(f.sent))

    # prepare inputs
    def prepare(self, insts: Union[List[Sent], List[Frame]], mlen: int, use_cache: bool):
        conf: AnchorExtractorConf = self.conf
        # get info
        if use_cache:
            zobjs = []
            attr_name = f"_acache_{conf.ftag}"  # should be unique
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
        arr_seq_labs = np.full((bsize, mlen), 0, dtype=np.int)
        arr_group_widxes = np.full((bsize, mlen, mnum), 0, dtype=np.int)
        arr_group_masks = np.full((bsize, mlen, mnum), 0., dtype=np.float)
        for zidx, zobj in enumerate(zobjs):
            arr_items[zidx, :len(zobj.items)] = zobj.items
            iidx_offset = zidx * mlen2  # note: offset for valid ones!
            arr_seq_iidxes[zidx, :len(zobj.seq_iidxes)] = [(iidx_offset+ii) if ii>=0 else ii for ii in zobj.seq_iidxes]
            arr_seq_labs[zidx, :len(zobj.seq_labs)] = zobj.seq_labs
            for zidx2, zwidxes in enumerate(zobj.group_widxes):
                arr_group_widxes[zidx, zidx2, :len(zwidxes)] = zwidxes
                arr_group_masks[zidx, zidx2, :len(zwidxes)] = 1.
        # final setup things
        expr_seq_iidxes = BK.input_idx(arr_seq_iidxes)  # [*, slen]
        expr_seq_labs = BK.input_idx(arr_seq_labs)  # [*, slen]
        expr_group_widxes = BK.input_idx(arr_group_widxes)  # [*, slen, MW]
        expr_group_masks = BK.input_real(arr_group_masks)  # [*, slen, MW]
        expr_loss_weight_non = BK.input_real([z.loss_weight_non for z in zobjs])  # [*]
        return arr_items, expr_seq_iidxes, expr_seq_labs, expr_group_widxes, expr_group_masks, expr_loss_weight_non

# --
# b msp2/tasks/zsfp/extract/extractors/anchor.py:73
