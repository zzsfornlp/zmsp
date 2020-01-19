#

# the graph(v2, order>=2) parser

from typing import List
import numpy as np

from msp.utils import Constants, zlog, Helper
from msp.data import VocabPackage
from msp.nn import BK
from msp.zext.seq_helper import DataPadder

from ...common.data import ParseInstance
from ...common.model import BaseParserConf, BaseInferenceConf, BaseTrainingConf, BaseParser
from ..scorer import Scorer, ScorerConf, SL0Layer, SL0Conf
from ...algo import hop_decode

from .g1p import G1Parser, PreG1Conf, PruneG1Conf

# =====
# conf

# todo(note): notes about g2p/efp and g1p
# 1) first, we may need g1p as pruner/scorer (g2p requires pruner), which may come from g1model or aux-score-files
# -- for usage for training, should use aux files which comes from leave-one-out styled scoring
# RELATED-OPTIONS: PruneG1Conf, PreG1Conf.{g1_use_aux_scores, lambda_g1_training, lambda_g1_testing}
# 2) then, there is also an option for init the current models with pre-trained go1 model
# RELATED-OPTIONS: PreG1Conf.g1_pretrain_*

# decoding conf
class G2InferenceConf(BaseInferenceConf):
    def __init__(self):
        super().__init__()
        # prune
        self.pruning_conf = PruneG1Conf()
        # mini-dec budget
        self.mb_dec_lb = 128  # batch decoding length budget
        self.mb_dec_sb = Constants.INT_PRAC_MAX  # scoring once budget (number of scoring parts)

# training conf
class G2TraningConf(BaseTrainingConf):
    def __init__(self):
        super().__init__()
        # loss functions
        self.loss_div_tok = True        # loss divide by token or by sent?
        # filtering out pruned parts
        self.filter_pruned = True
        self.filter_margin = True

# overall parser conf
class G2ParserConf(BaseParserConf):
    def __init__(self):
        super().__init__(G2InferenceConf(), G2TraningConf())
        self.sc_conf = ScorerConf()
        self.sl_conf = SL0Conf()
        self.pre_g1_conf = PreG1Conf()
        self.system_labeled = True
        # which graph-based model?
        self.gm_type = "o3gsib"  # o1/o2sib/o2g/o3gsib
        self.gm_projective = False

    def do_validate(self):
        # specific scorer
        self.sl_conf.use_label_feat = False  # no label features, otherwise will explode
        # currently only support 1 sib
        self.sl_conf.chs_num = 1
        self.sl_conf.chs_f = "sum"

# =====
# model

# the model
# todo(note): for simplicity, aggregate the features (sib, grandparent) at the head-node's srepr
class G2Parser(BaseParser):
    def __init__(self, conf: G2ParserConf, vpack: VocabPackage):
        super().__init__(conf, vpack)
        # todo(note): the neural parameters are exactly the same as the EF one
        # ===== basic G1 Parser's loading
        # todo(note): there can be parameter mismatch (but all of them in non-trained part, thus will be fine)
        self.g1parser = G1Parser.pre_g1_init(self, conf.pre_g1_conf)
        self.lambda_g1_arc_training = conf.pre_g1_conf.lambda_g1_arc_training
        self.lambda_g1_arc_testing = conf.pre_g1_conf.lambda_g1_arc_testing
        self.lambda_g1_lab_training = conf.pre_g1_conf.lambda_g1_lab_training
        self.lambda_g1_lab_testing = conf.pre_g1_conf.lambda_g1_lab_testing
        #
        self.add_slayer()
        self.dl = G2DL(self.scorer, self.slayer, conf)
        #
        self.predict_padder = DataPadder(2, pad_vals=0)
        self.num_label = self.label_vocab.trg_len(True)  # todo(WARN): use the original idx

    def build_decoder(self):
        conf = self.conf
        # ===== Decoding Scorer =====
        conf.sc_conf._input_dim = self.enc_output_dim
        conf.sc_conf._num_label = self.label_vocab.trg_len(True)  # todo(WARN): use the original idx
        return Scorer(self.pc, conf.sc_conf)

    def build_slayer(self):
        conf = self.conf
        # ===== Structured Layer =====
        conf.sl_conf._input_dim = self.enc_output_dim
        conf.sl_conf._num_label = self.label_vocab.trg_len(True)  # todo(WARN): use the original idx
        return SL0Layer(self.pc, conf.sl_conf)

    # =====
    # main procedures

    # get g1 prunings and extra-scores
    # -> if no g1parser provided, then use aux scores; otherwise, it depends on g1_use_aux_scores
    def _get_g1_pack(self, insts: List[ParseInstance], score_arc_lambda: float, score_lab_lambda: float):
        pconf = self.conf.iconf.pruning_conf
        if self.g1parser is None or self.g1parser.g1_use_aux_scores:
            valid_mask, arc_score, lab_score, _, _ = G1Parser.score_and_prune(insts, self.num_label, pconf)
        else:
            valid_mask, arc_score, lab_score, _, _ = self.g1parser.prune_on_batch(insts, pconf)
        if score_arc_lambda <= 0. and score_lab_lambda <= 0.:
            go1_pack = None
        else:
            arc_score *= score_arc_lambda
            lab_score *= score_lab_lambda
            go1_pack = (arc_score, lab_score)
        # [*, slen, slen], ([*, slen, slen], [*, slen, slen, Lab])
        return valid_mask, go1_pack

    # make real valid masks (inplaced): valid(byte): [bs, len-m, len-h]; mask(float): [bs, len]
    def _make_final_valid(self, valid_expr, mask_expr):
        maxlen = BK.get_shape(mask_expr, -1)
        # first apply masks
        mask_expr_byte = mask_expr.byte()
        valid_expr &= mask_expr_byte.unsqueeze(-1)
        valid_expr &= mask_expr_byte.unsqueeze(-2)
        # then diag
        mask_diag = 1-BK.eye(maxlen).byte()
        valid_expr &= mask_diag
        # root not as mod
        valid_expr[:, 0] = 0
        # only allow root->root (for grandparent feature)
        valid_expr[:, 0, 0] = 1
        return valid_expr

    # decoding
    def inference_on_batch(self, insts: List[ParseInstance], **kwargs):
        # iconf = self.conf.iconf
        with BK.no_grad_env():
            self.refresh_batch(False)
            # pruning and scores from g1
            valid_mask, go1_pack = self._get_g1_pack(insts, self.lambda_g1_arc_testing, self.lambda_g1_lab_testing)
            # encode
            input_repr, enc_repr, jpos_pack, mask_arr = self.bter.run(insts, False)
            mask_expr = BK.input_real(mask_arr)
            # decode
            final_valid_expr = self._make_final_valid(valid_mask, mask_expr)
            ret_heads, ret_labels, _, _ = self.dl.decode(insts, enc_repr, final_valid_expr, go1_pack, False, 0.)
            # collect the results together
            all_heads = Helper.join_list(ret_heads)
            if ret_labels is None:
                # todo(note): simply get labels from the go1-label classifier; must provide g1parser
                if go1_pack is None:
                    _, go1_pack = self._get_g1_pack(insts, 1., 1.)
                _, go1_label_max_idxes = go1_pack[1].max(-1)  # [bs, slen, slen]
                pred_heads_arr, _ = self.predict_padder.pad(all_heads)  # [bs, slen]
                pred_heads_expr = BK.input_idx(pred_heads_arr)
                pred_labels_expr = BK.gather_one_lastdim(go1_label_max_idxes, pred_heads_expr).squeeze(-1)
                all_labels = BK.get_value(pred_labels_expr)  # [bs, slen]
            else:
                all_labels = np.concatenate(ret_labels, 0)
            # ===== assign, todo(warn): here, the labels are directly original idx, no need to change
            for one_idx, one_inst in enumerate(insts):
                cur_length = len(one_inst)+1
                one_inst.pred_heads.set_vals(all_heads[one_idx][:cur_length])  # directly int-val for heads
                one_inst.pred_labels.build_vals(all_labels[one_idx][:cur_length], self.label_vocab)
                # one_inst.pred_par_scores.set_vals(all_scores[one_idx][:cur_length])
            # =====
            # put jpos result (possibly)
            self.jpos_decode(insts, jpos_pack)
            # -----
            info = {"sent": len(insts), "tok": sum(map(len, insts))}
            return info

    # training
    def fb_on_batch(self, annotated_insts: List[ParseInstance], training=True, loss_factor=1., **kwargs):
        self.refresh_batch(training)
        # pruning and scores from g1
        valid_mask, go1_pack = self._get_g1_pack(annotated_insts, self.lambda_g1_arc_training, self.lambda_g1_lab_training)
        # encode
        input_repr, enc_repr, jpos_pack, mask_arr = self.bter.run(annotated_insts, training)
        mask_expr = BK.input_real(mask_arr)
        # the parsing loss
        final_valid_expr = self._make_final_valid(valid_mask, mask_expr)
        parsing_loss, parsing_scores, info = \
            self.dl.loss(annotated_insts, enc_repr, final_valid_expr, go1_pack, True, self.margin.value)
        info["loss_parse"] = BK.get_value(parsing_loss).item()
        final_loss = parsing_loss
        # other loss?
        jpos_loss = self.jpos_loss(jpos_pack, mask_expr)
        if jpos_loss is not None:
            info["loss_jpos"] = BK.get_value(jpos_loss).item()
            final_loss = parsing_loss + jpos_loss
        if parsing_scores is not None:
            reg_loss = self.reg_scores_loss(*parsing_scores)
            if reg_loss is not None:
                final_loss = final_loss + reg_loss
        info["fb"] = 1
        if training:
            BK.backward(final_loss, loss_factor)
        return info

# =====
# helper for different models

# conventions:
# [m, h, sib, gp]: sib=m means first child; ROOT's parent is ROOT again; and force single root
# warnings:
# use the same scorer as ef, but different semantics

# decoder and losser
class G2DL:
    def __init__(self, scorer: Scorer, slayer: SL0Layer, conf: G2ParserConf):
        self.scorer = scorer
        self.slayer = slayer
        self.gm_type = conf.gm_type
        self.projective = conf.gm_projective
        self.system_labeled = conf.system_labeled
        # system specific
        self.helper = {"o1": G2O1Helper(), "o2sib": G2O2sibHelper(), "o2g": G2O2gHelper(), "o3gsib": G2O3gsibHelper()}[self.gm_type]
        self.margin_div = {"o1": 1, "o2sib": 2, "o2g": 2, "o3gsib": 3}[self.gm_type]
        self.use_sib = "sib" in self.gm_type
        self.use_gp = "g" in self.gm_type
        # others
        tconf: G2TraningConf = conf.tconf
        self.filter_pruned = tconf.filter_pruned
        self.filter_margin = tconf.filter_margin
        self.loss_div_tok = tconf.loss_div_tok
        iconf: G2InferenceConf = conf.iconf
        self.mb_dec_lb = iconf.mb_dec_lb
        self.mb_dec_sb = iconf.mb_dec_sb

    # get parsing loss (perceptron styled)
    # todo(+3): assume the same margin for both arc and label
    def loss(self, insts: List[ParseInstance], enc_expr, final_valid_expr, go1_pack, training: bool, margin: float):
        # first do decoding and related preparation
        with BK.no_grad_env():
            _, _, g_packs, p_packs = self.decode(insts, enc_expr, final_valid_expr, go1_pack, training, margin)
            # flatten the packs (remember to rebase the indexes)
            gold_pack = self._flatten_packs(g_packs)
            pred_pack = self._flatten_packs(p_packs)
            if self.filter_pruned:
                # filter out non-valid (pruned) edges, to avoid prune error
                mod_unpruned_mask, gold_mask = self.helper.get_unpruned_mask(final_valid_expr, gold_pack)
                pred_mask = mod_unpruned_mask[pred_pack[0], pred_pack[1]]  # filter by specific mod
                gold_pack = [(None if z is None else z[gold_mask]) for z in gold_pack]
                pred_pack = [(None if z is None else z[pred_mask]) for z in pred_pack]
        # calculate the scores for loss
        gold_b_idxes, gold_m_idxes, gold_h_idxes, gold_sib_idxes, gold_gp_idxes, gold_lab_idxes = gold_pack
        pred_b_idxes, pred_m_idxes, pred_h_idxes, pred_sib_idxes, pred_gp_idxes, pred_lab_idxes = pred_pack
        gold_arc_score, gold_label_score_all = self._get_basic_score(enc_expr, gold_b_idxes, gold_m_idxes, gold_h_idxes,
                                                                     gold_sib_idxes, gold_gp_idxes)
        pred_arc_score, pred_label_score_all = self._get_basic_score(enc_expr, pred_b_idxes, pred_m_idxes, pred_h_idxes,
                                                                     pred_sib_idxes, pred_gp_idxes)
        # whether have labeled scores
        if self.system_labeled:
            gold_label_score = BK.gather_one_lastdim(gold_label_score_all, gold_lab_idxes).squeeze(-1)
            pred_label_score = BK.gather_one_lastdim(pred_label_score_all, pred_lab_idxes).squeeze(-1)
            ret_scores = (gold_arc_score, pred_arc_score, gold_label_score, pred_label_score)
            pred_full_scores, gold_full_scores = pred_arc_score+pred_label_score, gold_arc_score+gold_label_score
        else:
            ret_scores = (gold_arc_score, pred_arc_score)
            pred_full_scores, gold_full_scores = pred_arc_score, gold_arc_score
        # hinge loss: filter-margin by loss*margin to be aware of search error
        if self.filter_margin:
            with BK.no_grad_env():
                mat_shape = BK.get_shape(enc_expr)[:2]  # [bs, slen]
                heads_gold = self._get_tmp_mat(mat_shape, 0, BK.int64, gold_b_idxes, gold_m_idxes, gold_h_idxes)
                heads_pred = self._get_tmp_mat(mat_shape, 0, BK.int64, pred_b_idxes, pred_m_idxes, pred_h_idxes)
                error_count = (heads_gold != heads_pred).float()
                if self.system_labeled:
                    labels_gold = self._get_tmp_mat(mat_shape, 0, BK.int64, gold_b_idxes, gold_m_idxes, gold_lab_idxes)
                    labels_pred = self._get_tmp_mat(mat_shape, 0, BK.int64, pred_b_idxes, pred_m_idxes, pred_lab_idxes)
                    error_count += (labels_gold != labels_pred).float()
                scores_gold = self._get_tmp_mat(mat_shape, 0., BK.float32, gold_b_idxes, gold_m_idxes, gold_full_scores)
                scores_pred = self._get_tmp_mat(mat_shape, 0., BK.float32, pred_b_idxes, pred_m_idxes, pred_full_scores)
                # todo(note): here, a small 0.1 is to exclude zero error: anyway they will get zero gradient
                sent_mask = ((scores_gold.sum(-1) - scores_pred.sum(-1)) <= (margin * error_count.sum(-1) + 0.1)).float()
                num_valid_sent = float(BK.get_value(sent_mask.sum()))
            final_loss_sum = (pred_full_scores*sent_mask[pred_b_idxes] - gold_full_scores*sent_mask[gold_b_idxes]).sum()
        else:
            num_valid_sent = len(insts)
            final_loss_sum = (pred_full_scores - gold_full_scores).sum()
        # prepare final loss
        # divide loss by what?
        num_sent = len(insts)
        num_valid_tok = sum(len(z) for z in insts)
        if self.loss_div_tok:
            final_loss = final_loss_sum / num_valid_tok
        else:
            final_loss = final_loss_sum / num_sent
        final_loss_sum_val = float(BK.get_value(final_loss_sum))
        info = {"sent": num_sent, "sent_valid": num_valid_sent, "tok": num_valid_tok, "loss_sum": final_loss_sum_val}
        return final_loss, ret_scores, info

    def _flatten_packs(self, packs):
        NUM_RET_PACK = 6  # discard the first mb-size
        ret_packs = [[] for _ in range(NUM_RET_PACK)]
        cur_base_idx = 0
        for one_pack in packs:
            mb_size = one_pack[0]
            ret_packs[0].append(one_pack[1]+cur_base_idx)
            for i in range(1, NUM_RET_PACK):
                ret_packs[i].append(one_pack[i+1])
            cur_base_idx += mb_size
        ret = [(None if z[0] is None else BK.concat(z, 0)) for z in ret_packs]
        return ret

    # get tmp mat of (bsize, mod) for special calculations
    def _get_tmp_mat(self, shape, val, dtype, idx0, idx1, vals):
        x = BK.constants(shape, val, dtype=dtype)
        x[idx0, idx1] = vals
        return x

    # enc: [bs, len, D], valid: [bs, len-m, len-h], mask: [bs, len]
    # todo(+N): currently does not support multiple high-order parts
    def decode(self, insts: List[ParseInstance], enc_expr, final_valid_expr, go1_pack, training: bool, margin: float):
        # =====
        has_go1 = go1_pack is not None
        if has_go1:
            go1_arc_scores, go1_lab_scores = go1_pack
        else:
            go1_arc_scores = go1_lab_scores = None
        # collect scores (split into mini-batches to save memory)
        batch_size, max_length = BK.get_shape(final_valid_expr)[:2]
        cur_base_bidx = 0
        ret_heads, ret_labels, ret_gold_packs, ret_pred_packs = [], [], [], []
        # dynamically deciding the mini-batch size based on budget
        cur_mb_batch_size = max(int(self.mb_dec_lb / max_length), 1)
        while cur_base_bidx < batch_size:
            next_base_bidx = min(batch_size, cur_base_bidx+cur_mb_batch_size)
            # decode for current mini-batch
            mb_insts = insts[cur_base_bidx:next_base_bidx]
            mb_enc_expr = enc_expr[cur_base_bidx:next_base_bidx]  # [mb, slen, D]
            mb_final_valid_expr = final_valid_expr[cur_base_bidx:next_base_bidx]  # [mb, slen, slen]
            if has_go1:
                mb_go1_pack = (go1_arc_scores[cur_base_bidx:next_base_bidx], go1_lab_scores[cur_base_bidx:next_base_bidx])
            else:
                mb_go1_pack = None
            res_heads, res_labels, gold_pack, pred_pack = \
                self._decode(mb_insts, mb_enc_expr, mb_final_valid_expr, mb_go1_pack, training, margin)
            # todo(note): temply put them like this, later will flatten specifically
            ret_heads.append(res_heads)  # List[List[List[Int]]]
            ret_labels.append(res_labels)  # List[np.array[mb_size, slen]]
            ret_gold_packs.append(gold_pack)  # List[idxes-pack]
            ret_pred_packs.append(pred_pack)  # List[idxes-pack]
            cur_base_bidx = next_base_bidx
        if not self.system_labeled:
            ret_labels = None
        return ret_heads, ret_labels, ret_gold_packs, ret_pred_packs

    # decode for one mini-batch, to be implemented
    # list[insts], [*, slen, D], [*, slen, slen], go1_pack(arc/label scores) -> List[results/heads]
    def _decode(self, mb_insts: List[ParseInstance], mb_enc_expr, mb_valid_expr, mb_go1_pack, training: bool, margin: float):
        # =====
        use_sib, use_gp = self.use_sib, self.use_gp
        # =====
        mb_size = len(mb_insts)
        mat_shape = BK.get_shape(mb_valid_expr)
        max_slen = mat_shape[-1]
        # step 1: extract the candidate features
        batch_idxes, m_idxes, h_idxes, sib_idxes, gp_idxes = self.helper.get_cand_features(mb_valid_expr)
        # =====
        # step 2: high order scoring
        # step 2.1: basic scoring, [*], [*, Lab]
        arc_scores, lab_scores = self._get_basic_score(mb_enc_expr, batch_idxes, m_idxes, h_idxes, sib_idxes, gp_idxes)
        cur_system_labeled = (lab_scores is not None)
        # step 2.2: margin
        # get gold labels, which can be useful for later calculating loss
        if training:
            gold_b_idxes, gold_m_idxes, gold_h_idxes, gold_sib_idxes, gold_gp_idxes, gold_lab_idxes = \
                [(None if z is None else BK.input_idx(z)) for z in self._get_idxes_from_insts(mb_insts, use_sib, use_gp)]
            # add the margins to the scores: (m,h), (m,sib), (m,gp)
            cur_margin = margin / self.margin_div
            self._add_margin_inplaced(mat_shape, gold_b_idxes, gold_m_idxes, gold_h_idxes, gold_lab_idxes,
                                      batch_idxes, m_idxes, h_idxes, arc_scores, lab_scores, cur_margin, cur_margin)
            if use_sib:
                self._add_margin_inplaced(mat_shape, gold_b_idxes, gold_m_idxes, gold_sib_idxes, gold_lab_idxes,
                                          batch_idxes, m_idxes, sib_idxes, arc_scores, lab_scores, cur_margin, cur_margin)
            if use_gp:
                self._add_margin_inplaced(mat_shape, gold_b_idxes, gold_m_idxes, gold_gp_idxes, gold_lab_idxes,
                                          batch_idxes, m_idxes, gp_idxes, arc_scores, lab_scores, cur_margin, cur_margin)
            # may be useful for later training
            gold_pack = (mb_size, gold_b_idxes, gold_m_idxes, gold_h_idxes, gold_sib_idxes, gold_gp_idxes, gold_lab_idxes)
        else:
            gold_pack = None
        # step 2.3: o1scores
        if mb_go1_pack is not None:
            go1_arc_scores, go1_lab_scores = mb_go1_pack
            # todo(note): go1_arc_scores is not added here, but as the input to the dec-algo
            if cur_system_labeled:
                lab_scores += go1_lab_scores[batch_idxes, m_idxes, h_idxes]
        else:
            go1_arc_scores = None
        # step 2.4: max out labels; todo(+N): or using logsumexp here?
        if cur_system_labeled:
            max_lab_scores, max_lab_idxes = lab_scores.max(-1)
            final_scores = arc_scores + max_lab_scores  # [*], final input arc scores
        else:
            max_lab_idxes = None
            final_scores = arc_scores
        # =====
        # step 3: actual decode
        res_heads = []
        for sid, inst in enumerate(mb_insts):
            slen = len(inst) + 1  # plus one for the art-root
            arr_o1_masks = BK.get_value(mb_valid_expr[sid, :slen, :slen].int())
            arr_o1_scores = BK.get_value(go1_arc_scores[sid, :slen, :slen].double()) if (go1_arc_scores is not None) else None
            cur_bidx_mask = (batch_idxes == sid)
            input_pack = [m_idxes, h_idxes, sib_idxes, gp_idxes, final_scores]
            one_heads = self.helper.decode_one(slen, self.projective, arr_o1_masks, arr_o1_scores, input_pack, cur_bidx_mask)
            res_heads.append(one_heads)
        # =====
        # step 4: get labels back and pred_pack
        pred_b_idxes, pred_m_idxes, pred_h_idxes, pred_sib_idxes, pred_gp_idxes, _ = \
            [(None if z is None else BK.input_idx(z)) for z in self._get_idxes_from_preds(res_heads, None, use_sib, use_gp)]
        if cur_system_labeled:
            # obtain hit components
            pred_hit_mask = self._get_hit_mask(mat_shape, pred_b_idxes, pred_m_idxes, pred_h_idxes, batch_idxes, m_idxes, h_idxes)
            if use_sib:
                pred_hit_mask &= self._get_hit_mask(mat_shape, pred_b_idxes, pred_m_idxes, pred_sib_idxes, batch_idxes, m_idxes, sib_idxes)
            if use_gp:
                pred_hit_mask &= self._get_hit_mask(mat_shape, pred_b_idxes, pred_m_idxes, pred_gp_idxes, batch_idxes, m_idxes, gp_idxes)
            # get pred labels (there should be only one hit per mod!)
            pred_labels = BK.constants_idx([mb_size, max_slen], 0)
            pred_labels[batch_idxes[pred_hit_mask], m_idxes[pred_hit_mask]] = max_lab_idxes[pred_hit_mask]
            res_labels = BK.get_value(pred_labels)
            pred_lab_idxes = pred_labels[pred_b_idxes, pred_m_idxes]
        else:
            res_labels = None
            pred_lab_idxes = None
        pred_pack = (mb_size, pred_b_idxes, pred_m_idxes, pred_h_idxes, pred_sib_idxes, pred_gp_idxes, pred_lab_idxes)
        # return
        return res_heads, res_labels, gold_pack, pred_pack

    # =====
    # helper functions

    # get high order scores: [*, slen, D], *[*]
    # split into multiple times if necessary
    def _get_basic_score(self, mb_enc_expr, batch_idxes, m_idxes, h_idxes, sib_idxes, gp_idxes):
        allp_size = BK.get_shape(batch_idxes, 0)
        all_arc_scores, all_lab_scores = [], []
        cur_pidx = 0
        while cur_pidx < allp_size:
            next_pidx = min(allp_size, cur_pidx + self.mb_dec_sb)
            # first calculate srepr
            s_enc = self.slayer
            cur_batch_idxes = batch_idxes[cur_pidx:next_pidx]
            h_expr = mb_enc_expr[cur_batch_idxes, h_idxes[cur_pidx:next_pidx]]
            m_expr = mb_enc_expr[cur_batch_idxes, m_idxes[cur_pidx:next_pidx]]
            s_expr = mb_enc_expr[cur_batch_idxes, sib_idxes[cur_pidx:next_pidx]].unsqueeze(-2) \
                if (sib_idxes is not None) else None  # [*, 1, D]
            g_expr = mb_enc_expr[cur_batch_idxes, gp_idxes[cur_pidx:next_pidx]] if (gp_idxes is not None) else None
            head_srepr = s_enc.calculate_repr(h_expr, g_expr, None, None, s_expr, None, None, None)
            mod_srepr = s_enc.forward_repr(m_expr)
            # then get the scores
            arc_score = self.scorer.transform_and_arc_score_plain(mod_srepr, head_srepr).squeeze(-1)
            all_arc_scores.append(arc_score)
            if self.system_labeled:
                lab_score = self.scorer.transform_and_label_score_plain(mod_srepr, head_srepr)
                all_lab_scores.append(lab_score)
            cur_pidx = next_pidx
        final_arc_score = BK.concat(all_arc_scores, 0)
        final_lab_score = BK.concat(all_lab_scores, 0) if self.system_labeled else None
        return final_arc_score, final_lab_score

    # prepare the idxes from gold annotations of instances
    def _get_idxes_from_insts(self, insts: List[ParseInstance], use_sib, use_gp):
        b_idxes, m_idxes, h_idxes = [], [], []
        sib_idxes, gp_idxes = ([] if use_sib else None), ([] if use_gp else None)
        lab_idxes = []
        for b, inst in enumerate(insts):
            cur_len = len(inst)  # length is len(heads)-1
            b_idxes.extend(b for _ in range(cur_len))
            # exclude the Node 0 as art-root
            m_idxes.extend(range(1, cur_len+1))
            h_idxes.extend(inst.heads.vals[1:])
            lab_idxes.extend(inst.labels.idxes[1:])
            if use_sib:
                sib_idxes.extend(inst.sibs[1:])
            if use_gp:
                gp_idxes.extend(inst.gps[1:])
        return b_idxes, m_idxes, h_idxes, sib_idxes, gp_idxes, lab_idxes

    # prepare the idxes from predicted heads
    def _get_idxes_from_preds(self, heads: List[List], labels: List[List], use_sib, use_gp):
        has_label = labels is not None
        b_idxes, m_idxes, h_idxes = [], [], []
        sib_idxes, gp_idxes = ([] if use_sib else None), ([] if use_gp else None)
        lab_idxes = ([] if has_label else None)
        b = 0
        for idx in range(len(heads)):
            one_heads = heads[idx]
            cur_len = len(one_heads)-1  # length is len(heads)-1
            b_idxes.extend(b for _ in range(cur_len))
            # exclude the Node 0 as art-root
            m_idxes.extend(range(1, cur_len + 1))
            h_idxes.extend(one_heads[1:])
            if has_label:
                lab_idxes.extend(labels[idx][1:])
            if use_sib:
                children_left, children_right = ParseInstance.get_children(one_heads)
                sib_idxes.extend(ParseInstance.get_sibs(children_left, children_right)[1:])
            if use_gp:
                gp_idxes.extend(ParseInstance.get_gps(one_heads)[1:])
            b += 1
        return b_idxes, m_idxes, h_idxes, sib_idxes, gp_idxes, lab_idxes

    # add margin for one group of features, modified inplaced!
    def _add_margin_inplaced(self, shape, hit_idxes0, hit_idxes1, hit_idxes2, hit_labels,
                    query_idxes0, query_idxes1, query_idxes2, arc_scores, lab_scores,
                    arc_margin: float, lab_margin: float):
        # arc
        gold_arc_mat = BK.constants(shape, 0.)
        gold_arc_mat[hit_idxes0, hit_idxes1, hit_idxes2] = arc_margin
        gold_arc_margins = gold_arc_mat[query_idxes0, query_idxes1, query_idxes2]
        arc_scores -= gold_arc_margins
        if lab_scores is not None:
            # label
            gold_lab_mat = BK.constants_idx(shape, 0)  # 0 means the padding idx
            gold_lab_mat[hit_idxes0, hit_idxes1, hit_idxes2] = hit_labels
            gold_lab_margin_idxes = gold_lab_mat[query_idxes0, query_idxes1, query_idxes2]
            lab_scores[BK.arange_idx(BK.get_shape(gold_lab_margin_idxes, 0)), gold_lab_margin_idxes] -= lab_margin
        return

    # get hit mask
    def _get_hit_mask(self, shape, hit_idxes0, hit_idxes1, hit_idxes2, query_idxes0, query_idxes1, query_idxes2):
        hit_mat = BK.constants(shape, 0, dtype=BK.uint8)
        hit_mat[hit_idxes0, hit_idxes1, hit_idxes2] = 1
        hit_mask = hit_mat[query_idxes0, query_idxes1, query_idxes2]
        return hit_mask

# =====
# for specific systems

# [m, h]
class G2O1Helper:
    def get_cand_features(self, final_valid_expr):
        # expand to get features: (m,h)
        batch_idxes, m_idxes, h_idxes = [x.squeeze(-1) for x in final_valid_expr.nonzero().split(1, -1)]
        vmask = (m_idxes != 0)  # m cannot be 0
        return batch_idxes[vmask], m_idxes[vmask], h_idxes[vmask], None, None

    def get_unpruned_mask(self, valid_expr, gold_pack):
        batch_idxes, m_idxes, h_idxes, _, _, _ = gold_pack
        gold_mask = valid_expr[batch_idxes, m_idxes, h_idxes]
        gold_mask = gold_mask.byte()
        mod_unpruned_mask = BK.constants(BK.get_shape(valid_expr)[:2], 0, dtype=BK.uint8)
        mod_unpruned_mask[batch_idxes[gold_mask], m_idxes[gold_mask]] = 1
        return mod_unpruned_mask, gold_mask

    def decode_one(self, slen: int, projective: bool, arr_o1_masks, arr_o1_scores, input_pack, cur_bidx_mask):
        m_idxes, h_idxes, _, _, final_scores = input_pack
        if arr_o1_scores is None:
            arr_o1_scores = np.full([slen, slen], 0., dtype=np.double)
        # direct add to the scores
        m_idxes, h_idxes, final_scores = m_idxes[cur_bidx_mask], h_idxes[cur_bidx_mask], final_scores[cur_bidx_mask]
        arr_o1_scores[BK.get_value(m_idxes), BK.get_value(h_idxes)] += BK.get_value(final_scores)
        return hop_decode(slen, projective, arr_o1_masks, arr_o1_scores, None, None, None)

# [m, h, sib]
class G2O2sibHelper:
    def get_cand_features(self, final_valid_expr):
        # expand to get features: (m,h), (h,sib) -> [m, h, sib]
        expanded_valid_expr = final_valid_expr.unsqueeze(-1) * final_valid_expr.transpose(-1, -2).unsqueeze(-3)
        batch_idxes, m_idxes, h_idxes, sib_idxes = [x.squeeze(-1) for x in expanded_valid_expr.nonzero().split(1, -1)]
        del expanded_valid_expr
        vmask = (m_idxes != 0)  # m cannot be 0
        vmask &= (sib_idxes != 0)  # sib cannot be 0
        # sib is inner than m
        vmask &= (((m_idxes <= sib_idxes) & (sib_idxes < h_idxes)) | ((h_idxes < sib_idxes) & (sib_idxes <= m_idxes)))
        return batch_idxes[vmask], m_idxes[vmask], h_idxes[vmask], sib_idxes[vmask], None

    def get_unpruned_mask(self, valid_expr, gold_pack):
        batch_idxes, m_idxes, h_idxes, sib_idxes, _, _ = gold_pack
        gold_mask = valid_expr[batch_idxes, m_idxes, h_idxes]
        gold_mask *= valid_expr[batch_idxes, sib_idxes, h_idxes]
        gold_mask = gold_mask.byte()
        mod_unpruned_mask = BK.constants(BK.get_shape(valid_expr)[:2], 0, dtype=BK.uint8)
        mod_unpruned_mask[batch_idxes[gold_mask], m_idxes[gold_mask]] = 1
        return mod_unpruned_mask, gold_mask

    def decode_one(self, slen: int, projective: bool, arr_o1_masks, arr_o1_scores, input_pack, cur_bidx_mask):
        m_idxes, h_idxes, sib_idxes, _, final_scores = input_pack
        o2sib_pack = [m_idxes[cur_bidx_mask].int(), h_idxes[cur_bidx_mask].int(), sib_idxes[cur_bidx_mask].int(), final_scores[cur_bidx_mask].double()]
        o2sib_arr_pack = [BK.get_value(z) for z in o2sib_pack]
        return hop_decode(slen, projective, arr_o1_masks, arr_o1_scores, o2sib_arr_pack, None, None)

# [m, h, gp]
class G2O2gHelper:
    def get_cand_features(self, final_valid_expr):
        # expand to get features: (m,h), (h,gp) -> [m, h, gp]
        expanded_valid_expr = final_valid_expr.unsqueeze(-1) * final_valid_expr.unsqueeze(-3)
        batch_idxes, m_idxes, h_idxes, gp_idxes = [x.squeeze(-1) for x in expanded_valid_expr.nonzero().split(1, -1)]
        del expanded_valid_expr
        vmask = (m_idxes != 0)  # m cannot be 0
        # sib is inner than m
        vmask &= (m_idxes != gp_idxes)  # gp cannot be m
        return batch_idxes[vmask], m_idxes[vmask], h_idxes[vmask], None, gp_idxes[vmask]

    def get_unpruned_mask(self, valid_expr, gold_pack):
        batch_idxes, m_idxes, h_idxes, _, gp_idxes, _ = gold_pack
        gold_mask = valid_expr[batch_idxes, m_idxes, h_idxes]
        gold_mask *= valid_expr[batch_idxes, h_idxes, gp_idxes]
        gold_mask = gold_mask.byte()
        mod_unpruned_mask = BK.constants(BK.get_shape(valid_expr)[:2], 0, dtype=BK.uint8)
        mod_unpruned_mask[batch_idxes[gold_mask], m_idxes[gold_mask]] = 1
        return mod_unpruned_mask, gold_mask

    def decode_one(self, slen: int, projective: bool, arr_o1_masks, arr_o1_scores, input_pack, cur_bidx_mask):
        m_idxes, h_idxes, _, gp_idxes, final_scores = input_pack
        o2g_pack = [m_idxes[cur_bidx_mask].int(), h_idxes[cur_bidx_mask].int(), gp_idxes[cur_bidx_mask].int(), final_scores[cur_bidx_mask].double()]
        o2g_arr_pack = [BK.get_value(z) for z in o2g_pack]
        return hop_decode(slen, projective, arr_o1_masks, arr_o1_scores, None, o2g_arr_pack, None)

# [m, h, sib, gp]
class G2O3gsibHelper:
    def get_cand_features(self, final_valid_expr):
        # expand to get features: (m,h), (h,sib), (h,gp) -> [m, h, sib, gp]
        expanded_valid_expr = final_valid_expr.unsqueeze(-1).unsqueeze(-1) * \
                              final_valid_expr.transpose(-1, -2).unsqueeze(-3).unsqueeze(-1) * \
                              final_valid_expr.unsqueeze(-3).unsqueeze(-2)
        batch_idxes, m_idxes, h_idxes, sib_idxes, gp_idxes = [x.squeeze(-1) for x in
                                                              expanded_valid_expr.nonzero().split(1, -1)]
        del expanded_valid_expr
        vmask = (m_idxes != 0)  # m cannot be 0
        vmask &= (sib_idxes != 0)  # sib cannot be 0
        # sib is inner than m
        vmask &= (((m_idxes<=sib_idxes) & (sib_idxes<h_idxes)) | ((h_idxes<sib_idxes) & (sib_idxes<=m_idxes)))
        vmask &= (m_idxes != gp_idxes)  # gp cannot be m
        vmask &= (sib_idxes != gp_idxes)  # gp cannot be sib
        return batch_idxes[vmask], m_idxes[vmask], h_idxes[vmask], sib_idxes[vmask], gp_idxes[vmask]

    def get_unpruned_mask(self, valid_expr, gold_pack):
        batch_idxes, m_idxes, h_idxes, sib_idxes, gp_idxes, _ = gold_pack
        gold_mask = valid_expr[batch_idxes, m_idxes, h_idxes]
        gold_mask *= valid_expr[batch_idxes, sib_idxes, h_idxes]
        gold_mask *= valid_expr[batch_idxes, h_idxes, gp_idxes]
        gold_mask = gold_mask.byte()
        mod_unpruned_mask = BK.constants(BK.get_shape(valid_expr)[:2], 0, dtype=BK.uint8)
        mod_unpruned_mask[batch_idxes[gold_mask], m_idxes[gold_mask]] = 1
        return mod_unpruned_mask, gold_mask

    def decode_one(self, slen: int, projective: bool, arr_o1_masks, arr_o1_scores, input_pack, cur_bidx_mask):
        m_idxes, h_idxes, sib_idxes, gp_idxes, final_scores = input_pack
        o3gsib_pack = [m_idxes[cur_bidx_mask].int(), h_idxes[cur_bidx_mask].int(), sib_idxes[cur_bidx_mask].int(), gp_idxes[cur_bidx_mask].int(), final_scores[cur_bidx_mask].double()]
        o3gsib_arr_pack = [BK.get_value(z) for z in o3gsib_pack]
        return hop_decode(slen, projective, arr_o1_masks, arr_o1_scores, None, None, o3gsib_arr_pack)

# b tasks/zdpar/ef/parser/g2p.py:316
