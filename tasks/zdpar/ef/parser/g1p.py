#

# first-order graph parser: as init & pruner

from typing import List
import numpy as np
from copy import deepcopy
import traceback

from msp.utils import Constants, FileHelper, zlog, Conf
from msp.data import VocabPackage
from msp.nn import BK
from msp.zext.seq_helper import DataPadder

from ...common.data import ParseInstance
from ...common.model import BaseParserConf, BaseInferenceConf, BaseTrainingConf, BaseParser
from ..scorer import Scorer, ScorerConf
from ...algo import nmst_unproj, nmarginal_unproj

# =====
# conf

class PruneG1Conf(Conf):
    def __init__(self):
        self.pruning_labeled = True  # todo(note): for marginal mode, do not change this which may cause decoding errors
        # simple topk pruning
        self.pruning_use_topk = False
        self.pruning_perc = 1.0
        self.pruning_topk = 4
        self.pruning_gap = 20.
        # marginal pruning
        self.pruning_use_marginal = True
        self.pruning_mthresh = 0.02
        self.pruning_mthresh_rel = True

# decoding conf
class G1InferenceConf(BaseInferenceConf):
    def __init__(self):
        super().__init__()
        # for plus pruning mode
        self.use_pruning = False
        self.pruning_conf = PruneG1Conf()
        # extra outputs
        self.output_marginals = True

# training conf
class G1TraningConf(BaseTrainingConf):
    def __init__(self):
        super().__init__()
        # loss functions
        self.loss_function = "hinge"
        self.loss_div_tok = True        # loss divide by token or by sent?

# overall parser conf
class G1ParserConf(BaseParserConf):
    def __init__(self):
        super().__init__(G1InferenceConf(), G1TraningConf())
        self.sc_conf = ScorerConf()
        self.debug_use_aux_scores = False  # only for pruning/scoring mode

# for g1 as pre-training
class PreG1Conf(Conf):
    def __init__(self):
        # for basic model
        self.g1_pretrain_path = ""  # basic g1 model for prune/init/score
        self.g1_pretrain_init = False  # whether init g1 model to the current model
        # whether add basic scores
        self.lambda_g1_arc_training = 0.
        self.lambda_g1_arc_testing = 0.
        self.lambda_g1_lab_training = 0.
        self.lambda_g1_lab_testing = 0.
        # use aux scores for g1 (must be attached to the inst); only for pruning/scoring mode
        self.g1_use_aux_scores = False

# =====
# model

# the model
class G1Parser(BaseParser):
    def __init__(self, conf: G1ParserConf, vpack: VocabPackage):
        super().__init__(conf, vpack)
        # todo(note): the neural parameters are exactly the same as the EF one
        self.scorer_helper = GScorerHelper(self.scorer)
        self.predict_padder = DataPadder(2, pad_vals=0)
        #
        self.g1_use_aux_scores = conf.debug_use_aux_scores  # assining here is only for debugging usage, otherwise assigning outside
        self.num_label = self.label_vocab.trg_len(True)  # todo(WARN): use the original idx
        #
        self.loss_hinge = (self.conf.tconf.loss_function == "hinge")
        if not self.loss_hinge:
            assert self.conf.tconf.loss_function == "prob", "This model only supports hinge or prob"

    def build_decoder(self):
        conf = self.conf
        # ===== Decoding Scorer =====
        conf.sc_conf._input_dim = self.enc_output_dim
        conf.sc_conf._num_label = self.label_vocab.trg_len(True)  # todo(WARN): use the original idx
        return Scorer(self.pc, conf.sc_conf)

    # =====
    # main procedures

    # decoding
    def inference_on_batch(self, insts: List[ParseInstance], **kwargs):
        iconf = self.conf.iconf
        pconf = iconf.pruning_conf
        with BK.no_grad_env():
            self.refresh_batch(False)
            if iconf.use_pruning:
                # todo(note): for the testing of pruning mode, use the scores instead
                if self.g1_use_aux_scores:
                    valid_mask, arc_score, label_score, mask_expr, _ = G1Parser.score_and_prune(insts, self.num_label, pconf)
                else:
                    valid_mask, arc_score, label_score, mask_expr, _ = self.prune_on_batch(insts, pconf)
                valid_mask_f = valid_mask.float()  # [*, len, len]
                mask_value = Constants.REAL_PRAC_MIN
                full_score = arc_score.unsqueeze(-1) + label_score
                full_score += (mask_value * (1. - valid_mask_f)).unsqueeze(-1)
                info_pruning = G1Parser.collect_pruning_info(insts, valid_mask_f)
                jpos_pack = [None, None, None]
            else:
                input_repr, enc_repr, jpos_pack, mask_arr = self.bter.run(insts, False)
                mask_expr = BK.input_real(mask_arr)
                full_score = self.scorer_helper.score_full(enc_repr)
                info_pruning = None
            # =====
            self._decode(insts, full_score, mask_expr, "g1")
            # put jpos result (possibly)
            self.jpos_decode(insts, jpos_pack)
            # -----
            info = {"sent": len(insts), "tok": sum(map(len, insts))}
            if info_pruning is not None:
                info.update(info_pruning)
            return info

    # training
    def fb_on_batch(self, annotated_insts: List[ParseInstance], training=True, loss_factor=1., **kwargs):
        self.refresh_batch(training)
        # encode
        input_repr, enc_repr, jpos_pack, mask_arr = self.bter.run(annotated_insts, training)
        mask_expr = BK.input_real(mask_arr)
        # the parsing loss
        arc_score = self.scorer_helper.score_arc(enc_repr)
        lab_score = self.scorer_helper.score_label(enc_repr)
        full_score = arc_score + lab_score
        parsing_loss, info = self._loss(annotated_insts, full_score, mask_expr)
        # other loss?
        jpos_loss = self.jpos_loss(jpos_pack, mask_expr)
        reg_loss = self.reg_scores_loss(arc_score, lab_score)
        #
        info["loss_parse"] = BK.get_value(parsing_loss).item()
        final_loss = parsing_loss
        if jpos_loss is not None:
            info["loss_jpos"] = BK.get_value(jpos_loss).item()
            final_loss = parsing_loss + jpos_loss
        if reg_loss is not None:
            final_loss = final_loss + reg_loss
        info["fb"] = 1
        if training:
            BK.backward(final_loss, loss_factor)
        return info

    # =====
    def _decode(self, insts: List[ParseInstance], full_score, mask_expr, misc_prefix):
        # decode
        mst_lengths = [len(z) + 1 for z in insts]  # +=1 to include ROOT for mst decoding
        mst_lengths_arr = np.asarray(mst_lengths, dtype=np.int32)
        mst_heads_arr, mst_labels_arr, mst_scores_arr = nmst_unproj(full_score, mask_expr, mst_lengths_arr,
                                                                    labeled=True, ret_arr=True)
        if self.conf.iconf.output_marginals:
            # todo(note): here, we care about marginals for arc
            # lab_marginals = nmarginal_unproj(full_score, mask_expr, None, labeled=True)
            arc_marginals = nmarginal_unproj(full_score, mask_expr, None, labeled=True).sum(-1)
            bsize, max_len = BK.get_shape(mask_expr)
            idxes_bs_expr = BK.arange_idx(bsize).unsqueeze(-1)
            idxes_m_expr = BK.arange_idx(max_len).unsqueeze(0)
            output_marg = arc_marginals[idxes_bs_expr, idxes_m_expr, BK.input_idx(mst_heads_arr)]
            mst_marg_arr = BK.get_value(output_marg)
        else:
            mst_marg_arr = None
        # ===== assign, todo(warn): here, the labels are directly original idx, no need to change
        for one_idx, one_inst in enumerate(insts):
            cur_length = mst_lengths[one_idx]
            one_inst.pred_heads.set_vals(mst_heads_arr[one_idx][:cur_length])  # directly int-val for heads
            one_inst.pred_labels.build_vals(mst_labels_arr[one_idx][:cur_length], self.label_vocab)
            one_scores = mst_scores_arr[one_idx][:cur_length]
            one_inst.pred_par_scores.set_vals(one_scores)
            # extra output
            one_inst.extra_pred_misc[misc_prefix+"_score"] = one_scores.tolist()
            if mst_marg_arr is not None:
                one_inst.extra_pred_misc[misc_prefix+"_marg"] = mst_marg_arr[one_idx][:cur_length].tolist()

    # here, only adopt hinge(max-margin) loss; mostly adopted from previous graph parser
    def _loss(self, annotated_insts: List[ParseInstance], full_score_expr, mask_expr, valid_expr=None):
        bsize, max_len = BK.get_shape(mask_expr)
        # gold heads and labels
        gold_heads_arr, _ = self.predict_padder.pad([z.heads.vals for z in annotated_insts])
        # todo(note): here use the original idx of label, no shift!
        gold_labels_arr, _ = self.predict_padder.pad([z.labels.idxes for z in annotated_insts])
        gold_heads_expr = BK.input_idx(gold_heads_arr)  # [BS, Len]
        gold_labels_expr = BK.input_idx(gold_labels_arr)  # [BS, Len]
        #
        idxes_bs_expr = BK.arange_idx(bsize).unsqueeze(-1)
        idxes_m_expr = BK.arange_idx(max_len).unsqueeze(0)
        # scores for decoding or marginal
        margin = self.margin.value
        decoding_scores = full_score_expr.clone().detach()
        decoding_scores = self.scorer_helper.postprocess_scores(decoding_scores, mask_expr, margin, gold_heads_expr, gold_labels_expr)
        if self.loss_hinge:
            mst_lengths_arr = np.asarray([len(z)+1 for z in annotated_insts], dtype=np.int32)
            pred_heads_expr, pred_labels_expr, _ = nmst_unproj(decoding_scores, mask_expr, mst_lengths_arr, labeled=True, ret_arr=False)
            # ===== add margin*cost, [bs, len]
            gold_final_scores = full_score_expr[idxes_bs_expr, idxes_m_expr, gold_heads_expr, gold_labels_expr]
            pred_final_scores = full_score_expr[idxes_bs_expr, idxes_m_expr, pred_heads_expr, pred_labels_expr] + margin*(gold_heads_expr!=pred_heads_expr).float() + margin*(gold_labels_expr!=pred_labels_expr).float()  # plus margin
            hinge_losses = pred_final_scores - gold_final_scores
            valid_losses = ((hinge_losses * mask_expr)[:, 1:].sum(-1) > 0.).float().unsqueeze(-1)  # [*, 1]
            final_losses = hinge_losses * valid_losses
        else:
            lab_marginals = nmarginal_unproj(decoding_scores, mask_expr, None, labeled=True)
            lab_marginals[idxes_bs_expr, idxes_m_expr, gold_heads_expr, gold_labels_expr] -= 1.
            grads_masked = lab_marginals*mask_expr.unsqueeze(-1).unsqueeze(-1)*mask_expr.unsqueeze(-2).unsqueeze(-1)
            final_losses = (full_score_expr * grads_masked).sum(-1).sum(-1)  # [bs, m]
        # divide loss by what?
        num_sent = len(annotated_insts)
        num_valid_tok = sum(len(z) for z in annotated_insts)
        # exclude non-valid ones: there can be pruning error
        if valid_expr is not None:
            final_valids = valid_expr[idxes_bs_expr, idxes_m_expr, gold_heads_expr]  # [bs, m] of (0. or 1.)
            final_losses = final_losses * final_valids
            tok_valid = float(BK.get_value(final_valids[:, 1:].sum()))
            assert tok_valid <= num_valid_tok
            tok_prune_err = num_valid_tok - tok_valid
        else:
            tok_prune_err = 0
        # collect loss with mask, also excluding the first symbol of ROOT
        final_losses_masked = (final_losses * mask_expr)[:, 1:]
        final_loss_sum = BK.sum(final_losses_masked)
        if self.conf.tconf.loss_div_tok:
            final_loss = final_loss_sum / num_valid_tok
        else:
            final_loss = final_loss_sum / num_sent
        final_loss_sum_val = float(BK.get_value(final_loss_sum))
        info = {"sent": num_sent, "tok": num_valid_tok, "tok_prune_err": tok_prune_err, "loss_sum": final_loss_sum_val}
        return final_loss, info

    # =====
    # special preloading
    @staticmethod
    def special_pretrain_load(m: BaseParser, path, strict):
        if FileHelper.isfile(path):
            try:
                zlog(f"Trying to load pretrained model from {path}")
                m.load(path, strict)
                zlog(f"Finished loading pretrained model from {path}")
                return True
            except:
                zlog(traceback.format_exc())
                zlog("Failed loading, keep the original ones.")
        else:
            zlog(f"File does not exist for pretraining loading: {path}")
        return False

    # init for the pre-trained G1, return g1parser and also modify m's params
    @staticmethod
    def pre_g1_init(m: BaseParser, pg1_conf: PreG1Conf, strict=True):
        # ===== basic G1 Parser's loading
        # todo(WARN): construct the g1conf here instead of loading for simplicity, since the scorer architecture should be the same
        g1conf = G1ParserConf()
        g1conf.bt_conf = deepcopy(m.conf.bt_conf)
        g1conf.sc_conf = deepcopy(m.conf.sc_conf)
        g1conf.validate()
        # todo(note): specific setting
        g1conf.g1_use_aux_scores = pg1_conf.g1_use_aux_scores
        #
        g1parser = G1Parser(g1conf, m.vpack)
        if not G1Parser.special_pretrain_load(g1parser, pg1_conf.g1_pretrain_path, strict):
            g1parser = None
        # current init
        if pg1_conf.g1_pretrain_init:
            G1Parser.special_pretrain_load(m, pg1_conf.g1_pretrain_path, strict)
        return g1parser

    # collect and batch scores
    @staticmethod
    def collect_aux_scores(insts: List[ParseInstance], output_num_label):
        score_tuples = [z.extra_features["aux_score"] for z in insts]
        num_label = score_tuples[0][1].shape[-1]
        max_len = max(len(z)+1 for z in insts)
        mask_value = Constants.REAL_PRAC_MIN
        bsize = len(insts)
        arc_score_arr = np.full([bsize, max_len, max_len], mask_value, dtype=np.float32)
        lab_score_arr = np.full([bsize, max_len, max_len, output_num_label], mask_value, dtype=np.float32)
        mask_arr = np.full([bsize, max_len], 0., dtype=np.float32)
        for bidx, one_tuple in enumerate(score_tuples):
            one_score_arc, one_score_lab = one_tuple
            one_len = one_score_arc.shape[1]
            arc_score_arr[bidx, :one_len, :one_len] = one_score_arc
            lab_score_arr[bidx, :one_len, :one_len, -num_label:] = one_score_lab
            mask_arr[bidx, :one_len] = 1.
        return BK.input_real(arc_score_arr).unsqueeze(-1), BK.input_real(lab_score_arr), BK.input_real(mask_arr)

    # pruner: [bs, slen, slen], [bs, slen, slen, Lab]
    @staticmethod
    def prune_with_scores(arc_score, label_score, mask_expr, pconf: PruneG1Conf, arc_marginals=None):
        prune_use_topk, prune_use_marginal, prune_labeled, prune_perc, prune_topk, prune_gap, prune_mthresh, prune_mthresh_rel = \
            pconf.pruning_use_topk, pconf.pruning_use_marginal, pconf.pruning_labeled, pconf.pruning_perc, pconf.pruning_topk, \
            pconf.pruning_gap, pconf.pruning_mthresh, pconf.pruning_mthresh_rel
        full_score = arc_score + label_score
        final_valid_mask = BK.constants(BK.get_shape(arc_score), 0, dtype=BK.uint8).squeeze(-1)
        # (put as argument) arc_marginals = None  # [*, mlen, hlen]
        if prune_use_marginal:
            if arc_marginals is None:  # does not provided, calculate from scores
                if prune_labeled:
                    # arc_marginals = nmarginal_unproj(full_score, mask_expr, None, labeled=True).max(-1)[0]
                    # use sum of label marginals instead of max
                    arc_marginals = nmarginal_unproj(full_score, mask_expr, None, labeled=True).sum(-1)
                else:
                    arc_marginals = nmarginal_unproj(arc_score, mask_expr, None, labeled=True).squeeze(-1)
            if prune_mthresh_rel:
                # relative value
                max_arc_marginals = arc_marginals.max(-1)[0].log().unsqueeze(-1)
                m_valid_mask = (arc_marginals.log() - max_arc_marginals) > float(np.log(prune_mthresh))
            else:
                # absolute value
                m_valid_mask = (arc_marginals > prune_mthresh)  # [*, len-m, len-h]
            final_valid_mask |= m_valid_mask
        if prune_use_topk:
            # prune by "in topk" and "gap-to-top less than gap" for each mod
            if prune_labeled:  # take argmax among label dim
                tmp_arc_score, _ = full_score.max(-1)
            else:
                # todo(note): may be modified inplaced, but does not matter since will finally be masked later
                tmp_arc_score = arc_score.squeeze(-1)
            # first apply mask
            mask_value = Constants.REAL_PRAC_MIN
            mask_mul = (mask_value * (1. - mask_expr))  # [*, len]
            tmp_arc_score += mask_mul.unsqueeze(-1)
            tmp_arc_score += mask_mul.unsqueeze(-2)
            maxlen = BK.get_shape(tmp_arc_score, -1)
            tmp_arc_score += mask_value * BK.eye(maxlen)
            prune_topk = min(prune_topk, int(maxlen * prune_perc + 1), maxlen)
            if prune_topk >= maxlen:
                topk_arc_score = tmp_arc_score
            else:
                topk_arc_score, _ = BK.topk(tmp_arc_score, prune_topk, dim=-1, sorted=False)  # [*, len, k]
            min_topk_arc_score = topk_arc_score.min(-1)[0].unsqueeze(-1)  # [*, len, 1]
            max_topk_arc_score = topk_arc_score.max(-1)[0].unsqueeze(-1)  # [*, len, 1]
            arc_score_thresh = BK.max_elem(min_topk_arc_score, max_topk_arc_score - prune_gap)  # [*, len, 1]
            t_valid_mask = (tmp_arc_score > arc_score_thresh)  # [*, len-m, len-h]
            final_valid_mask |= t_valid_mask
        return final_valid_mask, arc_marginals

    # combining the above two
    @staticmethod
    def score_and_prune(insts: List[ParseInstance], output_num_label, pconf: PruneG1Conf):
        arc_score, lab_score, mask_expr = G1Parser.collect_aux_scores(insts, output_num_label)
        valid_mask, arc_marginals = G1Parser.prune_with_scores(arc_score, lab_score, mask_expr, pconf)
        return valid_mask, arc_score.squeeze(-1), lab_score, mask_expr, arc_marginals

    # =====
    # special mode: first-order model as scorer/pruner
    def score_on_batch(self, insts: List[ParseInstance]):
        with BK.no_grad_env():
            self.refresh_batch(False)
            input_repr, enc_repr, jpos_pack, mask_arr = self.bter.run(insts, False)
            # mask_expr = BK.input_real(mask_arr)
            arc_score = self.scorer_helper.score_arc(enc_repr)
            label_score = self.scorer_helper.score_label(enc_repr)
            return arc_score.squeeze(-1), label_score

    # todo(note): the union of two types of pruner!
    def prune_on_batch(self, insts: List[ParseInstance], pconf: PruneG1Conf):
        with BK.no_grad_env():
            self.refresh_batch(False)
            # encode
            input_repr, enc_repr, jpos_pack, mask_arr = self.bter.run(insts, False)
            mask_expr = BK.input_real(mask_arr)
            arc_score = self.scorer_helper.score_arc(enc_repr)
            label_score = self.scorer_helper.score_label(enc_repr)
            final_valid_mask, arc_marginals = G1Parser.prune_with_scores(arc_score, label_score, mask_expr, pconf)
            return final_valid_mask, arc_score.squeeze(-1), label_score, mask_expr, arc_marginals

    # =====
    @staticmethod
    def collect_pruning_info(insts: List[ParseInstance], valid_mask_f):
        # two dimensions: coverage and pruning-effect
        maxlen = BK.get_shape(valid_mask_f, -1)
        # 1. coverage
        valid_mask_f_flattened = valid_mask_f.view([-1, maxlen])  # [bs*len, len]
        cur_mod_base = 0
        all_mods, all_heads = [], []
        for cur_idx, cur_inst in enumerate(insts):
            for m, h in enumerate(cur_inst.heads.vals[1:], 1):
                all_mods.append(m+cur_mod_base)
                all_heads.append(h)
            cur_mod_base += maxlen
        cov_count = len(all_mods)
        cov_valid = BK.get_value(valid_mask_f_flattened[all_mods, all_heads].sum()).item()
        # 2. pruning-rate
        # todo(warn): to speed up, these stats are approximate because of including paddings
        # edges
        pr_edges = int(np.prod(BK.get_shape(valid_mask_f)))
        pr_edges_valid = BK.get_value(valid_mask_f.sum()).item()
        # valid as structured heads
        pr_o2_sib = pr_o2_g = pr_edges
        pr_o3_gsib = maxlen*pr_edges
        valid_chs_counts, valid_par_counts = valid_mask_f.sum(-2), valid_mask_f.sum(-1)  # [*, len]
        valid_gsibs = valid_chs_counts*valid_par_counts
        pr_o2_sib_valid = BK.get_value(valid_chs_counts.sum()).item()
        pr_o2_g_valid = BK.get_value(valid_par_counts.sum()).item()
        pr_o3_gsib_valid = BK.get_value(valid_gsibs.sum()).item()
        return {"cov_count": cov_count, "cov_valid": cov_valid, "pr_edges": pr_edges, "pr_edges_valid": pr_edges_valid,
                "pr_o2_sib": pr_o2_sib, "pr_o2_g": pr_o2_g, "pr_o3_gsib": pr_o3_gsib,
                "pr_o2_sib_valid": pr_o2_sib_valid, "pr_o2_g_valid": pr_o2_g_valid, "pr_o3_gsib_valid": pr_o3_gsib_valid}

# =====
class GScorerHelper:
    def __init__(self, scorer: Scorer):
        self.scorer = scorer

    # first order full score: return [*, len-m, len-h, label]
    def score_full(self, enc_repr):
        arc_score = self.scorer.transform_and_arc_score(enc_repr)
        label_score = self.scorer.transform_and_label_score(enc_repr)
        # todo(note): apply masks/margins later
        return arc_score + label_score

    def score_arc(self, enc_repr):
        arc_score = self.scorer.transform_and_arc_score(enc_repr)
        # todo(note): apply masks/margins later
        return arc_score

    def score_label(self, enc_repr):
        label_score = self.scorer.transform_and_label_score(enc_repr)
        # todo(note): apply masks/margins later
        return label_score

    # apply mask and add margins for the score in training
    # todo(note): inplaced process!
    def postprocess_scores(self, scores_expr, mask_expr, margin, gold_heads_expr, gold_labels_expr):
        final_full_scores = scores_expr
        # first apply mask
        mask_value = Constants.REAL_PRAC_MIN
        mask_mul = (mask_value * (1.-mask_expr)).unsqueeze(-1)  # [*, len, 1]
        final_full_scores += mask_mul.unsqueeze(-2)
        final_full_scores += mask_mul.unsqueeze(-3)
        # then margin
        if margin > 0.:
            full_shape = BK.get_shape(final_full_scores)
            # combine the first two dim, and minus margin correspondingly
            combined_size = full_shape[0] * full_shape[1]
            combiend_score_expr = final_full_scores.view([combined_size] + full_shape[-2:])
            arange_idx_expr = BK.arange_idx(combined_size)
            combiend_score_expr[arange_idx_expr, gold_heads_expr.view(-1)] -= margin
            combiend_score_expr[arange_idx_expr, gold_heads_expr.view(-1), gold_labels_expr.view(-1)] -= margin
            final_full_scores = combiend_score_expr.view(full_shape)
        return final_full_scores

# tasks/zdpar/ef/parser/g1p.py:138
