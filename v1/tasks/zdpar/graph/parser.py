#

from typing import List
import numpy as np

from msp.utils import zfatal, Constants
from msp.data import VocabPackage
from msp.nn import BK
from msp.zext.seq_helper import DataPadder

from .scorer import GraphScorerConf, GraphScorer
from ..common.data import ParseInstance
from ..common.model import BaseParserConf, BaseInferenceConf, BaseTrainingConf, BaseParser
from ..algo import nmst_unproj, nmst_proj, nmst_greedy, nmarginal_unproj, nmarginal_proj

# =====
# confs

# decoding conf
class InferenceConf(BaseInferenceConf):
    def __init__(self):
        super().__init__()
        # method
        self.dec_algorithm = "unproj"       # proj/unproj/greedy
        self.dec_single_neg = False         # also consider neg links for single-norm (but this might make it unstable for labels?)

# training conf
class TraningConf(BaseTrainingConf):
    def __init__(self):
        super().__init__()
        # loss functions & general
        self.loss_div_tok = True        # loss divide by token or by sent?
        self.loss_function = "prob"     # prob/hinge/mr: probability or hinge/perceptron based or min-risk(partially supported)
        self.loss_single_sample = 2.0   # sampling negative ones (<1: rate, >=2: number) to balance for single-norm

# overall parser conf
class GraphParserConf(BaseParserConf):
    def __init__(self):
        super().__init__(InferenceConf(), TraningConf())
        # decoding part
        self.sc_conf = GraphScorerConf()
        # =====
        # output modeling (local/global/single, proj/unproj, prob/hinge)
        self.output_normalizing = "local"       # local/global/single/hlocal
        # self.iconf.dec_algorithm = "?"
        # self.tconf.loss_function = "?"

# =====
# model

# the model (handle inference & fber by itself, since usually does not involve complex search problem)
class GraphParser(BaseParser):
    def __init__(self, conf: GraphParserConf, vpack: VocabPackage):
        super().__init__(conf, vpack)
        # ===== Input Specification =====
        # both head/label padding with 0 (does not matter what, since will be masked)
        self.predict_padder = DataPadder(2, pad_vals=0)
        self.hlocal_padder = DataPadder(3, pad_vals=0.)
        #
        # todo(warn): adding-styled hlocal has problems intuitively, maybe not suitable for graph-parser
        self.norm_single, self.norm_local, self.norm_global, self.norm_hlocal = \
            [conf.output_normalizing==z for z in ["single", "local", "global", "hlocal"]]
        self.loss_prob, self.loss_hinge, self.loss_mr = [conf.tconf.loss_function==z for z in ["prob", "hinge", "mr"]]
        self.alg_proj, self.alg_unproj, self.alg_greedy = [conf.iconf.dec_algorithm==z for z in ["proj", "unproj", "greedy"]]
    #

    def build_decoder(self):
        conf = self.conf
        conf.sc_conf._input_dim = self.enc_output_dim
        conf.sc_conf._num_label = self.label_vocab.trg_len(False)
        return GraphScorer(self.pc, conf.sc_conf)

    # shared calculations for final scoring
    # -> the scores are masked with PRAC_MIN (by the scorer) for the paddings, but not handling diag here!

    def _prepare_score(self, insts, training):
        input_repr, enc_repr, jpos_pack, mask_arr = self.bter.run(insts, training)
        mask_expr = BK.input_real(mask_arr)
        # am_expr, ah_expr, lm_expr, lh_expr = self.scorer.transform_space(enc_repr)
        scoring_expr_pack = self.scorer.transform_space(enc_repr)
        return scoring_expr_pack, mask_expr, jpos_pack

    # scoring procedures
    def _score_arc_full(self, scoring_expr_pack, mask_expr, training, margin, gold_heads_expr=None):
        am_expr, ah_expr, _, _ = scoring_expr_pack
        # [BS, len-m, len-h]
        full_arc_score = self.scorer.score_arc_all(am_expr, ah_expr, mask_expr, mask_expr)
        # # set diag to small values # todo(warn): handled specifically in algorithms
        # maxlen = BK.get_shape(full_arc_score, 1)
        # full_arc_score += BK.diagflat(BK.constants([maxlen], Constants.REAL_PRAC_MIN))
        # margin?
        if training and margin>0.:
            full_arc_score = BK.minus_margin(full_arc_score, gold_heads_expr, margin)
        return full_arc_score

    def _score_label_full(self, scoring_expr_pack, mask_expr, training, margin, gold_heads_expr=None, gold_labels_expr=None):
        _, _, lm_expr, lh_expr = scoring_expr_pack
        # [BS, len-m, len-h, L]
        full_label_score = self.scorer.score_label_all(lm_expr, lh_expr, mask_expr, mask_expr)
        # # set diag to small values # todo(warn): handled specifically in algorithms
        # maxlen = BK.get_shape(full_label_score, 1)
        # full_label_score += BK.diagflat(BK.constants([maxlen], Constants.REAL_PRAC_MIN)).unsqueeze(-1)
        # margin? -- specially reshaping
        if training and margin>0.:
            full_shape = BK.get_shape(full_label_score)
            # combine last two dim
            combiend_score_expr = full_label_score.view(full_shape[:-2] + [-1])
            combined_idx_expr = gold_heads_expr * full_shape[-1] + gold_labels_expr
            combined_changed_score = BK.minus_margin(combiend_score_expr, combined_idx_expr, margin)
            full_label_score = combined_changed_score.view(full_shape)
        return full_label_score

    def _score_label_selected(self, scoring_expr_pack, mask_expr, training, margin, gold_heads_expr, gold_labels_expr=None):
        _, _, lm_expr, lh_expr = scoring_expr_pack
        # [BS, len-m, D]
        lh_expr_shape = BK.get_shape(lh_expr)
        selected_lh_expr = BK.gather(lh_expr, gold_heads_expr.unsqueeze(-1).expand(*lh_expr_shape), dim=len(lh_expr_shape)-2)
        # [BS, len-m, L]
        select_label_score = self.scorer.score_label_select(lm_expr, selected_lh_expr, mask_expr)
        # margin?
        if training and margin>0.:
            select_label_score = BK.minus_margin(select_label_score, gold_labels_expr, margin)
        return select_label_score

    # for global-norm + hinge(perceptron-like)-loss
    # [*, m, h, L], [*, m]
    def _losses_global_hinge(self, full_score_expr, gold_heads_expr, gold_labels_expr, pred_heads_expr, pred_labels_expr, mask_expr):
        return GraphParser.get_losses_global_hinge(full_score_expr, gold_heads_expr, gold_labels_expr, pred_heads_expr, pred_labels_expr, mask_expr)

    # ===== export
    @staticmethod
    def get_losses_global_hinge(full_score_expr, gold_heads_expr, gold_labels_expr, pred_heads_expr, pred_labels_expr, mask_expr, clamping=True):
        # combine the last two dimension
        full_shape = BK.get_shape(full_score_expr)
        # [*, m, h*L]
        last_size = full_shape[-1]
        combiend_score_expr = full_score_expr.view(full_shape[:-2] + [-1])
        # [*, m]
        gold_combined_idx_expr = gold_heads_expr * last_size + gold_labels_expr
        pred_combined_idx_expr = pred_heads_expr * last_size + pred_labels_expr
        # [*, m]
        gold_scores = BK.gather_one_lastdim(combiend_score_expr, gold_combined_idx_expr).squeeze(-1)
        pred_scores = BK.gather_one_lastdim(combiend_score_expr, pred_combined_idx_expr).squeeze(-1)
        # todo(warn): be aware of search error!
        # hinge_losses = BK.clamp(pred_scores - gold_scores, min=0.)  # this is previous version
        hinge_losses = pred_scores - gold_scores  # [*, len]
        if clamping:
            valid_losses = ((hinge_losses*mask_expr)[:, 1:].sum(-1) > 0.).float().unsqueeze(-1)  # [*, 1]
            return hinge_losses * valid_losses
        else:
            # for this mode, will there be problems of search error? Maybe rare.
            return hinge_losses
    # =====

    # for global-norm + prob-loss
    def _losses_global_prob(self, full_score_expr, gold_heads_expr, gold_labels_expr, marginals_expr, mask_expr):
        # combine the last two dimension
        full_shape = BK.get_shape(full_score_expr)
        last_size = full_shape[-1]
        # [*, m, h*L]
        combined_marginals_expr = marginals_expr.view(full_shape[:-2] + [-1])
        # # todo(warn): make sure sum to 1., handled in algorithm instead
        # combined_marginals_expr = combined_marginals_expr / combined_marginals_expr.sum(dim=-1, keepdim=True)
        # [*, m]
        gold_combined_idx_expr = gold_heads_expr * last_size + gold_labels_expr
        # [*, m, h, L]
        gradients = BK.minus_margin(combined_marginals_expr, gold_combined_idx_expr, 1.).view(full_shape)
        # the gradients on h are already 0. from the marginal algorithm
        gradients_masked = gradients * mask_expr.unsqueeze(-1).unsqueeze(-1) * mask_expr.unsqueeze(-2).unsqueeze(-1)
        # for the h-dimension, need to divide by the real length.
        # todo(warn): this values should be directly summed rather than averaged, since directly from loss
        fake_losses = (full_score_expr*gradients_masked).sum(-1).sum(-1)        # [BS, m]
        # todo(warn): be aware of search-error-like output constrains;
        #  but this clamp for all is not good for loss-prob, dealt at outside with unproj-mask.
        # <bad> fake_losses = BK.clamp(fake_losses, min=0.)
        return fake_losses

    # for single-norm: 0-1 loss
    # [*, L], [*], float
    def _losses_single(self, score_expr, gold_idxes_expr, single_sample, is_hinge=False, margin=0.):
        # expand the idxes to 0/1
        score_shape = BK.get_shape(score_expr)
        expanded_idxes_expr = BK.constants(score_shape, 0.)
        expanded_idxes_expr = BK.minus_margin(expanded_idxes_expr, gold_idxes_expr, -1.)        # minus -1 means +1
        # todo(+N): first adjust margin, since previously only minus margin for golds?
        if margin > 0.:
            adjusted_scores = margin + BK.minus_margin(score_expr, gold_idxes_expr, margin)
        else:
            adjusted_scores = score_expr
        # [*, L]
        if is_hinge:
            # multiply pos instances with -1
            flipped_scores = adjusted_scores * (1. - 2*expanded_idxes_expr)
            losses_all = BK.clamp(flipped_scores, min=0.)
        else:
            losses_all = BK.binary_cross_entropy_with_logits(adjusted_scores, expanded_idxes_expr, reduction='none')
        # special interpretation (todo(+2): there can be better implementation)
        if single_sample < 1.:
            # todo(warn): lower bound of sample_rate, ensure 2 samples
            real_sample_rate = max(single_sample, 2. / score_shape[-1])
        elif single_sample >= 2.:
            # including the positive one
            real_sample_rate = max(single_sample, 2.) / score_shape[-1]
        else:   # [1., 2.)
            real_sample_rate = single_sample
        #
        if real_sample_rate < 1.:
            sample_weight = BK.random_bernoulli(score_shape, real_sample_rate, 1.)
            # make sure positive is valid
            sample_weight = (sample_weight + expanded_idxes_expr.float()).clamp_(0., 1.)
            #
            final_losses = (losses_all*sample_weight).sum(-1) / sample_weight.sum(-1)
        else:
            final_losses = losses_all.mean(-1)
        return final_losses

    # =====
    # expr[BS, m, h, L], arr[BS] -> arr[BS, m]
    def _decode(self, full_score_expr, maske_expr, lengths_arr):
        if self.alg_unproj:
            return nmst_unproj(full_score_expr, maske_expr, lengths_arr, labeled=True, ret_arr=True)
        elif self.alg_proj:
            return nmst_proj(full_score_expr, maske_expr, lengths_arr, labeled=True, ret_arr=True)
        elif self.alg_greedy:
            return nmst_greedy(full_score_expr, maske_expr, lengths_arr, labeled=True, ret_arr=True)
        else:
            zfatal("Unknown decoding algorithm " + self.conf.iconf.dec_algorithm)
            return None

    # expr[BS, m, h, L], arr[BS] -> expr[BS, m, h, L]
    def _marginal(self, full_score_expr, maske_expr, lengths_arr):
        if self.alg_unproj:
            marginals_expr = nmarginal_unproj(full_score_expr, maske_expr, lengths_arr, labeled=True)
        elif self.alg_proj:
            marginals_expr = nmarginal_proj(full_score_expr, maske_expr, lengths_arr, labeled=True)
        else:
            zfatal("Unsupported marginal-calculation for the decoding algorithm of " + self.conf.iconf.dec_algorithm)
            marginals_expr = None
        return marginals_expr

    # ===== main methods: training and decoding
    # only getting scores
    def score_on_batch(self, insts: List[ParseInstance]):
        with BK.no_grad_env():
            self.refresh_batch(False)
            scoring_expr_pack, mask_expr, jpos_pack = self._prepare_score(insts, False)
            full_arc_score = self._score_arc_full(scoring_expr_pack, mask_expr, False, 0.)
            full_label_score = self._score_label_full(scoring_expr_pack, mask_expr, False, 0.)
            return full_arc_score, full_label_score

    # full score and inference
    def inference_on_batch(self, insts: List[ParseInstance], **kwargs):
        with BK.no_grad_env():
            self.refresh_batch(False)
            # ===== calculate
            scoring_expr_pack, mask_expr, jpos_pack = self._prepare_score(insts, False)
            full_arc_score = self._score_arc_full(scoring_expr_pack, mask_expr, False, 0.)
            full_label_score = self._score_label_full(scoring_expr_pack, mask_expr, False, 0.)
            # normalizing scores
            full_score = None
            final_exp_score = False         # whether to provide PROB by exp
            if self.norm_local and self.loss_prob:
                full_score = BK.log_softmax(full_arc_score, -1).unsqueeze(-1) + BK.log_softmax(full_label_score, -1)
                final_exp_score = True
            elif self.norm_hlocal and self.loss_prob:
                # normalize at m dimension, ignore each nodes's self-finish step.
                full_score = BK.log_softmax(full_arc_score, -2).unsqueeze(-1) + BK.log_softmax(full_label_score, -1)
            elif self.norm_single and self.loss_prob:
                if self.conf.iconf.dec_single_neg:
                    # todo(+2): add all-neg for prob explanation
                    full_arc_probs = BK.sigmoid(full_arc_score)
                    full_label_probs = BK.sigmoid(full_label_score)
                    fake_arc_scores = BK.log(full_arc_probs) - BK.log(1.-full_arc_probs)
                    fake_label_scores = BK.log(full_label_probs) - BK.log(1.-full_label_probs)
                    full_score = fake_arc_scores.unsqueeze(-1) + fake_label_scores
                else:
                    full_score = BK.logsigmoid(full_arc_score).unsqueeze(-1) + BK.logsigmoid(full_label_score)
                    final_exp_score = True
            else:
                full_score = full_arc_score.unsqueeze(-1) + full_label_score
            # decode
            mst_lengths = [len(z)+1 for z in insts]  # +=1 to include ROOT for mst decoding
            mst_heads_arr, mst_labels_arr, mst_scores_arr = self._decode(full_score, mask_expr, np.asarray(mst_lengths, dtype=np.int32))
            if final_exp_score:
                mst_scores_arr = np.exp(mst_scores_arr)
            # jpos prediction (directly index, no converting as in parsing)
            jpos_preds_expr = jpos_pack[2]
            has_jpos_pred = jpos_preds_expr is not None
            jpos_preds_arr = BK.get_value(jpos_preds_expr) if has_jpos_pred else None
            # ===== assign
            info = {"sent": len(insts), "tok": sum(mst_lengths)-len(insts)}
            mst_real_labels = self.pred2real_labels(mst_labels_arr)
            for one_idx, one_inst in enumerate(insts):
                cur_length = mst_lengths[one_idx]
                one_inst.pred_heads.set_vals(mst_heads_arr[one_idx][:cur_length])      # directly int-val for heads
                one_inst.pred_labels.build_vals(mst_real_labels[one_idx][:cur_length], self.label_vocab)
                one_inst.pred_par_scores.set_vals(mst_scores_arr[one_idx][:cur_length])
                if has_jpos_pred:
                    one_inst.pred_poses.build_vals(jpos_preds_arr[one_idx][:cur_length], self.bter.pos_vocab)
            return info

    # list(mini-batch) of annotated instances
    # optional results are written in-place? return info.
    def fb_on_batch(self, annotated_insts, training=True, loss_factor=1, **kwargs):
        self.refresh_batch(training)
        margin = self.margin.value
        # gold heads and labels
        gold_heads_arr, _ = self.predict_padder.pad([z.heads.vals for z in annotated_insts])
        gold_labels_arr, _ = self.predict_padder.pad([self.real2pred_labels(z.labels.idxes) for z in annotated_insts])
        gold_heads_expr = BK.input_idx(gold_heads_arr)  # [BS, Len]
        gold_labels_expr = BK.input_idx(gold_labels_arr)  # [BS, Len]
        # ===== calculate
        scoring_expr_pack, mask_expr, jpos_pack = self._prepare_score(annotated_insts, training)
        full_arc_score = self._score_arc_full(scoring_expr_pack, mask_expr, training, margin, gold_heads_expr)
        #
        final_losses = None
        if self.norm_local or self.norm_single:
            select_label_score = self._score_label_selected(scoring_expr_pack, mask_expr, training, margin,
                                                            gold_heads_expr, gold_labels_expr)
            # already added margin previously
            losses_heads = losses_labels = None
            if self.loss_prob:
                if self.norm_local:
                    losses_heads = BK.loss_nll(full_arc_score, gold_heads_expr)
                    losses_labels = BK.loss_nll(select_label_score, gold_labels_expr)
                elif self.norm_single:
                    single_sample = self.conf.tconf.loss_single_sample
                    losses_heads = self._losses_single(full_arc_score, gold_heads_expr, single_sample, is_hinge=False)
                    losses_labels = self._losses_single(select_label_score, gold_labels_expr, single_sample, is_hinge=False)
                # simply adding
                final_losses = losses_heads + losses_labels
            elif self.loss_hinge:
                if self.norm_local:
                    losses_heads = BK.loss_hinge(full_arc_score, gold_heads_expr)
                    losses_labels = BK.loss_hinge(select_label_score, gold_labels_expr)
                elif self.norm_single:
                    single_sample = self.conf.tconf.loss_single_sample
                    losses_heads = self._losses_single(full_arc_score, gold_heads_expr, single_sample, is_hinge=True, margin=margin)
                    losses_labels = self._losses_single(select_label_score, gold_labels_expr, single_sample, is_hinge=True, margin=margin)
                # simply adding
                final_losses = losses_heads + losses_labels
            elif self.loss_mr:
                # special treatment!
                probs_heads = BK.softmax(full_arc_score, dim=-1)        # [bs, m, h]
                probs_labels = BK.softmax(select_label_score, dim=-1)   # [bs, m, h]
                # select
                probs_head_gold = BK.gather_one_lastdim(probs_heads, gold_heads_expr).squeeze(-1)       # [bs, m]
                probs_label_gold = BK.gather_one_lastdim(probs_labels, gold_labels_expr).squeeze(-1)    # [bs, m]
                # root and pad will be excluded later
                # Reward = \sum_i 1.*marginal(GEdge_i); while for global models, need to gradient on marginal-functions
                # todo(warn): have problem since steps will be quite small, not used!
                final_losses = (mask_expr - probs_head_gold*probs_label_gold)           # let loss>=0
        elif self.norm_global:
            full_label_score = self._score_label_full(scoring_expr_pack, mask_expr, training, margin,
                                                      gold_heads_expr, gold_labels_expr)
            # for this one, use the merged full score
            full_score = full_arc_score.unsqueeze(-1) + full_label_score        # [BS, m, h, L]
            # +=1 to include ROOT for mst decoding
            mst_lengths_arr = np.asarray([len(z) + 1 for z in annotated_insts], dtype=np.int32)
            # do inference
            if self.loss_prob:
                marginals_expr = self._marginal(full_score, mask_expr, mst_lengths_arr)     # [BS, m, h, L]
                final_losses = self._losses_global_prob(full_score, gold_heads_expr, gold_labels_expr, marginals_expr, mask_expr)
                if self.alg_proj:
                    # todo(+N): deal with search-error-like problem, discard unproj neg losses (score>weighted-avg),
                    #  but this might be too loose, although the unproj edges are few?
                    gold_unproj_arr, _ = self.predict_padder.pad([z.unprojs for z in annotated_insts])
                    gold_unproj_expr = BK.input_real(gold_unproj_arr)  # [BS, Len]
                    comparing_expr = Constants.REAL_PRAC_MIN * (1. - gold_unproj_expr)
                    final_losses = BK.max_elem(final_losses, comparing_expr)
            elif self.loss_hinge:
                pred_heads_arr, pred_labels_arr, _ = self._decode(full_score, mask_expr, mst_lengths_arr)
                pred_heads_expr = BK.input_idx(pred_heads_arr)  # [BS, Len]
                pred_labels_expr = BK.input_idx(pred_labels_arr)  # [BS, Len]
                #
                final_losses = self._losses_global_hinge(full_score, gold_heads_expr, gold_labels_expr,
                                                         pred_heads_expr, pred_labels_expr, mask_expr)
            elif self.loss_mr:
                # todo(+N): Loss = -Reward = \sum marginals, which requires gradients on marginal-one-edge, or marginal-two-edges
                raise NotImplementedError("Not implemented for global-loss + mr.")
        elif self.norm_hlocal:
            # firstly label losses are the same
            select_label_score = self._score_label_selected(scoring_expr_pack, mask_expr, training, margin,
                                                            gold_heads_expr, gold_labels_expr)
            losses_labels = BK.loss_nll(select_label_score, gold_labels_expr)
            # then specially for arc loss
            children_masks_arr, _ = self.hlocal_padder.pad([z.get_children_mask_arr() for z in annotated_insts])
            children_masks_expr = BK.input_real(children_masks_arr)     # [bs, h, m]
            # [bs, h]
            # todo(warn): use prod rather than sum, but still only an approximation for the top-down
            # losses_arc = -BK.log(BK.sum(BK.softmax(full_arc_score, -2).transpose(-1, -2) * children_masks_expr, dim=-1) + (1-mask_expr))
            losses_arc = -BK.sum(BK.log_softmax(full_arc_score, -2).transpose(-1, -2) * children_masks_expr, dim=-1)
            # including the root-head is important
            losses_arc[:, 1] += losses_arc[:, 0]
            final_losses = losses_arc + losses_labels
        #
        # jpos loss? (the same mask as parsing)
        jpos_losses_expr = jpos_pack[1]
        if jpos_losses_expr is not None:
            final_losses += jpos_losses_expr
        # collect loss with mask, also excluding the first symbol of ROOT
        final_losses_masked = (final_losses * mask_expr)[:, 1:]
        final_loss_sum = BK.sum(final_losses_masked)
        # divide loss by what?
        num_sent = len(annotated_insts)
        num_valid_tok = sum(len(z) for z in annotated_insts)
        if self.conf.tconf.loss_div_tok:
            final_loss = final_loss_sum / num_valid_tok
        else:
            final_loss = final_loss_sum / num_sent
        #
        final_loss_sum_val = float(BK.get_value(final_loss_sum))
        info = {"sent": num_sent, "tok": num_valid_tok, "loss_sum": final_loss_sum_val}
        if training:
            BK.backward(final_loss, loss_factor)
        return info

# pdb:
# b tasks/zdpar/graph/parser:232
