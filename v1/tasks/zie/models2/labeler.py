#

# the slightly more complex labeler (corresponding to hlvocab)

# the hierarchical labeling system
# todo(note): the labeling str format is L1.L2.L3..., each layer has only 26-characters

from typing import List, Tuple, Iterable
import numpy as np

from msp.utils import Conf, Random, zlog
from msp.nn import BK
from msp.nn.layers import BasicNode, Affine, Embedding, Dropout, NoDropRop

from msp.zext.ie import HLabelIdx, HLabelVocab

#
class HLabelNodeConf(Conf):
    def __init__(self):
        self.n_dim = 300
        # modeling
        self.pool_init_hint = False  # init the pools with the lexicon hints' pre-trained embeddings
        self.tie_embeds = False  # tie input/output embeddings
        self.nodrop_pred_embeds = True  # no dropout for pred embeds (also True for lookup if tied)
        self.nodrop_lookup_embeds = True  # no dropout for lookup embeds (only effective if not tied)
        self.strategy_predict = "sum"  # currently only support sum
        self.strategy_lookup = "sum"  # add/ff
        self.bias_predict = True  # add bias for prediction
        self.zero_nil = True  # make layered-scores of NIL(None) to be zero
        # training (padding 1. for loss, 0. for margin)
        self.loss_lambdas = []
        self.margin_lambdas = [1., 1., 1.]  # currently with 1.
        self.loss_function = "prob"  # hinge/prob
        # INS CS loss as in P19-1521
        self.loss_prob_entropy_lambda = 0.  # the special extra loss: only effective for prob loss, 0. means nope
        self.loss_prob_reweight = False
        # others
        self.loss_fullnil_weight = 1.  # down-sampling full-nill?
        # different margins for different y/yt
        # if at the current layer, NIL is the gold, further multiple margin by this special scale
        self.margin_lambda_P = 0.  # false positive: gold==nil and pred!=nil
        self.margin_lambda_R = 0.  # false negative: gold!=nil and pred==nil
        self.margin_lambda_T = 0.  # wrong label: gold!=nil and pred!=nil and gold!=pred
        # if gold is the max after margin, then no loss.
        # This is implicitly for hinge, since gold will minus by itself,
        # but let us see if it is useful for prob (this is similar to `no_loss_satisfy_margin`)
        self.no_loss_max_gold = False
        # byproduct of prediction
        self.use_lookup_soft = False  # use soft lookup in predicting
        self.lookup_soft_alphas = []  # alphas for soft lookup

    def do_validate(self):
        self.loss_lambdas = [float(z) for z in self.loss_lambdas]
        self.margin_lambdas = [float(z) for z in self.margin_lambdas]

# module
# todo(note): if using linear, then it is the same to add embeddings or add scores, currently choose to add scores
# todo(+N): currently only consider single-class for each trigger
class HLabelNode(BasicNode):
    # pooled embeddings -> label embeddings for different layers -> add together
    # also optionally init the pooled embeddings with pre-trained embeddings
    def __init__(self, pc, conf: HLabelNodeConf, hl_vocab: HLabelVocab, eff_max_layer=None):
        super().__init__(pc, None, None)
        self.conf = conf
        self.hl_vocab = hl_vocab
        assert self.hl_vocab.nil_as_zero  # for each layer, the idx=0 is the full-NIL
        # basic pool embeddings
        npvec = hl_vocab.pool_init_vec
        if not conf.pool_init_hint:
            npvec = None
        else:
            assert npvec is not None, "pool-init not provided by the Vocab!"
        n_dim, n_pool = conf.n_dim, len(hl_vocab.pools_k)
        self.pool_pred = self.add_sub_node("pp", Embedding(pc, n_pool, n_dim, fix_row0=conf.zero_nil, npvec=npvec,
                                                           init_rop=(NoDropRop() if conf.nodrop_pred_embeds else None)))
        if conf.tie_embeds:
            self.pool_lookup = self.pool_pred
        else:
            self.pool_lookup = self.add_sub_node("pl", Embedding(pc, n_pool, n_dim, fix_row0=conf.zero_nil, npvec=npvec,
                                                                 init_rop=(NoDropRop() if conf.nodrop_lookup_embeds else None)))
        # layered labels embeddings (to be refreshed)
        self.max_layer = hl_vocab.max_layer
        self.layered_embeds_pred = [None] * self.max_layer
        self.layered_embeds_lookup = [None] * self.max_layer
        self.layered_prei = [None] * self.max_layer  # previous layer i, for score combining
        self.layered_isnil = [None] * self.max_layer  # whether is nil(None)
        self.zero_nil = conf.zero_nil
        # lookup summer
        assert conf.strategy_predict == "sum"
        self.lookup_is_sum, self.lookup_is_ff = [conf.strategy_lookup == z for z in ["sum", "ff"]]
        if self.lookup_is_ff:
            self.lookup_summer = self.add_sub_node("summer", Affine(pc, [n_dim]*self.max_layer, n_dim, act="tanh"))
        elif self.lookup_is_sum:
            self.sum_dropout = self.add_sub_node("sdrop", Dropout(pc, (n_dim,)))
            self.lookup_summer = lambda embeds: self.sum_dropout(BK.stack(embeds, 0).sum(0))
        else:
            raise NotImplementedError(f"UNK strategy_lookup: {conf.strategy_lookup}")
        # bias for prediction
        self.prediction_sizes = [len(hl_vocab.layered_pool_links_padded[i]) for i in range(self.max_layer)]
        if conf.bias_predict:
            self.biases_pred = [self.add_param(name="B", shape=(x, )) for x in self.prediction_sizes]
        else:
            self.biases_pred = [None] * self.max_layer
        # =====
        # training
        self.is_hinge_loss, self.is_prob_loss = [conf.loss_function==z for z in ["hinge", "prob"]]
        self.loss_lambdas = conf.loss_lambdas + [1.]*(self.max_layer-len(conf.loss_lambdas))  # loss scale
        self.margin_lambdas = conf.margin_lambdas + [0.]*(self.max_layer-len(conf.margin_lambdas))  # margin scale
        self.lookup_soft_alphas = conf.lookup_soft_alphas + [1.]*(self.max_layer-len(conf.lookup_soft_alphas))
        self.loss_fullnil_weight = conf.loss_fullnil_weight
        # ======
        # set current effective max_layer
        self.eff_max_layer = self.max_layer
        if eff_max_layer is not None:
            self.set_eff_max_layer(eff_max_layer)

    # todo(note): should start from idx=1 since idx=0 will include nothing
    def set_eff_max_layer(self, eff_max_layer=None):
        if eff_max_layer is None:
            return self.eff_max_layer  # simply query
        if eff_max_layer<0:
            eff_max_layer = self.max_layer + 1 + eff_max_layer
        assert eff_max_layer>0 and eff_max_layer<=self.max_layer, f"Err: layer our of range {eff_max_layer}"
        if self.eff_max_layer != eff_max_layer:
            zlog(f"Set current layer from {self.eff_max_layer} -> {eff_max_layer}!")
            self.eff_max_layer = eff_max_layer
        return self.eff_max_layer

    def refresh(self, rop=None):
        super().refresh(rop)
        # no need to fix0 for None since already done in the Embedding
        # refresh layered embeddings (in training, we should not be in no-grad mode)
        # todo(note): here, there can be dropouts
        layered_prei_arrs = self.hl_vocab.layered_prei
        layered_pool_links_padded_arrs = self.hl_vocab.layered_pool_links_padded
        layered_pool_links_mask_arrs = self.hl_vocab.layered_pool_links_mask
        layered_isnil = self.hl_vocab.layered_pool_isnil
        for i in range(self.max_layer):
            # [N, ?, D] -> [N, D] -> [D, N]
            self.layered_embeds_pred[i] = (BK.input_real(layered_pool_links_mask_arrs[i]).unsqueeze(-1)
                                           * self.pool_pred(layered_pool_links_padded_arrs[i])).sum(-2).transpose(0, 1).contiguous()
            # [N, ?, D] -> [N, D]
            self.layered_embeds_lookup[i] = (BK.input_real(layered_pool_links_mask_arrs[i]).unsqueeze(-1)
                                             * self.pool_lookup(layered_pool_links_padded_arrs[i])).sum(-2)
            # [?] of idxes/masks
            self.layered_prei[i] = BK.input_idx(layered_prei_arrs[i])
            self.layered_isnil[i] = BK.input_real(layered_isnil[i])  # is nil mask

    # =====
    # useful for outside prediction

    # which_set: 0 for pred, 1 for lookup; cascaded means summing up
    def get_embeddings(self, which_set, cascaded=True):
        layered_embeds = [self.layered_embeds_pred, self.layered_embeds_lookup][which_set]
        if cascaded:
            prev_embed = layered_embeds[0]
            for i in range(1, self.eff_max_layer):
                cur_embed = layered_embeds[i]
                cur_prei = self.layered_prei[i]
                prev_embed = cur_embed + prev_embed[cur_prei]
            return prev_embed
        else:
            return layered_embeds[self.eff_max_layer-1]  # [?, D]

    # [*, D] -> List[(*, ?)]
    def _raw_scores(self, input_expr):
        all_scores = []
        for i in range(self.eff_max_layer):
            # first, the scores of the current layer; here no dropout!
            pred_w, pred_b = self.layered_embeds_pred[i], self.biases_pred[i]  # [?, D], [?]
            cur_score = BK.matmul(input_expr, pred_w)  # [*, ?]
            if pred_b is not None:
                cur_score += pred_b
            # apply None mask (make it score 0., must be before adding prev)
            if self.zero_nil:
                cur_score *= (1. - self.layered_isnil[i])  # make it zero for NIL(None) types
            all_scores.append(cur_score)
        return all_scores

    # adding previous scores, split into two procedures for the convenience of margin
    def _cascade_scores(self, raw_scores: List):
        all_scores = []
        for i, one_scores in enumerate(raw_scores):
            if i>0:
                cur_prei = self.layered_prei[i]  # [?]
                prev_score = BK.select(all_scores[-1], cur_prei, -1)  # [*, ?]
                cur_score = prev_score + one_scores
            else:
                cur_score = one_scores
            all_scores.append(cur_score)
        return all_scores

    # for convenient
    def _score(self, input_expr):
        return self._cascade_scores(self._raw_scores(input_expr))

    # [*, D]
    # todo(note): here simply max or force-select on the final layer
    def predict(self, input_expr, force_idxes, return_final_scores=False):
        # scoring
        all_scores = self._score(input_expr)
        if return_final_scores:
            return all_scores[self.eff_max_layer-1]  # [*, D]
        else:
            return self._predict(all_scores, force_idxes)  # 2x[*, D], [*, DEmb]

    def _predict(self, all_scores, force_idxes):
        # predicting on the last one
        last_score = BK.log_softmax(all_scores[self.eff_max_layer-1], -1)  # [*, ?]
        if force_idxes is None:
            # todo(note): currently only do max
            res_logprobs, res_idxes = last_score.max(-1)  # [*]
        else:
            res_idxes = force_idxes
            res_logprobs = last_score.gather(-1, res_idxes.unsqueeze(-1)).squeeze(-1)  # [*]
        # lookup: [*, D]
        conf = self.conf
        if conf.use_lookup_soft:
            ret_lab_embeds = self.lookup_soft(all_scores)
        else:
            ret_lab_embeds = self.lookup(res_idxes)
        return res_logprobs, res_idxes, ret_lab_embeds

    #
    def _get_all_idxes(self, input_idxes):
        all_idxes = []
        cur_idxes = input_idxes  # [*]
        for i in range(self.eff_max_layer-1, -1, -1):  # reversely
            all_idxes.append(cur_idxes)
            cur_idxes = self.layered_prei[i][cur_idxes]  # [*]
        all_idxes.reverse()
        return all_idxes

    # todo(note): this lookup means the looking-up procedure for Embedding-like...
    def lookup(self, lookup_idxes):
        all_embeds = []
        all_idxes = self._get_all_idxes(lookup_idxes)
        for i in range(self.eff_max_layer):
            cur_embeds = self.layered_embeds_lookup[i][all_idxes[i]]  # [*, D]
            all_embeds.append(cur_embeds)
        ret_embed = self.lookup_summer(all_embeds)
        return ret_embed

    def lookup_soft(self, cascade_scores: List):
        all_embeds = []
        for i in range(self.eff_max_layer):
            cur_scores = cascade_scores[i] * self.lookup_soft_alphas[i]  # [*, ?]
            cur_embeds = BK.matmul(cur_scores, self.layered_embeds_lookup[i])  # [*, D]
            all_embeds.append(cur_embeds)
        ret_embed = self.lookup_summer(all_embeds)
        return ret_embed

    # [*, D], [*], [*]; todo(note): here no option for replace with gold!
    def loss(self, input_expr, loss_mask, gold_idxes, margin=0.):
        gold_all_idxes = self._get_all_idxes(gold_idxes)
        # scoring
        raw_scores = self._raw_scores(input_expr)
        raw_scores_aug = []
        margin_P, margin_R, margin_T = self.conf.margin_lambda_P, self.conf.margin_lambda_R, self.conf.margin_lambda_T
        #
        gold_shape = BK.get_shape(gold_idxes)  # [*]
        gold_bsize_prod = np.prod(gold_shape)
        # gold_arange_idxes = BK.arange_idx(gold_bsize_prod)
        # margin
        for i in range(self.eff_max_layer):
            cur_gold_inputs = gold_all_idxes[i]
            # add margin
            cur_scores = raw_scores[i]  # [*, ?]
            cur_margin = margin * self.margin_lambdas[i]
            if cur_margin > 0.:
                cur_num_target = self.prediction_sizes[i]
                cur_isnil = self.layered_isnil[i].byte()  # [NLab]
                cost_matrix = BK.constants([cur_num_target, cur_num_target], margin_T)  # [gold, pred]
                cost_matrix[cur_isnil, :] = margin_P
                cost_matrix[:, cur_isnil] = margin_R
                diag_idxes = BK.arange_idx(cur_num_target)
                cost_matrix[diag_idxes, diag_idxes] = 0.
                margin_mat = cost_matrix[cur_gold_inputs]
                cur_aug_scores = cur_scores + margin_mat  # [*, ?]
            else:
                cur_aug_scores = cur_scores
            raw_scores_aug.append(cur_aug_scores)
        # cascade scores
        final_scores = self._cascade_scores(raw_scores_aug)
        # loss weight, todo(note): asserted self.hl_vocab.nil_as_zero before
        loss_weights = ((gold_idxes==0).float() * (self.loss_fullnil_weight - 1.) + 1.) if self.loss_fullnil_weight<1. else 1.
        # calculate loss
        loss_prob_entropy_lambda = self.conf.loss_prob_entropy_lambda
        loss_prob_reweight = self.conf.loss_prob_reweight
        final_losses = []
        no_loss_max_gold = self.conf.no_loss_max_gold
        if loss_mask is None:
            loss_mask = BK.constants(BK.get_shape(input_expr)[:-1], 1.)
        for i in range(self.eff_max_layer):
            cur_final_scores, cur_gold_inputs = final_scores[i], gold_all_idxes[i]  # [*, ?], [*]
            # collect the loss
            if self.is_hinge_loss:
                cur_pred_scores, cur_pred_idxes = cur_final_scores.max(-1)
                cur_gold_scores = BK.gather(cur_final_scores, cur_gold_inputs.unsqueeze(-1), -1).squeeze(-1)
                cur_loss = cur_pred_scores - cur_gold_scores  # [*], todo(note): this must be >=0
                if no_loss_max_gold:  # this should be implicit
                    cur_loss = cur_loss * (cur_loss>0.).float()
            elif self.is_prob_loss:
                # cur_loss = BK.loss_nll(cur_final_scores, cur_gold_inputs)  # [*]
                cur_loss = self._my_loss_prob(cur_final_scores, cur_gold_inputs,
                                              loss_prob_entropy_lambda, loss_mask, loss_prob_reweight)  # [*]
                if no_loss_max_gold:
                    cur_pred_scores, cur_pred_idxes = cur_final_scores.max(-1)
                    cur_gold_scores = BK.gather(cur_final_scores, cur_gold_inputs.unsqueeze(-1), -1).squeeze(-1)
                    cur_loss = cur_loss * (cur_gold_scores>cur_pred_scores).float()
            else:
                raise NotImplementedError(f"UNK loss {self.conf.loss_function}")
            # here first summing up, divided at the outside
            one_loss_sum = (cur_loss * (loss_mask * loss_weights)).sum() * self.loss_lambdas[i]
            final_losses.append(one_loss_sum)
        # final sum
        final_loss_sum = BK.stack(final_losses).sum()
        _, ret_lab_idxes, ret_lab_embeds = self._predict(final_scores, None)
        return [[final_loss_sum, loss_mask.sum()]], ret_lab_idxes, ret_lab_embeds

    # special loss function
    def _my_loss_prob(self, score_expr, gold_idxes_expr, entropy_lambda: float, loss_mask, neg_reweight: bool):
        probs = BK.softmax(score_expr, -1)  # [*, NLab]
        log_probs = BK.log(probs + 1e-8)
        # first plain NLL loss
        nll_loss = -BK.gather_one_lastdim(log_probs, gold_idxes_expr).squeeze(-1)
        # next the special loss
        if entropy_lambda>0.:
            negative_entropy = probs * log_probs  # [*, NLab]
            last_dim = BK.get_shape(score_expr, -1)
            confusion_matrix = 1. - BK.eye(last_dim)  # [Nlab, Nlab]
            entropy_mask = confusion_matrix[gold_idxes_expr]  # [*, Nlab]
            entropy_loss = (negative_entropy * entropy_mask).sum(-1)
            final_loss = nll_loss + entropy_lambda * entropy_loss
        else:
            final_loss = nll_loss
        # reweight?
        if neg_reweight:
            golden_prob = BK.gather_one_lastdim(probs, gold_idxes_expr).squeeze(-1)
            is_full_nil = (gold_idxes_expr==0.).float()
            not_full_nil = 1. - is_full_nil
            count_pos = (loss_mask * not_full_nil).sum()
            count_neg = (loss_mask * is_full_nil).sum()
            prob_pos = (loss_mask * not_full_nil * golden_prob).sum()
            prob_neg = (loss_mask * is_full_nil * golden_prob).sum()
            neg_weight = prob_pos / (count_pos + count_neg - prob_neg + 1e-8)
            final_weights = not_full_nil + is_full_nil * neg_weight
            # todo(note): final mask will be applied at outside
            final_loss = final_loss * final_weights
        return final_loss
