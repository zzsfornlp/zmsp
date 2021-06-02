#

# direct sentence classification style: token-level trigger as evidence

import numpy as np
from typing import List
from msp.utils import Conf, Constants, zlog
from msp.nn import BK
from msp.nn.layers import Affine, NoDropRop, get_mlp
from msp.zext.ie import HLabelVocab
from .base import NodeExtractorConfBase, NodeExtractorBase

# todo(+N): a large problem is each type is independently predicted, there can be similar types (same L1) all triggered!!
#  or is it there are too few neg examples for tok?
#  maybe trigger-based ones should just softmax at each token's label dimension
# todo(+N): type L1 exclusive, since there are less chance of multiple types sharing L1???
# -> can reach 53+ after some tuning r0=5 r1=7 or 4/6 ...: train_skip_noevt_rate:0.$r0 c_evt_h.train_min_rate:0.$r1
# -> ok, maybe previously skip too much, but this mode is not the focus...
class NodeExtractorConfGene0(NodeExtractorConfBase):
    def __init__(self):
        super().__init__()
        self._input_dim = 0
        self._lexi_dim = 0  # todo(note): not used here!
        # todo(note): and no encoders here
        # decoder
        self.lambda_score_tok = 0.5  # <0 means auto adjusted by a gate
        self.score_sigmoid = True  # whether sigmoid on final score
        # prediction
        # sent-level label
        self.pred_sent_max_num = 5  # max number of types per sentence
        self.pred_sent_abs_thresh = 0.5  # sent-level prob absolute thresh
        # tok+label
        self.pred_tok_max_num = 4  # max triggers for type
        self.pred_tok_rel_thresh = 0.5  # trigger prob must be >=max-prob * this
        self.pred_tok_abs_thresh = 1.0  # trigger prob must be >=max-prob - this
        # training
        self.train_gold_corr = True  # totally use gold in training, and no prediction; otherwise pred+gold
        self.train_min_rate = 0.5  # training selecting rate for neg sents (selecting sent-L pair)
        self.positive_beta = 1.  # positive instances receive weights of (1+beta)
        self.lambda_tok = 0.5  # for tok trigger prediction (on slen+label dim)
        self.lambda_tok2 = 0.  # for tok trigger labeling (on label dim)
        self.lambda_sent = 1.  # for sent level ones

# extracting at sentence level (although still looking at token evidence)
class NodeExtractorGene0(NodeExtractorBase):
    def __init__(self, pc, conf: NodeExtractorConfGene0, vocab: HLabelVocab, extract_type: str):
        super().__init__(pc, conf, vocab, extract_type)
        # decoding
        # -----
        # the two parts: actually in biaffine attention forms
        # transform embeddings for attention match (token evidence)
        self.T_tok = self.add_sub_node("at", Affine(pc, conf.lab_conf.n_dim, conf._input_dim, init_rop=NoDropRop()))
        # transform embeddings for global match (sent evidence)
        self.T_sent = self.add_sub_node("as", Affine(pc, conf.lab_conf.n_dim, conf._input_dim, init_rop=NoDropRop()))
        # to be refreshed
        self.query_tok = None  # [L, D]
        self.query_sent = None  # [L, D]
        # -----
        # how to combine the two parts: fix lambda or dynamic gated (with the input features)
        self.lambda_score_tok = conf.lambda_score_tok
        if self.lambda_score_tok<0.:  # auto mode: using an MLP (make hidden size equal to input//4)
            self.score_gate = self.add_sub_node("mix", get_mlp(pc, [conf._input_dim]*4, 1, conf._input_dim, hidden_act="elu", final_act="sigmoid", final_init_rop=NoDropRop(), hidden_which_affine=3))
        else:
            self.score_gate = None

    def refresh(self, rop=None):
        super().refresh(rop)
        # first get the output embeddings, todo(note): currently do not use layered ones
        embed_tok = self.hl.get_embeddings(0, cascaded=False).transpose(0, 1).contiguous()
        embed_sent = self.hl.get_embeddings(1, cascaded=False)
        # tranform to the dim of input_expr, [L, D]
        self.query_tok = self.T_tok(embed_tok)
        self.query_sent = self.T_sent(embed_sent)

    # =====
    # scoring
    # [*, slen, D], [*, slen] -> att[*, slen, D, L], binary-score[*, slen, L]
    def _score(self, input_expr, input_mask, scores_aug_tok=None, scores_aug_sent=None):
        # token level attention and score
        # calculate the attention
        query_tok = self.query_tok  # [L, D]
        query_tok_t = query_tok.transpose(0, 1)  # [D, L]
        att_scores = BK.matmul(input_expr, query_tok_t)  # [*, slen, L]
        att_scores += (1.-input_mask).unsqueeze(-1) * Constants.REAL_PRAC_MIN
        if scores_aug_tok is not None:  # margin
            att_scores += scores_aug_tok
        attn = BK.softmax(att_scores, -2)  # [*, slen, L]
        score_tok = (att_scores * attn).sum(-2)  # [*, L]
        # token level labeling softmax
        attn2 = BK.softmax(att_scores.view(BK.get_shape(att_scores)[:-2]+[-1]), -1)  # [*, slen*L]
        # sent level score
        query_sent = self.query_sent  # [L, D]
        context_sent = input_expr[:, 0] + input_expr[:, -1]  # [*, D], simply adding the two ends
        score_sent = BK.matmul(context_sent, self.query_sent.transpose(0, 1))  # [*, L]
        # combine
        if self.lambda_score_tok < 0.:
            context_tok = BK.matmul(input_expr.transpose(-1, -2), attn).transpose(-1, -2).contiguous()  # [*, L, D]
            # 4*[*,L,D] -> [*, L]
            cur_lambda_score_tok = self.score_gate(
                [context_tok, query_tok.unsqueeze(0), context_sent.unsqueeze(-2), query_sent.unsqueeze(0)]).squeeze(-1)
        else:
            cur_lambda_score_tok = self.lambda_score_tok
        final_score = score_tok * cur_lambda_score_tok + score_sent * (1.-cur_lambda_score_tok)
        if scores_aug_sent is not None:
            final_score += scores_aug_sent
        if self.conf.score_sigmoid:  # margin
            final_score = BK.sigmoid(final_score)
        return final_score, attn, attn2  # [*, L], [*, slen, L], [*, slen*L]

    # =====

    def _predict(self, final_score, attn, attn2, input_mask):
        # attn[:, 0] = 0.  # not for artificial root # todo(note): input_mask should already exclude idx=0
        attn *= input_mask.unsqueeze(-1)  # not for invalid tokens
        conf = self.conf
        # get sentence level types
        pred_sent_max_num = conf.pred_sent_max_num
        pred_sent_abs_thresh = conf.pred_sent_abs_thresh
        # todo(note): kthvalue is not available at gpu and only support smallest!!
        # s_thresh00 = final_score.kthvalue(pred_sent_max_num, -1, keepdim=True)[0]  # [*, 1]
        s_thresh0 = final_score.topk(pred_sent_max_num, -1)[0].min(-1, keepdim=True)[0]  # [*, 1]
        s_thresh = s_thresh0.clamp(min=pred_sent_abs_thresh)
        sent_pred_mask = (final_score >= s_thresh).float()  # [*, L]
        # get token level as triggers (each type how many triggers && each token how many types)
        pred_tok_max_num = conf.pred_tok_max_num
        pred_tok_rel_thresh = conf.pred_tok_rel_thresh
        pred_tok_abs_thresh = conf.pred_tok_abs_thresh
        all_tok_pred_masks = []
        # for a, r in zip([attn, attn2], [-2, -1]):
        for a, r in zip([attn], [-2]):
            # t_thresh00 = attn.kthvalue(pred_tok_max_num, r, keepdim=True)[0]  # [*, 1, L]
            t_thresh0 = a.topk(pred_tok_max_num, r)[0].min(r, keepdim=True)[0]  # [*, 1, L]
            t_max_value = a.max(r, keepdim=True)[0]  # [*, 1, L]
            t_thresh1 = BK.max_elem(t_thresh0, t_max_value * pred_tok_rel_thresh)
            t_thresh = BK.max_elem(t_thresh1, t_max_value - pred_tok_abs_thresh)
            one_tok_pred_mask = (a >= t_thresh).float()  # [*, slen, L]
            if r == -1:
                one_tok_pred_mask = one_tok_pred_mask.view(BK.get_shape(attn))  # back to dim=3
            all_tok_pred_masks.append(one_tok_pred_mask)
        # put them together
        # final_pred_mask = all_tok_pred_masks[0] * all_tok_pred_masks[1] * (sent_pred_mask.unsqueeze(-2))  # [*, slen, L]
        final_pred_mask = all_tok_pred_masks[0] * (sent_pred_mask.unsqueeze(-2))  # [*, slen, L]
        final_pred_mask[:, :, 0] = 0.  # not for nil-type
        final_pred_mask *= input_mask.unsqueeze(-1)
        return final_pred_mask

    # from pred-mask to sel-idxes: [*, slen, L] -> [*, ?]
    def _pmask2idxes(self, pred_mask):
        orig_shape = BK.get_shape(pred_mask)
        dim_type = orig_shape[-1]
        flattened_mask = pred_mask.view(orig_shape[:-2]+[-1])  # [*, slen*L]
        f_idxes, sel_valid_mask = BK.mask2idx(flattened_mask)  # [*, max-count]
        # then back to the two dimensions
        sel_idxes, sel_lab_idxes = f_idxes // dim_type, f_idxes % dim_type
        # the embeddings
        sel_shape = BK.get_shape(sel_idxes)
        if sel_shape[-1] == 0:
            sel_lab_embeds = BK.zeros(sel_shape + [self.conf.lab_conf.n_dim])
        else:
            assert not self.hl.conf.use_lookup_soft, "Cannot do soft-lookup in this mode"
            sel_lab_embeds = self.hl.lookup(sel_lab_idxes)
        return sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds

    # prepare training mask: input_mask should be 1-dim less than gold_mask
    def _prepare_tmask(self, input_mask, gold_mask, trate):
        # todo(+3): currently simple sampling
        sel_mask = (BK.rand(BK.get_shape(gold_mask)) < trate).float()
        # add gold and exclude pad
        sel_mask += gold_mask
        sel_mask.clamp_(max=1.)
        if input_mask.dim() < sel_mask.dim():
            input_mask = input_mask.unsqueeze(-1)
        sel_mask *= input_mask
        return sel_mask

    def predict(self, insts: List, input_lexi, input_expr, input_mask):
        input_mask[:, 0] = 0.  # no artificial root
        final_score, attn, attn2 = self._score(input_expr, input_mask)
        pred_mask = self._predict(final_score, attn, attn2, input_mask)  # [*, slen, L]
        sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds = self._pmask2idxes(pred_mask)
        all_logprobs = final_score.log().unsqueeze(-2) + (attn + 1e-10).log()  # [*, slen, L]
        bsize = len(insts)
        sel_lab_logprobs = all_logprobs[BK.arange_idx(bsize).unsqueeze(-1), sel_idxes, sel_lab_idxes]  # [*, ?]
        return sel_lab_logprobs, sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds

    def loss(self, insts: List, input_lexi, input_expr, input_mask, margin=0.):
        conf = self.conf
        bsize = len(insts)
        # first get gold info, also multiple valid-masks
        gold_masks, _, gold_items_arr, gold_valid = self.batch_inputs_g0(insts)  # [*, slen, L]
        input_mask = input_mask * gold_valid.unsqueeze(-1)  # [*, slen]
        input_mask[:, 0] = 0.  # no artificial root
        # gold masks
        gold_mask_tok = gold_masks  # [*, slen, L]
        gold_mask_tok2 = gold_masks.sum(-1).clamp(max=1.)  # [*, slen]
        gold_mask_sent = gold_masks.sum(-2).clamp(max=1.)  # [*, L]
        # step 1: scoring
        if margin>0.:
            # prepare margin: -1 for pos and 1 for neg
            scores_aug_tok = -2 * gold_mask_tok + 1.
            scores_aug_sent = -2 * gold_mask_sent + 1.
        else:
            scores_aug_tok = None
            scores_aug_sent = None
        # the output should be after sigmoid / softmax
        final_score, attn, attn2 = self._score(input_expr, input_mask, scores_aug_tok, scores_aug_sent)  # [*, L], [*, slen, L]
        # step 2: collect loss
        # todo(+3): different loss functions, currently all use (margin-)softmax
        tmask_sent = self._prepare_tmask(gold_valid, gold_mask_sent, conf.train_min_rate)  # [*, L]
        tmask_sent[:, 0] = 0.  # no nil
        # 2.1 tok based ones
        # sum of log, we want every trigger to be large, only non-zero for has-type sents
        loss_tok1 = - (attn + 1e-10).log() * gold_mask_tok
        loss_tok_sum = loss_tok1.sum() * conf.lambda_tok
        loss_tok_count = gold_mask_tok.sum() + 1e-5
        # 2.1.5 tok based ones for each token
        # =====
        # nil_aug_gold_mask_tok = BK.copy(gold_mask_tok)
        # nil_aug_gold_mask_tok[:, :, 0] += (1. - gold_mask_tok2)  # [*, slen, L]
        # tmask_tok2 = self._prepare_tmask(input_mask, gold_mask_tok2, conf.train_min_rate)  # [*, slen]
        # nil_aug_gold_mask_tok *= tmask_tok2.unsqueeze(-1)
        # loss_tok2 = - (attn2 + 1e-10).log() * nil_aug_gold_mask_tok
        # loss_tok2_sum = loss_tok2.sum() * conf.lambda_tok2
        # loss_tok2_count = nil_aug_gold_mask_tok.sum() + 1e-5
        # =====
        loss_tok2 = - (attn2 + 1e-10).log().view(BK.get_shape(attn)) * gold_mask_tok
        loss_tok2_sum = loss_tok2.sum() * conf.lambda_tok2
        loss_tok2_count = gold_mask_tok.sum() + 1e-5
        # 2.1 sent based ones
        loss_sent = BK.binary_cross_entropy(final_score, gold_mask_sent, reduction='none')  # [*, L]
        loss_sent_sum = (loss_sent * (tmask_sent + gold_mask_sent*conf.positive_beta)).sum() * conf.lambda_sent
        loss_sent_count = tmask_sent.sum()
        # return separately since divided by different things
        ret_losses = [[loss_tok_sum, loss_tok_count], [loss_tok2_sum, loss_tok2_count], [loss_sent_sum, loss_sent_count]]
        # step 3: return (similar to lookup)
        if conf.train_gold_corr:
            pred_mask = gold_masks
        else:
            pred_mask = self._predict(final_score, attn, attn2, input_mask)  # [*, slen, L]
            # add gold
            pred_mask += gold_masks
            pred_mask.clamp_(max=1.)
        sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds = self._pmask2idxes(pred_mask)
        ret_items = gold_items_arr[np.arange(bsize)[:, np.newaxis], BK.get_value(sel_idxes), BK.get_value(sel_lab_idxes)]
        return ret_losses, ret_items, sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds

    def lookup(self, insts: List, input_lexi, input_expr, input_mask):
        bsize = len(insts)
        # get gold or pre-set ones, again [*, slen, L] -> [*, mc]
        gold_masks, _, gold_items_arr, gold_valid = self.batch_inputs_g0(insts)
        sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds = self._pmask2idxes(gold_masks)
        ret_items = gold_items_arr[np.arange(bsize)[:, np.newaxis], BK.get_value(sel_idxes), BK.get_value(sel_lab_idxes)]
        return ret_items, sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds
