#

# Selection Based Head Extractor

import numpy as np
from typing import List
from copy import deepcopy

from msp.utils import Conf, Constants, zlog
from msp.nn import BK
from msp.nn.layers import Affine, BasicNode, get_mlp, NoDropRop, Embedding, RelPosiEmbedding, Dropout
from msp.nn.modules import EncConf, MyEncoder
from msp.zext.ie import HLabelVocab
from .base import NodeExtractorConfBase, NodeExtractorBase, HLabelNode

#
class NodeExtractorConfHead(NodeExtractorConfBase):
    def __init__(self):
        super().__init__()
        self._input_dim = 0
        self._lexi_dim = 0
        # the first pass selector
        self.use_selector = False
        self.sel_conf = NodeSelectorConf()
        # the specific encoder
        # basic modeling mode 1) especially model for each cand position (DMXNN); 2) one general encoder
        self.dmxnn = False
        # specific to dmxnn mode
        self.posi_dim = 5  # dimension for PF
        self.posi_cut = 20  # [-cut, cut]
        # encoder
        self.e_enc = EncConf().init_from_kwargs(enc_hidden=300, enc_cnn_layer=0, enc_cnn_windows='[3]',
                                                enc_rnn_layer=0, no_final_dropout=True)
        # before labeler
        self.use_lab_f = True
        self.lab_f_use_lexi = False  # also include lexi repr for lab_f
        self.lab_f_act = "elu"
        # whether exclude nil
        self.exclude_nil = True
        # selection for training (only if use_selector=False)
        self.train_ng_ratio = 1.  # neg sample's ratio to gold's count
        self.train_min_num = 1.  # min number per sentence
        self.train_min_rate = 0.5  # a min selecting rate for neg sentences
        self.train_min_rate_s2 = 0.5  # sampling for the secondary type, within gold_mask1
        # correct labels for training return
        self.train_gold_corr = True
        # loss weights for NodeSelector and NodeExtractor
        self.lambda_ns = 0.5  # if use_selector=True
        self.lambda_ne = 1.
        self.lambda_ne2 = 1.  # secondary type
        # secondary type
        self.use_secondary_type = False  # the second type for each trigger
        self.sectype_reuse_hl = False  # use the same predictor?
        self.sectype_t2ift1 = True  # only output t2 if there are t1
        self.sectype_noback_enc = False  # stop grad to enc

# extracting single input elem as the head
class NodeExtractorHead(NodeExtractorBase):
    def __init__(self, pc, conf: NodeExtractorConfHead, vocab: HLabelVocab, extract_type: str):
        super().__init__(pc, conf, vocab, extract_type)
        # node selector
        conf.sel_conf._input_dim = conf._input_dim  # make dims fit
        self.sel: NodeSelector = self.add_sub_node("sel", NodeSelector(pc, conf.sel_conf))
        # encoding
        self.dmxnn = conf.dmxnn
        self.posi_embed = self.add_sub_node("pe", RelPosiEmbedding(pc, conf.posi_dim, max=conf.posi_cut))
        if self.dmxnn:
            conf.e_enc._input_dim = conf._input_dim + conf.posi_dim
        else:
            conf.e_enc._input_dim = conf._input_dim
        self.e_encoder = self.add_sub_node("ee", MyEncoder(pc, conf.e_enc))
        e_enc_dim = self.e_encoder.get_output_dims()[0]
        # decoding
        # todo(note): dropout after pooling; todo(+N): cannot go to previous layers if there are no encoders
        self.special_drop = self.add_sub_node("sd", Dropout(pc, (e_enc_dim,)))
        self.use_lab_f = conf.use_lab_f
        self.lab_f_use_lexi = conf.lab_f_use_lexi
        if self.use_lab_f:
            lab_f_input_dims = [e_enc_dim]*3 if self.dmxnn else [e_enc_dim]
            if self.lab_f_use_lexi:
                lab_f_input_dims.append(conf._lexi_dim)
            self.lab_f = self.add_sub_node("lab", Affine(pc, lab_f_input_dims, conf.lab_conf.n_dim, act=conf.lab_f_act))
        else:
            self.lab_f = lambda x: x[0]  # only use the first one
        # secondary type
        self.use_secondary_type = conf.use_secondary_type
        if self.use_secondary_type:
            # todo(note): re-use vocab; or totally reuse the predictor?
            if conf.sectype_reuse_hl:
                self.hl2: HLabelNode = self.hl
            else:
                new_lab_conf = deepcopy(conf.lab_conf)
                new_lab_conf.zero_nil = False  # todo(note): not zero_nil here!
                self.hl2: HLabelNode = self.add_sub_node("hl", HLabelNode(pc, new_lab_conf, vocab))
            # enc+t1 -> t2
            self.t1tot2 = self.add_sub_node("1to2", Embedding(pc, self.hl_output_size, conf.lab_conf.n_dim))
        else:
            self.hl2 = None
            self.t1tot2 = None

    # encoding
    # the main modeling part, returning the repr right before labeling
    # enc: [*, slen, D], [*, slen]; cand: [*, ?] -> [*, ?, DLab]
    def _enc(self, input_lexi, input_expr, input_mask, sel_idxes):
        if self.dmxnn:
            bsize, slen = BK.get_shape(input_mask)
            if sel_idxes is None:
                sel_idxes = BK.arange_idx(slen).unsqueeze(0)  # select all, [1, slen]
            ncand = BK.get_shape(sel_idxes, -1)
            # enc_expr aug with PE
            rel_dist = BK.arange_idx(slen).unsqueeze(0).unsqueeze(0) - sel_idxes.unsqueeze(-1)  # [*, ?, slen]
            pe_embeds = self.posi_embed(rel_dist)  # [*, ?, slen, Dpe]
            aug_enc_expr = BK.concat([pe_embeds.expand(bsize, -1, -1, -1),
                                      input_expr.unsqueeze(1).expand(-1, ncand, -1, -1)], -1)  # [*, ?, slen, D+Dpe]
            # [*, ?, slen, Denc]
            hidden_expr = self.e_encoder(aug_enc_expr.view(bsize * ncand, slen, -1),
                                         input_mask.unsqueeze(1).expand(-1, ncand, -1).contiguous().view(bsize*ncand, slen))
            hidden_expr = hidden_expr.view(bsize, ncand, slen, -1)
            # dynamic max-pooling (dist<0, dist=0, dist>0)
            NEG = Constants.REAL_PRAC_MIN
            mp_hiddens = []
            mp_masks = [rel_dist < 0, rel_dist == 0, rel_dist > 0]
            for mp_mask in mp_masks:
                float_mask = mp_mask.float() * input_mask.unsqueeze(-2)  # [*, ?, slen]
                valid_mask = (float_mask.sum(-1) > 0.).float().unsqueeze(-1)  # [*, ?, 1]
                mask_neg_val = (1. - float_mask).unsqueeze(-1) * NEG  # [*, ?, slen, 1]
                # todo(+2): or do we simply multiply mask?
                mp_hid0 = (hidden_expr + mask_neg_val).max(-2)[0]
                mp_hid = mp_hid0 * valid_mask  # [*, ?, Denc]
                mp_hiddens.append(self.special_drop(mp_hid))
                # mp_hiddens.append(mp_hid)
            final_hiddens = mp_hiddens
        else:
            hidden_expr = self.e_encoder(input_expr, input_mask)  # [*, slen, D']
            if sel_idxes is None:
                hidden_expr1 = hidden_expr
            else:
                hidden_expr1 = BK.gather_first_dims(hidden_expr, sel_idxes, -2)  # [*, ?, D']
            final_hiddens = [self.special_drop(hidden_expr1)]
        if self.lab_f_use_lexi:
            final_hiddens.append(BK.gather_first_dims(input_lexi, sel_idxes, -2))  # [*, ?, DLex]
        ret_expr = self.lab_f(final_hiddens)  # [*, ?, DLab]
        return ret_expr

    # return the same things as input, but remove nil ones
    def _exclude_nil(self, sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds, sel_logprobs=None, sel_items_arr=None):
        # todo(note): assure that nil is 0
        sel_valid_mask = sel_valid_mask * (sel_lab_idxes != 0).float()  # not inplaced
        # idx on idx
        s2_idxes, s2_valid_mask = BK.mask2idx(sel_valid_mask)
        sel_idxes = sel_idxes.gather(-1, s2_idxes)
        sel_valid_mask = s2_valid_mask
        sel_lab_idxes = sel_lab_idxes.gather(-1, s2_idxes)
        sel_lab_embeds = BK.gather_first_dims(sel_lab_embeds, s2_idxes, -2)
        sel_logprobs = None if sel_logprobs is None else sel_logprobs.gather(-1, s2_idxes)
        sel_items_arr = None if sel_items_arr is None \
            else sel_items_arr[np.arange(len(sel_items_arr))[:, np.newaxis], BK.get_value(s2_idxes)]
        return sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds, sel_logprobs, sel_items_arr

    # # if not use selector, then random sampling training instances
    # def _select_cands_training(self, input_mask, gold_mask):
    #     conf = self.conf
    #     # first select examples (randomly)
    #     sel_num = gold_mask.sum(-1) * conf.train_ng_ratio  # how many neg according to ratio
    #     sel_num.clamp_(min=conf.train_min_num)  # min num of neg
    #     sel_rate = sel_num / (input_mask.sum(-1) + 1e-3)  # [*]
    #     sel_rate.clamp_(min=conf.train_min_rate)  # min rate to slen
    #     sel_mask = (BK.rand(BK.get_shape(input_mask)) < sel_rate.unsqueeze(-1)).float()  # [*, slen]
    #     # add gold and exclude pad
    #     sel_mask += gold_mask
    #     sel_mask.clamp_(max=1.)
    #     sel_mask *= input_mask
    #     return sel_mask

    # todo(note): now we only use train_min_rate
    # if not use selector, then random sampling training instances
    def _select_cands_training(self, input_mask, gold_mask, train_min_rate):
        # first select examples (randomly)
        sel_mask = (BK.rand(BK.get_shape(input_mask)) < train_min_rate).float()  # [*, slen]
        # add gold and exclude pad
        sel_mask += gold_mask
        sel_mask.clamp_(max=1.)
        sel_mask *= input_mask
        return sel_mask

    # [*, slen, D], [*, slen]
    def predict(self, insts: List, input_lexi, input_expr, input_mask):
        conf = self.conf
        # step 1: select mention candidates
        if conf.use_selector:
            sel_mask = self.sel.predict(input_expr, input_mask)
        else:
            sel_mask = input_mask
        sel_idxes, sel_valid_mask = BK.mask2idx(sel_mask)  # [*, max-count]
        # step 2: encoding and labeling
        sel_hid_exprs = self._enc(input_lexi, input_expr, input_mask, sel_idxes)
        sel_lab_logprobs, sel_lab_idxes, sel_lab_embeds = self.hl.predict(sel_hid_exprs, None)  # [*, mc], [*, mc, D]
        # =====
        if self.use_secondary_type:
            sectype_embeds = self.t1tot2(sel_lab_idxes)  # [*, mc, D]
            sel2_input = sel_hid_exprs + sectype_embeds  # [*, mc, D]
            sel2_lab_logprobs, sel2_lab_idxes, sel2_lab_embeds = self.hl.predict(sel2_input, None)
            if conf.sectype_t2ift1:
                sel2_lab_idxes *= (sel_lab_idxes>0).long()  # pred t2 only if t1 is not 0 (nil)
            # first concat here and then exclude nil at one pass # [*, mc*2, ~]
            if sel2_lab_idxes.sum().item() > 0:  # if there are any predictions
                sel_lab_logprobs = BK.concat([sel_lab_logprobs, sel2_lab_logprobs], -1)
                sel_idxes = BK.concat([sel_idxes, sel_idxes], -1)
                sel_valid_mask = BK.concat([sel_valid_mask, sel_valid_mask], -1)
                sel_lab_idxes = BK.concat([sel_lab_idxes, sel2_lab_idxes], -1)
                sel_lab_embeds = BK.concat([sel_lab_embeds, sel2_lab_embeds], -2)
        # =====
        # step 3: exclude nil and return
        if conf.exclude_nil:  # [*, mc', ...]
            sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds, sel_lab_logprobs, _ = \
                self._exclude_nil(sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds, sel_logprobs=sel_lab_logprobs)
        # sel_enc_expr = BK.gather_first_dims(input_expr, sel_idxes, -2)  # [*, mc', D]
        return sel_lab_logprobs, sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds

    # [*], [*, slen], [*, slen]
    def loss(self, insts: List, input_lexi, input_expr, input_mask, margin=0.):
        conf = self.conf
        bsize = len(insts)
        # first get gold info, also multiple valid-masks
        gold_masks, gold_idxes, gold_items_arr, gold_valid, gold_idxes2, gold_items2_arr = self.batch_inputs_h(insts)
        input_mask = input_mask * gold_valid.unsqueeze(-1)  # [*, slen]
        # step 1: selector
        if conf.use_selector:
            sel_loss, sel_mask = self.sel.loss(input_expr, input_mask, gold_masks, margin=margin)
        else:
            sel_loss, sel_mask = None, self._select_cands_training(input_mask, gold_masks, conf.train_min_rate)
        sel_idxes, sel_valid_mask = BK.mask2idx(sel_mask)  # [*, max-count]
        sel_gold_idxes = gold_idxes.gather(-1, sel_idxes)
        sel_gold_idxes2 = gold_idxes2.gather(-1, sel_idxes)
        # todo(+N): only get items by head position!
        _tmp_i0, _tmp_i1 = np.arange(bsize)[:, np.newaxis], BK.get_value(sel_idxes)
        sel_items = gold_items_arr[_tmp_i0, _tmp_i1]  # [*, mc]
        sel2_items = gold_items2_arr[_tmp_i0, _tmp_i1]
        # step 2: encoding and labeling
        # if we select nothing
        # ----- debug
        # zlog(f"fb-extractor 1: shape sel_idxes = {sel_idxes.shape}")
        # -----
        sel_shape = BK.get_shape(sel_idxes)
        if sel_shape[-1] == 0:
            lab_loss = [[BK.zeros([]), BK.zeros([])]]
            sel2_lab_loss = [[BK.zeros([]), BK.zeros([])]] if self.use_secondary_type else None
            sel_lab_idxes = sel_gold_idxes
            sel_lab_embeds = BK.zeros(sel_shape + [conf.lab_conf.n_dim])
            ret_items = sel_items  # dim-1==0
        else:
            sel_hid_exprs = self._enc(input_lexi, input_expr, input_mask, sel_idxes)  # [*, mc, DLab]
            lab_loss, sel_lab_idxes, sel_lab_embeds = self.hl.loss(sel_hid_exprs, sel_valid_mask, sel_gold_idxes, margin=margin)
            if conf.train_gold_corr:
                sel_lab_idxes = sel_gold_idxes
                if not self.hl.conf.use_lookup_soft:
                    sel_lab_embeds = self.hl.lookup(sel_lab_idxes)
            ret_items = sel_items
            # =====
            if self.use_secondary_type:
                sectype_embeds = self.t1tot2(sel_lab_idxes)  # [*, mc, D]
                if conf.sectype_noback_enc:
                    sel2_input = sel_hid_exprs.detach() + sectype_embeds  # [*, mc, D]
                else:
                    sel2_input = sel_hid_exprs + sectype_embeds  # [*, mc, D]
                # =====
                # sepcial for the sectype mask (sample it within the gold ones)
                sel2_valid_mask = self._select_cands_training((sel_gold_idxes>0).float(),
                                                              (sel_gold_idxes2>0).float(), conf.train_min_rate_s2)
                # =====
                sel2_lab_loss, sel2_lab_idxes, sel2_lab_embeds = self.hl.loss(sel2_input, sel2_valid_mask,
                                                                              sel_gold_idxes2, margin=margin)
                if conf.train_gold_corr:
                    sel2_lab_idxes = sel_gold_idxes2
                    if not self.hl.conf.use_lookup_soft:
                        sel2_lab_embeds = self.hl.lookup(sel2_lab_idxes)
                if conf.sectype_t2ift1:
                    sel2_lab_idxes = sel2_lab_idxes * (sel_lab_idxes > 0).long()  # pred t2 only if t1 is not 0 (nil)
                # combine the two
                if sel2_lab_idxes.sum().item() > 0:  # if there are any gold sectypes
                    ret_items = np.concatenate([ret_items, sel2_items], -1)  # [*, mc*2]
                    sel_idxes = BK.concat([sel_idxes, sel_idxes], -1)
                    sel_valid_mask = BK.concat([sel_valid_mask, sel2_valid_mask], -1)
                    sel_lab_idxes = BK.concat([sel_lab_idxes, sel2_lab_idxes], -1)
                    sel_lab_embeds = BK.concat([sel_lab_embeds, sel2_lab_embeds], -2)
            else:
                sel2_lab_loss = None
            # =====
            # step 3: exclude nil and return
            if conf.exclude_nil:  # [*, mc', ...]
                sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds, _, ret_items = \
                    self._exclude_nil(sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds, sel_items_arr=ret_items)
        # sel_enc_expr = BK.gather_first_dims(input_expr, sel_idxes, -2)  # [*, mc', D]
        # step 4: finally prepare loss and items
        for one_loss in lab_loss:
            one_loss[0] *= conf.lambda_ne
        ret_losses = lab_loss
        if sel2_lab_loss is not None:
            for one_loss in sel2_lab_loss:
                one_loss[0] *= conf.lambda_ne2
            ret_losses = ret_losses + sel2_lab_loss
        if sel_loss is not None:
            for one_loss in sel_loss:
                one_loss[0] *= conf.lambda_ns
            ret_losses = ret_losses + sel_loss
        # ----- debug
        # zlog(f"fb-extractor 2: shape sel_idxes = {sel_idxes.shape}")
        # -----
        # mask out invalid items with None
        ret_items[BK.get_value(1.-sel_valid_mask).astype(np.bool)] = None
        return ret_losses, ret_items, sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds

    def lookup(self, insts: List, input_lexi, input_expr, input_mask):
        conf = self.conf
        bsize = len(insts)
        # first get gold/input info, also multiple valid-masks
        gold_masks, gold_idxes, gold_items_arr, gold_valid, gold_idxes2, gold_items2_arr = self.batch_inputs_h(insts)
        # step 1: no selection, simply forward using gold_masks
        sel_idxes, sel_valid_mask = BK.mask2idx(gold_masks)  # [*, max-count]
        sel_gold_idxes = gold_idxes.gather(-1, sel_idxes)
        sel_gold_idxes2 = gold_idxes2.gather(-1, sel_idxes)
        # todo(+N): only get items by head position!
        _tmp_i0, _tmp_i1 = np.arange(bsize)[:, np.newaxis], BK.get_value(sel_idxes)
        sel_items = gold_items_arr[_tmp_i0, _tmp_i1]  # [*, mc]
        sel2_items = gold_items2_arr[_tmp_i0, _tmp_i1]
        # step 2: encoding and labeling
        sel_shape = BK.get_shape(sel_idxes)
        if sel_shape[-1] == 0:
            sel_lab_idxes = sel_gold_idxes
            sel_lab_embeds = BK.zeros(sel_shape + [conf.lab_conf.n_dim])
            ret_items = sel_items  # dim-1==0
        else:
            # sel_hid_exprs = self._enc(input_expr, input_mask, sel_idxes)  # [*, mc, DLab]
            sel_lab_idxes = sel_gold_idxes
            sel_lab_embeds = self.hl.lookup(sel_lab_idxes)  # todo(note): here no softlookup?
            ret_items = sel_items
            # second type
            if self.use_secondary_type:
                sel2_lab_idxes = sel_gold_idxes2
                sel2_lab_embeds = self.hl.lookup(sel2_lab_idxes)  # todo(note): here no softlookup?
                sel2_valid_mask = (sel2_lab_idxes>0).float()
                # combine the two
                if sel2_lab_idxes.sum().item() > 0:  # if there are any gold sectypes
                    ret_items = np.concatenate([ret_items, sel2_items], -1)  # [*, mc*2]
                    sel_idxes = BK.concat([sel_idxes, sel_idxes], -1)
                    sel_valid_mask = BK.concat([sel_valid_mask, sel2_valid_mask], -1)
                    sel_lab_idxes = BK.concat([sel_lab_idxes, sel2_lab_idxes], -1)
                    sel_lab_embeds = BK.concat([sel_lab_embeds, sel2_lab_embeds], -2)
        # step 3: exclude nil assuming no deliberate nil in gold/inputs
        if conf.exclude_nil:  # [*, mc', ...]
            sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds, _, ret_items = \
                self._exclude_nil(sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds, sel_items_arr=ret_items)
        # step 4: return
        # sel_enc_expr = BK.gather_first_dims(input_expr, sel_idxes, -2)  # [*, mc', D]
        # mask out invalid items with None
        ret_items[BK.get_value(1.-sel_valid_mask).astype(np.bool)] = None
        return ret_items, sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds

# todo(note): how to compose all the modules in training? What will they return:
# 1. for node selector, there are two modes (controlled by ns_add_gold):
#   only_pred / pred+gold, since only_gold can be forced by labeler
# 2. for labler, also two modes (controlled by ne_gold_corr):
#   keep_pred / gold_correction; together with what features for output (soft/hard-argmax)

#
# =====
# the (binary) unlabeled selector possibly as a first filter
# this module will be mask based
class NodeSelectorConf(Conf):
    def __init__(self):
        self._input_dim = 0
        # mlp hp
        self.mlp_hidden_layer = 1
        self.mlp_hidden_dim = 256
        self.mlp_hidden_act = "elu"
        # selecting method (for output)
        self.topk_ratio = 0.4  # <=0 means select all
        self.thresh_k = -100.
        # training for first-pass node-selector
        self.train_ratio2gold = 0.  # neg samples' ratio to gold counts, <=0 means use testing sel (but can be unbalanced)
        self.train_return_loss_mask = True  # return the ones (loss_mask) used for training
        self.ns_add_gold = True  # adding gold selections in training even if not selected
        self.ns_loss = "prob"  # prob/margin?
        self.ns_no_back = False  # no back-prop to enc_repr/inputs
        # margin
        self.margin_pos = 1.  # minus gold by this
        self.margin_neg = 1.  # plus neg by this
        self.no_loss_satisfy_margin = False  # this is implicit for hinge, but may be also possible for prob

class NodeSelector(BasicNode):
    def __init__(self, pc, conf: NodeSelectorConf):
        super().__init__(pc, None, None)
        self.conf = conf
        self.input_dim = conf._input_dim
        self.scorer = self.add_sub_node("sc", get_mlp(pc, self.input_dim, 1, conf.mlp_hidden_dim, n_hidden_layer=conf.mlp_hidden_layer, hidden_act=conf.mlp_hidden_act, final_init_rop=NoDropRop()))
        # loss function
        self.loss_prob, self.loss_hinge = [conf.ns_loss==z for z in ["prob", "hinge"]]

    # =====
    # for selecting methods
    def _select_topk(self, masked_scores, pad_mask, ratio_mask, topk_ratio, thresh_k):
        slen = BK.get_shape(masked_scores, -1)
        sel_mask = BK.copy(pad_mask)
        # first apply the absolute thresh
        if thresh_k is not None:
            sel_mask *= (masked_scores>thresh_k).float()
        # then ratio-ed topk
        if topk_ratio > 0.:
            # prepare number
            cur_topk_num = ratio_mask.sum(-1)  # [*]
            cur_topk_num = (cur_topk_num * topk_ratio).long()  # [*]
            cur_topk_num.clamp_(min=1, max=slen)  # at least one, at most all
            # topk
            actual_max_k = max(cur_topk_num.max().item(), 1)
            topk_score, _ = BK.topk(masked_scores, actual_max_k, dim=-1, sorted=True)  # [*, k]
            thresh_score = topk_score.gather(-1, cur_topk_num.clamp(min=1).unsqueeze(-1)-1)  # [*, 1]
            # get mask and apply
            sel_mask *= (masked_scores >= thresh_score).float()
        return sel_mask

    # score and return predicting styled selections
    # input: enc_expr: [*, slen, D], pad_mask: [*, slen]
    # output: res_mask: [*, slen], all_scores: [*, slen]
    def score_and_select(self, enc_expr, pad_mask):
        conf = self.conf
        if conf.ns_no_back:
            enc_expr = enc_expr.detach()
        all_scores = self.scorer(enc_expr).squeeze(-1)  # [*, slen]
        # only for getting the mask
        with BK.no_grad_env():
            masked_all_scores = all_scores + (1.-pad_mask) * Constants.REAL_PRAC_MIN
            res_mask = self._select_topk(masked_all_scores, pad_mask, pad_mask, conf.topk_ratio, conf.thresh_k)
        return res_mask, all_scores

    # different calls for training and testing
    def loss(self, enc_expr, pad_mask, gold_mask, margin: float):
        conf = self.conf
        # =====
        # first testing-mode scoring and selecting
        res_mask, all_scores = self.score_and_select(enc_expr, pad_mask)
        # add gold
        if conf.ns_add_gold:
            res_mask += gold_mask
            res_mask.clamp_(max=1.)
        # =====
        with BK.no_grad_env():
            # how to select instances for training
            if conf.train_ratio2gold > 0.:
                # use gold-ratio for training
                masked_all_scores = all_scores + (1.-pad_mask+gold_mask) * Constants.REAL_PRAC_MIN
                loss_mask = self._select_topk(masked_all_scores, pad_mask, gold_mask, conf.train_ratio2gold, None)
                loss_mask += gold_mask
                loss_mask.clamp_(max=1.)
            elif not conf.ns_add_gold:
                loss_mask = res_mask+gold_mask
                loss_mask.clamp_(max=1.)
            else:
                # we already have the gold
                loss_mask = res_mask
        # ===== calculating losses [*, L]
        # first aug scores by margin
        aug_scores = all_scores - (conf.margin_pos*margin) * gold_mask + (conf.margin_neg*margin) * (1.-gold_mask)
        if self.loss_hinge:
            # multiply pos instances with -1
            flipped_scores = aug_scores * (1.-2*gold_mask)
            losses_all = BK.clamp(flipped_scores, min=0.)
        elif self.loss_prob:
            losses_all = BK.binary_cross_entropy_with_logits(aug_scores, gold_mask, reduction='none')
            if conf.no_loss_satisfy_margin:
                unsatisfy_mask = ((aug_scores * (1.-2*gold_mask))>0.).float()  # those still with hinge loss
                losses_all *= unsatisfy_mask
        else:
            raise NotImplementedError()
        # return prediction and loss(sum/count)
        loss_sum = (losses_all*loss_mask).sum()
        if conf.train_return_loss_mask:
            return [[loss_sum, loss_mask.sum()]], loss_mask
        else:
            return [[loss_sum, loss_mask.sum()]], res_mask

    def predict(self, enc_expr, pad_mask):
        res_mask, _ = self.score_and_select(enc_expr, pad_mask)
        return res_mask

# b tasks/zie/models2/extract/head:177
