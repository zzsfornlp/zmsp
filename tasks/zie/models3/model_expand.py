#

# expand args from head to span

from typing import List
import numpy as np

from msp.utils import Conf, Constants
from msp.nn import BK
from msp.nn.layers import BasicNode, Affine, get_mlp, NoDropRop, RnnNode

#
class ArgSpanExpanderConf(Conf):
    def __init__(self):
        self.max_range = 5  # including self, maximumly how much to extend
        self.hid_dim = 512  # score = linear(elu(linear(head, cand)))
        # use lstm or simple mlp
        self.use_lstm_scorer = False
        # use binary mode
        self.use_binary_scorer = False
        self.binary_scorer_skip = 0  # allow skip how many 0s?

#
class ArgSpanExpander(BasicNode):
    def __init__(self, pc, conf: ArgSpanExpanderConf, input_enc_dims):
        super().__init__(pc, None, None)
        self.conf = conf
        # assert not conf.use_binary_scorer, "this mode seems problematic!!"
        # todo(note): only using bert's ones, simply flatten
        bert_input_dims, _ = input_enc_dims
        bert_dim, bert_fold = bert_input_dims
        flatten_bert_dim = bert_dim * bert_fold
        self.flatten_bert_dim = flatten_bert_dim
        # scoring params
        self.use_lstm_scorer = conf.use_lstm_scorer
        if self.use_lstm_scorer:
            self.llstm = self.add_sub_node("llstm", RnnNode.get_rnn_node("lstm2", pc, flatten_bert_dim, conf.hid_dim))
            self.rlstm = self.add_sub_node("rlstm", RnnNode.get_rnn_node("lstm2", pc, flatten_bert_dim, conf.hid_dim))
            self.lscorer = self.add_sub_node("ls", Affine(pc, conf.hid_dim, 1, init_rop=NoDropRop()))
            self.rscorer = self.add_sub_node("rs", Affine(pc, conf.hid_dim, 1, init_rop=NoDropRop()))
        else:
            self.lscorer = self.add_sub_node("ls", get_mlp(pc, [flatten_bert_dim, flatten_bert_dim], 1, hidden_which_affine=3,
                                                           n_hidden=conf.hid_dim, n_hidden_layer=1, hidden_act='elu',
                                                           final_act="linear", final_bias=False, final_init_rop=NoDropRop()))
            self.rscorer = self.add_sub_node("rs", get_mlp(pc, [flatten_bert_dim, flatten_bert_dim], 1, hidden_which_affine=3,
                                                           n_hidden=conf.hid_dim, n_hidden_layer=1, hidden_act='elu',
                                                           final_act="linear", final_bias=False, final_init_rop=NoDropRop()))

    # collect the arguments according to center event
    def _collect_insts(self, ms_items: List, training):
        max_range = self.conf.max_range
        ret_efs, ret_sents, ret_bidxes, ret_head_idxes, ret_left_dists, ret_right_dists = [], [], [], [], [], []
        for batch_idx, one_item in enumerate(ms_items):
            one_sents = one_item.sents
            sid2sents = {s.sid: s for s in one_sents}  # not the sid in this list
            sid2offsets = {s.sid: v for s,v in zip(one_sents, one_item.offsets)}  # not the sid in this list
            # assert one_sents[0].sid == 0, "Currently only support fake doc!"
            one_center_idx = one_item.center_idx
            one_center_sent = one_sents[one_center_idx]
            # get target events
            one_center_evts = one_center_sent.events if training else one_center_sent.pred_events
            if one_center_evts is not None and len(one_center_evts)>0:
                # todo(+N): is multi-event ok?
                # assert len(one_center_evts) == 1, "Currently only support one event at one sent!!"
                # get args
                for one_center_evt in one_center_evts:
                    if one_center_evt.links is None:
                        continue
                    for one_arg in one_center_evt.links:
                        one_ef = one_arg.ef
                        # only collect in-ranged ones
                        if one_ef.mention is not None and one_ef.mention.hard_span.sid in sid2sents:
                            hspan = one_ef.mention.hard_span
                            sid, head_wid, wid, wlen = hspan.sid, hspan.head_wid, hspan.wid, hspan.length
                            left_dist = head_wid - wid
                            right_dist = wid + wlen - 1 - head_wid
                            if training:
                                if left_dist>=max_range or right_dist>=max_range:
                                    continue  # skip long spans in training
                            else:
                                # clear wid and wlen for testing
                                hspan.wid = hspan.head_wid
                                hspan.length = 1
                                left_dist = right_dist = 0
                            # add one
                            ret_sents.append(sid2sents[sid])
                            ret_efs.append(one_ef)  # todo(note): may repeat but does not matter
                            ret_bidxes.append(batch_idx)
                            ret_head_idxes.append(sid2offsets[sid]+head_wid-1)  # minus ROOT offset
                            ret_left_dists.append(left_dist)
                            ret_right_dists.append(right_dist)
        return ret_efs, ret_sents, BK.input_idx(ret_bidxes), BK.input_idx(ret_head_idxes), \
               BK.input_idx(ret_left_dists), BK.input_idx(ret_right_dists)

    # score
    def _score(self, bert_expr, bidxes_t, hidxes_t):
        # ----
        # # debug
        # print(f"# ====\n Debug: {ArgSpanExpander._debug_count}")
        # ArgSpanExpander._debug_count += 1
        # ----
        bert_expr = bert_expr.view(BK.get_shape(bert_expr)[:-2] + [-1])  # flatten
        #
        max_range = self.conf.max_range
        max_slen = BK.get_shape(bert_expr, 1)
        # get candidates
        range_t = BK.arange_idx(max_range).unsqueeze(0)  # [1, R]
        bidxes_t = bidxes_t.unsqueeze(1)  # [N, 1]
        hidxes_t = hidxes_t.unsqueeze(1)  # [N, 1]
        left_cands = hidxes_t - range_t  # [N, R]
        right_cands = hidxes_t + range_t
        left_masks = (left_cands>=0).float()
        right_masks = (right_cands<max_slen).float()
        left_cands.clamp_(min=0)
        right_cands.clamp_(max=max_slen-1)
        # score
        head_exprs = bert_expr[bidxes_t, hidxes_t]  # [N, 1, D']
        left_cand_exprs = bert_expr[bidxes_t, left_cands]  # [N, R, D']
        right_cand_exprs = bert_expr[bidxes_t, right_cands]
        # actual scoring
        if self.use_lstm_scorer:
            batch_size = BK.get_shape(bidxes_t, 0)
            all_concat_outputs = []
            for cand_exprs, lstm_node in zip([left_cand_exprs, right_cand_exprs], [self.llstm, self.rlstm]):
                cur_state = lstm_node.zero_init_hidden(batch_size)
                step_size = BK.get_shape(cand_exprs, 1)
                all_outputs = []
                for step_i in range(step_size):
                    cur_state = lstm_node(cand_exprs[:,step_i], cur_state, None)
                    all_outputs.append(cur_state[0])  # using h
                concat_output = BK.stack(all_outputs, 1)  # [N, R, ?]
                all_concat_outputs.append(concat_output)
            left_hidden, right_hidden = all_concat_outputs
            left_scores = self.lscorer(left_hidden).squeeze(-1)  # [N, R]
            right_scores = self.rscorer(right_hidden).squeeze(-1)  # [N, R]
        else:
            left_scores = self.lscorer([left_cand_exprs, head_exprs]).squeeze(-1)  # [N, R]
            right_scores = self.rscorer([right_cand_exprs, head_exprs]).squeeze(-1)
        # mask
        left_scores += Constants.REAL_PRAC_MIN * (1.-left_masks)
        right_scores += Constants.REAL_PRAC_MIN * (1.-right_masks)
        return left_scores, right_scores

    _debug_count=0

    # List[ms_item], [bs, slen, fold, D]
    def loss(self, ms_items: List, bert_expr):
        conf = self.conf
        max_range = self.conf.max_range
        bsize = len(ms_items)
        # collect instances
        col_efs, _, col_bidxes_t, col_hidxes_t, col_ldists_t, col_rdists_t = self._collect_insts(ms_items, True)
        if len(col_efs)==0:
            zzz = BK.zeros([])
            return [[zzz, zzz, zzz], [zzz, zzz, zzz]]
        left_scores, right_scores = self._score(bert_expr, col_bidxes_t, col_hidxes_t)  # [N, R]
        if conf.use_binary_scorer:
            left_binaries, right_binaries = (BK.arange_idx(max_range)<=col_ldists_t.unsqueeze(-1)).float(), \
                                            (BK.arange_idx(max_range)<=col_rdists_t.unsqueeze(-1)).float()  # [N,R]
            left_losses = BK.binary_cross_entropy_with_logits(left_scores, left_binaries, reduction='none')[:,1:]
            right_losses = BK.binary_cross_entropy_with_logits(right_scores, right_binaries, reduction='none')[:,1:]
            left_count = right_count = BK.input_real(BK.get_shape(left_losses, 0) * (max_range-1))
        else:
            left_losses = BK.loss_nll(left_scores, col_ldists_t)
            right_losses = BK.loss_nll(right_scores, col_rdists_t)
            left_count = right_count = BK.input_real(BK.get_shape(left_losses, 0))
        return [[left_losses.sum(), left_count, left_count], [right_losses.sum(), right_count, right_count]]

    #
    def _binary_decide_dist(self, one_scores):
        binary_scorer_skip = self.conf.binary_scorer_skip
        max_range = self.conf.max_range
        d = 0
        last_hit = 0  # by default is 0: no expansion
        while True:
            if d>=max_range:
                break
            if d>0 and one_scores[d]<0:
                binary_scorer_skip -= 1
                if binary_scorer_skip<0:
                    break
            else:
                last_hit = d
            d += 1
        return last_hit

    def predict(self, ms_items: List, bert_expr):
        conf = self.conf
        bsize = len(ms_items)
        # collect instances
        col_efs, col_sents, col_bidxes_t, col_hidxes_t, _, _ = self._collect_insts(ms_items, False)
        if len(col_efs) == 0:
            return
        left_scores, right_scores = self._score(bert_expr, col_bidxes_t, col_hidxes_t)
        if conf.use_binary_scorer:
            lscores_arr, rscores_arr = BK.get_value(left_scores), BK.get_value(right_scores)
            #
            for one_ef, one_sent, one_lscores, one_rscores in zip(col_efs, col_sents, lscores_arr, rscores_arr):
                one_ldist, one_rdist = self._binary_decide_dist(one_lscores), self._binary_decide_dist(one_rscores)
                # set span
                hspan = one_ef.mention.hard_span
                sid, head_wid = hspan.sid, hspan.head_wid
                left_wid = max(1, head_wid-one_ldist)  # not the artificial root
                right_wid = min(one_sent.length-1, head_wid+one_rdist)
                hspan.wid = left_wid
                hspan.length = right_wid-left_wid+1
        else:
            # simply pick max
            _, left_max_dist = left_scores.max(-1)
            _, right_max_dist = right_scores.max(-1)
            lmax_arr, rmax_arr = BK.get_value(left_max_dist), BK.get_value(right_max_dist)
            #
            for one_ef, one_sent, one_ldist, one_rdist in zip(col_efs, col_sents, lmax_arr, rmax_arr):
                one_ldist, one_rdist = int(one_ldist), int(one_rdist)
                # set span
                hspan = one_ef.mention.hard_span
                sid, head_wid = hspan.sid, hspan.head_wid
                left_wid = max(1, head_wid-one_ldist)  # not the artificial root
                right_wid = min(one_sent.length-1, head_wid+one_rdist)
                hspan.wid = left_wid
                hspan.length = right_wid-left_wid+1

# b tasks/zie/models3/model_expand:37
# b tasks/zie/models3/model_expand:83, ArgSpanExpander._debug_count==25
