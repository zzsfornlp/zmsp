#

# direct event generation with a s2s style

import numpy as np
from typing import List
from msp.utils import Conf, Constants, zlog
from msp.nn import BK
from msp.nn.layers import Affine, NoDropRop, RnnNode
from msp.zext.ie import HLabelVocab
from .base import NodeExtractorConfBase, NodeExtractorBase

#
class NodeExtractorConfGene1(NodeExtractorConfBase):
    def __init__(self):
        super().__init__()
        self._input_dim = 0
        self._lexi_dim = 0  # todo(note): not used here!
        # -----
        # decoder
        self.hid_att = 300  # hidden dim before biaffine att
        self.hid_repo = 300  # hidden dim for the repo embedding like stuffs
        self.hid_state = 300  # hidden dim for state
        self.num_repo = 15  # number of repo embeddings
        self.zero_eos_score = True  # force att score of eos(idx=0) as 0
        # testing
        self.beam_size = 1  # beam size
        self.max_step = 10  # max number of prediction steps (outputs)
        self.len_alpha = 1.  # length normalization alpha
        # training
        self.train_force = True  # currently only do teacher-force training
        self.lambda_att = 1.  # for tok-att trigger selecting
        self.lambda_lab = 1.  # for labeling
        self.train_reverse_evetns = True  # from larger idx down to 0 (since 0 is eos)

# extracting as summarization styled generation
class NodeExtractorGene1(NodeExtractorBase):
    def __init__(self, pc, conf: NodeExtractorConfGene1, vocab: HLabelVocab, extract_type: str):
        super().__init__(pc, conf, vocab, extract_type)
        # -----
        # decoding
        # 1. attention for selecting token
        self.affine_k = self.add_sub_node("ak", Affine(pc, [conf._input_dim, 1], conf.hid_att,
                                                       bias=False, which_affine=3, init_rop=NoDropRop()))
        self.affine_q = self.add_sub_node("aq", Affine(pc, [conf.hid_repo, conf.hid_state], conf.hid_att,
                                                       bias=False, which_affine=3, init_rop=NoDropRop()))
        self.repos = self.add_param("r", [conf.num_repo, conf.hid_repo], lookup=True)
        # input is (last_hid_layer + lab_embed)
        self.rnn_unit = self.add_sub_node("rnn", RnnNode.get_rnn_node("lstm2", pc, 2*conf.lab_conf.n_dim, conf.hid_state))
        # 2. labeling
        self.lab_f = self.add_sub_node("lab", Affine(pc, [conf._input_dim, 1, conf.hid_repo, conf.hid_state],
                                                     conf.lab_conf.n_dim, which_affine=3, act="elu"))

    # lookup embeddings from hl
    def _hl_lookup(self, sel_lab_idxes):
        # the embeddings
        sel_shape = BK.get_shape(sel_lab_idxes)
        if sel_shape[-1] == 0:
            sel_lab_embeds = BK.zeros(sel_shape + [self.conf.lab_conf.n_dim])
        else:
            assert not self.hl.conf.use_lookup_soft, "Cannot do soft-lookup in this mode"
            sel_lab_embeds = self.hl.lookup(sel_lab_idxes)
        return sel_lab_embeds

    # one step for decoding (two modes: force & free)
    # todo(+N): would it be useful to add previous token-level predictions?
    # [*, slen, D], [*, slen], [*, slen], tuple[*, D']; [*], [*]; -> widx, lidx, tok/lab-log-probs, next_state [*, ?]
    def _step(self, input_expr, input_mask, hard_coverage, prev_state, force_widx, force_lidx, free_beam_size):
        conf = self.conf
        free_mode = (force_widx is None)
        prev_state_h = prev_state[0]
        # =====
        # collect att scores
        key_up = self.affine_k([input_expr, hard_coverage.unsqueeze(-1)])  # [*, slen, h]
        query_up = self.affine_q([self.repos.unsqueeze(0), prev_state_h.unsqueeze(-2)])  # [*, R, h]
        orig_scores = BK.matmul(key_up, query_up.transpose(-2, -1))  # [*, slen, R]
        orig_scores += (1.-input_mask).unsqueeze(-1) * Constants.REAL_PRAC_MIN  # [*, slen, R]
        # first maximum across the R dim (this step is hard max)
        maxr_scores, maxr_idxes = orig_scores.max(-1)  # [*, slen]
        if conf.zero_eos_score:
            # use mask to make it able to be backward
            tmp_mask = BK.constants(BK.get_shape(maxr_scores), 1.)
            tmp_mask.index_fill_(-1, BK.input_idx(0), 0.)
            maxr_scores *= tmp_mask
        # then select over the slen dim (this step is prob based)
        maxr_logprobs = BK.log_softmax(maxr_scores)  # [*, slen]
        if free_mode:
            cur_beam_size = min(free_beam_size, BK.get_shape(maxr_logprobs, -1))
            sel_tok_logprobs, sel_tok_idxes = maxr_logprobs.topk(cur_beam_size, dim=-1, sorted=False)  # [*, beam]
        else:
            sel_tok_idxes = force_widx.unsqueeze(-1)  # [*, 1]
            sel_tok_logprobs = maxr_logprobs.gather(-1, sel_tok_idxes)  # [*, 1]
        # then collect the info and perform labeling
        lf_input_expr = BK.gather_first_dims(input_expr, sel_tok_idxes, -2)  # [*, ?, ~]
        lf_coverage = hard_coverage.gather(-1, sel_tok_idxes).unsqueeze(-1)  # [*, ?, 1]
        lf_repos = self.repos[maxr_idxes.gather(-1, sel_tok_idxes)]  # [*, ?, ~]  # todo(+3): using soft version?
        lf_prev_state = prev_state_h.unsqueeze(-2)  # [*, 1, ~]
        lab_hid_expr = self.lab_f([lf_input_expr, lf_coverage, lf_repos, lf_prev_state])  # [*, ?, ~]
        # final predicting labels
        # todo(+N): here we select only max at labeling part, only beam at previous one
        if free_mode:
            sel_lab_logprobs, sel_lab_idxes, sel_lab_embeds = self.hl.predict(lab_hid_expr, None)  # [*, ?]
        else:
            sel_lab_logprobs, sel_lab_idxes, sel_lab_embeds = self.hl.predict(lab_hid_expr, force_lidx.unsqueeze(-1))
        # no lab-logprob (*=0) for eos (sel_tok==0)
        sel_lab_logprobs *= (sel_tok_idxes>0).float()
        # compute next-state [*, ?, ~]
        # todo(note): here we flatten the first two dims
        tmp_rnn_dims = BK.get_shape(sel_tok_idxes) + [-1]
        tmp_rnn_input = BK.concat([lab_hid_expr, sel_lab_embeds], -1)
        tmp_rnn_input = tmp_rnn_input.view(-1, BK.get_shape(tmp_rnn_input, -1))
        tmp_rnn_hidden = [z.unsqueeze(-2).expand(tmp_rnn_dims).contiguous().view(-1, BK.get_shape(z, -1))
                          for z in prev_state]  # [*, ?, ?, D]
        next_state = self.rnn_unit(tmp_rnn_input, tmp_rnn_hidden, None)
        next_state = [z.view(tmp_rnn_dims) for z in next_state]
        return sel_tok_idxes, sel_tok_logprobs, sel_lab_idxes, sel_lab_logprobs, sel_lab_embeds, next_state

    # with simple beam search
    def predict(self, insts: List, input_lexi, input_expr, input_mask):
        conf = self.conf
        bsize, slen = BK.get_shape(input_mask)
        bsize_arange_t_1d = BK.arange_idx(bsize)  # [*]
        bsize_arange_t_2d = bsize_arange_t_1d.unsqueeze(-1)  # [*, 1]
        beam_size = conf.beam_size
        # prepare things with an extra beam dimension
        beam_input_expr, beam_input_mask = input_expr.unsqueeze(-3).expand(-1, beam_size, -1, -1).contiguous(), \
                                           input_mask.unsqueeze(-2).expand(-1, beam_size, -1).contiguous()  # [*, beam, slen, D?]
        # -----
        # recurrent states
        beam_hard_coverage = BK.zeros([bsize, beam_size, slen])  # [*, beam, slen]
        # tuple([*, beam, D], )
        beam_prev_state = [z.unsqueeze(-2).expand(-1, beam_size, -1) for z in self.rnn_unit.zero_init_hidden(bsize)]
        # frozen after reach eos
        beam_noneos = 1.-BK.zeros([bsize, beam_size])  # [*, beam]
        beam_logprobs = BK.zeros([bsize, beam_size])  # [*, beam], sum of logprobs
        beam_logprobs_paths = BK.zeros([bsize, beam_size, 0])  # [*, beam, step]
        beam_tok_paths = BK.zeros([bsize, beam_size, 0]).long()
        beam_lab_paths = BK.zeros([bsize, beam_size, 0]).long()
        # -----
        for cstep in range(conf.max_step):
            # get things of [*, beam, beam]
            sel_tok_idxes, sel_tok_logprobs, sel_lab_idxes, sel_lab_logprobs, sel_lab_embeds, next_state = \
                self._step(beam_input_expr, beam_input_mask, beam_hard_coverage, beam_prev_state, None, None, beam_size)
            sel_logprobs = sel_tok_logprobs + sel_lab_logprobs  # [*, beam, beam]
            if cstep == 0:
                # special for the first step, only select for the first element
                cur_selections = BK.arange_idx(beam_size).unsqueeze(0).expand(bsize, beam_size)  # [*, beam]
            else:
                # then select the topk in beam*beam (be careful about the frozen ones!!)
                beam_noneos_3d = beam_noneos.unsqueeze(-1)
                # eos can only followed by eos
                sel_tok_idxes *= beam_noneos_3d.long()
                sel_lab_idxes *= beam_noneos_3d.long()
                # numeric tricks to keep the frozen ones ([0] with 0. score, [1:] with -inf scores)
                sel_logprobs *= beam_noneos_3d
                tmp_exclude_mask = 1. - beam_noneos_3d.expand_as(sel_logprobs)
                tmp_exclude_mask[:, :, 0] = 0.
                sel_logprobs += tmp_exclude_mask * Constants.REAL_PRAC_MIN
                # select for topk
                topk_logprobs = (beam_noneos * beam_logprobs).unsqueeze(-1) + sel_logprobs
                _, cur_selections = topk_logprobs.view([bsize, -1]).topk(beam_size, dim=-1, sorted=True)  # [*, beam]
            # read and write the selections
            # gathering previous ones
            cur_sel_previ = cur_selections // beam_size  # [*, beam]
            prev_hard_coverage = beam_hard_coverage[bsize_arange_t_2d, cur_sel_previ]  # [*, beam]
            prev_noneos = beam_noneos[bsize_arange_t_2d, cur_sel_previ]  # [*, beam]
            prev_logprobs = beam_logprobs[bsize_arange_t_2d, cur_sel_previ]  # [*, beam]
            prev_logprobs_paths = beam_logprobs_paths[bsize_arange_t_2d, cur_sel_previ]  # [*, beam, step]
            prev_tok_paths = beam_tok_paths[bsize_arange_t_2d, cur_sel_previ]  # [*, beam, step]
            prev_lab_paths = beam_lab_paths[bsize_arange_t_2d, cur_sel_previ]  # [*, beam, step]
            # prepare new ones
            cur_sel_newi = cur_selections % beam_size
            new_tok_idxes = sel_tok_idxes[bsize_arange_t_2d, cur_sel_previ, cur_sel_newi]  # [*, beam]
            new_lab_idxes = sel_lab_idxes[bsize_arange_t_2d, cur_sel_previ, cur_sel_newi]  # [*, beam]
            new_logprobs = sel_logprobs[bsize_arange_t_2d, cur_sel_previ, cur_sel_newi]  # [*, beam]
            new_prev_state = [z[bsize_arange_t_2d, cur_sel_previ, cur_sel_newi] for z in next_state]  # [*, beam, ~]
            # update
            prev_hard_coverage[bsize_arange_t_2d, BK.arange_idx(beam_size).unsqueeze(0), new_tok_idxes] += 1.
            beam_hard_coverage = prev_hard_coverage
            beam_prev_state = new_prev_state
            beam_noneos = prev_noneos * (new_tok_idxes!=0).float()
            beam_logprobs = prev_logprobs + new_logprobs
            beam_logprobs_paths = BK.concat([prev_logprobs_paths, new_logprobs.unsqueeze(-1)], -1)
            beam_tok_paths = BK.concat([prev_tok_paths, new_tok_idxes.unsqueeze(-1)], -1)
            beam_lab_paths = BK.concat([prev_lab_paths, new_lab_idxes.unsqueeze(-1)], -1)
        # finally force an extra eos step to get ending tok-logprob (no need to update other things)
        final_eos_idxes = BK.zeros([bsize, beam_size]).long()
        _, eos_logprobs, _, _, _, _ = self._step(beam_input_expr, beam_input_mask, beam_hard_coverage, beam_prev_state, final_eos_idxes, final_eos_idxes, None)
        beam_logprobs += eos_logprobs.squeeze(-1) * beam_noneos  # [*, beam]
        # select and return the best one
        beam_tok_valids = (beam_tok_paths > 0).float()  # [*, beam, steps]
        final_scores = beam_logprobs / ((beam_tok_valids.sum(-1) + 1.) ** conf.len_alpha)  # [*, beam]
        _, best_beam_idx = final_scores.max(-1)  # [*]
        # -----
        # prepare returns; cut by max length: [*, all_step] -> [*, max_step]
        ret0_valid_mask = beam_tok_valids[bsize_arange_t_1d, best_beam_idx]
        cur_max_step = ret0_valid_mask.long().sum(-1).max().item()
        ret_valid_mask = ret0_valid_mask[:, :cur_max_step]
        ret_logprobs = beam_logprobs_paths[bsize_arange_t_1d, best_beam_idx][:, :cur_max_step]
        ret_tok_idxes = beam_tok_paths[bsize_arange_t_1d, best_beam_idx][:, :cur_max_step]
        ret_lab_idxes = beam_lab_paths[bsize_arange_t_1d, best_beam_idx][:, :cur_max_step]
        # embeddings
        ret_lab_embeds = self._hl_lookup(ret_lab_idxes)
        return ret_logprobs, ret_tok_idxes, ret_valid_mask, ret_lab_idxes, ret_lab_embeds

    #
    def loss(self, insts: List, input_lexi, input_expr, input_mask, margin=0.):
        # todo(+N): currently margin is not used
        conf = self.conf
        bsize = len(insts)
        arange_t = BK.arange_idx(bsize)
        assert conf.train_force, "currently only have forced training"
        # get the gold ones
        gold_widxes, gold_lidxes, gold_vmasks, ret_items, _ = self.batch_inputs_g1(insts)  # [*, ?]
        # for all the steps
        num_step = BK.get_shape(gold_widxes, -1)
        # recurrent states
        hard_coverage = BK.zeros(BK.get_shape(input_mask))  # [*, slen]
        prev_state = self.rnn_unit.zero_init_hidden(bsize)  # tuple([*, D], )
        all_tok_logprobs, all_lab_logprobs = [], []
        for cstep in range(num_step):
            slice_widx, slice_lidx = gold_widxes[:,cstep], gold_lidxes[:,cstep]
            _, sel_tok_logprobs, _, sel_lab_logprobs, _, next_state = \
                self._step(input_expr, input_mask, hard_coverage, prev_state, slice_widx, slice_lidx, None)
            all_tok_logprobs.append(sel_tok_logprobs)  # add one of [*, 1]
            all_lab_logprobs.append(sel_lab_logprobs)
            hard_coverage = BK.copy(hard_coverage)  # todo(note): cannot modify inplace!
            hard_coverage[arange_t, slice_widx] += 1.
            prev_state = [z.squeeze(-2) for z in next_state]
        # concat all the loss and mask
        # todo(note): no need to use gold_valid since things are telled in vmasks
        cat_tok_logprobs = BK.concat(all_tok_logprobs, -1) * gold_vmasks  # [*, steps]
        cat_lab_logprobs = BK.concat(all_lab_logprobs, -1) * gold_vmasks
        loss_sum = - (cat_tok_logprobs.sum() * conf.lambda_att + cat_lab_logprobs.sum() * conf.lambda_lab)
        # todo(+N): here we are dividing lab_logprobs with the all-count, do we need to separate?
        loss_count = gold_vmasks.sum()
        ret_losses = [[loss_sum, loss_count]]
        # =====
        # make eos unvalid for return
        ret_valid_mask = gold_vmasks * (gold_widxes>0).float()
        # embeddings
        sel_lab_embeds = self._hl_lookup(gold_lidxes)
        return ret_losses, ret_items, gold_widxes, ret_valid_mask, gold_lidxes, sel_lab_embeds

    def lookup(self, insts: List, input_lexi, input_expr, input_mask):
        # basically directly return will be fine
        sel_idxes, sel_lab_idxes, sel_valid_mask, ret_items, _ = self.batch_inputs_g1(insts)
        # further make eos unvalid
        sel_valid_mask *= (sel_idxes>0).float()
        # embeddings
        sel_lab_embeds = self._hl_lookup(sel_lab_idxes)
        return ret_items, sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds
