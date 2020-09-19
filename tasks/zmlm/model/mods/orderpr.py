#

# the module for order predictor

from typing import Dict
import numpy as np
from msp.utils import Conf, Random
from msp.nn import BK
from msp.nn.layers import PairScorerConf, PairScorer, Affine, NoDropRop
from ..base import BaseModuleConf, BaseModule, LossHelper
from .embedder import Inputter

# -----

# conf
class OrderPredNodeConf(BaseModuleConf):
    def __init__(self):
        super().__init__()
        self.loss_lambda.val = 0.  # by default no such loss
        # --
        self.disturb_mode = "abs_bin"  # abs_noise/abs_bin/rel_noise/rel_bin
        self.disturb_ranges = [0, 1]  # [a, b) range for either abs or rel
        self.abs_noise_sort = True  # for "abs_noise", sort the idxes, making it like local shuffling
        self.abs_keep_cls = True  # if using "abs_*", keep cls/arti_root/... as it is
        self.bin_rand_offset = True  # add random offsets [0, R) to idxes (only in blur1)
        self.bin_blur_method = 2  # which blurring method for bin
        self.bin_blur_bag_lbound = 100  # <=0 means [R+this, R], >0 means (auto) [R//2+1, R]
        self.bin_blur_bag_keep_rate = 0.  # keep the original order in what percentage of the bags
        # pairwise scorer
        self.ps_conf = PairScorerConf().init_from_kwargs(use_bias=False, use_input_pair=False,
                                                         use_biaffine=False, use_ff1=False, use_ff2=True)
        self.pred_abs = False  # predicting only absolute distance?
        self.pred_range = 0  # [-pr, 0) + (0, pr], not predicting self
        self.cand_range = -1  # at least as pred_range
        # simple clasification losses
        self.lambda_n1 = 1.  # reduce on dim=-1(2*R): classify dist for each cand
        self.lambda_n2 = 1.  # reduce on dim=-2(cand): classify cand for each dist

    def do_validate(self):
        self.disturb_ranges = [int(z) for z in self.disturb_ranges]
        assert len(self.disturb_ranges) == 2 and self.disturb_ranges[1]>self.disturb_ranges[0]
        self.cand_range = max(self.cand_range, self.pred_range)  # at least as pred_range

# module
class OrderPredNode(BaseModule):
    def __init__(self, pc: BK.ParamCollection, input_dim: int, inputp_dim: int, conf: OrderPredNodeConf, inputter: Inputter):
        super().__init__(pc, conf, name="ORP")
        self.conf = conf
        self.inputter = inputter
        self.input_dim = input_dim
        self.inputp_dim = inputp_dim
        # -----
        # this step is performed at the embedder, thus still does not influence the inputter
        self.add_root_token = self.inputter.embedder.add_root_token
        # --
        self._disturb_f = getattr(self, "_disturb_"+conf.disturb_mode)  # shortcut
        self.disturb_range_low, self.disturb_range_high = conf.disturb_ranges
        self._blur_idxes_f = [None, self._blur_idxes1, self._blur_idxes2][conf.bin_blur_method]
        self._bin_blur_bag_lbound = conf.bin_blur_bag_lbound
        self._bin_blur_bag_keep_rate = conf.bin_blur_bag_keep_rate
        # -----
        # simple dist predicting: inputs -> [*, len_q, len_k, 2*PredRange]
        self.ps_node = self.add_sub_node("ps", PairScorer(pc, input_dim, input_dim, 2*conf.pred_range,
                                                          conf=conf.ps_conf, in_size_pair=inputp_dim))
        # --
        self.speical_hid_layer = None

    # =====
    # different disturbing methods

    # add_noise then bin: either stay at this bag or jump to the next
    def _disturb_local_shuffle(self, input_mask_arr, disturb_range, rand_gen):
        conf = self.conf
        assert self._bin_blur_bag_keep_rate == 0., "Currently not supporting this one for this mode!"
        batch_size, max_len = input_mask_arr.shape
        # --
        R = disturb_range
        # first assign ordinary range idxes
        orig_idxes = np.arange(max_len)[np.newaxis, :]  # [1, max_len]
        # add noise: either stay at this bag or jump to the next one
        noised_idxes = orig_idxes + R * rand_gen.randint(2, size=(batch_size, max_len))  # [bs, len]
        blur_idxes = noised_idxes // R * R  # [bs, len]
        torank_values = blur_idxes + rand_gen.random_sample(size=(batch_size, max_len))  # sample small from [0,1)
        if conf.abs_keep_cls and self.add_root_token:
            torank_values[:, 0] = -R  # make it rank 0
        torank_values += (R*100) * (1.-input_mask_arr)  # make invalid ones stay there
        ret_abs_idxes = torank_values.argsort(axis=-1)  # [bs, len]
        # [bs, 1, len] - [bs, len, 1] = [bs, len, len]
        ret_rel_dists = ret_abs_idxes[:, np.newaxis, :] - ret_abs_idxes[:, :, np.newaxis]
        return ret_abs_idxes, ret_rel_dists

    # first split into local bags and then shuffle for local bags
    def _disturb_local_shuffle2(self, input_mask_arr, disturb_range, rand_gen):
        conf = self.conf
        batch_size, max_len = input_mask_arr.shape
        # --
        R = disturb_range
        arange_arr = np.arange(batch_size)[:, np.newaxis]  # [bs, 1]
        arange_arr2 = np.arange(max_len)[np.newaxis, :]  # [1, bs]
        # sample bag sizes
        orig_size = (batch_size, max_len)
        # sample bag sizes
        if self._bin_blur_bag_lbound <= 0:
            _L = max(1, R + self._bin_blur_bag_lbound)  # at least bag_size >= 1
        else:
            _L = R // 2 + 1  # automatically
        bag_sizes = rand_gen.randint(_L, 1+R, size=orig_size)  # [bs, len]
        mark_idxes = np.cumsum(bag_sizes, -1)  # [bs*len]
        mark_idxes = mark_idxes * (mark_idxes<max_len)  # [bs*len], make invalid ones 0
        # marked seq
        marked_seq = np.zeros(orig_size, dtype=np.int)  # [bs, len]
        marked_seq[arange_arr, mark_idxes] = 1  # [bs, len]
        marked_seq[:,0] = 1  # make them all start at 1, since some invalid ones can have idx=0
        # again cumsum to get values
        cumsum_values = np.cumsum(marked_seq.reshape(orig_size), -1) - 1  # [bs, len], minus 1 to make it start with 0
        torank_values = cumsum_values.copy()
        if conf.abs_keep_cls and self.add_root_token:
            torank_values[:, 0] = -R  # make it rank 0
        # keep some bags?
        _bin_blur_bag_keep_rate = self._bin_blur_bag_keep_rate
        if _bin_blur_bag_keep_rate>0.:
            bag_keeps = (rand_gen.random_sample(size=orig_size)<_bin_blur_bag_keep_rate).astype(np.float)  # [bs, len], whether keep bag
            keep_valid = bag_keeps[arange_arr, cumsum_values]  # [bs, len], whether keep token
            keep_valid_plus = np.clip(keep_valid + (1. - input_mask_arr), 0, 1)
            keep_adding = arange_arr2 * keep_valid_plus  # adding this to sorting values
        else:
            keep_valid = None
            keep_adding = (R*max(10, max_len)) * (1.-input_mask_arr)  # make invalid ones stay there
        # shuffle as noise and rank
        torank_values = torank_values*max(10, max_len) + keep_adding
        noised_to_rank_values = torank_values + rand_gen.random_sample(size=orig_size)  # [bs, len]
        sorted_idxes = noised_to_rank_values.argsort(axis=-1)  # [bs, len]
        ret_abs_idxes = sorted_idxes.argsort(axis=-1)  # [bs, len], get real ranking
        # [bs, 1, len] - [bs, len, 1] = [bs, len, len]
        ret_rel_dists = ret_abs_idxes[:, np.newaxis, :] - ret_abs_idxes[:, :, np.newaxis]
        return ret_abs_idxes, ret_rel_dists, keep_valid

    # orig_idx + noise
    def _disturb_abs_noise(self, input_mask_arr, disturb_range, rand_gen):
        conf = self.conf
        assert self._bin_blur_bag_keep_rate == 0., "Currently not supporting this one for this mode!"
        batch_size, max_len = input_mask_arr.shape
        # --
        R = disturb_range * 2  # +/-range
        # first assign ordinary range idxes
        orig_idxes = np.arange(max_len)[np.newaxis, :]  # [1, max_len]
        # add noises: [batch_size, max_len]
        noised_idxes = orig_idxes + R * (rand_gen.random_sample((batch_size, max_len))-0.5)  # +/- disturb_range
        if conf.abs_noise_sort:  # simple argsort
            if conf.abs_keep_cls and self.add_root_token:
                noised_idxes[:, 0] = -R  # make it rank 0
            noised_idxes += (R*100) * (1.-input_mask_arr)  # make invalid ones stay there
            ret_abs_idxes = noised_idxes.argsort(axis=-1)
        else:  # clip and round
            # todo(note): this can lead to repeated idxes
            ret_abs_idxes = noised_idxes.clip(0, max_len-1).round().astype(np.int64)
            if conf.abs_keep_cls and self.add_root_token:
                ret_abs_idxes[:, 0] = 0  # directly put it 0
        # [bs, 1, len] - [bs, len, 1] = [bs, len, len]
        ret_rel_dists = ret_abs_idxes[:, np.newaxis, :] - ret_abs_idxes[:, :, np.newaxis]
        return ret_abs_idxes, ret_rel_dists

    # orig_dist + noise
    def _disturb_rel_noise(self, input_mask_arr, disturb_range, rand_gen):
        conf = self.conf
        assert self._bin_blur_bag_keep_rate == 0., "Currently not supporting this one for this mode!"
        batch_size, max_len = input_mask_arr.shape
        # --
        R = disturb_range * 2  # +/-range
        # first assign ordinary range idxes
        orig_idxes0 = np.arange(max_len)  # [max_len]
        orig_idxes = orig_idxes0[np.newaxis, :]  # [1, max_len]
        orig_rel_idxes = orig_idxes[:, np.newaxis, :] - orig_idxes[:, :, np.newaxis]  # [1, len, len]
        # add noises +/- disturb_range: [batch_size, max_len, max_len]
        noised_rel_idxes_f = orig_rel_idxes + R * (rand_gen.random_sample([batch_size, max_len, max_len])-0.5)
        # round
        noised_rel_idxes = noised_rel_idxes_f.round().astype(np.int64)  # [bs, len, len]
        # keep diag as 0 and others not 0 (>0:+1 <0:-1)
        noised_rel_idxes += (2*(noised_rel_idxes_f>=0) - 1) * (noised_rel_idxes==0)  # += (1 if f>=0 else -1) * (i==0)
        noised_rel_idxes[:, orig_idxes0, orig_idxes0] = 0
        # no abs_idxes available
        return None, noised_rel_idxes

    # =====

    # helper for *bin*, orig_idxes should be >=0; keep the original ones if <keep_thresh
    def blur_idxes(self, orig_idxes, R: int, keep_thresh: int, rand_offset_size, rand_gen):
        offset_orig_idxes = (orig_idxes-keep_thresh).clip(min=0)  # offset by keep_thresh
        # bin the idxes
        blur_idxes, blur_keeps = self._blur_idxes_f(offset_orig_idxes, R, rand_offset_size, rand_gen)
        # keep original ones if <thresh & starting with this for the blur ones
        ret_idxes = np.where(orig_idxes < keep_thresh, orig_idxes, blur_idxes + keep_thresh)
        return ret_idxes, blur_keeps

    # simple blur
    def _blur_idxes1(self, input_idxes, R: int, rand_offset_size, rand_gen):
        bin_rand_offset = self.conf.bin_rand_offset
        # add offsets (1): make it group slightly differently at the start
        if bin_rand_offset:  # but keep at least a relatively large group >R/2
            toblur_idxes = input_idxes + rand_gen.randint(1 + R // 2, size=rand_offset_size)
        else:
            toblur_idxes = input_idxes
        blur_idxes = toblur_idxes // R * R
        assert self._bin_blur_bag_keep_rate==0., "Currently not supporting this one for this mode!"
        # add random offsets (2)
        if bin_rand_offset:
            blur_idxes += rand_gen.randint(R, size=rand_offset_size)
        return blur_idxes, None

    # random split bags with unequal sizes and random pick one as representative
    # todo(note): input idxes cannot be >= shape[1]+R, otherwise will oor-error
    def _blur_idxes2(self, orig_idxes, R: int, rand_offset_size, rand_gen):
        _bin_blur_bag_keep_rate = self._bin_blur_bag_keep_rate
        # conf = self.conf
        batch_size, max_len = orig_idxes.shape[:2]  # todo(note): assume the first two dims means this!!
        # --
        prep_size = (batch_size, max_len+R)  # make slightly larger prep sizes since the input make be larger
        arange_arr = np.arange(batch_size)[:, np.newaxis]  # [bs, 1]
        # sample bag sizes
        if self._bin_blur_bag_lbound <= 0:
            _L = max(1, R+self._bin_blur_bag_lbound)  # at least bag_size >= 1
        else:
            _L = R//2+1  # automatically
        bag_sizes = rand_gen.randint(_L, 1+R, size=prep_size)  # [bs, L+R], (3,4,3,5,...)
        if _bin_blur_bag_keep_rate>0.:
            bag_keeps = (rand_gen.random_sample(size=prep_size)<_bin_blur_bag_keep_rate)  # [bs, L+R], (1,0,1,1,...)
        else:
            bag_keeps = None  # no keeps
        # csum for the idxes
        mark_idxes = np.cumsum(bag_sizes, -1)  # [bs, L+R], (3,7,10,15,...)
        mark_idxes_i = mark_idxes * (mark_idxes<max_len+R)  # [bs, L+R], make invalid ones 0
        # mark idxes for later selecting
        marked_seq = np.zeros(prep_size, dtype=np.int)  # [bs, L+R], (0,0,0,1,0,0,0,0,1,...)
        marked_seq[arange_arr, mark_idxes_i] = 1  # [bs, L+R]
        marked_seq[:, 0] = 0  # initial ones can only be marked by invalid ones
        # get idxes for range(max_len)
        range_idxes = np.cumsum(marked_seq, -1)  # [bs, L+R], (0,0,0,1,1,1,1,1,2,...)
        # get representatives for each idx (each bag?): (3-?,7-?,10-?,15-?,...)
        repr_idxes = np.floor(mark_idxes - (bag_sizes * rand_gen.random_sample(prep_size))).astype(np.int)  # [bs, L+R]
        # --
        # final select (twice)
        sel_range_idxes = repr_idxes[arange_arr, range_idxes]  # [bs, L+R], (3-?,3-?,3-?,7-?,7-?,7-?,7-?,7-?,10-?,...)
        if _bin_blur_bag_keep_rate>0.:
            sel_range_keeps = bag_keeps[arange_arr, range_idxes]  # [bs, L+R], whether keep for each idx
            sel_range_idxes = np.where(sel_range_keeps, np.arange(max_len+R)[np.newaxis,:], sel_range_idxes)  # [bs,L+R]
            sel_ret_keeps = sel_range_keeps[arange_arr, orig_idxes.reshape([batch_size, -1])].reshape(orig_idxes.shape).astype(np.float)  # [bs, L]
        else:
            sel_ret_keeps = None
        sel_ret_idxes = sel_range_idxes[arange_arr, orig_idxes.reshape([batch_size, -1])].reshape(orig_idxes.shape)  # [bs, L]
        return sel_ret_idxes, sel_ret_keeps

    # bin(orig_idx)
    def _disturb_abs_bin(self, input_mask_arr, disturb_range, rand_gen):
        conf = self.conf
        batch_size, max_len = input_mask_arr.shape
        # --
        R = disturb_range  # bin size
        rand_offset_size = (batch_size, 1)
        # first assign ordinary range idxes
        orig_idxes = np.arange(max_len) + np.zeros(rand_offset_size, dtype=np.int)  # [bs, len]
        blur_keep_thresh = int(conf.abs_keep_cls and self.add_root_token)  # whether keep ARTI_ROOT as 0
        ret_abs_idxes, ret_abs_keeps = self.blur_idxes(orig_idxes, R, blur_keep_thresh, (batch_size, 1), rand_gen)  # [bs, max_len]
        # [bs, 1, len] - [bs, len, 1] = [bs, len, len]
        ret_rel_dists = ret_abs_idxes[:, np.newaxis, :] - ret_abs_idxes[:, :, np.newaxis]
        return ret_abs_idxes, ret_rel_dists, ret_abs_keeps

    # bin(orig_dist)
    def _disturb_rel_bin(self, input_mask_arr, disturb_range, rand_gen):
        # conf = self.conf
        batch_size, max_len = input_mask_arr.shape
        # --
        R = disturb_range  # bin size
        rand_offset_size = (batch_size, 1, 1)
        # first assign ordinary range idxes
        orig_idxes = np.arange(max_len)  # [max_len]
        orig_rel_idxes = orig_idxes[np.newaxis, :] - orig_idxes[:, np.newaxis]  # [len, len]
        abs_rel_idxes = np.abs(orig_rel_idxes) + np.zeros(rand_offset_size, dtype=np.int)  # [bs, len, len]
        # always keep 0 as special
        blur_abs_rel_idxes, _ = self.blur_idxes(abs_rel_idxes, R, 1, rand_offset_size, rand_gen)  # [bs, len, len]
        # recover sign
        blur_rel_idxes = blur_abs_rel_idxes * (2*(orig_rel_idxes>=0) - 1)  # recover signs
        # no abs_idxes available
        return None, blur_rel_idxes

    # -----
    def disturb_input(self, input_map: Dict, rand_gen=None):
        if rand_gen is None:
            rand_gen = Random
        # -----
        # get shape
        input_mask_arr = input_map["mask"]  # [bs, len]
        disturb_range = rand_gen.randint(self.disturb_range_low, self.disturb_range_high)  # random range
        if self.add_root_token:  # add +1 here!!
            input_mask_arr = np.pad(input_mask_arr, ((0,0),(1,0)), constant_values=1.)  # [bs, 1+len]
        # get disturbed idxes/dists
        disturb_ret = self._disturb_f(input_mask_arr, disturb_range, rand_gen)
        abs_idxes, rel_dists = disturb_ret[:2]
        # shallow copy
        ret_map = input_map.copy()
        ret_map["posi"] = abs_idxes
        ret_map["rel_dist"] = rel_dists
        if len(disturb_ret)>2:
            ret_map["disturb_keep"] = disturb_ret[2]
        # # debug
        # self.see_disturbed_inputs(input_map, abs_idxes)
        # # -----
        return ret_map

    # helper function to see
    def see_disturbed_inputs(self, input_map, abs_idxes):
        input_word_idxes = input_map["word"]  # [bs, len?]
        if self.add_root_token:
            input_word_idxes = np.pad(input_word_idxes, ((0,0),(1,0)), constant_values=0)  # [bs, 1+len]
        # argsort again to get inversed idxes
        inversed_idxes = np.argsort(abs_idxes, -1)  # [bs, len?]
        disturbed_word_idxes = input_word_idxes[np.arange(abs_idxes.shape[0])[:,np.newaxis], inversed_idxes]  # [bs, len?]
        tshape = disturbed_word_idxes.shape
        # -----
        word_voc = self.inputter.vpack.get_voc("word")
        orig_word_strs = np.array([word_voc.idx2word(z) for z in input_word_idxes.reshape(-1)], dtype=object).reshape(tshape)
        disturbed_word_strs = np.array([word_voc.idx2word(z) for z in disturbed_word_idxes.reshape(-1)], dtype=object).reshape(tshape)
        return orig_word_strs, disturbed_word_strs

    # [bs, slen, *], [bs, len_q, len_k, head], [bs, slen], [bs, slen]
    def loss(self, repr_t, attn_t, mask_t, disturb_keep_arr, **kwargs):
        conf = self.conf
        CR, PR = conf.cand_range, conf.pred_range
        # -----
        mask_single = BK.copy(mask_t)
        # no predictions for ARTI_ROOT
        if self.add_root_token:
            mask_single[:, 0] = 0.  # [bs, slen]
        # casting predicting range
        cur_slen = BK.get_shape(mask_single, -1)
        arange_t = BK.arange_idx(cur_slen)  # [slen]
        # [1, len] - [len, 1] = [len, len]
        reldist_t = (arange_t.unsqueeze(-2) - arange_t.unsqueeze(-1))  # [slen, slen]
        mask_pair = ((reldist_t.abs() <= CR) & (reldist_t != 0)).float()  # within CR-range; [slen, slen]
        mask_pair = mask_pair * mask_single.unsqueeze(-1) * mask_single.unsqueeze(-2)  # [bs, slen, slen]
        if disturb_keep_arr is not None:
            mask_pair *= BK.input_real(1.-disturb_keep_arr).unsqueeze(-1)  # no predictions for the kept ones!
        # get all pair scores
        score_t = self.ps_node.paired_score(repr_t, repr_t, attn_t, maskp=mask_pair)  # [bs, len_q, len_k, 2*R]
        # -----
        # loss: normalize on which dim?
        # get the answers first
        if conf.pred_abs:
            answer_t = reldist_t.abs()  # [1,2,3,...,PR]
            answer_t.clamp_(min=0, max=PR-1)  # [slen, slen], clip in range, distinguish using masks
        else:
            answer_t = BK.where((reldist_t>=0), reldist_t-1, reldist_t+2*PR)  # [1,2,3,...PR,-PR,-PR+1,...,-1]
            answer_t.clamp_(min=0, max=2*PR-1)  # [slen, slen], clip in range, distinguish using masks
        # expand answer into idxes
        answer_hit_t = BK.zeros(BK.get_shape(answer_t) + [2*PR])  # [len_q, len_k, 2*R]
        answer_hit_t.scatter_(-1, answer_t.unsqueeze(-1), 1.)
        answer_valid_t = ((reldist_t.abs() <= PR) & (reldist_t != 0)).float().unsqueeze(-1)  # [bs, len_q, len_k, 1]
        answer_hit_t = answer_hit_t * mask_pair.unsqueeze(-1) * answer_valid_t  # clear invalid ones; [bs, len_q, len_k, 2*R]
        # get losses sum(log(answer*prob))
        # -- dim=-1 is standard 2*PR classification, dim=-2 usually have 2*PR candidates, but can be less at edges
        all_losses = []
        for one_dim, one_lambda in zip([-1, -2], [conf.lambda_n1, conf.lambda_n2]):
            if one_lambda > 0.:
                # since currently there can be only one or zero correct answer
                logprob_t = BK.log_softmax(score_t, one_dim)  # [bs, len_q, len_k, 2*R]
                sumlogprob_t = (logprob_t * answer_hit_t).sum(one_dim)  # [bs, len_q, len_k||2*R]
                cur_dim_mask_t = (answer_hit_t.sum(one_dim)>0.).float()  # [bs, len_q, len_k||2*R]
                # loss
                cur_dim_loss = - (sumlogprob_t * cur_dim_mask_t).sum()
                cur_dim_count = cur_dim_mask_t.sum()
                # argmax and corr (any correct counts)
                _, cur_argmax_idxes = score_t.max(one_dim)
                cur_corrs = answer_hit_t.gather(one_dim, cur_argmax_idxes.unsqueeze(one_dim))  # [bs, len_q, len_k|1, 2*R|1]
                cur_dim_corr_count = cur_corrs.sum()
                # compile loss
                one_loss = LossHelper.compile_leaf_info(f"d{one_dim}", cur_dim_loss, cur_dim_count,
                                                        loss_lambda=one_lambda, corr=cur_dim_corr_count)
                all_losses.append(one_loss)
        return self._compile_component_loss("orp", all_losses)

    # =====
    # add a linear layer for speical loss
    def add_node_special(self, masklm_node):
        self.speical_hid_layer = self.add_sub_node("shid", Affine(
            self.pc, self.input_dim, masklm_node.conf.hid_dim, act=masklm_node.conf.hid_act))

    # borrow masklm and predict for local_shuffle2
    def loss_special(self, repr_t, mask_t, disturb_keep_arr, input_map, masklm_node):
        pred_mask_t = BK.copy(mask_t)
        pred_mask_t *= BK.input_real(1.-disturb_keep_arr)  # not for the non-shuffled ones
        abs_posi = input_map["posi"]  # shuffled positions
        # no predictions for ARTI_ROOT
        if self.add_root_token:
            pred_mask_t[:, 0] = 0.  # [bs, slen]
            abs_posi = (abs_posi[:,1:]-1)  # offset by 1
            pred_mask_t = pred_mask_t[:,1:]  # remove root
        corr_targets_arr = input_map["word"][np.arange(len(abs_posi))[:,np.newaxis], abs_posi]
        repr_t_hid = self.speical_hid_layer(repr_t)  # go through hid here!!
        loss_item = masklm_node.loss([repr_t_hid], pred_mask_t, {"word": corr_targets_arr}, active_hid=False)
        if len(loss_item) == 0:
            return loss_item
        # todo(note): simply change its name
        vs = [v for v in loss_item.values()]
        assert len(vs)==1
        return {"orp.d-2": vs[0]}

# b tasks/zmlm/model/mods/orderpr:305
