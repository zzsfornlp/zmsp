#

# the vertical recursive/recurrent module

from typing import List, Union, Dict, Iterable
import numpy as np
import math
from msp.utils import Constants, Conf, zwarn, Helper
from msp.nn import BK
from msp.nn.layers import BasicNode, LstmNode2, Affine, Dropout, NoDropRop, LayerNorm
from msp.zext.process_train import SVConf, ScheduledValue
from .matt import MAttConf, MAttNode
from .fcomb import FCombConf, FCombNode
from .base import AffineCombiner
from ...base import BaseModuleConf, BaseModule, LossHelper
# from ...helper import EntropyHelper

# =====
# single node

# conf
class VRecConf(Conf):
    def __init__(self):
        # layer norm
        self.use_pre_norm = False  # combine(ss(LN(x)), x)
        self.use_post_norm = False  # LN(combine(ss(x), x)
        # -----
        # selfatt conf
        self.feat_mod = "matt"  # matt/fcomb
        self.matt_conf = MAttConf()
        self.fc_conf = FCombConf()
        # what to feed
        self.feat_qk_lambda_orig = 0.  # for Q/K, this*orig_t + (1-this)*rec_t
        self.feat_v_lambda_orig = 0.  # for V(k), ...
        # -----
        # combiner
        self.comb_mode = "affine"  # affine/lstm
        # affine mode
        self.comb_affine_q = False
        self.comb_affine_v = False  # if d_v != dim_q, should affine this
        self.comb_affine_act = 'linear'  # maybe linear is fine since at outside there may be LayerNorm
        self.comb_affine_drop = 0.1
        # what to feed
        self.comb_q_lambda_orig = 0.  # for q, ...
        # -----
        # further ff?
        self.ff_dim = 0
        self.ff_drop = 0.1
        self.ff_act = "relu"

    @property
    def attn_count(self):
        if self.feat_mod == "matt":
            return self.matt_conf.head_count
        elif self.feat_mod == "fcomb":
            return self.fc_conf.fc_count
        else:
            raise NotImplementedError()

# cache
class VRecCache:
    def __init__(self):
        # values for cur step: "orig" is the very first input, "rec" is the recusively built one
        self.orig_t = None  # [*, len_q, D]
        self.rec_t = None  # [*, len_q, D]
        self.accu_attn = None  # [*, len_q, len_v, head]
        self.rec_lstm_c_t = None  # optional for lstm mode
        # list on steps
        self.list_hidden = []  # List[*, len_q, D]
        self.list_score = []  # List[*, len_q, len_v, head]
        self.list_attn = []  # List[*, len_q, len_v, head]
        self.list_accu_attn = []  # List[*, len_q, len_v, head]
        self.list_attn_info = []  # Tuple(attn, [prob_valid], [prob_noop], [prob_full])

# one node (one step) in the recursion
class VRecNode(BasicNode):
    def __init__(self, pc, dim: int, conf: VRecConf):
        super().__init__(pc, None, None)
        self.conf = conf
        self.dim = dim
        # =====
        # Feat
        if conf.feat_mod == "matt":
            self.feat_node = self.add_sub_node("feat", MAttNode(pc, dim, dim, dim, conf.matt_conf))
            self.attn_count = conf.matt_conf.head_count
        elif conf.feat_mod == "fcomb":
            self.feat_node = self.add_sub_node("feat", FCombNode(pc, dim, dim, dim, conf.fc_conf))
            self.attn_count = conf.fc_conf.fc_count
        else:
            raise NotImplementedError()
        feat_out_dim = self.feat_node.get_output_dims()[0]
        # =====
        # Combiner
        if conf.comb_mode == "affine":
            self.comb_aff = self.add_sub_node("aff", AffineCombiner(
                pc, [dim, feat_out_dim], [conf.comb_affine_q, conf.comb_affine_v], dim,
                out_act=conf.comb_affine_act, out_drop=conf.comb_affine_drop))
            self.comb_f = lambda q, v, c: (self.comb_aff([q,v]), None)
        elif conf.comb_mode == "lstm":
            self.comb_lstm = self.add_sub_node("lstm", LstmNode2(pc, feat_out_dim, dim))
            self.comb_f = self._call_lstm
        else:
            raise NotImplementedError()
        # =====
        # ff
        if conf.ff_dim > 0:
            self.has_ff = True
            self.linear1 = self.add_sub_node("l1", Affine(pc, dim, conf.ff_dim, act=conf.ff_act, init_rop=NoDropRop()))
            self.dropout1 = self.add_sub_node("d1", Dropout(pc, (conf.ff_dim, ), fix_rate=conf.ff_drop))
            self.linear2 = self.add_sub_node("l2", Affine(pc, conf.ff_dim, dim, act="linear", init_rop=NoDropRop()))
            self.dropout2 = self.add_sub_node("d2", Dropout(pc, (dim, ), fix_rate=conf.ff_drop))
        else:
            self.has_ff = False
        # layer norms
        if conf.use_pre_norm:
            self.att_pre_norm = self.add_sub_node("aln1", LayerNorm(pc, dim))
            self.ff_pre_norm = self.add_sub_node("fln1", LayerNorm(pc, dim))
        else:
            self.att_pre_norm = self.ff_pre_norm = None
        if conf.use_post_norm:
            self.att_post_norm = self.add_sub_node("aln2", LayerNorm(pc, dim))
            self.ff_post_norm = self.add_sub_node("fln2", LayerNorm(pc, dim))
        else:
            self.att_post_norm = self.ff_post_norm = None

    def get_output_dims(self, *input_dims):
        return (self.dim, )

    # -----
    def _call_lstm(self, q, v, c):
        prev_shapes = BK.get_shape(q)[:-1]  # [*, len_q]
        in_reshape = [np.prod(prev_shapes), -1]
        in_q, in_v, in_c = [z.view(in_reshape) for z in [q,v,c]]
        ret_h, ret_c = self.comb_lstm(in_v, (in_q, in_c), None)  # v acts as input, q acts as hidden
        out_reshape = prev_shapes + [-1]  # [*, len_q, D]
        return ret_h.view(out_reshape), ret_c.view(out_reshape)

    # init cache
    def init_call(self, src):
        # init accumulated attn: all 0.
        src_shape = BK.get_shape(src)
        attn_shape = src_shape[:-1] + [src_shape[-2], self.attn_count]  # [*, len_q, len_k, head]
        cache = VRecCache()
        cache.orig_t = src
        cache.rec_t = src  # initially, the same as "orig_t"
        cache.rec_lstm_c_t = BK.zeros(BK.get_shape(src))  # initially zero
        cache.accu_attn = BK.zeros(attn_shape)
        return cache

    # one update step: *[bs, slen, dim]
    def update_call(self, cache: VRecCache, src_mask=None, qk_mask=None,
                    attn_range=None, rel_dist=None, temperature=1., forced_attn=None):
        conf = self.conf
        # -----
        # first call matt to get v
        matt_input_qk = cache.orig_t * conf.feat_qk_lambda_orig + cache.rec_t * (1. - conf.feat_qk_lambda_orig)
        if self.att_pre_norm:
            matt_input_qk = self.att_pre_norm(matt_input_qk)
        matt_input_v = cache.orig_t * conf.feat_v_lambda_orig + cache.rec_t * (1. - conf.feat_v_lambda_orig)
        # todo(note): currently no pre-norm for matt_input_v
        # put attn_range as mask_qk
        if attn_range is not None and attn_range >= 0:  # <0 means not effective
            cur_slen = BK.get_shape(matt_input_qk, -2)
            tmp_arange_t = BK.arange_idx(cur_slen)  # [slen]
            # less or equal!!
            mask_qk = ((tmp_arange_t.unsqueeze(-1) - tmp_arange_t.unsqueeze(0)).abs() <= attn_range).float()
            if qk_mask is not None:  # further with input masks
                mask_qk *= qk_mask
        else:
            mask_qk = qk_mask
        scores, attn_info, result_value = self.feat_node(
            matt_input_qk, matt_input_qk, matt_input_v, cache.accu_attn, mask_k=src_mask, mask_qk=mask_qk,
            rel_dist=rel_dist, temperature=temperature, forced_attn=forced_attn)  # ..., [*, len_q, dv]
        # -----
        # then combine q(hidden) and v(input)
        comb_input_q = cache.orig_t * conf.comb_q_lambda_orig + cache.rec_t * (1. - conf.comb_q_lambda_orig)
        comb_result, comb_c = self.comb_f(comb_input_q, result_value, cache.rec_lstm_c_t)  # [*, len_q, dim]
        if self.att_post_norm:
            comb_result = self.att_post_norm(comb_result)
        # -----
        # ff
        if self.has_ff:
            if self.ff_pre_norm:
                ff_input = self.ff_pre_norm(comb_result)
            else:
                ff_input = comb_result
            ff_output = comb_result + self.dropout2(self.linear2(self.dropout1(self.linear1(ff_input))))
            if self.ff_post_norm:
                ff_output = self.ff_post_norm(ff_output)
        else:  # otherwise no ff
            ff_output = comb_result
        # -----
        # update cache and return output
        # cache.orig_t = cache.orig_t  # this does not change
        cache.rec_t = ff_output
        cache.accu_attn = cache.accu_attn + attn_info[0]  # accumulating attn
        cache.rec_lstm_c_t = comb_c  # optional C for lstm
        cache.list_hidden.append(ff_output)  # all hidden layers
        cache.list_score.append(scores)  # all un-normed scores
        cache.list_attn.append(attn_info[0])  # all normed scores
        cache.list_accu_attn.append(cache.accu_attn)  # all accumulated attns
        cache.list_attn_info.append(attn_info)  # all attn infos
        return ff_output

# =====
# encoder

class VRecEncoderConf(BaseModuleConf):
    def __init__(self):
        super().__init__()
        # -----
        # repeat of common layers
        self.vr_conf = VRecConf()  # vrec conf
        self.num_layer = 4
        self.share_layers = True  # recurrent/recursive if sharing
        self.attn_ranges = []
        # scheduled values
        self.temperature = SVConf().init_from_kwargs(val=1., which_idx="eidx", mode="none", min_val=0.1)
        self.lambda_noop_prob = SVConf().init_from_kwargs(val=0., which_idx="eidx", mode="none", min_val=0.)
        self.lambda_entropy = SVConf().init_from_kwargs(val=0., which_idx="eidx", mode="none", min_val=0.)
        self.lambda_attn_l1 = SVConf().init_from_kwargs(val=0., which_idx="eidx", mode="none", min_val=0.)
        # self.lambda_coverage = SVConf().init_from_kwargs(val=0., which_idx="eidx", mode="none", min_val=0.)
        # special noop loss
        self.noop_epsilon = 0.01  # if <0, means directly using prob themselves without log

    def do_validate(self):
        self.attn_ranges = [int(z) for z in self.attn_ranges]

class VRecEncoder(BaseModule):
    def __init__(self, pc, dim: int, conf: VRecEncoderConf):
        super().__init__(pc, conf, output_dims=(dim,))
        # -----
        if conf.share_layers:
            node = self.add_sub_node("m", VRecNode(pc, dim, conf.vr_conf))
            self.layers = [node for _ in range(conf.num_layer)]  # use the same node!!
        else:
            self.layers = [self.add_sub_node("m", VRecNode(pc, dim, conf.vr_conf)) for _ in range(conf.num_layer)]
        self.attn_ranges = conf.attn_ranges.copy()
        self.attn_ranges.extend([None] * (conf.num_layer - len(self.attn_ranges)))  # None means no restrictions
        # scheduled values
        self.temperature = ScheduledValue(f"{self.name}:temperature", conf.temperature)
        self.lambda_noop_prob = ScheduledValue(f"{self.name}:lambda_noop_prob", conf.lambda_noop_prob)
        self.lambda_entropy = ScheduledValue(f"{self.name}:lambda_entropy", conf.lambda_entropy)
        self.lambda_attn_l1 = ScheduledValue(f"{self.name}:lambda_attn_l1", conf.lambda_attn_l1)
        # self.lambda_coverage = ScheduledValue(f"{self.name}:lambda_coverage", conf.lambda_coverage)

    @property
    def attn_count(self):
        if len(self.layers) > 0:
            all_counts = [z.attn_count for z in self.layers]
            assert all(z==all_counts[0] for z in all_counts), "attn_count disagree!!"
            return all_counts[0]
        else:
            return self.conf.vr_conf.attn_count

    def get_scheduled_values(self):
        return super().get_scheduled_values() + [self.temperature, self.lambda_noop_prob, self.lambda_entropy, self.lambda_attn_l1]

    def __call__(self, src, src_mask=None, qk_mask=None, rel_dist=None, forced_attns=None, collect_loss=False):
        if src_mask is not None:
            src_mask = BK.input_real(src_mask)
        if forced_attns is None:
            forced_attns = [None] * len(self.layers)
        # -----
        # forward
        temperature = self.temperature.value
        cur_hidden = src
        if len(self.layers) > 0:
            cache = self.layers[0].init_call(src)
            for one_lidx, one_layer in enumerate(self.layers):
                cur_hidden = one_layer.update_call(cache, src_mask=src_mask, qk_mask=qk_mask, attn_range=self.attn_ranges[one_lidx],
                                                   rel_dist=rel_dist, temperature=temperature, forced_attn=forced_attns[one_lidx])
        else:
            cache = VRecCache()  # empty one
        # -----
        # collect loss
        if collect_loss:
            loss_item = self._collect_losses(cache)
        else:
            loss_item = None
        return cur_hidden, cache, loss_item

    # =====
    # helpers
    def _loss_noop_prob(self, t):
        noop_epsilon = self.conf.noop_epsilon
        if noop_epsilon >= 0.:
            ret_loss = ((1. + noop_epsilon - t).log() - math.log(noop_epsilon))
        else:
            ret_loss = (1. - t)
        return ret_loss

    def _loss_entropy(self, t_and_dim):
        # the entropy
        t, dim = t_and_dim
        all_ents = - (t * (t + 1e-10).log()).sum(dim)  # reduce on specific dim
        # all_ents = EntropyHelper.cross_entropy(t, t, dim)
        return all_ents

    def _loss_attn_l1(self, t):
        # simply L1 reg for all attn (but only for >=0 part)
        t_act = t * (t>=0.).float()
        return t_act.abs()  # same shape as input

    # collect loss from one layer
    def _collect_losses(self, cache: VRecCache):
        special_losses = []
        list_attn_info = cache.list_attn_info  # List[(attn, list_prob_valid, list_prob_noop, list_prob_full, list_dims)]
        # -----
        cur_lambda_noop_prob = self.lambda_noop_prob.value
        if cur_lambda_noop_prob > 0.:
            noop_losses = self.get_losses_from_attn_list(
                list_attn_info, lambda x: x[2], self._loss_noop_prob, "PNopS", cur_lambda_noop_prob)
            special_losses.extend(noop_losses)
        # -----
        cur_lambda_entropy = self.lambda_entropy.value
        if cur_lambda_entropy > 0.:
            ent_losses = self.get_losses_from_attn_list(
                list_attn_info, lambda x: list(zip(x[3],x[4])), self._loss_entropy, "EntRegS", cur_lambda_entropy)
            special_losses.extend(ent_losses)
        # -----
        cur_lambda_attn_l1 = self.lambda_attn_l1.value
        if cur_lambda_attn_l1 > 0.:
            attn_l1_losses = self.get_losses_from_attn_list(
                list_attn_info, lambda x: [x[0]], self._loss_attn_l1, "AttnL1S", cur_lambda_attn_l1)
            special_losses.extend(attn_l1_losses)
        # -----
        # here combine them into one
        ret_loss_item = self._compile_component_loss("vrec", special_losses)
        return ret_loss_item

    # common procedures for obtaining losses from "list_attn_info"
    @staticmethod
    def get_losses_from_attn_list(list_attn_info: List, ts_f, loss_f, loss_prefix, loss_lambda):
        loss_num = None
        loss_counts: List[int] = []
        loss_sums: List[List] = []
        rets = []
        # -----
        for one_attn_info in list_attn_info:  # each update step
            one_ts: List = ts_f(one_attn_info)  # get tensor list from attn_info
            # get number of losses
            if loss_num is None:
                loss_num = len(one_ts)
                loss_counts = [0] * loss_num
                loss_sums = [[] for _ in range(loss_num)]
            else:
                assert len(one_ts) == loss_num, "mismatched ts length"
            # iter them
            for one_t_idx, one_t in enumerate(one_ts):  # iter on the tensor list
                one_loss = loss_f(one_t)
                # need it to be in the corresponding shape
                loss_counts[one_t_idx] += np.prod(BK.get_shape(one_loss)).item()
                loss_sums[one_t_idx].append(one_loss.sum())
        # for different steps
        for i, one_loss_count, one_loss_sums in zip(range(len(loss_counts)), loss_counts, loss_sums):
            loss_leaf = LossHelper.compile_leaf_info(f"{loss_prefix}{i}", BK.stack(one_loss_sums, 0).sum(),
                                                     BK.input_real(one_loss_count), loss_lambda=loss_lambda)
            rets.append(loss_leaf)
        return rets
