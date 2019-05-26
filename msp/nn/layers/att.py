#

# about attentions
# todo(+2): for convenience, assume using torch-backend

import numpy as np
import math

from ..backends import BK
from .basic import ActivationHelper, BasicNode, Dropout, NoDropRop, NoFixRop, RefreshOptions
from .ff import Affine, Embedding
from msp.utils import Constants

# (key, query, value)
# todo(+2): use cache?
class AttentionNode(BasicNode):
    def __init__(self, pc, dim_k, dim_q, dim_v, att_dropout, init_rop=None, name=None):
        super().__init__(pc, name, init_rop)
        #
        self.dim_k = dim_k
        self.dim_q = dim_q
        self.dim_v = dim_v
        # todo(+3): here no fixed shape
        rr = NoFixRop()
        rr.add_fixed_value("hdrop", att_dropout)
        self.adrop = self.add_sub_node("adrop", Dropout(pc, (), init_rop=rr))

    # return re-weighted vs
    def get_output_dims(self, *input_dims):
        return (self.dim_v, )

    # #
    # def obtain_mask(self, mask_k, mask_q):
    #     if mask_k is None:
    #         if mask_q is None: return None
    #         else: return mask_q.unsqueeze(-1)
    #     else:
    #         if mask_q is None: mask_k.unsqueeze(-2)
    #         else: return mask_q.unsqueeze(-1)*mask_k.unsqueeze(-2)

# feed-forward attention
class FfAttentionNode(AttentionNode):
    def __init__(self, pc, dim_k, dim_q, dim_v, hidden_size, att_dropout, init_rop=None, name=None):
        super().__init__(pc, dim_k, dim_q, dim_v, att_dropout, init_rop, name)
        # parameters -- split mlp for possible cache-usage
        self.affine_k = self.add_sub_node("ak", Affine(pc, dim_k, hidden_size, bias=True, init_rop=NoDropRop(), affine2=True))
        self.affine_q = self.add_sub_node("aq", Affine(pc, dim_q, hidden_size, bias=False, init_rop=NoDropRop(), affine2=True))
        self.affine_top_w = self.add_param("w", (hidden_size, 1))       # special transpose

    # [*, len]
    def __call__(self, key, value, query, mask_k=None):
        # todo(+2): cache keys?
        key_up = self.affine_k(key)         # [*, len_k, hidden_size]
        query_up = self.affine_q(key)       # [*, len_q, hidden_size]
        att_hidden = BK.tanh(key_up.unsqueeze(-3)+query_up.unsqueeze(-2))       # [*, len_q, len_k, hidden_size]
        # todo(warn): dropout?
        scores = BK.matmul(att_hidden, self.affine_top_w).squeeze(-1)      # [*, len_q, len_k]
        if mask_k is not None:
            scores += (1. - mask_k).unsqueeze(-2).unsqueeze(-3) * Constants.REAL_PRAC_MIN
            # mask_bytes = (mask_k==0.)
            # mask = mask_bytes.unsqueeze(-2).expand_as(scores)   # mask: [*, len_q, len_k]
            # scores = scores.masked_fill(mask, Constants.REAL_PRAC_MIN)
        attn = BK.softmax(scores, -1)       # [*, len_q, len_k]
        drop_attn = self.adrop(attn)  # [*, len_q, len_k]
        context = BK.matmul(drop_attn, value)     # [*, len_q, dim_v]
        return context

# multi-head
class MultiHeadAttention(AttentionNode):
    def __init__(self, pc, dim_k, dim_q, dim_v, d_kqv, head_count, att_dropout, use_ranges, clip_dist, use_neg_dist, init_rop=None, name=None):
        super().__init__(pc, dim_k, dim_q, dim_v, att_dropout, init_rop, name)
        #
        self.head_count = head_count
        self.model_dim = dim_v
        self.d_kqv = d_kqv
        self.clip_dist = clip_dist
        self.use_neg_dist = use_neg_dist
        self.use_ranges = use_ranges
        self.att_ranges = [2**z-1 for z in range(self.head_count)]  # exclude if dist>rr
        # todo(+2): should we use more dropouts here?
        # no dropouts here
        self.affine_k = self.add_sub_node("ak", Affine(pc, dim_k, head_count*d_kqv, init_rop=NoDropRop(), bias=True, affine2=True))
        self.affine_q = self.add_sub_node("aq", Affine(pc, dim_q, head_count*d_kqv, init_rop=NoDropRop(), bias=True, affine2=True))
        self.affine_v = self.add_sub_node("av", Affine(pc, dim_v, head_count*d_kqv, init_rop=NoDropRop(), bias=True, affine2=True))
        # rel dist keys
        self.use_distance = (self.clip_dist > 0)
        if self.use_distance:
            self.edge_keys = self.add_sub_node("ek", Embedding(pc, 2*self.clip_dist+1, d_kqv, fix_row0=False, init_rop=NoDropRop()))
            self.edge_values = self.add_sub_node("ev", Embedding(pc, 2*self.clip_dist+1, d_kqv, fix_row0=False, init_rop=NoDropRop()))
        # todo(+2): with drops(y) & act(n) & bias(y)?
        self.final_linear = self.add_sub_node("fl", Affine(pc, head_count*d_kqv, dim_v, bias=True, affine2=True))

    #
    def _shape_project(self, x):
        x_size = BK.get_shape(x)[:-1] + [self.head_count, -1]
        return BK.transpose(BK.reshape(x, x_size), -2, -3)

    def _unshape_project(self, x):
        x_size = BK.get_shape(x)
        return BK.reshape(BK.transpose(x, -2, -3), x_size[:-3]+[x_size[-2], -1])

    # todo(+2): assuming (batch, len, dim)
    def __call__(self, key, value, query, mask_k=None):
        batch_dim_size = len(BK.get_shape(key))-2
        key_len = BK.get_shape(key, -2)
        query_len = BK.get_shape(query, -2)
        #
        # prepare distances if needed
        if self.use_distance:
            dist_x = BK.unsqueeze(BK.arange_idx(0, key_len), 0)
            dist_y = BK.unsqueeze(BK.arange_idx(0, query_len), 1)
            distance = dist_x - dist_y      # [query, key]
            if not self.use_neg_dist:
                distance = BK.abs(distance)
            distance = BK.clamp(distance, min=-self.clip_dist, max=self.clip_dist) + self.clip_dist
        else:
            distance = None
        #
        # 1. project the three
        key_up = self._shape_project(self.affine_k(key))        # [*, head, len_k, d_k]
        value_up = self._shape_project(self.affine_v(value))    # [*, head, len_k, d_v]
        query_up = self._shape_project(self.affine_q(query))    # [*, head, len_q, d_k]
        # 2. calculate and scale scores
        query_up = query_up / math.sqrt(self.d_kqv)
        scores = BK.matmul(query_up, BK.transpose(key_up, -1, -2))  # [*, head, len_q, len_k]
        # todo(+2): for convenience, assuming using pytorch here and *==batch_size
        if self.use_distance:
            distance_out = self.edge_keys(distance)     # [query, key, d_kqv]
            out = distance_out.view([1]*batch_dim_size + [1, query_len, key_len, self.d_kqv])
            # [*, head, len_q, 1, d_k] * [*, 1, query, key, d_kqv] = [*, head, len_q, 1, len_k]
            # add_term = BK.matmul(query_up.unsqueeze(-2), out.transpose(-1, -2)).squeeze(-2)
            add_term = BK.matmul(out, query_up.unsqueeze(-1)).squeeze(-1)
            scores += add_term
        # 3. apply attention
        if mask_k is not None:
            # todo(warn): mask as [*, len]
            scores += (1.-mask_k).unsqueeze(-2).unsqueeze(-3) * Constants.REAL_PRAC_MIN
            # mask_bytes = (mask_k==0.)
            # mask = mask_bytes.unsqueeze(-2).unsqueeze(-3).expand_as(scores)   # mask: [*, head, len_q, len_k]
            # scores = scores.masked_fill(mask, Constants.REAL_PRAC_MIN)
        # -- ranged clip
        if self.use_ranges:
            last_dims = [self.head_count, query_len, key_len]
            rr_x = BK.unsqueeze(BK.arange_idx(0, key_len), 0)
            rr_y = BK.unsqueeze(BK.arange_idx(0, query_len), 1)
            rr_xy = BK.abs(rr_x - rr_y).float().unsqueeze(0).expand(last_dims)  # [head, query, key]
            scores = scores.masked_fill(
                (rr_xy > BK.input_real(self.att_ranges).unsqueeze(-1).unsqueeze(-1)), Constants.REAL_PRAC_MIN)
        #
        attn = BK.softmax(scores, -1)       # [*, head, len_q, len_k]
        drop_attn = self.adrop(attn)  # [*, head, len_q, len_k]
        context = BK.matmul(drop_attn, value_up)     # [*, head, len_q, d_v]
        if self.use_distance:
            # specific rel-position values as in https://arxiv.org/pdf/1803.02155.pdf
            distance_out2 = self.edge_values(distance)      # [query, key, dim]
            out2 = distance_out2.view([1]*batch_dim_size + [1, query_len, key_len, self.d_kqv])
            add_term2 = BK.matmul(drop_attn.unsqueeze(-2), out2).squeeze(-2)  # [*, head, len_q, d_v]
            context += add_term2
        # 4. final
        context_merge = self._unshape_project(context)      # [*, len_q, d_v*head]
        output = self.final_linear(context_merge)           # [*, len_q, mdim]
        return output

# special form of multihead attention (fixed attention)
class MultiHeadFixedAttention(AttentionNode):
    def __init__(self, pc, dim_k, dim_q, dim_v, d_kqv, head_count, att_dropout, use_ranges, self_weight, init_rop=None, name=None):
        super().__init__(pc, dim_k, dim_q, dim_v, att_dropout, init_rop, name)
        #
        self.head_count = head_count
        self.model_dim = dim_v
        self.d_kqv = d_kqv
        # todo(+3): is this a good choice, or make it tunable or parameterized?
        self.att_scales = [0.1] * self.head_count
        self.att_ranges = [2**z-1 for z in range(self.head_count)]        # exclude if dist>rr
        self.use_ranges = use_ranges
        self.self_weight = self_weight              # should be negative!
        # no dropouts here
        self.affine_v = self.add_sub_node("av", Affine(pc, dim_v, head_count*d_kqv, init_rop=NoDropRop(), bias=True, affine2=True))
        # todo(+2): with drops(y) & act(n) & bias(y)?
        self.final_linear = self.add_sub_node("fl", Affine(pc, head_count*d_kqv, dim_v, bias=True, affine2=True))

    #
    def _shape_project(self, x):
        x_size = BK.get_shape(x)[:-1] + [self.head_count, -1]
        return BK.transpose(BK.reshape(x, x_size), -2, -3)

    def _unshape_project(self, x):
        x_size = BK.get_shape(x)
        return BK.reshape(BK.transpose(x, -2, -3), x_size[:-3] + [x_size[-2], -1])

    #
    def __call__(self, key, value, query, mask_k=None):
        batch_dim_size = len(BK.get_shape(key)) - 2
        value_len = BK.get_shape(value, -2)
        # 1. only project the value
        value_up = self._shape_project(self.affine_v(value))  # [*, head, len_k, d_v]
        # 2. get fixed attention scores (no parameters here)
        with BK.no_grad_env():
            last_dims = [self.head_count, value_len, value_len]
            #
            dist_x = BK.unsqueeze(BK.arange_idx(0, value_len), 0)
            dist_y = BK.unsqueeze(BK.arange_idx(0, value_len), 1)
            distance = BK.abs(dist_x-dist_y).float().unsqueeze(0).expand(last_dims)    # [head, query, key]
            # - multi range clip
            if self.use_ranges:
                clip_distance = distance.masked_fill((distance>BK.input_real(self.att_ranges).unsqueeze(-1).unsqueeze(-1)),
                                                     Constants.REAL_PRAC_MAX)
            else:
                clip_distance = distance
            # - settable self-weight (use special weight instead of 0.)
            neg_distance = BK.eye(value_len) * self.self_weight - clip_distance
            # - multi scales: [head, query, key]
            fixed_scores = neg_distance.unsqueeze(0) * BK.input_real(self.att_scales).unsqueeze(-1).unsqueeze(-1)
            # [*, head, q-len, k-len]
            fixed_scores = fixed_scores.view([1]*batch_dim_size + last_dims).expand(BK.get_shape(key)[:batch_dim_size] + last_dims)
        # 3. apply attention
        if mask_k is not None:
            # mask is [*, len], thus no need to expand here
            fixed_scores += (1.-mask_k).unsqueeze(-2).unsqueeze(-3) * Constants.REAL_PRAC_MIN
        #
        attn = BK.softmax(fixed_scores, -1)  # [*, head, len_q, len_k]
        drop_attn = self.adrop(attn)  # [*, head, len_q, len_k]
        context = BK.matmul(drop_attn, value_up)  # [*, head, len_q, d_v]
        # 4. final
        context_merge = self._unshape_project(context)  # [*, len_q, d_v*head]
        output = self.final_linear(context_merge)  # [*, len_q, mdim]
        return output
