#

# about attentions
# todo(+2): for convenience, assume using torch-backend

import numpy as np
import math

from msp.utils import Constants, Conf
from ..backends import BK
from .basic import BasicNode, Dropout, NoDropRop, NoFixRop
from .ff import Affine, Embedding, PosiEmbedding

# =====
# Attention Node Configure
# todo(warn): no keep of this conf, since
class AttConf(Conf):
    def __init__(self):
        self.type = "mh"
        self.d_kqv = 64  # also as hidden layer size if needed
        self.att_dropout = 0.1
        # mainly for multi-head
        self.head_count = 8
        self.use_ranges = False
        #
        self.clip_dist = 0
        self.use_neg_dist = False
        self.use_fix_dist = False
        #
        self.out_act = 'linear'
        # for relational one
        self.dim_r = 5
        # should be set specifically
        self._fixed_range_val = -1

# =====
# (key, query, value)
# todo(+2): use cache?
class AttentionNode(BasicNode):
    def __init__(self, pc, dim_k, dim_q, dim_v, aconf: AttConf, init_rop=None, name=None):
        super().__init__(pc, name, init_rop)
        self.dim_k = dim_k
        self.dim_q = dim_q
        self.dim_v = dim_v
        # todo(+3): here no fixed shape
        rr = NoFixRop()
        rr.add_fixed_value("hdrop", aconf.att_dropout)
        self.adrop = self.add_sub_node("adrop", Dropout(pc, (), init_rop=rr))

    def __repr__(self):
        return f"# AttNode({self.__class__.__name__}): k={self.dim_k},q={self.dim_q},v={self.dim_v}"

    # return re-weighted vs
    def get_output_dims(self, *input_dims):
        return (self.dim_v, )

    @staticmethod
    def get_att_node(node_type, pc, dim_k, dim_q, dim_v, aconf: AttConf, name=None, init_rop=None):
        _ATT_TYPES = {"ff": FfAttentionNode, "mh": MultiHeadAttention,
                      "mhr": MultiHeadRelationalAttention, "mhrs": MultiHeadSelfDistAttention}
        if isinstance(node_type, str):
            node_c = _ATT_TYPES[node_type]
        else:
            node_c = node_type
        ret = node_c(pc, dim_k, dim_q, dim_v, aconf, name=name, init_rop=init_rop)
        return ret

# feed-forward attention
class FfAttentionNode(AttentionNode):
    def __init__(self, pc, dim_k, dim_q, dim_v, aconf: AttConf, init_rop=None, name=None):
        super().__init__(pc, dim_k, dim_q, dim_v, aconf, init_rop, name)
        # parameters -- split mlp for possible cache-usage
        hidden_size = aconf.d_kqv
        self.affine_k = self.add_sub_node("ak", Affine(pc, dim_k, hidden_size, bias=True, init_rop=NoDropRop()))
        self.affine_q = self.add_sub_node("aq", Affine(pc, dim_q, hidden_size, bias=False, init_rop=NoDropRop()))
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
        attn = BK.softmax(scores, -1)       # [*, len_q, len_k]
        drop_attn = self.adrop(attn)  # [*, len_q, len_k]
        context = BK.matmul(drop_attn, value)     # [*, len_q, dim_v]
        return context

# =====
# MultiHeadAttention helpers

class AttRangeHelper(BasicNode):
    def __init__(self, pc, aconf: AttConf):
        super().__init__(pc, None, None)
        self.head_count = aconf.head_count
        # exclude if dist>rr
        fixed_range_val = aconf._fixed_range_val
        if fixed_range_val >= 0:
            self.att_ranges = [fixed_range_val] * self.head_count
        else:
            self.att_ranges = [2**z - 1 for z in range(self.head_count)]
        self._att_ranges_t = None

    def refresh(self, rop=None):
        super().refresh(rop)
        # refresh reuse-able tensors: [*, query, key]
        if self.att_ranges:
            self._att_ranges_t = BK.input_real(self.att_ranges).unsqueeze(-1).unsqueeze(-1)

    def __call__(self, query_len, key_len, att_scores_t):
        # -- ranged clip
        # todo(+3): can have repeated calculations with distances
        last_dims = [self.head_count, query_len, key_len]
        rr_x = BK.unsqueeze(BK.arange_idx(0, key_len), 0)
        rr_y = BK.unsqueeze(BK.arange_idx(0, query_len), 1)
        rr_xy = BK.abs(rr_x - rr_y).float().unsqueeze(0).expand(last_dims)  # [head, query, key]
        scores = att_scores_t.masked_fill(rr_xy > self._att_ranges_t, Constants.REAL_PRAC_MIN)
        return scores

class AttDistHelper(BasicNode):
    def __init__(self, pc, aconf: AttConf, dim: int):
        super().__init__(pc, None, None)
        self.use_neg_dist = aconf.use_neg_dist
        self.clip_dist = aconf.clip_dist
        if aconf.use_fix_dist:
            assert not self.use_neg_dist, "Currently does not support Neg fixed distance"
            posi_max_len = self.clip_dist + 1
            self.edge_atts = self.add_sub_node("ek", PosiEmbedding(pc, dim, posi_max_len))
            self.edge_values = self.add_sub_node("ev", PosiEmbedding(pc, dim, posi_max_len))
        else:
            self.edge_atts = self.add_sub_node("ek", Embedding(pc, 2 * self.clip_dist + 1, dim, fix_row0=False,
                                                               init_rop=NoDropRop()))
            self.edge_values = self.add_sub_node("ev", Embedding(pc, 2 * self.clip_dist + 1, dim, fix_row0=False,
                                                                 init_rop=NoDropRop()))

    def obatin_from_distance(self, distance):
        # use different ranges!
        if not self.use_neg_dist:
            distance = BK.clamp(BK.abs(distance), max=self.clip_dist)
        else:
            distance = BK.clamp(distance, min=-self.clip_dist, max=self.clip_dist) + self.clip_dist
        dist_atts = self.edge_atts(distance)
        dist_values = self.edge_values(distance)
        return dist_atts, dist_values

    def __call__(self, query_len, key_len):
        dist_x = BK.unsqueeze(BK.arange_idx(0, key_len), 0)
        dist_y = BK.unsqueeze(BK.arange_idx(0, query_len), 1)
        distance = dist_x - dist_y  # [query, key]
        dist_atts, dist_values = self.obatin_from_distance(distance)
        # [lenq, lenk], [lenq, lenk, dim], [lenq, lenk, dim]
        return distance, dist_atts, dist_values

# =====
# basic multi-head
class MultiHeadAttention(AttentionNode):
    def __init__(self, pc, dim_k, dim_q, dim_v, aconf: AttConf, init_rop=None, name=None):
        super().__init__(pc, dim_k, dim_q, dim_v, aconf, init_rop, name)
        self.model_dim = dim_v
        self.head_count = aconf.head_count
        self.d_kqv = aconf.d_kqv
        self.clip_dist = aconf.clip_dist
        self.use_neg_dist = aconf.use_neg_dist
        # =====
        # pre-fixed values for later computation
        self._att_scale = math.sqrt(self.d_kqv)
        if aconf.use_ranges:
            self.range_clipper = self.add_sub_node("rc", AttRangeHelper(pc, aconf))
        else:
            self.range_clipper = None
        # =====
        eff_hidden_size = self.head_count * self.d_kqv
        # todo(+2): should we use more dropouts here?
        # no dropouts here
        self.affine_k = self.add_sub_node("ak", Affine(pc, dim_k, eff_hidden_size, init_rop=NoDropRop()))
        self.affine_q = self.add_sub_node("aq", Affine(pc, dim_q, eff_hidden_size, init_rop=NoDropRop()))
        self.affine_v = self.add_sub_node("av", Affine(pc, dim_v, eff_hidden_size, init_rop=NoDropRop()))
        # rel dist keys
        self.use_distance = (self.clip_dist > 0)
        if self.use_distance:
            self.dist_helper = self.add_sub_node("dh", AttDistHelper(pc, aconf, self.d_kqv))
        else:
            self.dist_helper = None
        # todo(+2): with drops(y) & act(?) & bias(y)?
        self.final_linear = self.add_sub_node("fl", Affine(pc, eff_hidden_size, dim_v, act=aconf.out_act))

    # [*, len, head*dim] <-> [*, head, len, dim]
    def _shape_project(self, x):
        x_size = BK.get_shape(x)[:-1] + [self.head_count, -1]
        return BK.transpose(BK.reshape(x, x_size), -2, -3)

    def _unshape_project(self, x):
        x_size = BK.get_shape(x)
        return BK.reshape(BK.transpose(x, -2, -3), x_size[:-3]+[x_size[-2], -1])

    # todo(+2): assuming (batch, len, dim)
    # prob_qk is outside extra-prob for mixing by prob_mix_rate
    def __call__(self, key, value, query, mask_k=None, mask_qk=None, eprob_qk=None, eprob_mix_rate=0., eprob_head_count=0):
        batch_dim_size = len(BK.get_shape(key))-2
        query_len = BK.get_shape(query, -2)
        key_len = BK.get_shape(key, -2)
        #
        # prepare distances if needed
        if self.use_distance:
            distance, dist_atts, dist_values = self.dist_helper(query_len, key_len)
        else:
            distance, dist_atts, dist_values = None, None, None
        #
        # 1. project the three
        key_up = self._shape_project(self.affine_k(key))        # [*, head, len_k, d_k]
        value_up = self._shape_project(self.affine_v(value))    # [*, head, len_k, d_v]
        query_up = self._shape_project(self.affine_q(query))    # [*, head, len_q, d_k]
        # 2. calculate and scale scores
        query_up = query_up / self._att_scale
        scores = BK.matmul(query_up, BK.transpose(key_up, -1, -2))  # [*, head, len_q, len_k]
        # todo(+2): for convenience, assuming using pytorch here and *==batch_size
        if self.use_distance:
            distance_out = dist_atts     # [query, key, d_kqv]
            out = distance_out.view([1]*batch_dim_size + [1, query_len, key_len, self.d_kqv])
            # [*, head, len_q, 1, d_k] * [*, 1, query, key, d_kqv] = [*, head, len_q, 1, len_k]
            # add_term = BK.matmul(query_up.unsqueeze(-2), out.transpose(-1, -2)).squeeze(-2)
            add_term = BK.matmul(out, query_up.unsqueeze(-1)).squeeze(-1)
            scores += add_term
        # 3. apply attention
        if mask_k is not None:
            # todo(warn): mask as [*, len]
            scores += (1.-mask_k).unsqueeze(-2).unsqueeze(-3) * Constants.REAL_PRAC_MIN
        if mask_qk is not None:
            # todo(note): mask as [*, len-q, len-k]
            scores += (1.-mask_qk).unsqueeze(-3) * Constants.REAL_PRAC_MIN
        if self.range_clipper:
            scores = self.range_clipper(query_len, key_len, scores)
        attn = BK.softmax(scores, -1)       # [*, head, len_q, len_k]
        # mix eprob for the first several heads?
        if eprob_qk is not None and eprob_mix_rate>0. and eprob_head_count>0:
            # todo(warn): here assume that [*] == [bs]; cannot do inplaced operation here, otherwise backward error
            # attn[:, :eprob_head_count] = eprob_mix_rate*(eprob_qk.unsqueeze(-3)) + (1.-eprob_mix_rate)*(attn[:, :eprob_head_count])
            attn0 = eprob_mix_rate*(eprob_qk.unsqueeze(-3)) + (1.-eprob_mix_rate)*(attn[:, :eprob_head_count])
            attn1 = attn[:, eprob_head_count:]
            attn = BK.concat([attn0, attn1], 1)
        drop_attn = self.adrop(attn)  # [*, head, len_q, len_k]
        context = BK.matmul(drop_attn, value_up)     # [*, head, len_q, d_v]
        if self.use_distance:
            # specific rel-position values as in https://arxiv.org/pdf/1803.02155.pdf
            distance_out2 = dist_values      # [query, key, dim]
            out2 = distance_out2.view([1]*batch_dim_size + [1, query_len, key_len, self.d_kqv])
            add_term2 = BK.matmul(drop_attn.unsqueeze(-2), out2).squeeze(-2)  # [*, head, len_q, d_v]
            context += add_term2
        # 4. final
        context_merge = self._unshape_project(context)      # [*, len_q, d_v*head]
        output = self.final_linear(context_merge)           # [*, len_q, mdim]
        return output

# =====
# relational attention: adopt another input for the relational embeddings
# todo(+N): repeated codes with the class of "MultiHeadAttention"
# TODO(+N): still require too much memory
class MultiHeadRelationalAttention(AttentionNode):
    def __init__(self, pc, dim_k, dim_q, dim_v, aconf: AttConf, init_rop=None, name=None):
        super().__init__(pc, dim_k, dim_q, dim_v, aconf, init_rop, name)
        self.model_dim = dim_v
        self.head_count = aconf.head_count
        self.d_kqv = aconf.d_kqv
        self.d_r = aconf.dim_r
        # =====
        # pre-fixed values for later computation
        self._att_scale = math.sqrt(self.d_kqv)
        if aconf.use_ranges:
            self.range_clipper = self.add_sub_node("rc", AttRangeHelper(pc, aconf))
        else:
            self.range_clipper = None
        # =====
        eff_hidden_size = self.head_count * self.d_kqv
        # todo(+2): should we use more dropouts here?
        # no dropouts here
        self.affine_k = self.add_sub_node("ak", Affine(pc, dim_k, eff_hidden_size, init_rop=NoDropRop()))
        # todo(+N): need to add rel-aware values as well?
        self.affine_v = self.add_sub_node("av", Affine(pc, dim_v, eff_hidden_size, init_rop=NoDropRop()))
        # special relational one for query
        self.affine_q = self.add_sub_node("aq", Affine(pc, dim_q, eff_hidden_size*self.d_r, init_rop=NoDropRop()))
        # todo(+2): with drops(y) & act(?) & bias(y)?
        self.final_linear = self.add_sub_node("fl", Affine(pc, eff_hidden_size, dim_v, act=aconf.out_act))

    def _shape_project(self, x):
        x_size = BK.get_shape(x)[:-1] + [self.head_count, -1]
        return BK.transpose(BK.reshape(x, x_size), -2, -3)

    def _unshape_project(self, x):
        x_size = BK.get_shape(x)
        return BK.reshape(BK.transpose(x, -2, -3), x_size[:-3]+[x_size[-2], -1])

    # todo(+2): kqv=[*, len, dim], rel=[*, len_q, len_k, dim_r]
    def __call__(self, key, value, query, rel, mask_k=None):
        batch_dim_size = len(BK.get_shape(key)) - 2
        query_len = BK.get_shape(query, -2)
        key_len = BK.get_shape(key, -2)
        #
        # 1. project the three
        key_up = self._shape_project(self.affine_k(key))  # [*, head, len_k, d_k]
        value_up = self._shape_project(self.affine_v(value))  # [*, head, len_k, d_v]
        # special for query, select parameters by relation
        query_up = self._shape_project(self.affine_q(query))  # [*, head, len_q, d_k*d_r]
        # 2. calculate relational att scores
        key_up = (key_up / self._att_scale).unsqueeze(-3).unsqueeze(-2)  # [*, head, 1, len_k, 1, d_k]
        query_up = query_up.view(BK.get_shape(query_up)[:-1]+[1,self.d_kqv,self.d_r])  # [*, head, len_q, 1, d_k, d_r]
        scores0 = BK.matmul(key_up, query_up).squeeze(-2)  # [*, head, len_q, len_k, d_r]
        scores = BK.sum(scores0 * rel.unsqueeze(-4), -1)  # [*, head, len_q, len_k]
        # 3. attention
        if mask_k is not None:
            # todo(warn): mask as [*, len]
            scores += (1.-mask_k).unsqueeze(-2).unsqueeze(-3) * Constants.REAL_PRAC_MIN
        if self.range_clipper:
            scores = self.range_clipper(query_len, key_len, scores)
        attn = BK.softmax(scores, -1)  # [*, head, len_q, len_k]
        drop_attn = self.adrop(attn)  # [*, head, len_q, len_k]
        context = BK.matmul(drop_attn, value_up)  # [*, head, len_q, d_v]
        # todo(+N): do the values need to be aware of rel?
        # 4. final
        context_merge = self._unshape_project(context)  # [*, len_q, d_v*head]
        output = self.final_linear(context_merge)  # [*, len_q, mdim]
        return output

# =====
class MultiHeadSelfDistAttention(AttentionNode):
    def __init__(self, pc, dim_k, dim_q, dim_v, aconf: AttConf, init_rop=None, name=None):
        super().__init__(pc, dim_k, dim_q, dim_v, aconf, init_rop, name)
        self.dim_r = aconf.dim_r
        self.rel_att = self.add_sub_node("ra", MultiHeadRelationalAttention(pc, dim_k, dim_q, dim_v, aconf))
        assert (aconf.clip_dist > 0), "Must use distance here!"
        self.dist_helper = self.add_sub_node("dh", AttDistHelper(pc, aconf, aconf.dim_r))

    # todo(+2): assuming (batch, len, dim)
    def __call__(self, key, value, query, mask_k=None):
        batch_dim_size = len(BK.get_shape(key)) - 2
        key_len = BK.get_shape(key, -2)
        query_len = BK.get_shape(query, -2)
        # get distance embeddings
        distance, dist_atts, dist_values = self.dist_helper(query_len, key_len)  # [query, key, d_r]
        # todo(+N): make use of dist_values?
        # expand batch dimensions
        rel_input = dist_atts.view([1] * batch_dim_size + [query_len, key_len, self.dim_r])
        return self.rel_att(key, value, query, rel_input, mask_k=mask_k)
