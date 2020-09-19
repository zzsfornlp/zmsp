#

# Component 1 for MAtt: scorer
# (Q, K, accu_attns, rel_dist) => scores [*, len_q, len_k, head]

from typing import List, Union, Dict, Iterable
from msp.utils import Constants, Conf, zwarn, Helper
from msp.nn import BK
from msp.nn.layers import BasicNode, ActivationHelper, Affine, NoDropRop, NoFixRop, Dropout, PosiEmbedding2, LayerNorm
from .base import AffineHelperNode

# -----

# conf
class MAttScorerConf(Conf):
    def __init__(self):
        # special
        self.param_init_scale = 1.  # todo(+N): ugly extra scale
        self.use_piece4init = False
        # dot-product for Q*K
        self.d_qk = 64  # query*key -> score
        self.qk_act = 'linear'
        self.qk_drop = 0.
        self.score_scale_power = 0.5  # scale power for the dot product scoring
        self.use_unhead_score = False  # general score (untyped) for q*k
        # relative distances?
        self.use_rel_dist = False
        self.rel_emb_dim = 512  # this should be enough
        self.rel_emb_zero0 = True  # zero the 0 item
        self.rel_dist_clip = 1000  # this should be enough
        self.rel_dist_abs = False  # discard sign on rel dist, if True: [0, clip] else [-clip, clip]
        self.rel_init_sincos = True  # init and freeze rel_dist embeddings with sincos version
        # q(/h)-specific lambdas to minus accu_attns
        self.use_lambq = False
        self.lambq_hdim = 128
        self.lambq_hdrop = 0.1
        self.lambq_hact = "elu"
        self.lambq_fbias = 0.  # fixed extra bias
        self.lambq_fact = "relu"  # output activation after adding bias
        # q(/h)-specific score offset
        self.use_soff = False
        self.soff_hdim = 128
        self.soff_hdrop = 0.1
        self.soff_hact = "elu"
        self.soff_fbias = 0.  # fixed extra bias
        self.soff_fact = "relu"  # output activation after adding bias
        # no (masking out) self-loop?
        self.no_self_loop = False

    def get_att_scale_qk(self):
        return float(self.d_qk ** self.score_scale_power)

# the scorer
class MAttScorerNode(BasicNode):
    def __init__(self, pc: BK.ParamCollection, dim_q, dim_k, head_count, conf: MAttScorerConf):
        super().__init__(pc, None, None)
        self.conf = conf
        self.dim_q, self.dim_k = dim_q, dim_k
        self.head_count = head_count
        # -----
        self.use_unhead_score = conf.use_unhead_score
        _calc_head_count = (1+head_count) if self.use_unhead_score else head_count  # extra dim for overall unhead score
        _d_qk = conf.d_qk
        _hid_size_qk = _calc_head_count * _d_qk
        self._att_scale_qk = conf.get_att_scale_qk()  # scale for dot product
        self.split_dims = [_calc_head_count, -1]
        # affine-qk
        self.affine_q = self.add_sub_node("aq", AffineHelperNode(
            pc, dim_q, _hid_size_qk, hid_act=conf.qk_act, hid_drop=conf.qk_drop,
            hid_piece4init=(_calc_head_count if conf.use_piece4init else 1), init_scale=conf.param_init_scale))
        self.affine_k = self.add_sub_node("ak", AffineHelperNode(
            pc, dim_k, _hid_size_qk, hid_act=conf.qk_act, hid_drop=conf.qk_drop,
            hid_piece4init=(_calc_head_count if conf.use_piece4init else 1), init_scale=conf.param_init_scale))
        # rel positional
        if conf.use_rel_dist:
            self.dist_helper = self.add_sub_node("dh", RelDistHelperNode(pc, _calc_head_count, conf))
        else:
            self.dist_helper = None
        # lambq
        if conf.use_lambq:
            self.lambq_aff = self.add_sub_node("al", AffineHelperNode(
                pc, dim_q, hid_dim=conf.lambq_hdim, hid_act=conf.lambq_hact, hid_drop=conf.lambq_hdrop,
                out_dim=head_count, out_fbias=conf.lambq_fbias, out_fact=conf.lambq_fact,
                out_piece4init=(head_count if conf.use_piece4init else 1), init_scale=conf.param_init_scale))
        else:
            self.lambq_aff = None
        # soff
        if conf.use_soff:
            self.soff_aff = self.add_sub_node("as", AffineHelperNode(
                pc, dim_q, hid_dim=conf.soff_hdim, hid_act=conf.soff_hact, hid_drop=conf.soff_hdrop,
                out_dim=(1+head_count), out_fbias=conf.soff_fbias, out_fact=conf.soff_fact,
                out_piece4init=((1+head_count) if conf.use_piece4init else 1), init_scale=conf.param_init_scale))
        else:
            self.soff_aff = None
        # no (masking out) self loop?
        self.no_self_loop = conf.no_self_loop

    # -----
    # [*, len, head*dim] -> [*, head, len, dim] if do_transpose else [*, len, head, dim]
    def _shape_project(self, x, do_transpose: bool):
        x_size = BK.get_shape(x)[:-1] + self.split_dims
        x_reshaped = x.view(x_size)
        return x_reshaped.transpose(-2, -3) if do_transpose else x_reshaped

    # input is *[*, len?, Din], [*, len_q, len_k, head], [*, len_k], [*, len_q, len_k]
    def __call__(self, query, key, accu_attn, mask_k, mask_qk, rel_dist):
        conf = self.conf
        # == calculate the dot-product scores
        # calculate the three: # [bs, len_?, head*D]; and also add sta ones if needed
        query_up, key_up = self.affine_q(query), self.affine_k(key)  # [*, len?, head?*Dqk]
        query_up, key_up = self._shape_project(query_up, True), self._shape_project(key_up, True)  # [*, head?, len_?, D]
        # original scores
        scores = BK.matmul(query_up, BK.transpose(key_up, -1, -2)) / self._att_scale_qk  # [*, head?, len_q, len_k]
        # == adding rel_dist ones
        if conf.use_rel_dist:
            scores = self.dist_helper(query_up, key_up, rel_dist=rel_dist, input_scores=scores)
        # tranpose
        scores = scores.transpose(-2, -3).transpose(-1, -2)  # [*, len_q, len_k, head?]
        # == unhead score
        if conf.use_unhead_score:
            scores_t0, score_t1 = BK.split(scores, [1, self.head_count], -1)  # [*, len_q, len_k, 1|head]
            scores = scores_t0 + score_t1  # [*, len_q, len_k, head]
        # == combining with history accumulated attns
        if conf.use_lambq and accu_attn is not None:
            # todo(note): here we only consider "query" and "head", would it be necessary for "key"?
            lambq_vals = self.lambq_aff(query)  # [*, len_q, head], if for eg., using relu as fact, this>=0
            scores -= lambq_vals.unsqueeze(-2) * accu_attn
        # == score offset
        if conf.use_soff:
            # todo(note): here we only consider "query" and "head", key may be handled by "unhead_score"
            score_offset_t = self.soff_aff(query)  # [*, len_q, 1+head]
            score_offset_t0, score_offset_t1 = BK.split(score_offset_t, [1, self.head_count], -1)  # [*, len_q, 1|head]
            scores -= score_offset_t0.unsqueeze(-2)
            scores -= score_offset_t1.unsqueeze(-2)  # still [*, len_q, len_k, head]
        # == apply mask & no-self-loop
        # NEG_INF = Constants.REAL_PRAC_MIN
        NEG_INF = -1000.  # this should be enough
        NEG_INF2 = -2000.  # this should be enough
        if mask_k is not None:  # [*, 1, len_k, 1]
            scores += (1.-mask_k).unsqueeze(-2).unsqueeze(-1) * NEG_INF2
        if mask_qk is not None:  # [*, len_q, len_k, 1]
            scores += (1.-mask_qk).unsqueeze(-1) * NEG_INF2
        if self.no_self_loop:
            query_len = BK.get_shape(query, -2)
            assert query_len == BK.get_shape(key, -2), "Shape not matched for no_self_loop"
            scores += BK.eye(query_len).unsqueeze(-1) * NEG_INF  # [len_q, len_k, 1]
        return scores.contiguous()  # [*, len_q, len_k, head]

# -----
# RelDistHelper: here we apply (relative) positional info only to att/score calculation
# -- following the ones in xl-transformer/xlnet
class RelDistHelperNode(BasicNode):
    def __init__(self, pc, head_count: int, conf: MAttScorerConf):
        super().__init__(pc, None, None)
        self.rel_dist_abs = conf.rel_dist_abs
        _clip = conf.rel_dist_clip
        _head_count = head_count
        _d_qk = conf.d_qk
        _d_emb = conf.rel_emb_dim
        # scale
        self._att_scale_qk = conf.get_att_scale_qk()  # scale for dot product
        # positional embedding
        self.E = self.add_sub_node("e", PosiEmbedding2(
            pc, _d_emb, max_val=_clip, init_sincos=conf.rel_init_sincos, freeze=conf.rel_init_sincos, zero0=conf.rel_emb_zero0))
        # different parameters for each head
        self.split_dims = [_head_count, _d_qk]
        self.affine_rel = self.add_sub_node("ar", AffineHelperNode(
            pc, _d_emb, _head_count*_d_qk, hid_piece4init=(head_count if conf.use_piece4init else 1), init_scale=conf.param_init_scale))
        self.vec_u = self.add_param("vu", (_head_count, _d_qk), scale=conf.param_init_scale)
        self.vec_v = self.add_param("vv", (_head_count, _d_qk), scale=conf.param_init_scale)

    # [query, key]
    def get_rel_dist(self, len_q: int, len_k: int):
        dist_x = BK.arange_idx(0, len_k).unsqueeze(0)  # [1, len_k]
        dist_y = BK.arange_idx(0, len_q).unsqueeze(1)  # [len_q, 1]
        distance = dist_x - dist_y  # [len_q, len_k]
        return distance

    # [*, head, len_q/len_k, D]; could be inplaced adding "input_scores"
    def __call__(self, query_up, key_up, rel_dist=None, input_scores=None):
        _att_scale_qk = self._att_scale_qk
        # -----
        # get dim info
        len_q, len_k = BK.get_shape(query_up, -2), BK.get_shape(key_up, -2)
        # get distance embeddings
        if rel_dist is None:
            rel_dist = self.get_rel_dist(len_q, len_k)
        if self.rel_dist_abs:  # use abs?
            rel_dist = BK.abs(rel_dist)
        dist_embs = self.E(rel_dist)  # [len_q, len_k, Demb]
        # -----
        # dist_up
        dist_up0 = self.affine_rel(dist_embs)  # [len_q, len_k, head*D]
        # -> [head, len_q, len_k, D]
        dist_up1 = dist_up0.view(BK.get_shape(dist_up0)[:-1] + self.split_dims).transpose(-2, -3).transpose(-3, -4)
        # -----
        # all items are [*, head, len_q, len_k]
        posi_scores = (input_scores if (input_scores is not None) else 0.)
        # item (b): <query, dist>: [head, len_q, len_k, D] * [*, head, len_q, D, 1] -> [*, head, len_q, len_k]
        item_b = (BK.matmul(dist_up1, query_up.unsqueeze(-1)) / _att_scale_qk).squeeze(-1)
        posi_scores += item_b
        # todo(note): remove this item_c since it is not related with rel_dist
        # # item (c): <key, u>: [*, head, len_k, D] * [head, D, 1] -> [*, head, 1, len_k]
        # item_c = (BK.matmul(key_up, self.vec_u.unsqueeze(-1)) / _att_scale_qk).squeeze(-1).unsqueeze(-2)
        # posi_scores += item_c
        # item (d): <dist, v>: [head, len_q, len_k, D] * [head, 1, D, 1] -> [head, len_q, len_k]
        item_d = (BK.matmul(dist_up1, self.vec_v.unsqueeze(-2).unsqueeze(-1)) / _att_scale_qk).squeeze(-1)
        posi_scores += item_d
        return posi_scores
