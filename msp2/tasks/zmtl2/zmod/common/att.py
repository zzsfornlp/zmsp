#

# attention node (first half can be a scorer)

__all__ = [
    "RelDistConf", "RelDistNode", "ZAttConf", "ZAttNode",
]

import math
from typing import List, Dict
from collections import Counter
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.utils import Constants, zwarn

# =====
# attention

# relative distance helper
class RelDistConf(BasicConf):
    def __init__(self):
        super().__init__()
        self._dim = -1
        # --
        self.clip_dist = 16  # this seems ok
        self.use_dist_v = False  # add dist-values to V
        self.use_neg_dist = True  # otherwise, use ABS
        self.use_posi_dist = False  # use posi embedddings?
        self.posi_zero0 = False  # zero0 if using

@node_reg(RelDistConf)
class RelDistNode(BasicNode):
    def __init__(self, conf: RelDistConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: RelDistConf = self.conf
        dim = conf._dim
        # --
        self.use_posi_dist = conf.use_posi_dist
        self.use_neg_dist = conf.use_neg_dist
        self.clip_dist = conf.clip_dist
        if conf.use_posi_dist:
            self.edge_atts = PosiEmbeddingNode(None, osize=dim, max_val=conf.clip_dist, zero0=conf.posi_zero0, no_drop=True)
            # self.edge_values = PosiEmbeddingNode(None, osize=dim, max_val=conf.clip_dist, zero0=conf.posi_zero0, no_drop=True)
        else:
            self.edge_atts = EmbeddingNode(None, osize=dim, n_words=2*conf.clip_dist+1, no_drop=True)
            # self.edge_values = EmbeddingNode(None, osize=dim, n_words=2*conf.clip_dist+1, no_drop=True)

    # using provided
    def embed_rposi(self, distance: BK.Expr):
        ret_dist = distance
        # use different ranges!
        if not self.use_neg_dist:
            distance = BK.abs(distance)
        if not self.use_posi_dist:  # clamp and offset if using plain embeddings
            distance = BK.clamp(distance, min=-self.clip_dist, max=self.clip_dist) + self.clip_dist
        dist_atts = self.edge_atts(distance)
        # dist_values = self.edge_values(distance)
        dist_values = None  # note: not used!!
        return ret_dist, dist_atts, dist_values

    # auto one
    def embed_lens(self, query_len: int, key_len: int):
        a_q = BK.arange_idx(query_len).unsqueeze(1)  # [query, 1]
        a_k = BK.arange_idx(key_len).unsqueeze(0)  # [1, key]
        ret_dist = a_q - a_k  # [query, key]
        _, dist_atts, dist_values = self.embed_rposi(ret_dist)
        # [lenq, lenk], [lenq, lenk, dim], [lenq, lenk, dim]
        return ret_dist, dist_atts, dist_values

# attention (as well as a biaffine-styled scorer)
class ZAttConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.dim_q: int = -1
        self.dim_k: int = -1
        self.dim_v: int = -1
        # --
        # att
        self.nh_qk = 8  # head in
        self.d_qk = 64  # head dim at input
        self.init_scale_hin = 1.  # since we may have many heads
        self.nh_v = 8  # head out
        self.d_v = 64  # head dim at output
        self.useaff_qk2v = False  # further affine from qk to v
        self.att_drop = 0.1
        self.out_act = 'linear'
        self.use_out_att = True  # by default, use output (v)
        # rel
        self.use_rposi = False  # whether use rel_dist
        self.rel = RelDistConf()

@node_reg(ZAttConf)
class ZAttNode(BasicNode):
    def __init__(self, conf: ZAttConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZAttConf = self.conf
        dim_q, dim_k, dim_v, nh_qk, d_qk, nh_v, d_v = \
            conf.dim_q, conf.dim_k, conf.dim_v, conf.nh_qk, conf.d_qk, conf.nh_v, conf.d_v
        # --
        self._att_scale = math.sqrt(conf.d_qk)  # scale for score
        # pre-att affines (no dropouts here!)
        _eg_q = BK.get_inita_xavier_uniform((d_qk, dim_q)) / BK.get_inita_xavier_uniform((nh_qk*d_qk, dim_q))
        self.affine_q = AffineNode(None, isize=dim_q, osize=nh_qk*d_qk, no_drop=True, init_scale=_eg_q*conf.init_scale_hin)
        _eg_k = BK.get_inita_xavier_uniform((d_qk, dim_k)) / BK.get_inita_xavier_uniform((nh_qk*d_qk, dim_k))
        self.affine_k = AffineNode(None, isize=dim_k, osize=nh_qk*d_qk, no_drop=True, init_scale=_eg_k*conf.init_scale_hin)
        if conf.use_out_att:
            self.affine_v = AffineNode(None, isize=dim_v, osize=nh_v*d_v, no_drop=True)
        else:
            self.affine_v = None
        # rel dist keys
        self.rposi = RelDistNode(conf.rel, _dim=d_qk) if conf.use_rposi else None
        # att & output
        if conf.use_out_att and conf.useaff_qk2v:
            self.aff_qk2v = AffineNode(None, isize=nh_qk, osize=nh_v)
        else:
            # assert nh_qk == nh_v
            if nh_qk != nh_v:
                zwarn(f"Possible problems with AttNode since hin({nh_qk}) != hout({nh_v})")
        self.adrop = DropoutNode(None, drop_rate=conf.att_drop, fix_drop=False)
        if conf.use_out_att:
            # todo(note): with drops(y) & act(?) & bias(y)?
            self.final_linear = AffineNode(None, isize=nh_v*d_v, osize=dim_v, out_act=conf.out_act)
        else:
            self.final_linear = None
        # --

    def extra_repr(self) -> str:
        conf: ZAttConf = self.conf
        return f"Att({conf.dim_q}/{conf.dim_k}[{conf.nh_qk}*{conf.d_qk}]=>({conf.dim_v}[{conf.nh_v}*{conf.d_v}])"

    def get_output_dims(self, *input_dims):
        return (self.conf.dim_v, )

    # [*, len, head*dim] -> [*, head, len, dim]
    def _shape_project(self, x: BK.Expr, nhead: int):
        orig_shape = BK.get_shape(x)
        x_size = orig_shape[:-1] + [nhead, orig_shape[-1]//nhead]
        return BK.transpose(BK.reshape(x, x_size), -2, -3)

    # [*, head, len, dim] -> [*, len, head*dim]
    def _unshape_project(self, x: BK.Expr):
        orig_shape = BK.get_shape(x)
        x_size = orig_shape[:-3] + [orig_shape[-2], orig_shape[-1]*orig_shape[-3]]
        return BK.reshape(BK.transpose(x, -2, -3), x_size)

    # --
    # *[*, slen, D]
    def do_score(self, query, key):
        conf: ZAttConf = self.conf
        query_len = BK.get_shape(query, -2)
        key_len = BK.get_shape(key, -2)
        # --
        # 1. project
        query_up = self._shape_project(self.affine_q(query), conf.nh_qk)  # [*, Hin, len_q, d_qk]
        key_up = self._shape_project(self.affine_k(key), conf.nh_qk)  # [*, Hin, len_k, d_qk]
        # 2. score
        query_up = query_up / self._att_scale
        scores_t = BK.matmul(query_up, BK.transpose(key_up, -1, -2))  # [*, Hin, len_q, len_k]
        if conf.use_rposi:
            distance, distance_out, _ = self.rposi.embed_lens(query_len, key_len)
            # avoid broadcast!
            _d_bs, _d_h, _d_q, _d_d = BK.get_shape(query_up)
            query_up0 = BK.reshape(query_up.transpose(2, 1).transpose(1, 0), [_d_q, _d_bs * _d_h, _d_d])
            add_term0 = BK.matmul(query_up0, distance_out.transpose(-1, -2))  # [len_q, head*bs, len_k]
            add_term = BK.reshape(add_term0.transpose(0, 1), BK.get_shape(scores_t))
            # --
            scores_t += add_term  # [*, Hin, len_q, len_k]
        # todo(note): no dropout here, if use this at outside, need extra one!!
        return scores_t  # [*, Hin, len_q, len_k]

    def do_output(self, scores_t, value, mask_k=None, mask_qk=None):
        conf: ZAttConf = self.conf
        # --
        # aff_qk2v
        scores_in_t = scores_t
        if conf.useaff_qk2v:
            _trans_t = scores_in_t.transpose(-3, -2).transpose(-2, -1).contiguous()  # [*, len_q, len_k, Hin]
            _out_t = self.aff_qk2v(_trans_t)  # [*, len_q, len_k, Hout]
            scores_out_t = _out_t.transpose(-2, -1).transpose(-3, -2)  # [*, Hout, len_q, len_k]
        else:
            scores_out_t = scores_in_t  # only valid when Hin==Hout!
        # mask
        if mask_k is not None:
            # todo(note): mask as [*, len]
            scores_out_t = scores_out_t + (1.-mask_k).unsqueeze(-2).unsqueeze(-3) * Constants.REAL_PRAC_MIN
        if mask_qk is not None:
            # todo(note): mask as [*, len-q, len-k]
            scores_out_t = scores_out_t + (1.-mask_qk).unsqueeze(-3) * Constants.REAL_PRAC_MIN
        # attn
        attn = scores_out_t.softmax(-1)  # [*, Hout, len_q, len_k]
        drop_attn = self.adrop(attn)  # [*, Hout, len_q, len_k]
        # value up
        value_up = self._shape_project(self.affine_v(value), conf.nh_v)    # [*, Hout, len_k, D]
        context = BK.matmul(drop_attn, value_up)  # [*, head, len_q, d_v]
        # final
        context_merge = self._unshape_project(context)  # [*, len_q, d_v*Hout]
        output = self.final_linear(context_merge)  # [*, len_q, D]
        return output

    # combine the above two
    def forward(self, query, key, value, mask_k=None, mask_qk=None):
        scores_t = self.do_score(query, key)
        out_t = self.do_output(scores_t, value, mask_k=mask_k, mask_qk=mask_qk)
        return out_t

# --
