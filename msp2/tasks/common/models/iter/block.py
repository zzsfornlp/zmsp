#

# basic building blocks

__all__ = [
    "SingleBlockConf", "SingleBlockNode", "PairwiseBlockConf", "PairwiseBlockNode",
]

from typing import Union, List
import math
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.utils import Constants, zlog

# =====
# single labeling
# [ndim] -> ([hid_dim]) -> [nlab] -> ([hid_dim]) -> [ndim]

class SingleBlockConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self.ndim = -1  # dim
        self.nlab = -1  # number of classes/labels to predict at the middle
        self.e_tie_weights = True  # tie embeddings to scorer's output W (also inference e_dim)
        self.e_mul_scale = 1.  # multiplication after prob*weights
        self.p_init_scale = 1.  # for hid_in
        self.cf_init_scale = 1.  # scale for aff_cf
        self.hid_dim = 256  # hidden dim
        self.hid_aff = AffineConf().direct_update(out_act='elu')

@node_reg(SingleBlockConf)
class SingleBlockNode(BasicNode):
    def __init__(self, conf: SingleBlockConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: SingleBlockConf = self.conf
        ndim, hid_dim, nlab = conf.ndim, conf.hid_dim, conf.nlab
        # --
        self.e_mul_scale = conf.e_mul_scale if conf.e_mul_scale>0 else math.sqrt(hid_dim)  # special alpha
        self.hid_in = AffineNode(conf.hid_aff, isize=ndim, osize=hid_dim)  # ndim->hid_dim
        self.pred_in = AffineNode(None, isize=hid_dim, osize=nlab, no_drop=True, init_scale=conf.p_init_scale)  # hid_dim->nlab
        self.aff_cf = AffineNode(None, isize=hid_dim, osize=1, no_drop=True, init_scale=conf.cf_init_scale)  # confident score: hid -> 1
        if conf.e_tie_weights:
            tmp_W = self.pred_in.get_ws()[0]  # [nlab, hid_dim]
            assert BK.get_shape(tmp_W) == [nlab, hid_dim]
            self.W_getf = lambda: tmp_W
        else:
            self.W = BK.new_param([hid_dim, nlab])  # [nlab, hid_dim]
            self.reset_parameters()
            self.W_getf = lambda: self.W
        self.hid_out = AffineNode(None, isize=hid_dim, osize=ndim)  # hid_dim->ndim
        self.norm = LayerNormNode(None, osize=ndim)  # add&norm

    def reset_parameters(self):
        if not self.conf.e_tie_weights:
            BK.init_param(self.W, "glorot", lookup=True)

    def extra_repr(self) -> str:
        conf: SingleBlockConf = self.conf
        return f"SingleBlockNode({conf.ndim}->{conf.hid_dim}->{conf.nlab}(tie={conf.e_tie_weights}))"

    # [*, D], [*, L], bool, [*, L]
    def forward(self, expr_t: BK.Expr, fixed_scores_t: BK.Expr = None, feed_output=False, mask_t: BK.Expr = None):
        conf: SingleBlockConf = self.conf
        # --
        # pred
        if fixed_scores_t is not None:
            score_t = fixed_scores_t
            cf_t = None
        else:
            hid1_t = self.hid_in(expr_t)  # [*, hid]
            score_t = self.pred_in(hid1_t)  # [*, nlab]
            cf_t = self.aff_cf(hid1_t).squeeze(-1)  # [*]
        # --
        if mask_t is not None:
            shape0 = BK.get_shape(expr_t)
            shape1 = BK.get_shape(mask_t)
            if len(shape1) < len(shape0):
                mask_t = mask_t.unsqueeze(-1)  # [*, 1]
            score_t += Constants.REAL_PRAC_MIN * (1. - mask_t)  # [*, nlab]
        # --
        # output
        if feed_output:
            W = self.W_getf()  # [nlab, hid]
            prob_t = score_t.softmax(-1)  # [*, nlab]
            hid2_t = BK.matmul(prob_t, W) * self.e_mul_scale  # [*, hid], todo(+W): need dropout here?
            out_t = self.hid_out(hid2_t)  # [*, ndim]
            final_t = self.norm(out_t + expr_t)  # [*, ndim], add and norm
        else:
            final_t = expr_t  # [*, ndim], simply no change and use input!
        return score_t, cf_t, final_t  # [*, nlab], [*], [*, ndim]

# =====
# pairwise labeling
# [*, slen, ndim] -> Q: [*, slen, Hin, D'], K: [*, slen, Hin, D'] -> [*, lenq, lenk, Hin] -> [*, lenq, lenk, nlab]
# ... -> [*, lenq, lenk, Hout] -> .*[*, lenq, Hout, Dv] -> [*, lenq, ndim]

class PairwiseBlockConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self.ndim = -1  # dim
        self.nlab = -1  # number of classes/labels to predict at the middle
        # --
        self.nhead_in = 64  # head dim at input
        self.hin_init_scale = 1.  # since we may have many heads
        self.cf_init_scale = 1.  # scale for aff_cf
        self.dim_qk = 32
        self.nhead_out = 8  # head dim at output
        self.dim_v = 64
        # output
        self.att_drop = 0.1
        self.out_act = 'linear'

@node_reg(PairwiseBlockConf)
class PairwiseBlockNode(BasicNode):
    def __init__(self, conf: PairwiseBlockConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: PairwiseBlockConf = self.conf
        ndim, nlab, nhead_in, dim_qk, nhead_out, dim_v = \
            conf.ndim, conf.nlab, conf.nhead_in, conf.dim_qk, conf.nhead_out, conf.dim_v
        # --
        self._att_scale = math.sqrt(conf.dim_qk)  # scale for score; note: no scale here since already small
        # pre-att affines, (no dropouts here)
        _extra_gain = BK.get_inita_xavier_uniform((dim_qk, ndim)) / BK.get_inita_xavier_uniform((nhead_in*dim_qk, ndim))
        self.affine_q = AffineNode(None, isize=ndim, osize=nhead_in*dim_qk, no_drop=True, init_scale=_extra_gain*conf.hin_init_scale)
        self.affine_k = AffineNode(None, isize=ndim, osize=nhead_in*dim_qk, no_drop=True, init_scale=_extra_gain*conf.hin_init_scale)
        self.affine_v = AffineNode(None, isize=ndim, osize=nhead_out*dim_v, no_drop=True)
        # pred
        self.pred_in = AffineNode(None, isize=nhead_in, osize=nlab, no_drop=True)
        self.aff_cf = AffineNode(None, isize=nhead_in, osize=1, no_drop=True, init_scale=conf.cf_init_scale)  # for pairwise confident score
        # final layers
        self.adrop = DropoutNode(None, drop_rate=conf.att_drop, fix_drop=False)
        self.fl_score = AffineNode(None, isize=nlab, osize=nhead_out, no_drop=True)
        self.fl_expr = AffineNode(None, isize=nhead_out*dim_v, osize=ndim, out_act=conf.out_act)
        self.norm = LayerNormNode(None, osize=ndim)  # add&norm

    def extra_repr(self) -> str:
        conf: PairwiseBlockConf = self.conf
        return f"PairwiseBlockNode({conf.ndim}->[{conf.nhead_in}*{conf.dim_qk}]->{conf.nlab}->({conf.nhead_out}*{conf.dim_v})"

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
    # [*, slen, D], [*, slen, L], bool, [*, len_k], [*, len_q, len_k]
    def forward(self, expr_t: BK.Expr, fixed_scores_t: BK.Expr = None, feed_output=False, mask_k=None, mask_qk=None):
        conf: PairwiseBlockConf = self.conf
        ndim, nlab, nhead_in, dim_qk, nhead_out, dim_v = \
            conf.ndim, conf.nlab, conf.nhead_in, conf.dim_qk, conf.nhead_out, conf.dim_v
        # --
        # 1. pred
        if fixed_scores_t is not None:
            score_t = fixed_scores_t
            cf_t = None
        else:
            # 1.1: project
            query_up = self._shape_project(self.affine_q(expr_t), nhead_in)  # [*, Hin, len_q, d_qk]
            key_up = self._shape_project(self.affine_k(expr_t), nhead_in)  # [*, Hin, len_k, d_qk]
            # 1.2. calculate and scale scores
            query_up = query_up / self._att_scale
            scores0_t = BK.matmul(query_up, BK.transpose(key_up, -1, -2))  # [*, Hin, len_q, len_k]
            scores1_t = scores0_t.transpose(-3, -2).transpose(-2, -1).contiguous()  # [*, len_q, len_k, Hin]
            scores2_t = self.adrop(scores1_t)  # dropout!!
            score_t = self.pred_in(scores2_t)  # [*, len_q, len_k, nlab]
            cf_t = self.aff_cf(scores2_t).squeeze(-1)  # [*, len_q, len_k]
        # --
        # todo(+W): currently no masks at L-dim
        # --
        # 2. output
        if feed_output:
            # todo(+W): no pairwise mask currently
            prob_t = score_t.softmax(-1)  # [*, len_q, len_k, nlab]
            prob_score_t = self.fl_score(prob_t)  # [*, len_q, len_k, Hout]
            # attention
            att_score_t = prob_score_t.transpose(-2, -1).transpose(-3, -2)  # [*, Hout, len_q, len_k]
            if mask_k is not None:  # note: mask as [*, len]
                att_score_t += (1.-mask_k).unsqueeze(-2).unsqueeze(-3) * Constants.REAL_PRAC_MIN
            if mask_qk is not None:  # note: mask as [*, len_q, len_k]
                att_score_t += (1.-mask_qk).unsqueeze(-3) * Constants.REAL_PRAC_MIN
            attn = BK.softmax(att_score_t, -1)  # [*, Hout, len_q, len_k]
            drop_attn = self.adrop(attn)  # [*, Hout, len_q, len_k]
            # value
            value_up = self._shape_project(self.affine_v(expr_t), nhead_out)  # [*, Hout, len_k, d_v]
            context = BK.matmul(drop_attn, value_up)  # [*, Hout, len_q, d_v]
            # final
            context_merge = self._unshape_project(context)  # [*, len_q, d_v*Hout]
            out_t = self.fl_expr(context_merge)  # [*, len_q, ndim]
            final_t = self.norm(out_t + expr_t)  # [*, len_q, ndim], add and norm
        else:
            final_t = expr_t  # [*, slen, ndim], simply no change and use input!
        return score_t, cf_t, final_t  # [*, len_q, len_k, nlab], [*, slen, ndim]

# --
# b msp2/tasks/common/models/iter/block:64
