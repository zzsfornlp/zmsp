#

# Component 3 for MAtt: collector
# (V, normalized_scores [*, len_q, len_k, head], accu_attns) => collected-V [*, len_q, Dv]

from typing import List, Union, Dict, Iterable
from msp.utils import Constants, Conf, zwarn, Helper
from msp.nn import BK
from msp.nn.layers import BasicNode, ActivationHelper, Affine, NoDropRop, NoFixRop, Dropout, PosiEmbedding2, LayerNorm
from .base import AffineHelperNode

# -----

# conf
class MAttCollectorConf(Conf):
    def __init__(self):
        # special
        self.param_init_scale = 1.  # todo(+N): ugly extra scale
        self.use_piece4init = False
        # content of values
        self.use_affv = True  # if not using, then directly use input (dim_v)
        self.d_v = 64  # content reprs
        self.v_act = 'linear'
        self.v_drop = 0.
        # attn & accu_attn
        self.accu_lambda = 0.  # for eg., -1 means delete previous ones, +1 means consider previous ones
        self.accu_ceil = 1.  # clamp final applying scores by certain range
        self.accu_floor = 0.
        # model based clamp
        self.use_mclp = False
        self.mclp_hdim = 128
        self.mclp_hdrop = 0.1
        self.mclp_hact = "elu"
        self.mclp_fbias = 1.  # fixed extra bias (making mclp by default positive value)
        self.mclp_fact = "relu"  # output activation after adding bias
        # collecting: 1. weighted sum over certain dims, 2. further reduce if needed
        self.collect_mode = "ch"  # direct_2d(2d)/cand_head(ch)/head_cand(hc)
        self.collect_renorm = False  # renorm to avoid weights too large?
        self.collect_renorm_mindiv = 1.  # avoid renorm on small values
        self.collect_reducer_mode = "aff"  # max/sum/avg/aff; (aff is only valid for multihead-reduce)
        self.collect_red_aff_out = 512  # output for "aff" mode, <0 means the same as d_v
        self.collect_red_aff_act = 'linear'

# the collector: normalized_scores * V
class MAttCollectorNode(BasicNode):
    def __init__(self, pc: BK.ParamCollection, dim_q, dim_v, head_count, conf: MAttCollectorConf):
        super().__init__(pc, None, None)
        self.conf = conf
        self.dim_q, self.dim_v, self.head_count = dim_q, dim_v, head_count
        # -----
        _d_v = conf.d_v
        _hid_size_v = head_count * _d_v
        self.split_dims = [head_count, -1]
        # affine-v
        if conf.use_affv:
            self.out_dim = _d_v
            self.affine_v = self.add_sub_node("av", AffineHelperNode(
                pc, dim_v, _hid_size_v, hid_act=conf.v_act, hid_drop=conf.v_drop,
                hid_piece4init=(head_count if conf.use_piece4init else 1), init_scale=conf.param_init_scale))
        else:
            self.out_dim = dim_v
            self.affine_v = None
        # -----
        # model based clamp
        if conf.use_mclp:
            self.mclp_aff = self.add_sub_node("am", AffineHelperNode(
                pc, dim_q, hid_dim=conf.mclp_hdim, hid_act=conf.mclp_hact, hid_drop=conf.mclp_hdrop,
                out_dim=head_count, out_fbias=conf.mclp_fbias, out_fact=conf.mclp_fact,
                out_piece4init=(head_count if conf.use_piece4init else 1), init_scale=conf.param_init_scale))
        else:
            self.mclp_aff = None
        # -----
        # how to reduce the two dims of [len_k, head]
        self._collect_f = getattr(self, "_collect_"+conf.collect_mode)  # shortcut
        if conf.collect_reducer_mode == "aff":
            aff_out_dim = _d_v if (conf.collect_red_aff_out<=0) else conf.collect_red_aff_out
            aff_out_act = conf.collect_red_aff_act
            self.out_dim = aff_out_dim
            # no param_init_scale here, since output is not multihead
            self.multihead_aff_node = self.add_sub_node("ma", Affine(pc, _hid_size_v, aff_out_dim,
                                                                     act=aff_out_act, init_rop=NoDropRop()))
        else:
            self.multihead_aff_node = None
        # todo(note): always reduce on dim=-2!!
        self._collect_reduce_f = {
            "sum": lambda x: x.sum(-2), "avg": lambda x: x.mean(-2), "max": lambda x: x.max(-2)[0],
            "aff": lambda x: self.multihead_aff_node(x.view(BK.get_shape(x)[:-2] + [-1])),  # flatten last two and aff
            "cat": lambda x: x.view(BK.get_shape(x)[:-2] + [-1]),  # directly flatten
        }[conf.collect_reducer_mode]
        # special mode "cat"
        if conf.collect_reducer_mode == "cat":
            assert conf.collect_mode == "ch", "Must get rid of the var. dim of cand first"
            self.out_dim = _hid_size_v
        # =====

    def get_output_dims(self, *input_dims):
        return (self.out_dim, )

    # input is [*, len_k, D], *[*, len_q, len_k, head]
    def __call__(self, query, value, attn, accu_attn):
        conf = self.conf
        # get values
        if self.affine_v:
            value_up = self.affine_v(value).view(BK.get_shape(value)[:-1]+self.split_dims)  # [*, (len_k, head), Dv]
        else:
            value_up = value.unsqueeze(-2)  # [*, len_k, 1, D]
            value_up_expand_shape = BK.get_shape(value_up)
            value_up_expand_shape[-2] = self.head_count
            value_up = value_up.expand(value_up_expand_shape).contiguous()  # [*, len_k, head, D]
        # get final applying attn scores: [*, len_q, (len_k, head)]
        if conf.accu_lambda != 0.:
            combined_attn = attn + accu_attn * conf.accu_lambda
            clamped_attn = combined_attn.clamp(min=conf.accu_floor, max=conf.accu_ceil)
            # apply_attn = (clamped_attn - combined_attn).detach() + combined_attn  # todo(+N): make gradient flow?
            apply_accu_attn = clamped_attn
        else:
            apply_accu_attn = attn
        # model based clamp
        if self.mclp_aff:
            mclp_upper = self.mclp_aff(query).unsqueeze(-2)  # [*, len_q, 1, head]
            apply_attn = BK.min_elem(apply_accu_attn, mclp_upper)  # clamp on the ceil side
        else:
            apply_attn = apply_accu_attn
        # apply attn and get result
        result_value = self._collect_f(value_up, apply_attn)  # [*, len_q, Dv]
        return result_value

    # =====
    # specific strategies

    # 1. simply weighted sum over 2d [len_k, head]
    def _collect_2d(self, value_up, apply_attn):
        conf = self.conf
        fused_attn = apply_attn.view(BK.get_shape(apply_attn)[:-2]+[-1])  # [*, len_q, len_k*head]
        if conf.collect_renorm:
            fused_attn = fused_attn / fused_attn.sum(-1, keepdims=True).clamp(min=conf.collect_renorm_mindiv)
        value_up_shape = BK.get_shape(value_up)
        fused_value_up = value_up.view(value_up_shape[:-3] + [-1, value_up_shape[-1]])  # [*, len_k*head, Dv]
        result_value = BK.matmul(fused_attn, fused_value_up)  # [*, len_q, Dv]
        return result_value

    # 2. weighted sum over cand, then reduce on head (pool/affine)
    def _collect_ch(self, value_up, apply_attn):
        conf = self.conf
        # first weighted sum over cand
        cand_final_attn = apply_attn.transpose(-1, -2).transpose(-2, -3)  # [*, head, len_q, len_k]
        if conf.collect_renorm:
            cand_final_attn = cand_final_attn / cand_final_attn.sum(-1, keepdims=True).clamp(min=conf.collect_renorm_mindiv)
        transposed_value_up = value_up.transpose(-2, -3)  # [*, head, len_k, Dv]
        multihead_context = BK.matmul(cand_final_attn, transposed_value_up).transpose(-2, -3)  # [*, len_q, head, Dv]
        # then reduce head
        result_value = self._collect_reduce_f(multihead_context.contiguous())  # [*, len_q, Dv]
        return result_value

    # 3. weighted sum over head, then reduce on cand (pool) (cannot use affine here)
    def _collect_hc(self, value_up, apply_attn):
        conf = self.conf
        # first weighted sum over head
        head_final_attn = apply_attn.transpose(-2, -3)  # [*, len_k, len_q, head]
        if conf.collect_renorm:
            head_final_attn = head_final_attn / head_final_attn.sum(-1, keepdims=True).clamp(min=conf.collect_renorm_mindiv)
        apply_value_up = value_up  # [*, len_k, head, Dv]
        multicand_context = BK.matmul(head_final_attn, apply_value_up).transpose(-2, -3)  # [*, len_q, len_k, Dv]
        # then reduce cand
        result_value = self._collect_reduce_f(multicand_context)  # [*, len_q, Dv]
        return result_value
