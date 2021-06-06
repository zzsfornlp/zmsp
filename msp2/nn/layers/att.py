#

# Attention Nodes

__all__ = [
    "AttentionConf", "AttentionNode", "AttDistHelper",
]

import math
from ..backends import BK
from .base import *
from .ff import AffineConf, AffineNode, EmbeddingNode, PosiEmbeddingNode
from msp2.utils import Constants

# =====

class AttentionConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.dim_q: int = -1
        self.dim_k: int = -1
        self.dim_v: int = -1
        # --
        # att
        self.d_kqv = 64  # like hidden size
        self.att_drop = 0.1  # special att drop
        self.att_method = "dot"  # dot, ff, max
        self.head_count = 8
        # relative positional
        self.clip_dist = 0
        self.use_dist_v = False  # add dist-values to V
        self.use_neg_dist = True  # otherwise, use ABS
        self.use_posi_dist = False  # use posi embedddings?
        self.posi_zero0 = False  # zero0 if using
        # output
        self.out_act = 'linear'

    @property
    def need_rposi(self):
        return self.clip_dist > 0

@node_reg(AttentionConf)
class AttentionNode(BasicNode):
    def __init__(self, conf: AttentionConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: AttentionConf = self.conf
        # --
        eff_hidden_size = conf.head_count * conf.d_kqv
        self._att_scale = math.sqrt(conf.d_kqv) if conf.att_method=="dot" else 1.  # only scale if using dot
        # pre-att affines, (no dropouts here)
        self.affine_q = AffineNode(None, isize=conf.dim_q, osize=eff_hidden_size, no_drop=True)
        self.affine_k = AffineNode(None, isize=conf.dim_k, osize=eff_hidden_size, no_drop=True)
        self.affine_v = AffineNode(None, isize=conf.dim_v, osize=eff_hidden_size, no_drop=True)
        # att method: [*, head, len_q, D], [*, head, len_k, D] -> [*, head, len_q, len_k]
        self.score_aff_W = None
        if conf.att_method == "dot":
            self.score_f = lambda q,k: BK.matmul(q, BK.transpose(k, -1, -2))
        elif conf.att_method == "ff":  # todo(note): this cost memory!!
            self.score_aff_W = BK.new_param([conf.head_count, conf.d_kqv])
            self.score_f = self._score_ff
        elif conf.att_method == "max":  # todo(note): still cost memory!!
            self.score_f = lambda q,k: BK.max(q.unsqueeze(-2)+k.unsqueeze(-3), -1)[0]
        else:
            raise NotImplementedError(f"UNK att_method: {conf.att_method}")
        # att dropout (never fix_drop, thus no dims provided!)
        self.adrop = DropoutNode(None, drop_rate=conf.att_drop, fix_drop=False)
        # rel dist keys
        if conf.need_rposi:
            self.dist_helper = AttDistHelper(conf.d_kqv, conf)
        else:
            self.dist_helper = None
        # todo(note): with drops(y) & act(?) & bias(y)?
        self.final_linear = AffineNode(None, isize=eff_hidden_size, osize=conf.dim_v, out_act=conf.out_act)
        # --
        self.reset_parameters()

    # [*, len, head*dim] <-> [*, head, len, dim]
    def _shape_project(self, x):
        head_count = self.conf.head_count
        orig_shape = BK.get_shape(x)
        x_size = orig_shape[:-1] + [head_count, orig_shape[-1]//head_count]
        return BK.transpose(BK.reshape(x, x_size), -2, -3)

    def _unshape_project(self, x):
        x_size = BK.get_shape(x)
        return BK.reshape(BK.transpose(x, -2, -3), x_size[:-3]+[x_size[-2], x_size[-1]*x_size[-3]])

    # clip scores according to range (<=R as valid)
    @staticmethod
    def _range_scores(att_scores_t, R: int):
        len1, len2 = BK.get_shape(att_scores_t)[-2:]  # assume last two dim
        r1 = BK.arange_idx(len1).unsqueeze(-1)  # [len1, 1]
        r2 = BK.arange_idx(len2).unsqueeze(-2)  # [1, len2]
        rdiff = BK.abs(r1 - r2).float()
        scores = att_scores_t.masked_fill(rdiff > R, Constants.REAL_PRAC_MIN)
        return scores

    def _score_ff(self, q, k):
        hid = BK.elu(q.unsqueeze(-2)+k.unsqueeze(-3))  # [*, head, len_q, len_k, D]
        hid1 = hid.transpose(-4, -3).transpose(-3,-2)  # [*, len_q, len_k, head, D]
        score0 = BK.matmul(hid1.unsqueeze(-2), self.score_aff_W.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # [*, len_q, len_k, head]
        score = score0.transpose(-1, -2).transpose(-2, -3)  # [*, head, len_q, len_k]
        return score

    def reset_parameters(self):
        if self.score_aff_W is not None:
            BK.init_param(self.score_aff_W, "default", lookup=True)  # actually this is similar to lookup

    # --
    def extra_repr(self) -> str:
        conf: AttentionConf = self.conf
        return f"AttentionNode(k={conf.dim_k},q={conf.dim_q},v={conf.dim_v})"

    def get_output_dims(self, *input_dims):
        return (self.conf.dim_v, )

    # todo(+2): assuming (batch, len, dim)
    def forward(self, query, key, value, mask_k=None, mask_qk=None, rposi=None, cutoff_range:int=None, **kwargs):
        conf: AttentionConf = self.conf
        # --
        batch_dim_size = len(BK.get_shape(key))-2
        query_len = BK.get_shape(query, -2)
        key_len = BK.get_shape(key, -2)
        # --
        # prepare distances if needed
        if conf.need_rposi:  # [*, query, key]
            distance, dist_atts, dist_values = self.dist_helper.embed_lens(query_len, key_len) \
                if rposi is None else self.dist_helper.embed_rposi(rposi)
        else:
            distance, dist_atts, dist_values = None, None, None
        # --
        # 1. project the three
        query_up = self._shape_project(self.affine_q(query))    # [*, head, len_q, d]
        key_up = self._shape_project(self.affine_k(key))        # [*, head, len_k, d]
        value_up = self._shape_project(self.affine_v(value))    # [*, head, len_k, d]
        # 2. calculate and scale scores
        query_up = query_up / self._att_scale
        # scores = BK.matmul(query_up, BK.transpose(key_up, -1, -2))  # [*, head, len_q, len_k]
        scores = self.score_f(query_up, key_up)  # [*, head, len_q, len_k]
        if conf.need_rposi:
            distance_out = dist_atts  # [query, key, d_kqv]
            if len(BK.get_shape(distance_out)) <= 3:  # we can make it more mem efficient
                # adopted from T2T: https://github.com/tensorflow/tensor2tensor/blob/5f9dd2db6d7797162e53adf152310ed13e9fc711/tensor2tensor/layers/common_attention.py#L1705
                # rearrange to avoid broadcast: [len_q, bs*head, D] * [len_q, D, len_k] -> [len_q, head*bs, len_k] -> ..
                _d_bs, _d_h, _d_q, _d_d = BK.get_shape(query_up)
                query_up0 = BK.reshape(query_up.transpose(2,1).transpose(1,0), [_d_q, _d_bs*_d_h, _d_d])
                add_term0 = BK.matmul(query_up0, distance_out.transpose(-1,-2))  # [len_q, head*bs, len_k]
                add_term = BK.reshape(add_term0.transpose(0,1), BK.get_shape(scores))
            else:
                # let it broadcast: [..., len_q, len_k, d] * [*, head, len_q, d, 1] => [*, head, len_q, len_k, 1]
                add_term = BK.matmul(distance_out, query_up.unsqueeze(-1)).squeeze(-1)
            scores += add_term
        # 3. apply attention
        if mask_k is not None:
            # todo(note): mask as [*, len]
            scores += (1.-mask_k).unsqueeze(-2).unsqueeze(-3) * Constants.REAL_PRAC_MIN
        if mask_qk is not None:
            # todo(note): mask as [*, len-q, len-k]
            scores += (1.-mask_qk).unsqueeze(-3) * Constants.REAL_PRAC_MIN
        if cutoff_range is not None:  # cutoff scores by abs range, todo(note): mostly utilized for self-att!!
            scores = AttentionNode._range_scores(scores, cutoff_range)
        attn = BK.softmax(scores, -1)  # [*, head, len_q, len_k]
        drop_attn = self.adrop(attn)  # [*, head, len_q, len_k]
        context = BK.matmul(drop_attn, value_up)  # [*, head, len_q, d_v]
        if conf.need_rposi and conf.use_dist_v:
            # specific rel-position values as in https://arxiv.org/pdf/1803.02155.pdf
            distance_out2 = dist_values  # [query, key, dim]
            # --
            # todo(+W): we can do the same arrangements as the attention ones, but without dist_v it also works ...
            add_term2 = BK.matmul(drop_attn.unsqueeze(-2), distance_out2).squeeze(-2)  # [*, head, len_q, d_v]
            context += add_term2
        # 4. final
        context_merge = self._unshape_project(context)  # [*, len_q, d_v*head]
        output = self.final_linear(context_merge)  # [*, len_q, mdim]
        return output

# =====
# helpers

class AttDistHelper(BasicNode):
    def __init__(self, dim: int, aconf: AttentionConf):
        super().__init__(None)
        # directly borrow aconf
        self.use_posi_dist = aconf.use_posi_dist
        self.use_neg_dist = aconf.use_neg_dist
        self.clip_dist = aconf.clip_dist
        self.use_dist_v = aconf.use_dist_v
        # --
        self.edge_values = None
        if aconf.use_posi_dist:
            self.edge_atts = PosiEmbeddingNode(None, osize=dim, max_val=aconf.clip_dist, zero0=aconf.posi_zero0, no_drop=True)
            if self.use_dist_v:
                self.edge_values = PosiEmbeddingNode(None, osize=dim, max_val=aconf.clip_dist, zero0=aconf.posi_zero0, no_drop=True)
        else:
            self.edge_atts = EmbeddingNode(None, osize=dim, n_words=2*aconf.clip_dist+1, no_drop=True)
            if self.use_dist_v:
                self.edge_values = EmbeddingNode(None, osize=dim, n_words=2*aconf.clip_dist+1, no_drop=True)

    # using provided
    def embed_rposi(self, distance: BK.Expr):
        ret_dist = distance
        # use different ranges!
        if not self.use_neg_dist:
            distance = BK.abs(distance)
        if not self.use_posi_dist:  # clamp and offset if using plain embeddings
            distance = BK.clamp(distance, min=-self.clip_dist, max=self.clip_dist) + self.clip_dist
        dist_atts = self.edge_atts(distance)
        if self.use_dist_v:
            dist_values = self.edge_values(distance)
        else:
            dist_values = None
        return ret_dist, dist_atts, dist_values

    # auto one
    def embed_lens(self, query_len: int, key_len: int):
        a_q = BK.arange_idx(query_len).unsqueeze(1)  # [query, 1]
        a_k = BK.arange_idx(key_len).unsqueeze(0)  # [1, key]
        ret_dist = a_q - a_k  # [query, key]
        _, dist_atts, dist_values = self.embed_rposi(ret_dist)
        # [lenq, lenk], [lenq, lenk, dim], [lenq, lenk, dim]
        return ret_dist, dist_atts, dist_values
