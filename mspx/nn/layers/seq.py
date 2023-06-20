#

# for sequential modeling

__all__ = [
    "AttentionConf", "AttentionLayer", "RelpConf", "RelpLayer",
    "RnnCellConf", "RnnCellLayer", "RnnConf", "RnnLayer",
    "TfCellConf", "TfCellLayer", "TransformerConf", "TransformerLayer",
]

import math
import numpy as np

from mspx.utils import zlog, Constants, ConfEntryChoices, ZObject
from ..backends import BK
from .base import *
from .ff import *
from .misc import split_at_dim, unsqueeze_expand

# --
# Attention
@NnConf.rd('att')
class AttentionConf(NnConf):
    def __init__(self):
        super().__init__()
        self.dim_q = -1
        self.dim_k = -1
        self.dim_v = -1
        self.dim_out = -1
        # --
        # att
        self.att_aff = AffineConf()
        self.d_qkv = -1  # like hidden size, if -1 then dim // head_count
        self.head_count = 8
        self.att_drop = 0.1  # special att drop
        self.kv_static = False  # for cross-att, kv can be static and reuse through cache
        # relative posi
        self.relp = RelpConf()
        # output
        self.out_aff = AffineConf()

    @property
    def use_rel(self):
        return self.relp.rel_clip > 0

@AttentionConf.conf_rd()
class AttentionLayer(NnLayer):
    def __init__(self, conf: AttentionConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: AttentionConf = self.conf
        # --
        if conf.dim_out < 0:
            conf.dim_out = conf.dim_v
        if conf.d_qkv < 0:
            conf.d_qkv = conf.dim_out // conf.head_count
        eff_hidden_size = conf.head_count * conf.d_qkv
        self._att_scale = math.sqrt(conf.d_qkv)
        self.kv_static = conf.kv_static
        # pre-att affines
        self.affine_q = AffineLayer(conf.att_aff, isize=conf.dim_q, osize=eff_hidden_size)
        self.affine_k = AffineLayer(conf.att_aff, isize=conf.dim_k, osize=eff_hidden_size)
        self.affine_v = AffineLayer(conf.att_aff, isize=conf.dim_v, osize=eff_hidden_size)
        # att drop
        self.att_drop = BK.nn.Dropout(conf.att_drop)
        # rel dist keys
        if conf.use_rel:
            self.relp = RelpLayer(conf.relp, dim=conf.d_qkv)
        else:
            self.relp = None
        # final output
        self.final_linear = AffineLayer(conf.out_aff, isize=eff_hidden_size, osize=conf.dim_out)
        # --

    def extra_repr(self) -> str:
        conf: AttentionConf = self.conf
        return f"Att({(conf.dim_q, conf.dim_k, conf.dim_v)},R={conf.use_rel})"

    def get_output_dims(self, *input_dims):
        return (self.conf.dim_out, )

    # *[*, L, D{q/k/v}], [*, Lq, Lk] -> [*, L, Dv]
    def forward(self, query, key, value, mask_k=None, mask_qk=None, rposi=None, cache=None, ret_att=False, external_attn=None):
        conf: AttentionConf = self.conf
        query_len = BK.get_shape(query, -2)
        key_len = BK.get_shape(key, -2)
        # --
        # 1. project the three
        query_up = self._shape_project(self.affine_q(query))    # [*, H, Q, D]
        has_cache = cache is not None and len(cache) > 0
        if has_cache and self.kv_static:  # already calculated the static cache, directly reuse!
            key_up, value_up, mask_k = cache['k'], cache['v'], cache['m']
        else:
            key_up = self._shape_project(self.affine_k(key))        # [*, H, K, D]
            value_up = self._shape_project(self.affine_v(value))    # [*, H, K, D]
            if has_cache:  # concat previous ones!
                key_up = BK.concat([cache['k'], key_up], -2)  # [*, H, PREV+K, D]
                value_up = BK.concat([cache['v'], value_up], -2)  # [*, H, PREV+K, D]
                if mask_k is None:  # all 1s
                    mask_k = BK.constants(key.shape[:-1], value=1.)  # [*, LK]
                mask_k = BK.concat([cache['m'], mask_k], -1)  # [*, PREV+K]
                if mask_qk is not None and BK.get_shape(mask_qk)[-2:] == [query_len, key_len]:
                    mask_qk = BK.concat([unsqueeze_expand(cache['m'], -2, query_len), mask_qk], -1, True)  # [*, Q, PREV+K]
                    # breakpoint()
        # --
        if cache is not None:
            if mask_k is None:  # all 1s
                mask_k = BK.constants(key.shape[:-1], value=1.)  # [*, LK]
            cache.update({'k': key_up, 'v': value_up, 'm': mask_k})
        # --
        # relative position; note: need rposi input if using cache
        if self.relp is not None:
            if rposi is None:
                rposi = self.relp.len2dist(query_len, key_len)
                dist_atts, dist_values = self.relp.dist2emb(rposi)  # [Q, K, D]
            else:
                if len(BK.get_shape(rposi)) > 2:
                    rposi = rposi.unsqueeze(-3)  # add for H
                dist_atts, dist_values = self.relp.dist2emb(rposi)  # [*, H1, Q, K, D]
        else:
            dist_atts = dist_values = None
        # --
        # 2. calculate and scale scores
        query_up = query_up / self._att_scale
        scores = BK.matmul(query_up, BK.transpose(key_up, -1, -2))  # [*, head, Q, K]
        if dist_atts is not None:
            if len(BK.get_shape(dist_atts)) <= 3:  # we can make it more mem efficient
                # adopted from T2T: https://github.com/tensorflow/tensor2tensor/blob/5f9dd2db6d7797162e53adf152310ed13e9fc711/tensor2tensor/layers/common_attention.py#L1705
                # rearrange to avoid broadcast: [Q, **H, D] * [Q, D, K] -> [Q, **H, K] -> ..
                _query_shape = BK.get_shape(query_up)  # [*, H, Q, D]
                _query_up0 = query_up.view([np.prod(_query_shape[:-2]).item()] + _query_shape[-2:]).transpose(0, 1)
                add_term0 = BK.matmul(_query_up0, dist_atts.transpose(-1,-2))  # [Q, **H, K]
                add_term = add_term0.transpose(0,1).view_as(scores)  # [*, H, Q, K]
            else:  # todo(+2): not quite efficient if doing this!
                # let it broadcast: [..., Q, K, D] * [*, head, Q, D, 1] => [*, head, Q, K, 1]
                add_term = BK.matmul(dist_atts, query_up.unsqueeze(-1)).squeeze(-1)
            scores = scores + add_term
        # 3. apply attention
        if mask_k is not None:  # note: mask as [*, K]
            scores = scores + (1.-mask_k).unsqueeze(-2).unsqueeze(-3) * Constants.REAL_PRAC_MIN
        if mask_qk is not None:  # note: mask as [*, Q, K]
            scores = scores + (1.-mask_qk).unsqueeze(-3) * Constants.REAL_PRAC_MIN
        attn = BK.softmax(scores, -1)  # [*, H, Q, K]
        if external_attn is not None:  # [*, Q, K]
            mixed_attn = external_attn.unsqueeze(-3) * attn  # [*, H, Q, K]
            attn = mixed_attn / (mixed_attn.sum(-1,keepdims=True) + 1e-8)  # renorm!
        drop_attn = self.att_drop(attn)  # [*, H, Q, K]
        context = BK.matmul(drop_attn, value_up)  # [*, H, Q, D]
        if dist_values is not None:
            if len(BK.get_shape(dist_values)) <= 3:  # same as above
                _att_shape = BK.get_shape(attn)  # [*, H, Q, K]
                _attn0 = drop_attn.view([np.prod(_att_shape[:-2]).item()] + _att_shape[-2:]).transpose(0, 1)
                add_c0 = BK.matmul(_attn0, dist_values)  # [Q, **H, D]
                add_c = add_c0.transpose(0,1).view_as(context)  # [*, H, Q, D]
            else:
                add_c = BK.matmul(drop_attn.unsqueeze(-2), dist_values).squeeze(-2)  # [*, H, Q, D]
            context = context + add_c
        # 4. final
        context_merge = self._unshape_project(context)  # [*, Q, H*D]
        output = self.final_linear(context_merge)  # [*, len_q, mdim]
        return (output, attn) if ret_att else output

    # [*, L, H*D] <-> [*, H, L, D]
    def _shape_project(self, x):
        head_count = self.conf.head_count
        orig_shape = BK.get_shape(x)
        x_size = orig_shape[:-1] + [head_count, orig_shape[-1]//head_count]
        return x.view(x_size).transpose(-2, -3)  # [*, H, L, D]

    def _unshape_project(self, x):
        x_size = BK.get_shape(x)  # [*, H, L, D]
        return x.transpose(-2, -3).reshape(x_size[:-3]+[x_size[-2], x_size[-1]*x_size[-3]])  # [*, L, H*D]

# --
# Relative position
@NnConf.rd('relP')
class RelpConf(NnConf):
    def __init__(self):
        super().__init__()
        self.dim = -1
        self.rel_clip = 0  # use rel if >0!
        self.rel_use_v = False  # add dist-values to V
        self.rel_use_neg = True  # otherwise, use abs(dist)

@RelpConf.conf_rd()
class RelpLayer(NnLayer):
    def __init__(self, conf: RelpConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: RelpConf = self.conf
        _num = (2*conf.rel_clip+1) if conf.rel_use_neg else (conf.rel_clip+1)
        self.edge_atts = BK.nn.Embedding(_num, conf.dim)
        self.edge_values = None
        if conf.rel_use_v:
            self.edge_values = BK.nn.Embedding(_num, conf.dim)

    def extra_repr(self) -> str:
        conf: RelpConf = self.conf
        return f"Relp({conf.dim},V={conf.rel_use_v},N={conf.rel_use_neg})"

    def get_output_dims(self, *input_dims):
        return (self.conf.dim, )

    # from dist
    def dist2emb(self, distance: BK.Expr):
        conf: RelpConf = self.conf
        if conf.rel_use_neg:
            distance = distance.clamp(min=-conf.rel_clip, max=conf.rel_clip) + conf.rel_clip
        else:
            distance = distance.abs().clamp(max=conf.rel_clip)
        dist_atts = self.edge_atts(distance)
        if self.use_dist_v:
            dist_values = self.edge_values(distance)
        else:
            dist_values = None
        return dist_atts, dist_values

    # from lens
    def len2dist(self, query_len: int, key_len: int):
        a_q = BK.arange_idx(query_len).unsqueeze(1)  # [query, 1]
        a_k = BK.arange_idx(key_len).unsqueeze(0)  # [1, key]
        ret_dist = a_q - a_k  # [query, key]
        return ret_dist

# --
# RNN cell (note: a simple wrapper over *Cell)

@NnConf.rd('rnnC')
class RnnCellConf(NnConf):
    def __init__(self):
        super().__init__()
        self.rnn_type = 'lstm'
        self.isize = -1  # input size
        self.hsize = 256  # hidden size
        self.dropout = 0.1  # dropout

@RnnCellConf.conf_rd()
class RnnCellLayer(NnLayer):
    def __init__(self, conf: RnnCellConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: RnnCellConf = self.conf
        # --
        _node_type = {'lstm': BK.nn.LSTMCell, 'gru': BK.nn.GRUCell, 'rnn': BK.nn.RNNCell}[conf.rnn_type]
        self.is_lstm = conf.rnn_type == 'lstm'
        self.node = _node_type(conf.isize, conf.hsize)
        self.drop = BK.nn.Dropout(conf.dropout)

    def extra_repr(self) -> str:
        conf: RnnCellConf = self.conf
        return f"RNNCell[{conf.rnn_type}](I={conf.isize},H={conf.hsize})"

    def get_output_dims(self, *input_dims):
        return (self.conf.hsize, )

    # [*, D], [*] -> (*?)[*, D]
    def forward(self, t_input, t_prev=None, t_mask=None):
        conf: RnnCellConf = self.conf
        res = self.node(t_input, t_prev)  # forward rnn-cell
        if self.is_lstm:  # drop & mask
            res = (self.drop(res[0]), res[1])
            if t_mask is not None:
                if t_prev is None:
                    _tmp = BK.zeros(BK.get_shape(t_input)[:-1] + [conf.hsize])  # [*, H]
                    t_prev = (_tmp, _tmp)
                t_mask = t_mask.unsqueeze(-1)
                res = (t_mask * res[0] + (1.-t_mask) * t_prev[0], t_mask * res[1] + (1.-t_mask) * t_prev[1])
        else:
            res = self.drop(res)
            if t_mask is not None:
                if t_prev is None:
                    t_prev = BK.zeros(BK.get_shape(t_input)[:-1] + [conf.hsize])  # [*, H]
                t_mask = t_mask.unsqueeze(-1)
                res = t_mask * res + (1.-t_mask) * t_prev
        return res

    def res2hid(self, res):
        return res[0] if self.is_lstm else res

# --
# Rnn layer

@NnConf.rd('rnn')
class RnnConf(NnConf):
    def __init__(self):
        super().__init__()
        self.cell = RnnCellConf.direct_conf(_rm_names=('isize', 'hsize'))  # specify at outside!
        self.mdim = -1  # model dim (default for isize and osize)
        self.use_cross = False  # add cross-att?
        self.isize = -1  # input dim
        self.osize = -1  # hidden/output_dim
        self.n_layers = 3  # number of layers
        self.n_layer_lidxes = []  # specific names for the layers
        self.bidirectional = False  # bidirectional?
        self.sep_bidirection = False  # sep the two directions when stacking layers?

@RnnConf.conf_rd()
class RnnLayer(NnLayer):
    def __init__(self, conf: RnnConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: RnnConf = self.conf
        # --
        assert not conf.use_cross, "Not implemented yet!"  # todo(+N): add cross-att to enable it as a decoder
        # actually three modes: 1) uni-d, 2) bi-d + sep, 3) bi-d + non-sep
        if conf.mdim > 0:
            conf.isize = conf.osize = conf.mdim
        n_input, n_hidden = conf.isize, conf.osize
        if not conf.bidirection:  # 1) both full n_hidden
            one_hid_dim, one_inp_dim = n_hidden, n_hidden
        else:
            assert n_hidden % 2 == 0, f"Hidden-dim {n_hidden} not dividable by 2 for bidirection!"
            one_hid_dim = n_hidden // 2  # each hid only get half
            if conf.sep_bidirection:  # 2) bi-d + sep
                one_inp_dim = n_hidden // 2
            else:  # 3) bi-d + non-sep
                one_inp_dim = n_hidden  # combine as input!
        # --
        cur_inp_dim = n_input  # start with real input dim
        cur_out_dim = cur_inp_dim
        self.fnode_names, self.bnode_names = [], []
        _lidxes = list(conf.n_layer_lidxes) if conf.n_layer_lidxes else list(range(conf.n_layers))
        for layer_idx in range(conf.n_layers):
            _lidx = _lidxes[layer_idx]
            self.add_module(f"F{_lidx}", RnnCellLayer(conf.cell, isize=cur_inp_dim, hsize=one_hid_dim))
            self.fnode_names.append(f"F{_lidx}")
            if conf.bidirection:
                self.add_module(f"B{_lidx}", RnnCellLayer(conf.cell, isize=cur_inp_dim, hsize=one_hid_dim))
                self.bnode_names.append(f"B{_lidx}")
            cur_inp_dim = one_inp_dim
            cur_out_dim = n_hidden
        self.output_dim = cur_out_dim  # final output dim

    @property
    def fnodes(self):
        return [getattr(self, k) for k in self.fnode_names]
    @property
    def bnodes(self):
        return [getattr(self, k) for k in self.bnode_names]

    def extra_repr(self) -> str:
        conf: RnnConf = self.conf
        return f"RNN[{conf.rnn_type}](I={conf.isize},H={conf.hsize},L={conf.n_layers})"

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    # [*, L, D], [*, L]
    def forward(self, t_input, t_mask=None, cache=None, return_dict=False):
        conf: RnnConf = self.conf
        if conf.bidirectional:
            assert cache is None, "No incremental for bidirectional RNN!"
        # split inputs
        step_inputs = split_at_dim(t_input, -1, False)  # *[*, D]
        if t_mask is None:
            step_masks = [None] * len(step_inputs)
        else:
            step_masks = split_at_dim(t_mask, -1, False)  # *[*]
        # --
        # loop
        fnodes, bnodes = self.fnodes, self.bnodes
        f_outputs = [step_inputs]  # *(layer)[List(step)]
        b_outputs = [step_inputs]  # ...
        if cache is None:  # int -> H or (H, C)
            cache = [None] * conf.n_layers
        for layer_idx in range(conf.n_layers):
            f_node = fnodes[layer_idx]
            tmp_f = []  # forward
            tmp_b = []  # backward
            tmp_f_prev = cache[layer_idx]
            for e, m in zip(f_outputs[-1], step_masks):
                one_f = f_node(e, tmp_f_prev, m)
                tmp_f.append(f_node.res2hid(one_f))  # store hid
                tmp_f_prev = one_f  # recurrent
            cache[layer_idx] = tmp_f_prev  # note: directly assign!
            if conf.bidirection:
                b_node = bnodes[layer_idx]
                tmp_b_prev = None  # note: no backward init!
                for e, m in zip(reversed(b_outputs[-1]), reversed(step_masks)):
                    one_b = b_node(e, tmp_b_prev, m)
                    tmp_b.append(b_node.res2hid(one_b))  # store hid
                    tmp_b_prev = one_b  # recurrent
                tmp_b.reverse()  # note: always store in l2r order
            # output or for next layer
            if not conf.bidirection:  # 1) uni-d
                f_outputs.append(tmp_f)
            elif conf.sep_bidirection:  # 2) bi-d + sep
                f_outputs.append(tmp_f)
                b_outputs.append(tmp_b)
            else:  # 3) bi-d + non-sep
                all_ctx_slices = [BK.concat([f, b], -1) for f, b in zip(tmp_f, tmp_b)]
                f_outputs.append(all_ctx_slices)
                b_outputs.append(all_ctx_slices)
        # --
        # finally
        if return_dict:
            all_hids = [t_input]
            if conf.bidirectional:
                all_hids.extend([BK.concat([BK.stack(a,-2), BK.stack(b,-2)], -1)
                                 for a,b in zip(f_outputs[1:], b_outputs[1:])])
            else:
                all_hids.extend([BK.stack(vs, -2) for vs in f_outputs[1:]])
            return ZObject(last_hidden_state=all_hids[-1], hidden_states=all_hids, cache=cache)
        else:
            if conf.n_layers == 0:
                return t_input  # simply return inputs
            else:  # the final layer
                if conf.bidirection:
                    stacked_res_f, stacked_res_b = BK.stack(f_outputs[-1], -2), BK.stack(b_outputs[-1], -2)
                    stacked_res = BK.concat([stacked_res_f, stacked_res_b], -1)  # concat at last dim
                else:
                    stacked_res = BK.stack(f_outputs[-1], -2)
                return stacked_res  # [*, L, D]
        # --

# --
# Transformer

@NnConf.rd('tfC')
class TfCellConf(NnConf):
    def __init__(self):
        super().__init__()
        self.mdim = -1  # model dim
        self.cdim = -1  # cross dim
        self.use_cross = False  # add cross-att?
        self.aconf = AttentionConf()
        self.d_ff = 1024  # dim of FF, 0 for skipping ff
        self.ff_act = "gelu"
        self.dropout = 0.1  # dropout at various places
        self.layer_norm_eps = 1e-5  # same as pytorch's default

@TfCellConf.conf_rd()
class TfCellLayer(NnLayer):
    def __init__(self, conf: TfCellConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: TfCellConf = self.conf
        mdim = conf.mdim
        cdim = conf.cdim if conf.cdim>0 else mdim
        # --
        self.attention = AttentionLayer(conf.aconf, dim_q=mdim, dim_k=mdim, dim_v=mdim,
                                        out_aff__out_drop=conf.dropout)  # self-att
        self.att_ln = BK.nn.LayerNorm(mdim, eps=conf.layer_norm_eps)
        # --
        if conf.use_cross:
            self.crossattention = AttentionLayer(conf.aconf, dim_q=mdim, dim_k=cdim, dim_v=cdim, dim_out=mdim,
                                                 kv_static=True, out_aff__out_drop=conf.dropout)  # cross-att
            self.catt_ln = BK.nn.LayerNorm(mdim, eps=conf.layer_norm_eps)
        # --
        if conf.d_ff > 0:
            self.ffn_d1 = BK.nn.Linear(mdim, conf.d_ff)
            self.ffn_act = ActivationHelper.get_act(conf.ff_act)
            self.ffn_d2 = BK.nn.Linear(conf.d_ff, mdim)
            self.ffn_ln = BK.nn.LayerNorm(conf.mdim, eps=conf.layer_norm_eps)
            self.ffn_drop = BK.nn.Dropout(conf.dropout)
        # --

    def extra_repr(self) -> str:
        return f"TFC"

    def get_output_dims(self, *input_dims):
        return (self.conf.mdim, )

    # [*, L, D], [*, L]; [*, Lc, D], [*, Lc]
    def forward(self, t_input, t_mask=None, t_mask_qk=None, t_cross=None, t_cross_mask=None, cache=None):
        conf: TfCellConf = self.conf
        cache_self, cache_cross = None, None
        if cache is not None:
            if 'self' not in cache:
                cache['self'] = {}
            if 'cross' not in cache:
                cache['cross'] = {}
            cache_self, cache_cross = cache['self'], cache['cross']
        # self-att
        res0, att = self.attention(t_input, t_input, t_input,
                                   mask_k=t_mask, mask_qk=t_mask_qk, cache=cache_self, ret_att=True)
        # add & norm
        res0p = self.att_ln(t_input + res0)
        # cross-att
        if conf.use_cross:
            res1 = self.crossattention(res0p, t_cross, t_cross, mask_k=t_cross_mask, cache=cache_cross)
            res1p = self.catt_ln(res0p + res1)
        else:
            res1p = res0p
        # ffn
        if conf.d_ff > 0:
            t_hid = self.ffn_act(self.ffn_d1(res1p))
            t_hid = self.ffn_drop(self.ffn_d2(t_hid))
            res2p = self.ffn_ln(t_hid + res1p)
        else:
            res2p = res1p
        return res2p, att

@NnConf.rd('tf')
class TransformerConf(NnConf):
    def __init__(self):
        super().__init__()
        self.tf = TfCellConf.direct_conf(_rm_names=('mdim', 'cdim', 'use_cross'))
        self.mdim = -1
        self.cdim = -1
        self.use_cross = False  # add cross-att?
        self.n_layers = 6  # number of layers
        self.n_layer_lidxes = []  # specific names for the layers

@TransformerConf.conf_rd()
class TransformerLayer(NnLayer):
    def __init__(self, conf: TransformerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: TransformerConf = self.conf
        # --
        self.node_names = []
        _lidxes = list(conf.n_layer_lidxes) if conf.n_layer_lidxes else list(range(conf.n_layers))
        for layer_idx in range(conf.n_layers):
            lname = f"T{_lidxes[layer_idx]}"
            self.add_module(lname, TfCellLayer(conf.tf, mdim=conf.mdim, cdim=conf.cdim, use_cross=conf.use_cross))
            self.node_names.append(lname)
        # --

    @property
    def nodes(self):
        return [getattr(self, k) for k in self.node_names]

    def extra_repr(self) -> str:
        conf: TransformerConf = self.conf
        return f"Transformer[L={conf.n_layers}]"

    def get_output_dims(self, *input_dims):
        return (self.conf.mdim, )

    def forward(self, t_input, cache=None, return_dict=False, **kwargs):
        all_hids, all_atts = [t_input], []
        for layer_idx, layer_node in enumerate(self.nodes):
            l_cache = None
            if cache is not None:
                if layer_idx not in cache:
                    cache[layer_idx] = {}
                l_cache = cache[layer_idx]
            t_res, t_att = layer_node(all_hids[-1], **kwargs, cache=l_cache)
            all_hids.append(t_res)
            all_atts.append(t_att)
        if return_dict:
            return ZObject(last_hidden_state=all_hids[-1], hidden_states=all_hids, attentions=all_atts, cache=cache)
        else:
            return all_hids[-1]

# --
# b mspx/nn/layers/seq:??
