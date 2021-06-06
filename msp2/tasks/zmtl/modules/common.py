#

# some common modules

__all__ = [
    "RelDistConf", "RelDistNode", "AttentionPlainConf", "AttentionPlainNode", "ZMediator",
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
            self.edge_values = PosiEmbeddingNode(None, osize=dim, max_val=conf.clip_dist, zero0=conf.posi_zero0, no_drop=True)
        else:
            self.edge_atts = EmbeddingNode(None, osize=dim, n_words=2*conf.clip_dist+1, no_drop=True)
            self.edge_values = EmbeddingNode(None, osize=dim, n_words=2*conf.clip_dist+1, no_drop=True)

    # using provided
    def embed_rposi(self, distance: BK.Expr):
        ret_dist = distance
        # use different ranges!
        if not self.use_neg_dist:
            distance = BK.abs(distance)
        if not self.use_posi_dist:  # clamp and offset if using plain embeddings
            distance = BK.clamp(distance, min=-self.clip_dist, max=self.clip_dist) + self.clip_dist
        dist_atts = self.edge_atts(distance)
        dist_values = self.edge_values(distance)
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
class AttentionPlainConf(BasicConf):
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
        # rel
        self.use_rposi = True  # whether use rel_dist
        self.rel = RelDistConf()

@node_reg(AttentionPlainConf)
class AttentionPlainNode(BasicNode):
    def __init__(self, conf: AttentionPlainConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: AttentionPlainConf = self.conf
        dim_q, dim_k, dim_v, nh_qk, d_qk, nh_v, d_v = \
            conf.dim_q, conf.dim_k, conf.dim_v, conf.nh_qk, conf.d_qk, conf.nh_v, conf.d_v
        # --
        self._att_scale = math.sqrt(conf.d_qk)  # scale for score
        # pre-att affines (no dropouts here!)
        _eg_q = BK.get_inita_xavier_uniform((d_qk, dim_q)) / BK.get_inita_xavier_uniform((nh_qk*d_qk, dim_q))
        self.affine_q = AffineNode(None, isize=dim_q, osize=nh_qk*d_qk, no_drop=True, init_scale=_eg_q*conf.init_scale_hin)
        _eg_k = BK.get_inita_xavier_uniform((d_qk, dim_k)) / BK.get_inita_xavier_uniform((nh_qk*d_qk, dim_k))
        self.affine_k = AffineNode(None, isize=dim_k, osize=nh_qk*d_qk, no_drop=True, init_scale=_eg_k*conf.init_scale_hin)
        self.affine_v = AffineNode(None, isize=dim_v, osize=nh_v*d_v, no_drop=True)
        # rel dist keys
        self.rposi = RelDistNode(conf.rel, _dim=d_qk) if conf.use_rposi else None
        # att & output
        if conf.useaff_qk2v:
            self.aff_qk2v = AffineNode(None, isize=nh_qk, osize=nh_v)
        else:
            # assert nh_qk == nh_v
            if nh_qk != nh_v:
                zwarn(f"Possible problems with AttNode since hin({nh_qk}) != hout({nh_v})")
        self.adrop = DropoutNode(None, drop_rate=conf.att_drop, fix_drop=False)
        # todo(note): with drops(y) & act(?) & bias(y)?
        self.final_linear = AffineNode(None, isize=nh_v*d_v, osize=dim_v, out_act=conf.out_act)

    def extra_repr(self) -> str:
        conf: AttentionPlainConf = self.conf
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
        conf: AttentionPlainConf = self.conf
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
        conf: AttentionPlainConf = self.conf
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

# =====
# mediate (storing values) between enc and dec

class ZMediator:
    def __init__(self, encoder, decoders: List, force_min_layer=0, force_max_layer=Constants.INT_PRAC_MAX):
        # --
        # static: decoders
        self.encoder = encoder
        self.decoders = [z for z in decoders if z is not None]
        self.force_min_layer = force_min_layer  # (forced) min layer
        self.force_max_layer = force_max_layer  # (forced) max layer
        # caches
        self.insts = None
        self.force_lidx = None  # note: saved for special usage
        self.exit_lidxes: List = None
        self.lidx = 0  # current at which layer: L0 is emb, L1+ is enc
        self.mask_t = None  # [*, slen]
        self.valid_idxes_t = None  # [*, slen], selecting first tokens
        self.rev_f = None  # callable: to reverse values back
        self.attns = []  # store all attns
        self.embs = []  # store all embs
        self.c_caches = {}  # (consistent cache) name -> value
        self.l_caches = {}  # (inconsistent cache, cleared after each layer) name -> value
        self.main_scores = {}  # name -> List[value]
        self.aug_scores = {}  # name -> List[value]

    # --
    # helper
    @staticmethod
    def append_scores(from_dict: Dict, to_dict: Dict):
        # simply appending!!
        for k,v in from_dict.items():
            if k not in to_dict:
                to_dict[k] = []
            if isinstance(v, list):  # extend previous list!!
                to_dict[k].extend(v)
            else:  # append for leaf node!
                to_dict[k].append(v)
        # --
    # --

    # --
    def set_cache(self, k, v, assert_no_exist=True):
        if assert_no_exist:
            assert k not in self.c_caches  # no repeat set!
        self.c_caches[k] = v

    def get_cache(self, k, df=None):
        return self.c_caches.get(k, df)
    # --

    # last raw emb, [*, ??len, D]; note: when we want to get seq-level features
    def get_raw_last_emb(self):
        return self.embs[-1]  # simply the last one!

    # last emb, [*, slen, D]
    def get_last_emb(self):
        k = "last_emb"
        ret = self.l_caches.get(k)
        if ret is None:
            ret = self.embs[-1]
            valid_idxes_t = self.valid_idxes_t
            if valid_idxes_t is not None:
                arange2_t = BK.arange_idx(BK.get_shape(valid_idxes_t, 0)).unsqueeze(-1)  # [bsize, 1]
                ret = ret[arange2_t, valid_idxes_t]  # select!
            self.l_caches[k] = ret  # cache
        return ret

    # stack emb, [*, slen, D, NL]
    def get_stack_emb(self):
        k = "stack_emb"
        ret = self.l_caches.get(k)
        if ret is None:
            # note: excluding embeddings here to make it convenient!!
            ret = BK.stack(self.embs[1:], -1)  # [*, slen, D, NL]
            valid_idxes_t = self.valid_idxes_t
            if valid_idxes_t is not None:
                arange2_t = BK.arange_idx(BK.get_shape(valid_idxes_t, 0)).unsqueeze(-1)  # [bsize, 1]
                ret = ret[arange2_t, valid_idxes_t]  # select!
            self.l_caches[k] = ret  # cache
        return ret

    # stacked attn, [*, lenq, lenk, NL, H]
    def get_stack_att(self):
        k = "stack_att"
        ret = self.l_caches.get(k)
        if ret is None:
            ret = BK.stack(self.attns, -1).permute(0,2,3,4,1)  # NL*[*, H, lenq, lenk] -> [*, lenq, lenk, NL, H]
            valid_idxes_t = self.valid_idxes_t
            if valid_idxes_t is not None:
                arange3_t = BK.arange_idx(BK.get_shape(valid_idxes_t, 0)).unsqueeze(-1).unsqueeze(-1)  # [bsize, 1, 1]
                ret = ret[arange3_t, valid_idxes_t.unsqueeze(-1), valid_idxes_t.unsqueeze(-2)]  # select!
            self.l_caches[k] = ret  # cache
        return ret

    # mask_t: [*, slen]
    def get_mask_t(self):
        return self.mask_t

    # --
    def restart(self, insts=None, mask_t=None, valid_idxes_t=None, rev_f=None):  # [*, slen]
        self.insts = insts
        self.exit_lidxes = None if insts is None else [-1]*len(insts)  # start with -1!
        self.lidx = 0
        self.mask_t = mask_t
        self.valid_idxes_t = valid_idxes_t
        self.rev_f = rev_f
        self.attns = []
        self.embs = []
        self.c_caches = {}
        self.l_caches = {}
        self.main_scores = {}
        self.aug_scores = {}

    # end of one layer: [*, slen, D], [*, H, lenq, lenk]; return whether early exit?
    def layer_end(self, emb_t, attn_t):
        if attn_t is None:
            assert self.lidx == 0
        else:
            assert self.lidx == len(self.attns) + 1  # no attns for L0!!
            self.attns.append(attn_t)
        assert self.lidx == len(self.embs)
        self.embs.append(emb_t)
        # check decoders: use self to communicate!
        rets = []
        all_satisfied = (self.lidx >= self.force_min_layer)  # first must satisfy min_layer
        for d in self.decoders:
            if d.active:  # only forward if active!
                one_scores, one_rets, one_satisfied = d.layer_end(self)
                ZMediator.append_scores(one_scores, self.main_scores)  # add to med instead of nodes!!
                rets.extend(one_rets)
                all_satisfied = all_satisfied and one_satisfied
        all_satisfied = all_satisfied or (self.lidx >= self.force_max_layer)  # or satisfy max_layer
        all_satisfied = all_satisfied or (self.force_lidx is not None and self.lidx >= self.force_lidx)  # another special one!!
        # ret
        all_ret = sum_or_none(rets)
        if all_ret is not None:
            if self.rev_f is not None:  # need to reverse back
                all_ret = self.rev_f(all_ret)
        # go next
        self.lidx += 1
        self.l_caches.clear()
        return all_ret, all_satisfied

    # =====
    def do_losses(self):
        all_losses = []
        for dec in self.decoders:
            if dec.active:
                one_loss = dec.loss(self)
                all_losses.append(one_loss)
        return all_losses

    def do_preds(self):
        info = Counter()
        for dec in self.decoders:
            if dec.active:
                one_info = dec.predict(self)
                info += Counter(one_info)
        return info

    def do_scores(self, *args, **kwargs):
        info = Counter()
        for dec in self.decoders:
            if dec.active:
                one_info = dec.score(self, *args, **kwargs)
                info += Counter(one_info)
        return info

# --
# b msp2/tasks/zmtl/modules/common:301
