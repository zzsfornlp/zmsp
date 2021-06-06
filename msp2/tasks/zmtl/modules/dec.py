#

# the base of dec
# -- and dec helpers

__all__ = [
    "CoreScorerConf", "CoreScorerNode", "IdecConf", "IdecNode",
    "IdecSingleConf", "IdecSingleNode", "IdecPairwiseConf", "IdecPairwiseNode", "IdecAttConf", "IdecAttNode",
    "IdecConnectorConf", "IdecConnectorNode", "IdecConnectorAttConf", "IdecConnectorAttNode",
    "ZDecoderConf", "ZDecoder", "ZDecoderHelper",
]

from typing import List
from collections import OrderedDict
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.utils import ConfEntryChoices
from .common import *
from .dec_help import *

# =====
# the core scorer

class CoreScorerConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self._isize = -1  # input dim
        self._csize = -1  # number of class
        self._osize = -1  # output dim
        self.init_scale_in = 1.  # init scale for pred
        self.init_scale_out = 1.  # init scale for output

@node_reg(CoreScorerConf)
class CoreScorerNode(BasicNode):
    def __init__(self, conf: CoreScorerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: CoreScorerConf = self.conf
        isize, csize, osize = conf._isize, conf._csize, conf._osize
        # --
        self.aff_in = AffineNode(None, isize=isize, osize=csize, no_drop=True, init_scale=conf.init_scale_in)
        self.w_out = BK.new_param([csize, osize])  # [nlab, hid_dim]
        self.reset_parameters()

    def reset_parameters(self):
        BK.init_param(self.w_out, "glorot", lookup=True, scale=self.conf.init_scale_out)

    def extra_repr(self) -> str:
        conf: CoreScorerConf = self.conf
        isize, csize, osize = conf._isize, conf._csize, conf._osize
        return f"CoreScorer({isize}->{csize}=>{osize})"

    # [*, in] -> [*, L] -> [*, out]
    def forward(self, expr_t: BK.Expr, feed_output: bool, score_scale=1.):
        # pred
        score_t = self.aff_in(expr_t)  # [*, L]
        if score_scale != 1:
            score_t /= score_scale  # mainly for temperature scale!
        # out
        if feed_output:
            prob_t = score_t.softmax(-1)  # [*, L]
            out_t = BK.matmul(prob_t, self.w_out)  # [*, out]
        else:
            out_t = None
        return score_t, out_t

# =====
# the actual multi-layer scorers

# the base idec
class IdecConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self._isize = -1  # input size (emb)
        self._nhead = -1  # input size (num_head)
        self._csize = -1  # number of classes
        # core scorer
        self.cs = CoreScorerConf()
        self.share_cs = True
        # apply layers
        self.app_layers = []  # which layers to apply this idec
        self.app_ts = []  # temp-scales, mainly for prediction!! (by default 1., no ts)
        self.app_feeds = []  # whether allow feed (by default 0)
        self.app_detach_scales = []  # gradient scaling at detaching, 0 means full detach! (by default 1.)
        self.app_input_fixed_mask = []  # apply fixed mask at input, 0 means no mask (by default 0.)
        # loss & pred
        self.hconf: IdecHelperConf = ConfEntryChoices(
            {"simple": IdecHelperSimpleConf(), "simple2": IdecHelperSimple2Conf()}, "simple2")

@node_reg(IdecConf)
class IdecNode(BasicNode):
    def __init__(self, conf: IdecConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecConf = self.conf
        # --
        self.app_lidxes = sorted([int(z) for z in conf.app_layers])
        self.app_lidxes_set = set(self.app_lidxes)
        self.app_lidxes_map = {z:i for i,z in enumerate(self.app_lidxes)}  # lidx -> app_idx
        self.app_lidxes_map[-1] = len(self.app_lidxes) - 1
        # others
        self.app_ts_map = {z:float(t) for z,t in zip(self.app_lidxes, conf.app_ts)}  # by default 1.
        self.app_feeds_map = {z:int(t) for z,t in zip(self.app_lidxes, conf.app_feeds)}  # by default 0
        self.app_detach_scales_map = {z:float(t) for z,t in zip(self.app_lidxes, conf.app_detach_scales)}  # by default 1.
        self.app_input_fixed_mask_map = {z:float(t) for z,t in zip(self.app_lidxes, conf.app_input_fixed_mask)}  # by default 1.
        # --
        self.max_app_lidx = max(self.app_lidxes) if len(self.app_lidxes)>0 else -1
        # --
        self.connectors = [None] * (1 + self.max_app_lidx)
        self.scorers = [None] * (1 + self.max_app_lidx)
        # for loss/pred
        self.helper: IdecHelper = conf.hconf.make_node()

    def _make_connector(self, lidx: int, idec_node):  # note: simply pass self in ...
        raise NotImplementedError

    def _forw(self, med: ZMediator):
        raise NotImplementedError()

    def need_app_layer(self, lidx: int):
        return lidx in self.app_lidxes_set

    def forward(self, med: ZMediator):
        score_t, out_t = self._forw(med)
        # output both!!
        return score_t, out_t

    def _score(self, lidx: int, *args, **kwargs):
        _ts = self.app_ts_map.get(lidx, 1.)
        return self.scorers[lidx].forward(*args, score_scale=_ts, **kwargs)

    def make_connectors(self):
        for lidx in self.app_lidxes:
            conn = self._make_connector(lidx, self)
            self.add_module(f"C{lidx}", conn)  # reg it!
            self.connectors[lidx] = conn
        # --

    def make_scorers(self, *args, **kwargs):
        conf: IdecConf = self.conf
        # --
        if conf.share_cs:
            cs = CoreScorerNode(conf.cs, *args, **kwargs)
            self.add_module("cs", cs)
            for lidx in self.app_lidxes:
                self.scorers[lidx] = cs
        else:  # make one for each layer!
            for lidx in self.app_lidxes:
                cs = CoreScorerNode(conf.cs, *args, **kwargs)
                self.add_module(f"CS{lidx}", cs)
                self.scorers[lidx] = cs
        # --

# --
# plain connector
class IdecConnectorConf(BasicConf):
    pass

@node_reg(IdecConnectorConf)
class IdecConnectorNode(BasicNode):
    def __init__(self, conf: IdecConnectorConf, lidx: int, idec_node: IdecNode, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecConnectorConf = self.conf
        self.lidx = lidx
        self.detach_scale = idec_node.app_detach_scales_map.get(lidx, 1.)  # by default 1., no detach!!
        self.fixed_mask_rate = idec_node.app_input_fixed_mask_map.get(lidx, 1.)
        # --

    # go detach
    def _go_detach(self, x):
        return BK.go_detach(x, self.detach_scale, self.is_training())

    # [*, D], [*]
    def forward(self, med: ZMediator, **kwargs):
        x = med.get_last_emb()  # usually we only need this last emb
        return self._go_detach(x)

# plain one with some options
class IdecConnectorPlainConf(IdecConnectorConf):
    def __init__(self):
        super().__init__()
        self._isize = -1  # input size (#dim)
        # --
        self.use_nlayer = 1  # use how many layers, <=0 means all (mix with single weights)
        self.mid_dim = 256  # dim in-middle (before predict)
        self.mid_act = "elu"
        self.init_scale_mid = 1.  # init scale
        self.pre_mid_drop = 0.1
        # especially give it more context
        self.mid_extra_range = 0  # context window size to feed to the mid-layer [-r, r]
        # special mode: seq mode
        self.do_seq_mode = "none"  # none/sel/pool
        self.seq_sel_key = "evt_idx"  # note: this idx is over original seq (not subword seq)
        self.seq_pool = "idx0"  # pooling over all items: idx0/avg/max

    def get_out_dim(self, df: int):
        return self.mid_dim if self.mid_dim>0 else df

@node_reg(IdecConnectorPlainConf)
class IdecConnectorPlainNode(IdecConnectorNode):
    def __init__(self, conf: IdecConnectorPlainConf, lidx: int, idec_node: IdecNode, **kwargs):
        super().__init__(conf, lidx, idec_node, **kwargs)
        conf: IdecConnectorPlainConf = self.conf
        # --
        self.lstart = 0 if conf.use_nlayer<=0 else max(0, lidx-conf.use_nlayer)
        self.mixed_weights = BK.new_param([lidx-self.lstart])  # [NL]
        # --
        self.pre_mid_drop = DropoutNode(None, drop_rate=conf.pre_mid_drop, fix_drop=False)
        if conf.mid_dim > 0:
            self.mid_aff = AffineNode(None, isize=[conf._isize]*(2*conf.mid_extra_range+1), osize=conf.mid_dim,
                                      out_act=conf.mid_act, init_scale=conf.init_scale_mid)
        else:
            self.mid_aff = None
        # speical mode: seq
        self.do_seq_pool, self.do_seq_sel = [conf.do_seq_mode==z for z in ["pool", "sel"]]
        if self.do_seq_pool:
            assert conf.use_nlayer == 1
            self.pool_f = {
                "idx0": lambda x: x.narrow(-2,0,1).squeeze(-2), "avg": lambda x: x.mean(-2), "max": lambda x: x.max(-2)[0],
            }[conf.seq_pool]
        else:
            self.pool_f = None
        # note: special mask!!
        if self.fixed_mask_rate < 1.:
            self.input_mask = BK.new_param([conf._isize,])
        else:
            self.input_mask = None
        # --
        self.reset_parameters()

    def reset_parameters(self):
        BK.init_param(self.mixed_weights, "zero")  # make it all 0
        if self.input_mask is not None:
            with BK.no_grad_env():
                self.input_mask.set_(BK.random_bernoulli((self.conf._isize, ), self.fixed_mask_rate, 1.))
        # --

    # [*, D], [*]
    def forward(self, med: ZMediator, **kwargs):
        conf: IdecConnectorPlainConf = self.conf
        # --
        if self.do_seq_pool:
            # note: for pooling, use the raw emb!!
            mixed_emb0 = self._go_detach(med.get_raw_last_emb())  # [*, ??, D]
            mixed_emb = self.pool_f(mixed_emb0)  # [*, D]
        else:
            if conf.use_nlayer == 1:  # simply get the last one
                mixed_emb = self._go_detach(med.get_last_emb())
            else:  # mix them
                stacked_embs = self._go_detach(med.get_stack_emb())[:,:,:,-len(self.mixed_weights):]  # [*, slen, D, NL]
                mixed_emb = BK.matmul(stacked_embs, BK.softmax(self.mixed_weights, -1).unsqueeze(-1)).squeeze(-1)  # [*, slen, D]
            if self.do_seq_sel:
                _arange_t = BK.arange_idx(BK.get_shape(mixed_emb, 0))
                _idx_t = med.get_cache(conf.seq_sel_key)
                mixed_emb = mixed_emb[_arange_t, _idx_t]  # [*, D]
        # further affine
        if self.input_mask is not None:  # note: special input mask!!
            mixed_emb = mixed_emb * self.input_mask.detach()  # no grad for input_mask!!
        drop_emb = self.pre_mid_drop(mixed_emb)
        if conf.mid_dim > 0:
            # gather inputs
            _r = conf.mid_extra_range
            _detached_drop_emb = drop_emb.detach()
            _inputs = []
            for ii in range(-_r, _r+1):
                if ii<0:
                    _one = BK.pad(_detached_drop_emb[:,:ii], [0,0,-ii,0])
                elif ii==0:
                    _one = drop_emb  # no need to change!
                else:
                    _one = BK.pad(_detached_drop_emb[:,ii:], [0,0,0,ii])
                _inputs.append(_one)
            # --
            ret_t = self.mid_aff(_inputs)  # [*, slen, M] or [*, M]
        else:
            ret_t = drop_emb
        return ret_t
# --

# --
# single one over seq of embs (seq-labeling or seq-aggregating)
class IdecSingleConf(IdecConf):
    def __init__(self):
        super().__init__()
        # --
        self.conn = IdecConnectorPlainConf()

@node_reg(IdecSingleConf)
class IdecSingleNode(IdecNode):
    def __init__(self, conf: IdecSingleConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecSingleConf = self.conf
        isize, csize, osize = conf.conn.get_out_dim(conf._isize), conf._csize, conf._isize
        # --
        self.make_connectors()
        self.make_scorers(_isize=isize, _csize=csize, _osize=osize)

    # [*, D], [*]
    def _forw(self, med: ZMediator):
        cur_lidx = med.lidx
        features_t = self.connectors[cur_lidx].forward(med)  # actually check!
        return self._score(cur_lidx, features_t, feed_output=(self.app_feeds_map.get(cur_lidx, 0)))

    def _make_connector(self, lidx: int, idec_node):
        return IdecConnectorPlainNode(self.conf.conn, lidx, idec_node, _isize=self.conf._isize)
# --

# --
# plain pairwise one
class IdecPairwiseConf(IdecConf):
    def __init__(self):
        super().__init__()
        # --
        self.aconf = AttentionPlainConf().direct_update(nh_qk=64, d_qk=32)  # more heads for more features
        self.conn = IdecConnectorConf()
        self.pre_cs_drop = 0.1

@node_reg(IdecPairwiseConf)
class IdecPairwiseNode(IdecNode):
    def __init__(self, conf: IdecPairwiseConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecPairwiseConf = self.conf
        isize, csize = conf._isize, conf._csize
        # --
        # todo(+W): still sharing anode!
        self.anode = AttentionPlainNode(conf.aconf, dim_q=isize, dim_k=isize, dim_v=isize)
        self.pre_cs_drop = DropoutNode(None, drop_rate=conf.pre_cs_drop, fix_drop=False)
        self.make_connectors()
        self.make_scorers(_isize=conf.aconf.nh_qk, _csize=csize, _osize=conf.aconf.nh_v)

    # [*, slen, D], [*, slen]
    def _forw(self, med: ZMediator):
        cur_lidx = med.lidx
        _cur_feed = self.app_feeds_map.get(cur_lidx, 0)
        features_t = self.connectors[cur_lidx].forward(med)  # actually check!
        # --
        scores_t = self.anode.do_score(features_t, features_t)  # [*, Hin, len_q, len_k]
        scores1_t = scores_t.permute(0,2,3,1).contiguous()  # [*, len_q, len_k, Hin]
        # todo(note): especial score dropout!
        scores1_drop_t = self.pre_cs_drop(scores1_t)
        scores2_t, score_out_t = self._score(cur_lidx, scores1_drop_t, feed_output=_cur_feed)  # [*, len_q, len_k, Hout]
        if _cur_feed:
            score_out2_t = score_out_t.permute(0,3,1,2)  # [*, Hout, len_q, len_k]
            out_t = self.anode.do_output(score_out2_t, features_t, mask_k=med.get_mask_t())  # [*, len_q, D]
        else:
            out_t = None
        return scores2_t, out_t

    def _make_connector(self, lidx: int, idec_node):
        return IdecConnectorNode(self.conf.conn, lidx, idec_node)

# --
# att-based pairwise (with special connector!)

class IdecConnectorAttConf(IdecConnectorConf):
    def __init__(self):
        super().__init__()
        self._nhead = -1  # input size (#head)
        # --
        self.use_nlayer = 0  # use how many layers, <=0 means use all!
        self.head_end = 0  # use how many heads (slice them, simply use the first ones!, <=0 means all)
        self.mid_dim = 64  # dim in-middle (before predict)
        self.mid_act = "elu"
        self.init_scale_mid = 1.  # init scale
        self.pre_mid_drop = 0.1

@node_reg(IdecConnectorAttConf)
class IdecConnectorAttNode(IdecConnectorNode):
    def __init__(self, conf: IdecConnectorAttConf, lidx: int, idec_node: IdecNode, **kwargs):
        super().__init__(conf, lidx, idec_node, **kwargs)
        conf: IdecConnectorAttConf = self.conf
        # --
        assert lidx>=1, "L0 has not atts!"
        self.lstart = 0 if conf.use_nlayer<=0 else max(0, lidx-conf.use_nlayer)  # note: lidx starts with 1 since L0 has no attentions
        self.head_end = conf._nhead if conf.head_end<=0 else conf.head_end
        self.d_in = (lidx - self.lstart) * self.head_end
        # use both directions for more features!
        self.pre_mid_drop = DropoutNode(None, drop_rate=conf.pre_mid_drop, fix_drop=False)
        self.mid_aff = AffineNode(None, isize=self.d_in*2, osize=conf.mid_dim,
                                  out_act=conf.mid_act, init_scale=conf.init_scale_mid)

    # [*, len_q, len_k, [Layer, H]/[H]]
    def forward(self, med: ZMediator, **kwargs):
        conf: IdecConnectorAttConf = self.conf
        # --
        # get stack att: already transposed by zmed
        scores_t = med.get_stack_att()  # [*, len_q, len_k, NL, H]
        _d_bs, _dq, _dk, _d_nl, _d_nh = BK.get_shape(scores_t)
        in1_t = scores_t[:, :, :, self.lstart:, :self.head_end].reshape([_d_bs, _dq, _dk, self.d_in])  # [*, lenq, lenk, din]
        in2_t = in1_t.transpose(-3, -2)  # [*, lenk, lenq, din]
        cat_t = self._go_detach(BK.concat([in1_t, in2_t], -1))  # [*, lenk, lenq, din*2]
        # further affine
        cat_drop_t = self.pre_mid_drop(cat_t)  # [*, lenk, lenq, din*2]
        ret_t = self.mid_aff(cat_drop_t)  # [*, lenk, lenq, M]
        return ret_t

class IdecAttConf(IdecConf):
    def __init__(self):
        super().__init__()
        # --
        self.conn = IdecConnectorAttConf()

@node_reg(IdecAttConf)
class IdecAttNode(IdecNode):
    def __init__(self, conf: IdecAttConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecAttConf = self.conf
        # --
        csize, cs_in = conf._csize, conf.conn.mid_dim
        self.make_connectors()
        self.make_scorers( _isize=cs_in, _csize=csize, _osize=cs_in)

    # [*, lenq, lenk, D], ...
    def _forw(self, med: ZMediator):
        cur_lidx = med.lidx
        _cur_feed = self.app_feeds_map.get(cur_lidx, 0)
        features_t = self.connectors[cur_lidx].forward(med)  # actually check!, [*, lenq, lenk, M]
        # todo(+W): currently no feed for this one!
        assert not _cur_feed
        return self._score(cur_lidx, features_t, feed_output=_cur_feed)

    def _make_connector(self, lidx: int, idec_node):
        return IdecConnectorAttNode(self.conf.conn, lidx, idec_node, _nhead=self.conf._nhead)

# =====
# the overall decoder

class ZDecoderConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self.init_active = True  # active at init?

@node_reg(ZDecoderConf)
class ZDecoder(BasicNode):
    def __init__(self, conf: ZDecoderConf, name: str, **kwargs):
        super().__init__(conf, **kwargs)
        # --
        self.name = name
        self._max_app_lidx = None
        self._active = conf.init_active

    # active or not?
    @property
    def active(self):
        return self._active

    def set_active(self, a: bool):
        self._active = a

    # pass in the data/status, return whether satisfied
    def layer_end(self, med: ZMediator):
        raise NotImplementedError()

    def get_idec_nodes(self):
        raise NotImplementedError()

    @property
    def max_app_lidx(self):
        if self._max_app_lidx is None:
            r = -1  # simply get the highest one
            for n in self.get_idec_nodes():
                r = max(r, n.max_app_lidx)
            self._max_app_lidx = r
        return self._max_app_lidx

class ZDecoderHelper:
    @staticmethod
    def get_zobjs(insts: List, f, use_cache: bool, cache_name: str):
        # get info
        if use_cache:
            zobjs = []
            for s in insts:
                one = getattr(s, cache_name, None)
                if one is None:
                    one = f(s)
                    setattr(s, cache_name, one)  # set cache
                zobjs.append(one)
        else:
            zobjs = [f(s) for s in insts]
        return zobjs

# --
