#

# simplified idec nodes (can also serve as pure mid-(adapter)-layers)
# todo(+N): simply put all args in **kwargs, which is not a good design and may be error-prone!!

__all__ = ["IdecConf", "IdecNode"]

from typing import List
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.utils import ConfEntryChoices, ConfEntryTyped
from ..common import ZMediator, ZAttConf, ZAttNode

# =====
# the idec

class IdecConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        # general sizes from outside
        self._isize = -1  # input size (emb)
        self._nhead = -1  # input size (num_head)
        self._csize = -1  # number of classes
        # apply layers (allow different settings for different layers)
        self.app_layers = []  # which layers to apply this idec: List[int]
        self.app_feeds = []  # whether allow feed (by default 0)
        self.app_detachs = []  # gradient scaling at detaching, 0 means full detach! (by default 1.)
        # common mid-layer node setting
        self.node = MidLayerConf()
        # --

    # shortcut to make specific confs
    @staticmethod
    def make_conf(mode: str):
        ret = IdecConf()
        if mode == 'ff':  # ff adapter
            ret.node.conn.do_dsel = False
            ret.node.core = CoreFfConf()
        elif mode == 'satt':  # satt adapter
            ret.node.conn.do_dsel = False
            ret.node.core = CoreSattConf()
        elif mode == 'score':  # plain scorer
            ret.node.conn.do_dsel = True
            ret.node.core = CoreScorerConf()
        elif mode == 'pairwise':  # pairwise scorer
            ret.node.conn.do_dsel = True
            ret.node.core = CorePairwiseConf()
        elif mode == 'default':  # configurable!
            pass
        else:
            raise NotImplementedError(f"UNK mode: {mode}")
        return ret

@node_reg(IdecConf)
class IdecNode(BasicNode):
    def __init__(self, conf: IdecConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecConf = self.conf
        # --
        self.app_lidxes = [int(z) for z in conf.app_layers]  # app_idx -> lidx
        # check nonnegative and increasing
        assert all(z>0 for z in self.app_lidxes)
        assert all(self.app_lidxes[i+1]>self.app_lidxes[i] for i in range(len(self.app_lidxes)-1))
        self.app_lidxes_set = set(self.app_lidxes)
        self.max_app_lidx = max(self.app_lidxes) if len(self.app_lidxes)>0 else -1
        # --
        _NUM_LIDX = self.max_app_lidx + 1
        self.nodes: List = [None] * _NUM_LIDX
        self.lidx2aidx = [None] * _NUM_LIDX  # layer_idx -> app_idx
        self.lidx2feeds = [False] * _NUM_LIDX  # whether feed, by default False
        self.lidx2detachs = [1.] * _NUM_LIDX  # detach_scales, by default 1., no detach!
        for aidx, lidx in enumerate(self.app_lidxes):  # assign them all
            # info
            self.lidx2aidx[lidx] = aidx
            for _src_list, _trg_list, _ff in \
                    zip([conf.app_feeds, conf.app_detachs], [self.lidx2feeds, self.lidx2detachs],
                        [lambda x: bool(int(x)), float]):
                if aidx < len(_src_list):
                    _trg_list[lidx] = _ff(_src_list[aidx])
            # mod
            _node = conf.node.make_node(self.conf)  # put self.conf for info about external dims
            self.add_module(f"_M{lidx}", _node)  # add parameters
            self.nodes[lidx] = _node
            # check forced_feed for the adapters
            if _node.forced_feed:
                self.lidx2feeds[lidx] = True
        # --

    @property
    def num_app_layers(self):
        return len(self.app_lidxes)

    @property
    def app_layers(self):
        return self.app_lidxes

    def has_layer(self, lidx: int):
        return lidx in self.app_lidxes_set

    def has_feed(self, lidx: int):
        return self.has_layer(lidx) and self.lidx2feeds[lidx]

    def forward(self, med: ZMediator):
        cur_lidx = med.lidx
        # simply forward with layer specific options
        return self.nodes[cur_lidx].forward(
            med, feed=self.lidx2feeds[cur_lidx], detach_scale=self.lidx2detachs[cur_lidx])
        # --

# =====
# mid layers: combining conn + core

class MidLayerConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self.conn = ConnectorConf()
        self.core = ConfEntryChoices({"ff": CoreFfConf(), "satt": CoreSattConf(),
                                      "score": CoreScorerConf(), "pairwise": CorePairwiseConf()}, "ff")
        # --

@node_reg(MidLayerConf)
class MidLayerNode(BasicNode):
    def __init__(self, conf: MidLayerConf, idec_conf: IdecConf, **kwargs):
        super().__init__(conf, **kwargs)
        # --
        self.conn = conf.conn.make_node(idec_conf)
        self.core = conf.core.make_node(idec_conf)
        # --

    @property
    def forced_feed(self):  # these types will force feed!!
        return isinstance(self.core, (CoreFfNode, CoreSattNode))

    def forward(self, med: ZMediator, **kwargs):
        input_t, mask_t = self.conn.forward_input(med, **kwargs)
        scores_t, feeds_t = self.core.forward(input_t, mask_t=mask_t, **kwargs)
        ret_scores_t, ret_feeds_t = self.conn.forward_output(med, input_t, scores_t, feeds_t, **kwargs)
        return ret_scores_t, ret_feeds_t

# =====
# connector

# selctor from encoder's to decoder's
class DSelectorConf(BasicConf):
    def __init__(self):
        super().__init__()
        self._isize = -1
        self.dsel_method = "first"  # first/last/avg/sum/max, todo(+N): make add parameters?
        self.dsel_max_subtoks = 10  # for avg/max, take first how many ones
        # --

@node_reg(DSelectorConf)
class DSelector(BasicNode):
    def __init__(self, conf: DSelectorConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: DSelectorConf = self.conf
        # --
        self._f = getattr(self, "_forward_"+conf.dsel_method)

    @property
    def signature(self):
        return self.conf.dsel_method  # currently simply dsel name!!

    def _forward_first(self, repr_t: BK.Expr, dsel_seq_info):
        _arange_t, _sel_t = dsel_seq_info.arange2_t, dsel_seq_info.dec_sel_idxes
        _ss_t = _sel_t.clamp(max=repr_t.shape[1]-1)  # avoid OOR
        ret = repr_t[_arange_t, _ss_t]  # first subtok
        return ret

    def _forward_last(self, repr_t: BK.Expr, dsel_seq_info):
        _arange_t, _sel_t, _len_t = dsel_seq_info.arange2_t, dsel_seq_info.dec_sel_idxes, dsel_seq_info.dec_sel_lens
        _ss_t = (_sel_t+_len_t-1).clamp(max=repr_t.shape[1]-1)  # avoid OOR
        ret = repr_t[_arange_t, _ss_t]  # last subtok
        return ret

    # helper function: aggregate information for subtoks
    def _aggregate_subtoks(self, repr_t: BK.Expr, dsel_seq_info):
        conf: DSelectorConf = self.conf
        _arange_t, _sel_t, _len_t = dsel_seq_info.arange2_t, dsel_seq_info.dec_sel_idxes, dsel_seq_info.dec_sel_lens
        _max_len = 1 if BK.is_zero_shape(_len_t) else _len_t.max().item()
        _max_len = max(1, min(conf.dsel_max_subtoks, _max_len))  # truncate
        # --
        _tmp_arange_t = BK.arange_idx(_max_len)  # [M]
        _all_valids_t = (_tmp_arange_t < _len_t.unsqueeze(-1)).float()  # [*, dlen, M]
        _tmp_arange_t = _tmp_arange_t * _all_valids_t.long()  # note: pad as 0
        _all_idxes_t = _sel_t.unsqueeze(-1) + _tmp_arange_t  # [*, dlen, M]
        # --
        _ss_t = _all_idxes_t.clamp(max=repr_t.shape[1] - 1)  # avoid OOR
        _all_repr_t = repr_t[_arange_t.unsqueeze(-1), _ss_t]  # [*, dlen, M, D]
        while len(BK.get_shape(_all_valids_t)) < len(BK.get_shape(_all_repr_t)):
            _all_valids_t = _all_valids_t.unsqueeze(-1)
        _all_repr_t = _all_repr_t * _all_valids_t
        return _all_repr_t, _all_valids_t

    def _forward_avg(self, repr_t: BK.Expr, dsel_seq_info):
        RDIM = 2  # reduce dim
        # --
        _all_repr_t, _all_valids_t = self._aggregate_subtoks(repr_t, dsel_seq_info)
        div0, div1 = _all_repr_t.sum(RDIM), (_all_valids_t.sum(RDIM, keepdims=True) + 1e-5)
        ret = div0 / div1  # [*, dlen, D]
        return ret

    def _forward_sum(self, repr_t: BK.Expr, dsel_seq_info):
        RDIM = 2  # reduce dim
        # --
        _all_repr_t, _ = self._aggregate_subtoks(repr_t, dsel_seq_info)
        ret = _all_repr_t.sum(RDIM)
        return ret

    def _forward_max(self, repr_t: BK.Expr, dsel_seq_info):
        RDIM = 2  # reduce dim
        # --
        _all_repr_t, _ = self._aggregate_subtoks(repr_t, dsel_seq_info)
        ret = _all_repr_t.sum(RDIM) if BK.is_zero_shape(_all_repr_t) else _all_repr_t.max(RDIM)[0]  # [*, dlen, D]
        ret = BK.relu(ret)  # note: for simplicity, just make things>=0.
        return ret

    def forward(self, repr_t: BK.Expr, dsel_seq_info):
        return self._f(repr_t, dsel_seq_info)

# real connector
class ConnectorConf(BasicConf):
    def __init__(self):
        super().__init__()
        self._isize = -1
        # whether do dec's selection?
        self.do_dsel = True
        self.dsel_conf = DSelectorConf()
        self.dsel_nonhit_zero = True  # for the nonhit subwords, feed 0, otherwise feed start
        # special seq mode
        self.do_seq_mode = "none"  # none/sel/pool
        self.seq_sel_key = "None"  # this idx is over original seq (not subword seq)
        self.seq_pool = "idx0"  # pooling over all items: idx0/avg/max
        # --
        # extra gate for feeding
        _gate_conf = MLPConf().direct_conf(osize=1, dim_hid=256, n_hid_layer=1)
        _gate_conf.out_conf.no_drop = True  # no drop here
        self.gate = ConfEntryChoices({"yes": _gate_conf, "no": None}, "no")
        # --

@node_reg(ConnectorConf)
class ConnectorNode(BasicNode):
    def __init__(self, conf: ConnectorConf, idec_conf: IdecConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ConnectorConf = self.conf
        _isize = conf._isize if (conf._isize<0) else idec_conf._isize
        # --
        self.dsel = None
        if conf.do_dsel:
            self.dsel = DSelector(conf.dsel_conf, _isize=_isize)
        self.do_dsel, self.dsel_nonhit_zero = conf.do_dsel, conf.dsel_nonhit_zero
        self.do_seq_pool, self.do_seq_sel = [conf.do_seq_mode == z for z in ["pool", "sel"]]
        self.seq_sel_key = eval(conf.seq_sel_key)  # eval it!!
        if self.do_seq_pool:
            self.seq_pool_f = {
                "idx0": lambda x: x.narrow(-2, 0, 1).squeeze(-2), "avg": lambda x: x.mean(-2),
                "max": lambda x: x.max(-2)[0],
            }[conf.seq_pool]
        else:
            self.seq_pool_f = None
        # --
        self.gate = None
        if conf.gate is not None:
            self.gate = MLPNode(conf.gate, isize=_isize)
        # --

    # two forwards
    def forward_input(self, med: ZMediator, detach_scale: float, **kwargs):
        # get it
        if self.do_dsel:
            _dsel = self.dsel
            input_t0 = med.get_enc_cache_val(
                "hid", signature=_dsel.signature, function=(lambda x: _dsel.forward(x, med.ibatch.seq_info)))  # [*, ??, D]
        else:
            # input_t0 = med.get_enc_cache_val("hid", no_cache=True)  # [*, ??, D], note: no need for caching!
            input_t0 = med.get_enc_cache_val("hid")  # [*, ??, D]
        mask_t = med.get_mask(self.do_dsel)  # [*, ??]
        # extra processing?
        if self.do_seq_pool:
            input_t = self.seq_pool_f(input_t0)  # [*, D]
        elif self.do_seq_sel:
            _arange_t = BK.arange_idx(BK.get_shape(input_t0, 0))  # [*]
            _idx_t = med.get_cache(self.seq_sel_key)  # [*]
            input_t = input_t0[_arange_t, _idx_t]  # [*, D]
        else:
            input_t = input_t0
        # detach?
        ret_t = BK.go_detach(input_t, detach_scale, self.is_training())
        return ret_t, mask_t  # [*, (??), D], [*, ??]

    def forward_output(self, med: ZMediator, input_t: BK.Expr, scores_t: BK.Expr, feeds_t: BK.Expr, **kwargs):
        if feeds_t is None:  # if no feeding, simply return as it is
            return scores_t, feeds_t
        # --
        if self.gate is not None:
            gate_v = self.gate(input_t).sigmoid()  # [*, ??, 1]
            feeds_t = feeds_t * gate_v  # [*, ??, D]
        if self.do_dsel:  # transform back to the original seq_len
            dsel_seq_info = med.ibatch.seq_info
            _arange_t, _back_sel_idxes = dsel_seq_info.arange2_t, dsel_seq_info.enc_back_sel_idxes  # [*, ??]
            feeds_t = feeds_t[_arange_t, _back_sel_idxes]  # [*, ??, D]
            if self.dsel_nonhit_zero:  # zero out the nonhit ones
                feeds_t = feeds_t * dsel_seq_info.enc_back_hits.unsqueeze(-1)
        return scores_t, feeds_t

# =====
# core nodes

# simply stacked FF nodes
class CoreFfConf(BasicConf):
    def __init__(self):
        super().__init__()
        self._isize = -1  # input dim
        self._osize = -1  # output dim
        # --
        self.ff_dim = 256
        self.ff_act = 'relu'
        # --

@node_reg(CoreFfConf)
class CoreFfNode(BasicNode):
    def __init__(self, conf: CoreFfConf, idec_conf: IdecConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: CoreFfConf = self.conf
        # --
        _isize = conf._isize if (conf._isize>0) else idec_conf._isize
        _osize = conf._osize if (conf._osize>0) else _isize  # by default same as isize
        self.aff_in = AffineNode(None, isize=_isize, osize=conf.ff_dim, out_act=conf.ff_act)
        self.aff_out = AffineNode(None, isize=conf.ff_dim, osize=_isize)

    # [*, in] -> [*, ff] -> [*, in(out)]
    def forward(self, expr_t: BK.Expr, feed: bool, **kwargs):
        assert feed
        hid_t = self.aff_in(expr_t)
        out_t = self.aff_out(hid_t)
        return None, out_t  # no scores, always feed

# similar to ff, core_scorer
class CoreScorerConf(BasicConf):
    def __init__(self):
        super().__init__()
        self._isize = -1  # input dim
        self._csize = -1  # number of class to pred (0 means directly use input!!)
        self._osize = -1  # output dim
        # --
        self.hid_dim = 0  # optional one more hidden layer
        self.hid_act = 'elu'
        self.no_aff_in = False  # no aff_in!
        self.init_scale_in = 1.  # init scale for pred
        self.init_scale_out = 1.  # init scale for output
        self.out_act = 'linear'  # used for feed and output
        self.use_out_score = False  # note: by default, no adding of output params!
        # --

@node_reg(CoreScorerConf)
class CoreScorerNode(BasicNode):
    def __init__(self, conf: CoreScorerConf, idec_conf: IdecConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: CoreScorerConf = self.conf
        # --
        _isize = conf._isize if (conf._isize>0) else idec_conf._isize
        _csize = conf._csize if (conf._csize>0) else idec_conf._csize
        _osize = conf._osize if (conf._osize>0) else _isize  # by default same as isize
        # --
        self.aff_hid = None
        cur_size = _isize
        if conf.hid_dim > 0:
            self.aff_hid = AffineNode(None, isize=_isize, osize=conf.hid_dim, out_act=conf.hid_act)
            cur_size = conf.hid_dim
        if not conf.no_aff_in:
            self.aff_in = AffineNode(None, isize=cur_size, osize=_csize, no_drop=True, init_scale=conf.init_scale_in)
            cur_size = _csize
        else:
            self.aff_in = None
        assert cur_size == _csize  # no matter what, we should get _csize!!
        # --
        self.out_act_f = ActivationHelper.get_act(conf.out_act)
        if conf.use_out_score:
            self.aff_out = AffineNode(None, isize=cur_size, osize=_osize, init_scale=conf.init_scale_out)
        else:
            self.aff_out = None
        # --

    # [*, in] -> [*, L] -> [*, out]
    def forward(self, expr_t: BK.Expr, feed: bool, **kwargs):
        # hid?
        if self.aff_hid is not None:
            expr_t = self.aff_hid(expr_t)  # [*, D]
        # raw score
        if self.aff_in is not None:
            score_t = self.aff_in(expr_t)  # [*, L]
        else:  # directly feed input?
            score_t = expr_t
        # out
        out_t = None
        if feed:
            mid_t = self.out_act_f(score_t)  # [*, L]
            out_t = self.aff_out(mid_t)  # [*, D]
        return score_t, out_t

# simple satt node
class CoreSattConf(BasicConf):
    def __init__(self):
        super().__init__()
        self._isize = -1  # input dim
        self._osize = -1  # output dim
        # --
        self.satt = ZAttConf()

@node_reg(CoreSattConf)
class CoreSattNode(BasicNode):
    def __init__(self, conf: CoreSattConf, idec_conf: IdecConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: CoreSattConf = self.conf
        # --
        _isize = conf._isize if (conf._isize>0) else idec_conf._isize
        _osize = conf._osize if (conf._osize>0) else _isize  # by default same as isize
        self.satt = ZAttNode(conf.satt, dim_q=_isize, dim_k=_isize, dim_v=_osize)
        # --

    # [*, in] -> [*, in(out)]
    def forward(self, expr_t: BK.Expr, mask_t: BK.Expr, feed: bool, **kwargs):
        assert feed
        out_t = self.satt.forward(expr_t, expr_t, expr_t, mask_k=mask_t)
        return None, out_t  # no scores, always feed

# pairwise scoring
class CorePairwiseConf(BasicConf):
    def __init__(self):
        super().__init__()
        self._isize = -1  # input dim
        self._csize = -1  # number of class to pred
        self._osize = -1  # output dim
        # --
        self.satt = ZAttConf().direct_update(nh_qk=64, d_qk=32)  # more heads for more features
        self.no_scorer = False  # no scorer, directly use satt's output as output
        self.pre_cs_drop = 0.1
        self.score = CoreScorerConf()  # reuse!
        self.use_out_pairwise = False  # note: by default, no adding of output params!
        # --

@node_reg(CorePairwiseConf)
class CorePairwiseNode(BasicNode):
    def __init__(self, conf: CorePairwiseConf, idec_conf: IdecConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: CorePairwiseConf = self.conf
        # --
        _isize = conf._isize if (conf._isize>0) else idec_conf._isize
        _csize = conf._csize if (conf._csize>0) else idec_conf._csize
        _osize = conf._osize if (conf._osize>0) else _isize  # by default same as isize
        # --
        self.satt = ZAttNode(conf.satt, dim_q=_isize, dim_k=_isize, dim_v=_osize, use_out_att=conf.use_out_pairwise)
        if conf.no_scorer:
            self.pre_cs_drop = None
            self.scorer = None
        else:
            self.pre_cs_drop = DropoutNode(None, drop_rate=conf.pre_cs_drop, fix_drop=False)
            self.scorer = CoreScorerNode(conf.score, None, _isize=conf.satt.nh_qk, _csize=_csize, _osize=conf.satt.nh_v,
                                         use_out_score=conf.use_out_pairwise)
        # --

    # [*, slen, D] -> [*, len_q, len_k, H] -> [*, slen, D]
    def forward(self, expr_t: BK.Expr, mask_t: BK.Expr, feed: bool, **kwargs):
        satt = self.satt
        pairwise_t0 = satt.do_score(expr_t, expr_t)  # [*, Hin, len_q, len_k]
        pairwise_t = pairwise_t0.permute(0,2,3,1).contiguous()  # [*, len_q, len_k, Hin]
        if self.scorer is None:  # no scorer!
            scores_t, out_t = pairwise_t, None
        else:
            # note: especial score dropout!
            pairwise_t1 = self.pre_cs_drop(pairwise_t)  # [*, len_q, len_k, Hin]
            scores_t, out_t = self.scorer.forward(pairwise_t1, feed=feed)  # [*, len_q, len_k, C], [*, len_q, len_k, Hout]
        if feed:
            permute_out_t = out_t.permute(0,3,1,2)  # [*, Hout, len_q, len_k]
            out_t = satt.do_output(permute_out_t, expr_t, mask_k=mask_t)  # [*, slen, D]
        return scores_t, out_t

# --
# b msp2/tasks/zmtl2/zmod/dev/base_idec:
