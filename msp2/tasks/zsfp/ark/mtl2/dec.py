#

# the base of dec
# -- to be plugged into enc

__all__ = [
    "CoreScorerConf", "CoreScorerNode", "IdecConf", "IdecNode",
    "IdecSingleConf", "IdecSingleNode", "IdecPairwiseConf", "IdecPairwiseNode", "IdecAttConf", "IdecAttNode",
    "IdecConnectorConf", "IdecConnectorNode", "IdecConnectorAttConf", "IdecConnectorAttNode",
]

from typing import List
from msp2.nn.layers import *
from msp2.nn import BK
from collections import OrderedDict
from .common import AttentionPlainConf, AttentionPlainNode

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
        self.w_out = BK.new_param([osize, csize])  # [nlab, hid_dim]
        self.reset_parameters()

    def reset_parameters(self):
        BK.init_param(self.w_out, "glorot", lookup=True, scale=self.conf.init_scale_out)

    def extra_repr(self) -> str:
        conf: CoreScorerConf = self.conf
        isize, csize, osize = conf._isize, conf._csize, conf._osize
        return f"CoreScorer({isize}->{csize}=>{osize})"

    # [*, in] -> [*, L] -> [*, out]
    def forward(self, expr_t: BK.Expr, feed_output: bool):
        # pred
        score_t = self.aff_in(expr_t)  # [*, L]
        # out
        if feed_output:
            prob_t = score_t.softmax(-1)  # [*, L]
            out_t = BK.matmul(prob_t, self.w_out)  # [*, out]
        else:
            out_t = None
        return score_t, out_t

# =====
# the base idec
class IdecConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self._isize = -1  # input size (emb)
        self._nhead = -1  # input size (num_head)
        self._csize = -1  # number of classes
        self.conn = IdecConnectorConf()  # plain one is enough!
        # core scorer
        self.cs = CoreScorerConf()
        # apply layers
        self.app_layers = []  # which layers to apply this idec
        self.feed_layers = []  # which layers to further allow feed (first should have 'app')

@node_reg(IdecConf)
class IdecNode(BasicNode):
    def __init__(self, conf: IdecConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecConf = self.conf
        # --
        self.app_lidxes = sorted([int(z) for z in conf.app_layers])
        self.feed_lidxes_set = set([int(z) for z in conf.feed_layers])
        self.buffer_scores = OrderedDict()  # lidx -> scores
        self.buffer_inputs = OrderedDict()  # lidx -> inputs
        self.connectors = []

    def refresh(self, rop: RefreshOptions = None):
        super().refresh(rop)
        # --
        # clear buffer
        self.buffer_scores.clear()
        self.buffer_inputs.clear()
        # --

    def _make_connector(self, lidx: int, feed_output: bool):
        raise NotImplementedError

    def _forw(self, expr_t: BK.Expr, mask_t: BK.Expr, feed_output: bool, **kwargs):
        raise NotImplementedError()

    def forward(self, expr_t: BK.Expr, mask_t: BK.Expr, lidx: int, feed_output: bool, **kwargs):
        score_t, out_t = self._forw(expr_t, mask_t, feed_output, **kwargs)
        # store info
        self.buffer_scores[lidx] = score_t
        self.buffer_inputs[lidx] = expr_t
        return out_t

    def make_connectors(self):
        for lidx in self.app_lidxes:
            conn = self._make_connector(lidx, (lidx in self.feed_lidxes_set))
            self.add_module(f"C{lidx}", conn)  # reg it!
            self.connectors.append(conn)
        # --

# --
# plain connector
class IdecConnectorConf(BasicConf):
    pass

@node_reg(IdecConnectorConf)
class IdecConnectorNode(BasicNode):
    def __init__(self, conf: IdecConnectorConf, node: IdecNode, lidx: int, feed_output: bool, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecConnectorConf = self.conf
        self.lidx = lidx
        self.feed_output = feed_output
        self.setattr_borrow('node', node)
        # --

    # [*, D], [*]
    def forward(self, expr_t: BK.Expr, mask_t: BK.Expr, **kwargs):
        return self.node.forward(expr_t, mask_t, lidx=self.lidx, feed_output=self.feed_output, **kwargs)  # simply forwarding!
# --

# --
# single one over seq of embs
class IdecSingleConf(IdecConf):
    def __init__(self):
        super().__init__()
        # --

@node_reg(IdecSingleConf)
class IdecSingleNode(IdecNode):
    def __init__(self, conf: IdecSingleConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecSingleConf = self.conf
        isize, csize = conf._isize, conf._csize
        # --
        self.cs = CoreScorerNode(conf.cs, _isize=isize, _csize=csize, _osize=isize)
        self.make_connectors()

    # [*, D], [*]
    def _forw(self, expr_t: BK.Expr, mask_t: BK.Expr, feed_output: bool, **kwargs):
        return self.cs.forward(expr_t, feed_output)  # again simply fowarding!

    def _make_connector(self, lidx: int, feed_output: bool):
        return IdecConnectorNode(self.conf.conn, self, lidx, feed_output)

# --
# complex pairwise one
class IdecPairwiseConf(IdecConf):
    def __init__(self):
        super().__init__()
        # --
        self.aconf = AttentionPlainConf()

@node_reg(IdecPairwiseConf)
class IdecPairwiseNode(IdecNode):
    def __init__(self, conf: IdecPairwiseConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecPairwiseConf = self.conf
        isize, csize = conf._isize, conf._csize
        # --
        self.anode = AttentionPlainNode(conf.aconf, dim_q=isize, dim_k=isize, dim_v=isize)
        self.cs = CoreScorerNode(conf.cs, _isize=conf.aconf.nh_qk, _csize=csize, _osize=conf.aconf.nh_v)
        self.make_connectors()

    # [*, slen, D], [*, slen]
    def _forw(self, expr_t: BK.Expr, mask_t: BK.Expr, feed_output: bool, **kwargs):
        scores_t = self.anode.do_score(expr_t, expr_t)  # [*, Hin, len_q, len_k]
        scores1_t = scores_t.transpose(-3, -2).transpose(-2, -1).contiguous()  # [*, len_q, len_k, Hin]
        scores2_t = self.cs.forward(scores1_t, feed_output)  # [*, len_q, len_k, Hout]
        if feed_output:
            out_t = self.anode.do_output(scores2_t, expr_t, mask_k=mask_t)
        else:
            out_t = None
        return out_t

    def _make_connector(self, lidx: int, feed_output: bool):
        return IdecConnectorNode(self.conf.conn, self, lidx, feed_output)

# --
# att-based pairwise (with special connector!)

class IdecConnectorAttConf(IdecConnectorConf):
    def __init__(self):
        super().__init__()
        self._nhead = -1  # input size (#head)
        # --
        self.use_nlayer = 0  # use how many layers, <=0 means use all!
        self.head_end = 0  # use how many heads (slice them, simply use the first ones!, <=0 means all)
        self.mid_dim = 100  # dim in-middle (before predict)
        self.init_scale_mid = 1.  # init scale

@node_reg(IdecConnectorAttConf)
class IdecConnectorAttNode(IdecConnectorNode):
    def __init__(self, conf: IdecConnectorAttConf, node: IdecNode, lidx: int, feed_output: bool, **kwargs):
        super().__init__(conf, node, lidx, feed_output, **kwargs)
        conf: IdecConnectorAttConf = self.conf
        # --
        assert lidx>=1, "L0 has not atts!"
        self.lstart = 0 if conf.use_nlayer<=0 else max(0, lidx-conf.use_nlayer)  # note: lidx starts with 1 since L0 has no attentions
        self.head_end = conf._nhead if conf.head_end<=0 else conf.head_end
        self.d_in = (lidx - self.lstart) * conf.head_end  # use both directions!
        self.mid_aff = AffineNode(None, isize=self.d_in*2, osize=conf.mid_dim, init_scale=conf.init_scale_mid)

    # [*, slen, D], [*, slen]; [*, len_q, len_k, [Layer, H]/[H]]
    def forward(self, expr_t: BK.Expr, mask_t: BK.Expr, scores_t=None, **kwargs):
        conf: IdecConnectorAttConf = self.conf
        # --
        # prepare input
        _d_bs, _dq, _dk, _d_nl, _d_nh = BK.get_shape(scores_t)
        in1_t = scores_t[:, :, :, self.lstart:, :self.head_end].reshape([_d_bs, _dq, _dk, self.d_in])  # [*, lenq, lenk, din]
        in2_t = in1_t.transpose(-3, -2)  # [*, lenk, lenq, din]
        final_input_t = BK.concat([in1_t, in2_t], -1)  # [*, lenk, lenq, din*2]
        # forward
        node_ret_t = self.node.forward(final_input_t, mask_t, self.feed_output, self.lidx, **kwargs)  # [*, lenq, lenk, head_end]
        if self.feed_output:
            # pad zeros if necessary
            if self.head_end < _d_nh:
                pad_t = BK.zeros([_d_bs, _dq, _dk, _d_nh-self.head_end])
                node_ret_t = BK.concat([node_ret_t, pad_t], -3)  # [*, lenq, lenk, Hin]
            return node_ret_t
        else:
            return None

class IdecAttConf(IdecConf):
    def __init__(self):
        super().__init__()
        # --
        self.conn = IdecConnectorAttConf()  # simply overwrite!!

@node_reg(IdecAttConf)
class IdecAttNode(IdecNode):
    def __init__(self, conf: IdecAttConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecAttConf = self.conf
        # --
        csize, cs_in, cs_out = conf._csize, conf.conn.mid_dim, conf.conn.head_end if conf.conn.head_end>0 else conf._nhead
        self.cs = CoreScorerNode(conf.cs, _isize=cs_in, _csize=csize, _osize=cs_out)
        self.make_connectors()

    # [*, lenq, lenk, D], ...
    def _forw(self, input_t: BK.Expr, mask_t: BK.Expr, feed_output: bool, **kwargs):
        return self.cs.forward(input_t, feed_output)  # again simply fowarding!

    def _make_connector(self, lidx: int, feed_output: bool):
        return IdecConnectorAttNode(self.conf.conn, self, lidx, feed_output, _nhead=self.conf._nhead)

# =====
# loss and pred
