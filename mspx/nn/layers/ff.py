#

# some simple feed forward layers

__all__ = [
    "AffineConf", "AffineLayer", "MlpConf", "MlpLayer", "CombinerConf", "CombinerLayer",
    "ScalarConf", "ScalarLayer", "SimConf", "SimLayer", "SubPoolerConf", "SubPoolerLayer",
    "PairScoreConf", "PairScoreLayer", "BigramConf", "BigramLayer",
]

import math
from typing import Union, List
import numpy as np
from mspx.utils import zlog
from ..backends import BK
from .base import *

# --
# Linear + activation + dropout
@NnConf.rd('aff')
class AffineConf(NnConf):
    def __init__(self):
        super().__init__()
        self.isize: Union[int, List[int]] = [0]
        self.osize = -1
        self.use_bias = True  # bias or not
        self.out_act = "linear"  # activation
        self.in_drop = 0.  # dropout for input
        self.out_drop = 0.  # dropout for output

@AffineConf.conf_rd()
class AffineLayer(NnLayer):
    def __init__(self, conf: AffineConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: AffineConf = self.conf
        # --
        self.act_f = ActivationHelper.get_act(conf.out_act)
        self.isize = sum(conf.isize) if isinstance(conf.isize, list) else int(conf.isize)
        if conf.osize == 0:  # note: special code!!
            self.linear = (lambda x: x)
        else:
            self.linear = BK.nn.Linear(self.isize, conf.osize, bias=conf.use_bias)
        self.in_drop = BK.nn.Dropout(conf.in_drop)  # dropout node
        self.out_drop = BK.nn.Dropout(conf.out_drop)  # dropout node

    def extra_repr(self) -> str:
        conf: AffineConf = self.conf
        return f"Affine({conf.isize}->{conf.osize}+{super().extra_repr()})"

    def get_output_dims(self, *input_dims):
        conf: AffineConf = self.conf
        ret = conf.osize if conf.osize>0 else input_dims
        return (ret, )

    def forward(self, inputs):
        # conf: AffineConf = self.conf
        if BK.is_tensor(inputs):
            inputs = [inputs]
        inp0 = BK.concat(inputs)
        inp1 = self.in_drop(inp0)
        hid0 = self.linear(inp1)
        hid1 = self.act_f(hid0)
        hid2 = self.out_drop(hid1)
        return hid2

# --
# MLP
@NnConf.rd('mlp')
class MlpConf(NnConf):
    def __init__(self):
        super().__init__()
        self.isize: Union[int, List[int]] = [0]
        self.osize: int = -1
        self.dim_hid = 256
        self.n_hid_layer = 0
        self.hid_conf = AffineConf().direct_update(out_act="elu", out_drop=0.1)
        self.use_out = True  # whether using output layer
        self.out_conf = AffineConf()

@MlpConf.conf_rd()
class MlpLayer(NnLayer):
    def __init__(self, conf: MlpConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: MlpConf = self.conf
        # --
        self.node_names = []
        cur_dim = conf.isize
        self.mlp_sizes = [cur_dim]
        # hidden layers
        for idx in range(conf.n_hid_layer):  # hidden layers
            node = AffineLayer(conf.hid_conf, isize=cur_dim, osize=conf.dim_hid)
            cur_dim = conf.dim_hid
            self.add_module(f"H{idx}", node)
            self.node_names.append(f"H{idx}")
            self.mlp_sizes.append(cur_dim)
        # final layer
        if conf.use_out:
            fnode = AffineLayer(conf.out_conf, isize=cur_dim, osize=conf.osize)
            cur_dim = conf.osize
            self.add_module(f"F", fnode)
            self.node_names.append("F")
            self.mlp_sizes.append(cur_dim)
        self.output_dim = cur_dim

    @property
    def nodes(self):
        return [getattr(self, k) for k in self.node_names]

    def extra_repr(self) -> str:
        return f"MLP({'->'.join([str(z) for z in self.mlp_sizes])})"

    def get_output_dims(self, *input_dims):
        ret = input_dims
        for n in self.nodes:
            ret = n.get_output_dims(ret)
        return ret

    def forward(self, input_expr):
        cur_expr = input_expr
        for n in self.nodes:
            cur_expr = n(cur_expr)
        return cur_expr

# --
# horizontal combiners
@NnConf.rd('comb')
class CombinerConf(NnConf):
    def __init__(self):
        super().__init__()
        # --
        self.isizes: List[int] = [0]
        self.osize: int = -1
        self.comb_method = "concat"  # affine/concat/sum/weighted/stack/pick/gate
        self.aff_conf = AffineConf()  # for both affine/gate-affine
        self.stack_dim = -2  # if stack, at which dim?
        self.pick_idx = -1  # simply pick one

@CombinerConf.conf_rd()
class CombinerLayer(NnLayer):
    def __init__(self, conf: CombinerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: CombinerConf = self.conf
        # --
        output_dim: int = None
        if conf.comb_method == "affine":
            # note: if not provided, output the mean of input sizes
            _osize = conf.osize if conf.osize>0 else int(np.mean(conf.isizes))
            self.AFF = AffineLayer(conf.aff_conf, isize=conf.isizes, osize=_osize)
            self._f = lambda xs: self.AFF(xs)
            output_dim = _osize
        elif conf.comb_method == "concat":
            self._f = lambda xs: BK.concat(xs, -1)  # [*, sum(d)]
            output_dim = sum(conf.isizes)
        elif conf.comb_method == "sum":
            assert all(z==conf.isizes[0] for z in conf.isizes), "Must be the same size to use SUM!!"
            self._f = lambda xs: BK.stack(xs, -1).sum(-1)  # [*, d, N] -> [*, d]
            output_dim = conf.isizes[0]
        elif conf.comb_method == "weighted":  # weighted sum
            assert all(z==conf.isizes[0] for z in conf.isizes), "Must be the same size to use SUM!!"
            self.GAMMA = BK.nn.Parameter(BK.input_real(1.))  # scalar, []
            fold = len(conf.isizes)
            self.LAMBDAS = BK.nn.Parameter(BK.input_real([0.]*fold))  # [N]
            self._f = lambda xs: (BK.stack(xs, -1) * BK.softmax(self.LAMBDAS, -1)).sum(-1) * self.GAMMA  # [*, d, N] -> [*, d]
            output_dim = conf.isizes[0]
        elif conf.comb_method == "stack":
            assert all(z==conf.isizes[0] for z in conf.isizes), "Must be the same size to use STACK!!"
            self._f = lambda xs: BK.stack(xs, conf.stack_dim)
            output_dim = len(conf.isizes) if conf.stack_dim==-1 else conf.isizes[0]
        elif conf.comb_method == 'pick':
            self._f = lambda xs: xs[conf.pick_idx]
            output_dim = conf.isizes[conf.pick_idx]
        elif conf.comb_method == 'gate':  # gated combine
            assert all(z==conf.isizes[0] for z in conf.isizes), "Must be the same size to use GATE!!"
            # note: make it flexible for dynamic number of items
            self.GATE = AffineLayer(conf.aff_conf, isize=conf.isizes[0], osize=conf.isizes[0])
            self._f = self._forward_gate
            output_dim = conf.isizes[0]
        else:
            raise NotImplementedError(f"UNK comb_method: {conf.comb_method}")
        # --
        self.output_dim = output_dim

    def _forward_gate(self, exprs: List):
        weights = [self.GATE(z) for z in exprs]  # [..., D]
        t_alpha = BK.stack(weights, -1).softmax(-1)  # [..., D, K]
        t_expr = BK.stack(exprs, -1)  # [..., D, K]
        ret = (t_expr * t_alpha).sum(-1)  # [..., D]
        return ret

    def extra_repr(self) -> str:
        conf: CombinerConf = self.conf
        return f"Combiner({conf.comb_method},{conf.isizes}->{self.output_dim})"

    def forward(self, exprs: List):
        return self._f(exprs)

# --
# a special scalar for convenience

@NnConf.rd('sca')
class ScalarConf(NnConf):
    def __init__(self):
        super().__init__()
        self.init = 0.
        self.fixed = True

@ScalarConf.conf_rd()
class ScalarLayer(NnLayer):
    def __init__(self, conf: ScalarConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ScalarConf = self.conf
        if conf.fixed:
            self.v = conf.init
        else:
            self.v = BK.nn.Parameter(BK.as_tensor(conf.init))
        # --

    def extra_repr(self) -> str:
        return f"Scalar({self.v})"

    def get_output_dims(self, *input_dims):
        return (1, )

    def forward(self):
        return self.v

    def reset_value(self, v: float):
        zlog(f"Reset scalar value from {self.v} to {v}")
        if BK.is_tensor(self.v):
            BK.set_value(self.v, v)
        else:
            assert self.conf.fixed
            self.v = v
        # --

# --
# similarity function

@NnConf.rd('sim')
class SimConf(NnConf):
    def __init__(self):
        super().__init__()
        self.isize = -1  # input size (only utilized when proj_dim>0)
        # --
        self.func = 'dot'  # similarity function: dot/cos/dist
        self.dot_div = True  # div by sqrt(dim)
        self.dist_p = 2.  # p for dist
        self.scale = ScalarConf.direct_conf(init=1., fixed=True)
        self.split_dims = []  # the full vector composes of how many pieces, last-dim can be omitted
        self.proj_dim = -1  # whether adding a projection layer? (if >0)
        self.proj_aff = AffineConf()
        self.dropout = 0.  # input dropout?
        # --

@SimConf.conf_rd()
class SimLayer(NnLayer):
    def __init__(self, conf: SimConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: SimConf = self.conf
        # --
        self.scale = conf.scale.make_node()
        self.split_dims = [int(z) for z in conf.split_dims]
        # --
        if conf.proj_dim > 0:
            self.proj = AffineLayer(conf.proj_aff, isize=conf.isize, osize=conf.proj_dim)
        else:
            self.proj = (lambda x: x)
        if conf.dropout > 0:
            self.drop = BK.nn.Dropout(conf.dropout)  # dropout node
        else:
            self.drop = (lambda x: x)
        # --

    def extra_repr(self) -> str:
        return f"SimLayer({self.conf.func})"

    def get_output_dims(self, *input_dims):
        return (1, )

    # [..., N1, D], [..., N2, D] -> [..., N1, N2] or [..., D], [..., D] -> [...]
    def forward(self, repr1, repr2, is_pair=True):
        conf: SimConf = self.conf
        _ff = conf.func
        # --
        repr1, repr2 = self.drop(self.proj(repr1)), self.drop(self.proj(repr2))
        # --
        _fulldim = BK.get_shape(repr1, -1)
        _dims = self.split_dims + [_fulldim - sum(self.split_dims)]
        _dims = [z for z in _dims if z>0]  # keep valid ones!
        _ndim = len(_dims)
        if _ndim > 1:
            r1s, r2s = repr1.split(_dims, -1), repr2.split(_dims, -1)
        else:  # no need to split
            r1s, r2s = [repr1], [repr2]
        # --
        # loop
        score = 0.
        square1, square2 = 0., 0.
        count0, count1, count2 = 0., 0., 0.
        for r1, r2 in zip(r1s, r2s):  # [???, D]
            # check square & count
            s1, s2 = (r1**2).sum(-1), (r2**2).sum(-1)  # |...|^2
            c1, c2 = (s1!=0.).to(BK.DEFAULT_FLOAT), (s2!=0.).to(BK.DEFAULT_FLOAT)  # valid counts
            # add them in
            if is_pair:
                s1, s2, c1, c2 = s1.unsqueeze(-1), s2.unsqueeze(-2), c1.unsqueeze(-1), c2.unsqueeze(-2)
            c0 = c1 * c2  # both valid
            square1 = square1 + s1
            square2 = square2 + s2
            count0 = count0 + c0  # both valid
            count1 = count1 + c1
            count2 = count2 + c2
            # for the actual scores
            if _ff in ['dot', 'cos']:  # both do dot-product first
                if is_pair:
                    s0 = BK.matmul(r1, r2.transpose(-1,-2))
                else:
                    s0 = (r1 * r2).sum(-1)
            elif _ff == 'dist':  # neg L? distance
                if is_pair:
                    s0 = (r1.unsqueeze(-2) - r2.unsqueeze(-3)).abs() ** conf.dist_p
                else:
                    s0 = (r1-r2).abs() ** conf.dist_p
                s0 = (s0.sum(-1) * c0)  # zero out if invalid!
            else:
                raise NotImplementedError(f"Unknown sim function: {_ff}")
            score = score + s0
        # --
        # outside
        final_score0 = score / count0.clamp(min=1.) * _ndim  # extend to full ones
        if _ff == 'dot':  # simply dot product
            if conf.dot_div:  # divide things by sqrt(dim)
                final_score0 = final_score0 / math.sqrt(_fulldim)
        elif _ff == 'cos':
            norm1 = (square1 / count1.clamp(min=1.) * _ndim) ** 0.5
            norm2 = (square2 / count2.clamp(min=1.) * _ndim) ** 0.5
            final_score0 = final_score0 / (norm1 * norm2).clamp(min=1e-8)  # cosine
        elif _ff == 'dist':
            final_score0 = - (final_score0 ** (1./conf.dist_p))  # 1/p & negate!!
            # note: if no valid pairs
            _no_valid = (count0==0.).to(BK.DEFAULT_FLOAT)
            final_score0 = _no_valid * (-10000.) + (1.-_no_valid) * final_score0
        # --
        ret = final_score0 * self.scale.forward()  # final scale
        # breakpoint()
        return ret

# --
# subword2word pooler
@NnConf.rd('spool')
class SubPoolerConf(NnConf):
    def __init__(self):
        super().__init__()
        self.pool_hid_f = 'first'  # first/last/mean2/max2
        self.pool_att_f = 'max4'  # first/last/mean4/max4
        # --

@SubPoolerConf.conf_rd()
class SubPoolerLayer(NnLayer):
    def __init__(self, conf: SubPoolerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: SubPoolerConf = self.conf
        # --

    # [*, L, D], [*, ??]
    def forward_hid(self, hid_t, sublen_t, pool_f=None):
        if pool_f is None:
            pool_f = self.conf.pool_hid_f
        # --
        _arange_t = BK.arange_idx(len(hid_t)).unsqueeze(-1)  # [bs, 1]
        _idx1_t = sublen_t.cumsum(-1) - 1  # [bs, ??]
        _idx0_t = _idx1_t - (sublen_t-1).clamp(min=0)  # [bs, ??]
        # --
        if pool_f == 'first':
            ret = hid_t[_arange_t, _idx0_t]  # [bs, ??, D]
        elif pool_f == 'last':
            ret = hid_t[_arange_t, _idx1_t]  # [bs, ??, D]
        elif pool_f == 'mean2':
            ret = (hid_t[_arange_t, _idx0_t] + hid_t[_arange_t, _idx1_t]) / 2
        elif pool_f == 'max2':
            ret = BK.max_elem(hid_t[_arange_t, _idx0_t], hid_t[_arange_t, _idx1_t])
        else:
            raise NotImplementedError(f"UNK pool_f: {pool_f}")
        return ret

    # [*, Q, K, D], [*, Q], [*, K]
    def forward_att(self, att_t, sublen_qt, sublen_kt=None, pool_f=None):
        if pool_f is None:
            pool_f = self.conf.pool_att_f
        # --
        _arange_t = BK.arange_idx(len(att_t)).unsqueeze(-1).unsqueeze(-1)  # [bs, 1, 1]
        _idx1_qt = (sublen_qt.cumsum(-1) - 1).unsqueeze(-1)  # [bs, ??, 1]
        _idx0_qt = _idx1_qt - (sublen_qt - 1).clamp(min=0).unsqueeze(-1)  # [bs, ??, 1]
        if sublen_kt is None:  # same as Q
            _idx0_kt, _idx1_kt = _idx0_qt.squeeze(-1).unsqueeze(-2), _idx1_qt.squeeze(-1).unsqueeze(-2)  # [bs, 1, ??]
        else:
            _idx1_kt = (sublen_kt.cumsum(-1) - 1).unsqueeze(-2)  # [bs, 1, ??]
            _idx0_kt = _idx1_kt - (sublen_kt - 1).clamp(min=0).unsqueeze(-2)  # [bs, 1, ??]
        # --
        # => [bs, Q?, K?, D]
        if pool_f == 'first':
            ret = att_t[_arange_t, _idx0_qt, _idx0_kt]
        elif pool_f == 'last':
            ret = att_t[_arange_t, _idx1_qt, _idx1_kt]  # [bs, Q?, K?, D]
        else:
            all4 = [att_t[_arange_t, a, b] for a in [_idx0_qt, _idx1_qt] for b in [_idx0_kt, _idx1_kt]]
            if pool_f == 'mean4':
                ret = BK.stack(all4, -1).mean(-1)
            elif pool_f == 'max4':
                ret = BK.stack(all4, -1).max(-1)[0]
            else:
                raise NotImplementedError(f"UNK pool_f: {pool_f}")
        return ret

# --
# pairwise scoring (adopted from nn.layers.scorer.PairScorerNode)
@NnConf.rd('ps')
class PairScoreConf(NnConf):
    def __init__(self):
        super().__init__()
        # --
        self.osize = 0  # output size
        self.isize = -1  # common isize!
        self.isize0: int = -1  # input size 0
        self.isize1: int = -1  # input size 1
        # linear layer for the inputs
        self.aff_dim = 0  # no intermediate if <=0!
        self.aff0 = AffineConf.direct_conf(out_act='elu', out_drop=0.1)
        self.aff1 = AffineConf.direct_conf(out_act='elu', out_drop=0.1)
        # how to score?
        self.use_biaff = False  # use biaffine?
        self.biaff_ortho = True  # ortho init if using biaff?
        self.use_dot = False  # use dot?
        self.mlp_dim = 0  # use mlp if >0
        # --

@PairScoreConf.conf_rd()
class PairScoreLayer(NnLayer):
    def __init__(self, conf: PairScoreConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: PairScoreConf = self.conf
        # --
        if conf.isize > 0:  # shortcut for the same!
            conf.isize0 = conf.isize1 = conf.isize
        if conf.osize == 0:  # special mode!
            conf.osize = 1
            self.squeeze_final = True
        else:
            self.squeeze_final = False
        cur_dims = [conf.isize0, conf.isize1]
        # input proj
        if conf.aff_dim > 0:
            self.aff0 = AffineLayer(conf.aff0, isize=conf.isize0, osize=conf.aff_dim)
            self.aff1 = AffineLayer(conf.aff1, isize=conf.isize1, osize=conf.aff_dim)
            cur_dims = [conf.aff_dim, conf.aff_dim]
        else:
            self.aff0 = None
            self.aff1 = None
        # score
        if conf.use_biaff:
            self.BW = BK.nn.Linear(cur_dims[0], cur_dims[1]*conf.osize, bias=False)
            if conf.biaff_ortho:  # init with ortho!
                with BK.no_grad_env():
                    _W = self.BW.weight
                    _D = cur_dims[1]
                    for ii in range(conf.osize):
                        BK.nn.init.orthogonal_(_W[:, _D*ii:_D*(ii+1)])
        else:
            self.BW = None
        if conf.mlp_dim > 0:
            self.mlp = MlpLayer(None, isize=cur_dims, osize=conf.osize, dim_hid=conf.mlp_dim, n_hid_layer=1)
        else:
            self.mlp = None
        self.dot_div = (cur_dims[0] * cur_dims[1]) ** 0.25  # sqrt(sqrt(in1*in2))
        zlog(f"Adopt *_div of {self.dot_div} for the current PairScorer!")
        # --

    def forward(self, repr0, repr1, is_pair=True):
        conf: PairScoreConf = self.conf
        # --
        if self.aff0 is not None:
            repr0 = self.aff0(repr0)
        if self.aff1 is not None:
            repr1 = self.aff1(repr1)
        # --
        score = 0.
        if conf.use_dot:
            if is_pair:
                s0 = BK.matmul(repr0, repr1.transpose(-1,-2)).unsqueeze(-1)
            else:
                s0 = (repr0 * repr1).sum(-1, keepdims=True)
            score = score + s0 / self.dot_div
        if conf.use_biaff:
            if is_pair:
                s0 = self.BW(repr0).view(BK.get_shape(repr0)[:-1] + [BK.get_shape(repr1, -1), conf.osize])  # [*, L0, r1, out]
                s1 = BK.matmul(repr1.unsqueeze(-3), s0)  # [*, L0, L1, Out]
            else:
                s0 = self.BW(repr0).view(BK.get_shape(repr0)[:-1] + [BK.get_shape(repr1, -1), conf.osize])  # [*, r1, out]
                s1 = BK.matmul(repr1.unsqueeze(-2), s0).squeeze(-2)  # [*, Out]
            score = score + s1 / self.dot_div  # also divide by 'dot_div'!
        if self.mlp is not None:
            if is_pair:
                from .misc import unsqueeze_expand
                s0 = BK.concat([unsqueeze_expand(repr0, -2, BK.get_shape(repr1, -2)),
                                unsqueeze_expand(repr1, -3, BK.get_shape(repr0, -2))], -1)
            else:
                s0 = BK.concat([repr0, repr1], -1)
            s1 = self.mlp(s0)
            score = score + s1
        if self.squeeze_final:
            score = score.squeeze(-1)  # [*, L0, L1]
        # --
        return score  # [*, L0, L1, ??]

# --
# simple bigram transition matrix
@NnConf.rd('bigram')
class BigramConf(NnConf):
    def __init__(self):
        super().__init__()
        self.osize = -1  # number of entries
        self.lrank_k = -1  # if >0; then use E^T W E instead of full

@BigramConf.conf_rd()
class BigramLayer(NnLayer):
    def __init__(self, conf: BigramConf, extra_values: BK.Expr = None, **kwargs):
        super().__init__(conf, **kwargs)
        conf: BigramConf = self.conf
        # --
        _osize = conf.osize
        # self._M = None  # [out_prev, out_next]
        self.extra_values = None  # [out, out] extra ones like mask
        if extra_values is not None:
            self.set_extra_values(extra_values)
        if conf.lrank_k > 0:
            self.E = BK.new_param([_osize, conf.lrank_k], init='xavier_uniform')  # [out, K]
            self.W = BK.new_param([conf.lrank_k, conf.lrank_k])  # [K, K]
        else:  # direct
            self.E = None
            self.W = BK.new_param([_osize, _osize])
        # --

    def set_extra_values(self, extra_values):
        _osize = self.conf.osize
        assert BK.get_shape(extra_values) == [_osize, _osize]
        self.extra_values = BK.input_real(extra_values)

    @property
    def M(self):
        _extra_values = self.extra_values
        if _extra_values is None:
            _extra_values = 0.
        # --
        if self.E is not None:
            tmp_v = BK.matmul(self.E, self.W)  # [out, K]
            _M = BK.matmul(tmp_v, self.E.t()) + _extra_values  # [out, out]
        else:
            _M = self.W + _extra_values  # [out, out]
        return _M  # [out_prev, out_next]

    # [*] -> [*, out_next]
    def forward(self, t_in: BK.Expr):
        next_scores = self.M[t_in]
        return next_scores

# --
# b mspx/nn/layers/ff:247
