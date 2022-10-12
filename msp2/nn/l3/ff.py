#

# some simple feed forward layers

__all__ = [
    "AffineConf", "AffineLayer", "MlpConf", "MlpLayer", "CombinerConf", "CombinerLayer",
    "ScalarConf", "ScalarLayer", "SimConf", "SimLayer", "SubPoolerConf", "SubPoolerLayer",
    "PairReprConf", "PairReprLayer", "SimpleNormConf", "SimpleNormLayer", "PairScoreConf", "PairScoreLayer",
]

import math
from typing import Union, List
import numpy as np
from msp2.utils import zlog
from ..backends import BK
from .base import *

# --
# Linear + activation + dropout
class AffineConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        self.isize: Union[int, List[int]] = []
        self.osize = -1
        self.use_bias = True  # bias or not
        self.out_act = "linear"  # activation
        self.dropout = 0.  # 0. means no dropout

    @classmethod
    def _get_type_hints(cls):
        return {"isize": int}

@node_reg(AffineConf)
class AffineLayer(Zlayer):
    def __init__(self, conf: AffineConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: AffineConf = self.conf
        # --
        self.act_f = ActivationHelper.get_act(conf.out_act)
        self.isize = conf.isize if isinstance(conf.isize, int) else sum(conf.isize)
        if conf.osize == 0:  # note: special code!!
            self.linear = (lambda x: x)
        else:
            self.linear = BK.nn.Linear(self.isize, conf.osize, bias=conf.use_bias)
        self.dropout = BK.nn.Dropout(conf.dropout)  # dropout node

    def get_output_dim(self):
        conf: AffineConf = self.conf
        return conf.osize if conf.osize>0 else self.isize

    def extra_repr(self) -> str:
        conf: AffineConf = self.conf
        return f"Affine({conf.isize}->{conf.osize}+{super().extra_repr()})"

    def forward(self, inputs):
        # conf: AffineConf = self.conf
        if isinstance(inputs, BK.Expr):
            inputs = [inputs]
        inp = BK.concat(inputs)
        hid0 = self.linear(inp)
        hid1 = self.act_f(hid0)
        hid2 = self.dropout(hid1)
        return hid2

# --
# MLP
class MlpConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        self.isize: Union[int, List[int]] = []
        self.osize: int = -1
        self.dim_hid = 256
        self.n_hid_layer = 0
        self.hid_conf = AffineConf().direct_update(out_act="elu", dropout=0.1)
        self.use_out = True  # whether use output layer
        self.out_conf = AffineConf()

    @classmethod
    def _get_type_hints(cls):
        return {"isize": int}

@node_reg(MlpConf)
class MlpLayer(Zlayer):
    def __init__(self, conf: MlpConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: MlpConf = self.conf
        # --
        self.nodes = []
        cur_dim = conf.isize
        self.mlp_sizes = [cur_dim]
        # hidden layers
        for idx in range(conf.n_hid_layer):  # hidden layers
            node = AffineLayer(conf.hid_conf, isize=cur_dim, osize=conf.dim_hid)
            self.nodes.append(node)
            cur_dim = conf.dim_hid
            self.add_module(f"H{idx}", node)
            self.mlp_sizes.append(cur_dim)
        # final layer
        if conf.use_out:
            fnode = AffineLayer(conf.out_conf, isize=cur_dim, osize=conf.osize)
            self.nodes.append(fnode)
            cur_dim = conf.osize
            self.add_module(f"F", fnode)
            self.mlp_sizes.append(cur_dim)
        self.output_dim = cur_dim

    def extra_repr(self) -> str:
        return f"MLP({'->'.join([str(z) for z in self.mlp_sizes])})"

    def forward(self, input_expr):
        cur_expr = input_expr
        for n in self.nodes:
            cur_expr = n(cur_expr)
        return cur_expr

    @staticmethod
    def get_mlp(isize: Union[int, List[int]], osize: int, dim_hid: int, n_hid_layer: int,
                hid_conf: AffineConf = None, out_conf: AffineConf = None):
        conf = MlpConf().direct_update(isize=isize, osize=osize, dim_hid=dim_hid, n_hid_layer=n_hid_layer)
        if hid_conf is not None:
            conf.hid_conf = hid_conf
        if out_conf is not None:
            conf.out_conf = out_conf
        return MlpLayer(conf)

# --
# horizontal combiners
class CombinerConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        # --
        self.isizes: List[int] = []
        self.osize: int = -1
        self.comb_method = "concat"  # affine/concat/sum/weighted/stack/pick
        self.aff_conf = AffineConf()
        self.stack_dim = -2  # if stack, at which dim?
        self.pick_idx = -1  # simply pick one

    @classmethod
    def _get_type_hints(cls):
        return {"isizes": int}

@node_reg(CombinerConf)
class CombinerLayer(Zlayer):
    def __init__(self, conf: CombinerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: CombinerConf = self.conf
        # --
        output_dim: int = None
        if conf.comb_method == "affine":
            # todo(note): if not provided, output the mean of input sizes
            _osize = conf.osize if conf.osize>0 else int(np.mean(conf.isizes))
            aff_node = AffineLayer(conf.aff_conf, isize=conf.isizes, osize=_osize)
            self.add_module("_aff", aff_node)
            self._f = lambda xs: aff_node(xs)
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
        else:
            raise NotImplementedError(f"UNK comb_method: {conf.comb_method}")
        # --
        self.output_dim = output_dim

    def extra_repr(self) -> str:
        conf: CombinerConf = self.conf
        return f"Combiner({conf.comb_method},{conf.isizes}->{self.output_dim})"

    def forward(self, exprs: List):
        return self._f(exprs)

# --
# a special scalar for convenience

class ScalarConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        self.init = 0.
        self.fixed = True

@node_reg(ScalarConf)
class ScalarLayer(Zlayer):
    def __init__(self, conf: ScalarConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ScalarConf = self.conf
        # --
        if conf.fixed:
            self.v = conf.init
        else:
            self.v = BK.nn.Parameter(BK.as_tensor(conf.init))
        # --

    def forward(self):
        return self.v

    def reset_value(self, v: float):
        zlog(f"Reset scalar value from {self.v} to {v}")
        if isinstance(self.v, BK.Expr):
            BK.set_value(self.v, v)
        else:
            assert self.conf.fixed
            self.v = v
        # --

# --
# similarity function

class SimConf(ZlayerConf):
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
        self.dropout = 0.  # input dropout?
        # --

@node_reg(SimConf)
class SimLayer(Zlayer):
    def __init__(self, conf: SimConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: SimConf = self.conf
        # --
        self.scale = conf.scale.make_node()
        self.split_dims = [int(z) for z in conf.split_dims]
        # --
        if conf.proj_dim > 0:
            self.proj = BK.nn.Linear(conf.isize, conf.proj_dim, bias=False)
        else:
            self.proj = (lambda x: x)
        if conf.dropout > 0:
            self.drop = BK.nn.Dropout(conf.dropout)  # dropout node
        else:
            self.drop = (lambda x: x)
        # --

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
            c1, c2 = (s1!=0.).float(), (s2!=0.).float()  # valid counts
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
            _no_valid = (count0==0.).float()
            final_score0 = _no_valid * (-10000.) + (1.-_no_valid) * final_score0
        # --
        ret = final_score0 * self.scale.forward()  # final scale
        # breakpoint()
        return ret

    # do average: [*, N, D], [*, N]
    def do_average(self, x_t, weight_t):
        _fulldim = BK.get_shape(x_t, -1)
        _dims = self.split_dims + [_fulldim - sum(self.split_dims)]
        _dims = [z for z in _dims if z > 0]  # keep valid ones!
        _ndim = len(_dims)
        # --
        rs = x_t.split(_dims, -1) if _ndim>1 else [x_t]
        all_res = []
        for r in rs:
            r_all0 = (r.norm(dim=-1) != 0.).float()  # [*, N]
            w_t = weight_t * r_all0
            w_t = w_t / (w_t.sum(-1, keepdims=True).clamp(min=1))  # [*, N]
            res = (r * w_t.unsqueeze(-1)).sum(-2)  # [*]
            all_res.append(res)
        # --
        # for debugging
        # print(r_all0)
        # --
        ret = BK.concat(all_res, -1)
        return ret

    # simple kmeans: adopted from https://github.com/subhadarship/kmeans_pytorch
    # todo(+N): do we need to average over non-NIL for each components?
    # [*, N, D], [*, N] -> [*, k, D]
    # note: return all0 if there are extra unfound components
    def run_kmeans(self, x_t, mask_t, k: int, tol=1e-4, iter_limit=50, kmpp_alpha=2., return_idxes=False):
        if mask_t is None:
            mask_t = BK.constants(shape=BK.get_shape(x_t)[:-1], value=1.)  # [*, N]
        # --
        if k == 1:
            # um_t = mask_t / (mask_t.sum(-1, keepdims=True).clamp(min=1))  # uniform prob, [*, N]
            # ret = (x_t * um_t.unsqueeze(-1)).sum(-2, keepdims=True)  # [*, 1, D]
            ret = self.do_average(x_t, mask_t).unsqueeze(-2)  # [*, 1, D]
            ret_idxes = BK.constants_idx(mask_t.shape, 0)  # [*, N]
        else:
            x_t = x_t * mask_t.unsqueeze(-1)  # 1) contiguous, 2) make invalid ones all0
            # --
            _NEG = -10000.
            # init with kmeans++
            # first randomly sample one!
            tmp_mask = mask_t.clone()  # [*, N]
            tmp_ones = tmp_mask * 0. + 1.  # [*, N]
            cur_masks = (tmp_mask.sum(-1, keepdims=True) > 0.).float()  # [*, ->]
            cur_idxes = BK.multinomial_choice(BK.where(cur_masks>0, tmp_mask, tmp_ones), 1, False)  # [*, ->]
            cur_centers = BK.gather_first_dims(x_t, cur_idxes, -2)  # [*, ->, D]
            tmp_mask.scatter_(-1, cur_idxes, 0.)  # set zero
            # then iteratively select!
            for cur_i in range(1, k):
                # find other ones: [*, N, ->] -> [*, N]
                _sim = (self.forward(x_t, cur_centers) + _NEG * (1.-cur_masks).unsqueeze(-2)).max(-1)[0]
                _prob = ((_sim.max(-1, keepdims=True)[0] - _sim + 1.) ** kmpp_alpha) * tmp_mask  # [*, N]
                _slice_masks = (tmp_mask.sum(-1, keepdims=True) > 0.).float()  # [*, 1]
                _slice_idxes = BK.multinomial_choice(BK.where(_slice_masks>0, _prob, tmp_ones), 1, False)  # [*, 1]
                _slice_centers = BK.gather_first_dims(x_t, _slice_idxes, -2)  # [*, 1, D]
                tmp_mask.scatter_(-1, _slice_idxes, 0.)  # set zero
                # update
                cur_masks = BK.concat([cur_masks, _slice_masks], -1)  # [*, ->]
                cur_centers = BK.concat([cur_centers, _slice_centers], -2)  # [*, ->, D]
            cur_centers = cur_centers * cur_masks.unsqueeze(-1)
            # --
            # kmeans
            iteration = 0
            while True:
                # argmax sim score
                scores = self.forward(x_t, cur_centers)  # [*, N, k]
                _, _max_idxes = (scores + _NEG * (1.-cur_masks).unsqueeze(-2)).max(-1)  # [*, N]
                _hit = (_max_idxes.unsqueeze(-1) == BK.arange_idx(k)).float() * mask_t.unsqueeze(-1)  # [*, N, k]
                # _hit_count = _hit.sum(-2)  # [*, k]
                # _hit_sum = BK.matmul(_hit.transpose(-1, -2), x_t)  # [*, k, D]
                # new_cur_centers = _hit_sum / _hit_count.unsqueeze(-1).clamp(min=1)  # [*, k, D]
                # breakpoint()
                new_cur_centers = self.do_average(x_t.unsqueeze(-3), _hit.transpose(-1, -2))  # [*, k, D]
                # --
                # check finish
                ret_idxes = _max_idxes
                _shift = ((cur_centers - new_cur_centers) ** 2).sum(-1).sqrt()  # [*, k]
                iteration = iteration + 1
                cur_centers = new_cur_centers
                if iteration > iter_limit or (_shift<=tol).all():
                    break
            ret = cur_centers  # [*, k, D]
        # --
        if return_idxes:
            return ret, ret_idxes
        else:
            return ret  # [*, orig_k, D]

# --
# subword2word pooler
class SubPoolerConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        self.pool_hid_f = 'first'  # first/last/mean2/max2
        self.pool_att_f = 'max4'  # first/last/mean4/max4
        # --

@node_reg(SubPoolerConf)
class SubPoolerLayer(Zlayer):
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
# pair repr
class PairReprConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        # --
        self.isize1 = -1  # input size of repr1
        self.isize2 = -1  # input size of repr2
        self.hsize = -1   # hid size, if 0 then no hid_aff!
        self.osize = -1  # out size, if 0 then no out_aff!
        self.pair_func = "inner"  # inner/outer/...
        self.pair_piece = 8  # hsize//piece
        self.inner_div = 0.  # if <0, then sqrt(hsize//piece)
        self.hid_share = False  # whether share hid_aff
        self.hid_aff = AffineConf()
        self.out_aff = AffineConf()
        # --

@node_reg(PairReprConf)
class PairReprLayer(Zlayer):
    def __init__(self, conf: PairReprConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: PairReprConf = self.conf
        # --
        # repr1/repr2 --(hid_aff)--> hid1/hid2 --(pf)--> repr --(out_aff)--> output
        self.hid_aff1 = AffineLayer(conf.hid_aff, isize=conf.isize1, osize=conf.hsize)
        if conf.hid_share:
            assert conf.isize1 == conf.isize2
            self.hid_aff2 = None
        else:
            self.hid_aff2 = AffineLayer(conf.hid_aff, isize=conf.isize2, osize=conf.hsize)
        _dim = self.hid_aff1.get_output_dim()
        # --
        if conf.pair_func == 'inner':
            _dim = conf.pair_piece
        elif conf.pair_func == 'outer':
            _dim = conf.pair_piece * (_dim // conf.pair_piece) ** 2
        else:
            raise NotImplementedError(f"UNK pair_func: {conf.pair_func}")
        # --
        self.out_aff = AffineLayer(conf.out_aff, isize=_dim, osize=conf.osize)
        self.output_dim = self.out_aff.get_output_dim()
        # --

    def get_output_dim(self):
        return self.output_dim

    def _pf(self, hid1, hid2, is_pair: bool):
        conf: PairReprConf = self.conf
        # --
        # reshape
        _piece = conf.pair_piece
        r1 = hid1.view(BK.get_shape(hid1)[:-1] + [_piece, -1])  # [..., P, D']
        r2 = hid2.view(BK.get_shape(hid2)[:-1] + [_piece, -1])  # [..., P, D']
        # --
        if conf.pair_func == 'inner':
            if is_pair:
                res0 = BK.matmul(r1.transpose(-2,-3), r2.transpose(-2,-3).transpose(-1,-2))  # [..., P, N1, N2]
                res = res0.transpose(-2,-3).transpose(-1,-2)  # [..., N1, N2, P]
            else:
                res = (r1 * r2).sum(-1)  # [..., P]
            _div = conf.inner_div if conf.inner_div>0 else BK.get_shape(r1, -1) ** 0.5
            res = res / _div
        elif conf.pair_func == 'outer':
            if is_pair:
                r1, r2 = r1.unsqueeze(-3), r2.unsqueeze(-4)
            res = BK.matmul(r1.unsqueeze(-1), r2.unsqueeze(-2))  # [..., P, D', D']
            res = res.view(BK.get_shape(res)[:-3] + [-1])  # [..., P*D'*D']
        else:
            raise NotImplementedError(f"UNK pair_func: {conf.pair_func}")
        # --
        return res

    # [..., N1, D], [..., N2, D] -> [..., N1, N2, Dout] or [..., D], [..., D] -> [..., Dout]
    def forward(self, repr1, repr2, is_pair=True):
        hid1 = self.hid_aff1(repr1)
        hid2 = self.hid_aff1(repr2) if self.hid_aff2 is None else self.hid_aff2(repr2)
        hid_pf = self._pf(hid1, hid2, is_pair)
        out = self.out_aff(hid_pf)
        return out

# --
class SimpleNormConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        # --
        self.dim = -1
        self.freeze = True  # whether trainable?
        # --

@node_reg(SimpleNormConf)
class SimpleNormLayer(Zlayer):
    def __init__(self, conf: SimpleNormConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: SimpleNormConf = self.conf
        # --
        self.bias = BK.nn.Parameter(BK.input_real([0.] * conf.dim))
        self.weight = BK.nn.Parameter(BK.input_real([1.] * conf.dim))
        if conf.freeze:
            self.bias.requires_grad = False
            self.weight.requires_grad = False
        # --

    def set(self, b, w):
        with BK.no_grad_env():
            BK.set_value(self.bias, b)
            BK.set_value(self.weight, w)
        # --

    def forward(self, x):
        ret = (x-self.bias) * self.weight
        return ret

# --
# pairwise scoring (adopted from nn.layers.scorer.PairScorerNode)
class PairScoreConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        # --
        self.osize = 0  # output size
        self.isize = -1  # common isize!
        self.isize0: int = -1  # input size 0
        self.isize1: int = -1  # input size 1
        # linear layer for the inputs
        self.aff_dim = 0  # no intermediate if <=0!
        self.aff0 = AffineConf()
        self.aff1 = AffineConf()
        # how to score?
        self.use_biaff = False
        self.use_dot = False
        self.mlp_dim = 0  # use mlp if >0
        # --

@node_reg(PairScoreConf)
class PairScoreLayer(Zlayer):
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
            self.BW = BK.nn.Linear(cur_dims[0], cur_dims[1]*conf.osize)
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
            score = score + s1 / self.dot_div
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
# b msp2/nn/l3/ff:247
