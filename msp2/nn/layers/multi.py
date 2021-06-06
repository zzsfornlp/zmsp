#

# multi-ones: containers and wrappers

__all__ = [
    "SequentialConf", "SequentialNode", "WrapperConf", "WrapperNode",
    "MLPConf", "MLPNode", "get_mlp", "CombinerConf", "CombinerNode",
]

from typing import List, Union, Iterable
import numpy as np
import math
from ..backends import BK
from .base import *
from .ff import *
from msp2.utils import zlog, zwarn

# =====
# Sequence

class SequentialConf(BasicConf):
    def __init__(self):
        super().__init__()

@node_reg(SequentialConf)
class SequentialNode(BasicNode):
    def __init__(self, node_iter: Iterable[BasicNode], conf: SequentialConf, **kwargs):
        super().__init__(conf, **kwargs)
        # --
        _len = 0
        for ni, node in enumerate(node_iter):
            self.add_module(f"N{ni}", node)
            _len += 1
        self._len = _len
        self.ns = self.get_ns()

    def get_ns(self):
        return [getattr(self, f"N{i}") for i in range(self._len)]

    # --
    def get_output_dims(self, *input_dims):
        cur_dims = input_dims
        for n in self.ns:
            cur_dims = n.get_output_dims(*cur_dims)
        return cur_dims

    def extra_repr(self) -> str:
        return f"SequentialNode(L={self._len})"

    def forward(self, input_exp, *args, **kwargs):
        x = input_exp
        for n in self.ns:
            x = n(x, *args, **kwargs)
        return x

# =====
# wrappers

class WrapperConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize = -1
        self.strategy = ""  # addnorm, addact, highway, ...
        self.act = "linear"

@node_reg(WrapperConf)
class WrapperNode(BasicNode):
    def __init__(self, node: BasicNode, conf: WrapperConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: WrapperConf = self.conf
        # --
        self.N = node
        # --
        self.osize = node.get_output_dims(conf.isize)[0]
        assert self.osize == conf.isize, "Currently Wrapper must be same i/o size!"
        # todo(note): most of them simply use simple forms
        if conf.strategy == "addnorm":
            self.normer = LayerNormNode(None, osize=self.osize)
            self._f = lambda in_expr, out_expr: self.normer(in_expr+out_expr)
        elif conf.strategy == "addact":
            self.act = ActivationHelper.get_act(conf.act)
            self._f = lambda in_expr, out_expr: self.act(in_expr+out_expr)
        elif conf.strategy == "highway":
            self.gate = AffineNode(None, isize=self.osize, osize=self.osize)
            self._f = self._f_highway
        else:
            raise NotImplementedError(f"UNK strategy for WrapperNode: {conf.strategy}")

    def _f_highway(self, in_expr, out_expr):
        g = BK.sigmoid(out_expr)
        r = in_expr*g + out_expr*(1.-g)
        return r

    # --
    def get_output_dims(self, *input_dims):
        return (self.osize, )

    def extra_repr(self) -> str:
        return f"WrapperNode({self.conf.strategy})"

    def forward(self, x, *input, **kwargs):
        # todo(note): assume the first input is the input one!
        hid = self.N(x, *input, **kwargs)
        out = self._f(x, hid)
        return out

# =====
# MLP

class MLPConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize: Union[int, List[int]] = []
        self.osize: int = -1
        self.dim_hid = 256
        self.n_hid_layer = 0
        self.hid_conf = AffineConf().direct_update(out_act="elu")
        self.use_out = True  # whether use output layer
        self.out_conf = AffineConf()

    @classmethod
    def _get_type_hints(cls):
        return {"isize": int}

@node_reg(MLPConf)
class MLPNode(BasicNode):
    def __init__(self, conf: MLPConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: MLPConf = self.conf
        # --
        self.nodes = []
        cur_dim = conf.isize
        # hidden layers
        for idx in range(conf.n_hid_layer):  # hidden layers
            node = AffineNode(conf.hid_conf, isize=cur_dim, osize=conf.dim_hid)
            self.nodes.append(node)
            cur_dim = conf.dim_hid
            self.add_module(f"H{idx}", node)
        # final layer
        if conf.use_out:
            fnode = AffineNode(conf.out_conf, isize=cur_dim, osize=conf.osize)
            self.nodes.append(fnode)
            cur_dim = conf.osize
            self.add_module(f"F", fnode)
        self.output_dim = cur_dim

    def extra_repr(self) -> str:
        return f"MLP(L={len(self.nodes)})"

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    def forward(self, input_expr):
        cur_expr = input_expr
        for n in self.nodes:
            cur_expr = n(cur_expr)
        return cur_expr

# shortcut
def get_mlp(isize: Union[int, List[int]], osize: int, dim_hid: int, n_hid_layer: int,
            hid_conf: AffineConf = None, out_conf: AffineConf = None):
    conf = MLPConf().direct_update(isize=isize, osize=osize, dim_hid=dim_hid, n_hid_layer=n_hid_layer)
    if hid_conf is not None:
        conf.hid_conf = hid_conf
    if out_conf is not None:
        conf.out_conf = out_conf
    return MLPNode(conf)

# =====
# horizontal combiners

class CombinerConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self.isizes: List[int] = []
        self.osize: int = None
        self.comb_method = "concat"  # affine/concat/sum/weighted/stack
        self.aff_conf = AffineConf()
        self.stack_dim = -2  # if stack, at which dim?

    @classmethod
    def _get_type_hints(cls):
        return {"isizes": int, "osize": int}

@node_reg(CombinerConf)
class CombinerNode(BasicNode):
    def __init__(self, conf: CombinerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: CombinerConf = self.conf
        # --
        output_dim: int = None
        if conf.comb_method == "affine":
            # todo(note): if not provided, output the mean of input sizes
            aff_node = AffineNode(conf.aff_conf, isize=conf.isizes, osize=conf.osize if conf.osize else int(np.mean(conf.isizes)))
            self.add_module("_aff", aff_node)
            self._f = lambda xs: aff_node(xs)
            output_dim = aff_node.get_output_dims()[0]
        elif conf.comb_method == "concat":
            self._f = lambda xs: BK.concat(xs, -1)  # [*, sum(d)]
            output_dim = sum(conf.isizes)
        elif conf.comb_method == "sum":
            assert all(z==conf.isizes[0] for z in conf.isizes), "Must be the same size to use SUM!!"
            self._f = lambda xs: BK.stack(xs, -1).sum(-1)  # [*, d, N] -> [*, d]
            output_dim = conf.isizes[0]
        elif conf.comb_method == "weighted":  # weighted sum
            assert all(z==conf.isizes[0] for z in conf.isizes), "Must be the same size to use SUM!!"
            self.GAMMA = BK.new_param(())  # scalar, []
            self.LAMBDAS = BK.new_param((len(conf.isizes),))  # [N]
            self.reset_parameters()
            self._f = lambda xs: (BK.stack(xs, -1) * BK.softmax(self.LAMBDAS, -1)).sum(-1) * self.GAMMA  # [*, d, N] -> [*, d]
            output_dim = conf.isizes[0]
        elif conf.comb_method == "stack":
            assert all(z==conf.isizes[0] for z in conf.isizes), "Must be the same size to use STACK!!"
            self._f = lambda xs: BK.stack(xs, conf.stack_dim)
            output_dim = len(conf.isizes) if conf.stack_dim==-1 else conf.isizes[0]
        else:
            raise NotImplementedError(f"UNK comb_method: {conf.comb_method}")
        self.output_dim = output_dim

    def reset_parameters(self):
        conf: CombinerConf = self.conf
        if conf.comb_method == "weighted":
            BK.init_param(self.GAMMA, np.array(1.))
            fold = len(conf.isizes)
            BK.init_param(self.LAMBDAS, np.array([1./fold] * fold))

    def extra_repr(self) -> str:
        conf: CombinerConf = self.conf
        return f"Combiner({conf.comb_method},{conf.isizes}->{self.output_dim})"

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    def forward(self, exprs: List):
        return self._f(exprs)
