#

# simple feed forward layers

__all__ = [
    "FFConf", "FFNode", "AffineConf", "AffineNode", "LayerNormConf", "LayerNormNode",
    "EmbeddingConf", "EmbeddingNode", "PosiEmbeddingConf", "PosiEmbeddingNode"
]

from typing import List, Union, Iterable, Callable
import numpy as np
import math
from ..backends import BK
from .base import *
from msp2.utils import zlog, zwarn

# =====
# Basic FF Layer
# todo(note): should be a Wrapper class, but since it is usually static, make it super-class for convenience

class FFConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.osize = -1
        # --
        self.init_scale = 1.  # scale for init
        self.out_act = "linear"  # activation
        self.no_drop = False  # no dropout for output
        self.out_drop = DropoutConf()

@node_reg(FFConf)
class FFNode(BasicNode):
    def __init__(self, conf: FFConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: FFConf = self.conf
        # --
        # activations
        self._act_f = ActivationHelper.get_act(conf.out_act)
        # dropout
        if conf.no_drop:
            self.drop_node = lambda x: x
        else:
            self.drop_node = DropoutNode(conf.out_drop, osize=conf.osize)

    def get_output_dims(self, *input_dims):
        return (self.conf.osize, )

    def extra_repr(self) -> str:
        conf: FFConf = self.conf
        return f"FFNode(A={conf.out_act})"

    # todo(note): everyone must do this!
    def forward(self, x, **kwargs):
        h1 = self._act_f(x)
        h2 = self.drop_node(h1)
        return h2

# ===== Linear/Affine Nodes
# linear layer with selectable activation functions
# [inputs] or input -> output

class AffineConf(FFConf):
    def __init__(self):
        super().__init__()
        self.isize: Union[int, List[int]] = []
        # --
        self.use_bias = True  # use bias?
        self.which_affine = 2  # which affine to use

    @classmethod
    def _get_type_hints(cls):
        return {"isize": int}

@node_reg(AffineConf)
class AffineNode(FFNode):
    def __init__(self, conf: AffineConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: AffineConf = self.conf
        # --
        # dimensions
        _n_ins = [conf.isize] if isinstance(conf.isize, int) else list(conf.isize)
        _n_out = conf.osize
        conf.isize = _n_ins  # replace it
        # params
        self.use_bias = conf.use_bias
        if conf.use_bias:
            self.B = BK.new_param([_n_out])
        for wi, din in enumerate(_n_ins):
            # self.__setattr__(f"W{wi}", BK.new_param([_n_out, din]))
            self.register_parameter(f"W{wi}", BK.new_param([_n_out, din]))
        self.w_is_exteranl = [False] * len(_n_ins)
        self.direct_matmul = (len(_n_ins)==1 and not conf.use_bias)
        self._input_list = self._init_input_list()
        # --
        self._affine_f = {1: BK.affine, 2: BK.affine2, 3: BK.affine3}[conf.which_affine]
        self.reset_parameters()

    def reset_parameters(self):
        conf: AffineConf = self.conf
        # --
        if conf.use_bias:
            BK.init_param(self.B, "zero")
        for i, W in enumerate(self.get_ws()):
            if not self.w_is_exteranl[i]:  # note: need to be careful here, since some w can be external!
                BK.init_param(W, "glorot", scale=conf.init_scale)
        # --

    def get_ws(self):
        rets = [getattr(self, f"W{i}") for i in range(len(self.conf.isize))]
        rets = [(z if isinstance(z, BK.Expr) else z()) for z in rets]  # allow lazy-prepare!
        return rets

    def put_external_ws(self, external_ws: List):
        assert len(external_ws) == len(self.conf.isize)
        for i, w in enumerate(external_ws):
            if w is not None:
                name = f"W{i}"
                if hasattr(self, name):
                    delattr(self, name)  # delete original one!!
                self.setattr_borrow(name, w)  # set external one without adding parameters!!
                self.w_is_exteranl[i] = True
        # --

    def _init_input_list(self):
        # todo(note): should be called everytime param changes
        conf: AffineConf = self.conf
        if conf.use_bias:
            input_lists = [self.B]
        else:
            input_lists = [BK.zeros((conf.osize,))]
        for W in self.get_ws():
            input_lists.extend([W, None])
        return input_lists

    # fill in the list
    def _fill_input_list(self, inputs: List):
        for i, one in enumerate(inputs):
            self._input_list[2+2*i] = one
        return self._input_list

    # =====
    def extra_repr(self) -> str:
        conf: AffineConf = self.conf
        return f"Affine({conf.isize}->{conf.osize}+{super().extra_repr()})"

    def forward(self, input_exp, **kwargs):
        conf: AffineConf = self.conf
        # --
        _direct_matmul = self.direct_matmul
        # --
        if not isinstance(input_exp, (list, tuple)):
            input_exp = [input_exp]
        if _direct_matmul:
            assert len(input_exp) == 1
            h0 = BK.matmul(input_exp[0], BK.transpose(self._input_list[-2], 0, 1))  # at dim=-2
        else:
            assert len(input_exp) == len(conf.isize), "Unmatched input sizes!"
            input_lists = self._fill_input_list(input_exp)
            h0 = self._affine_f(input_lists)
        # output
        h1 = super().forward(h0)
        return h1

# =====
# LayerNorm

class LayerNormConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.osize = -1
        # --
        self.a_init = 1.
        self.eps = 1e-6
        self.std_no_grad = False

@node_reg(LayerNormConf)
class LayerNormNode(BasicNode):
    def __init__(self, conf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: LayerNormConf = self.conf
        # --
        # todo(+N): is this all right?
        size = conf.osize
        self.a_2 = BK.new_param(size)
        self.b_2 = BK.new_param(size)
        # no droput here
        self.reset_parameters()

    def reset_parameters(self):
        conf: LayerNormConf = self.conf
        a_init, size = conf.a_init, conf.osize
        a_init_v = np.sqrt(3./size) if a_init is None else a_init
        BK.init_param(self.a_2, init=np.full(size, a_init_v, dtype=np.float32))
        BK.init_param(self.b_2, init="zero")

    # =====

    def extra_repr(self) -> str:
        return f"LayerNorm(s={self.conf.osize})"

    def forward(self, x, **kwargs):
        conf: LayerNormConf = self.conf
        a_2, b_2 = self.a_2, self.b_2
        # --
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        if conf.std_no_grad:
            std = std.detach()
        # todo(+N): if std is small, (for example, zero input), then div 1 since otherwise there will be NAN
        #  currently depend on outside flag (for the very first (input) layer)
        to_div = std + conf.eps
        return a_2 * (x - mean) / to_div + b_2

# =====
# Embeddings

class EmbeddingConf(FFConf):
    def __init__(self):
        super().__init__()
        self.n_words = -1  # number of words, must provide!
        # --
        self.fix_row0 = False  # whether force row0 to be zero?
        self.freeze = False  # freeze param?

@node_reg(EmbeddingConf)
class EmbeddingNode(FFNode):
    def __init__(self, conf: EmbeddingConf, npvec=None, **kwargs):
        super().__init__(conf, **kwargs)
        conf: EmbeddingConf = self.conf
        # --
        n_words, n_dim, freeze = conf.n_words, conf.osize, conf.freeze
        # --
        self.has_npvec_init = False
        if npvec is None:  # no init
            self.E = BK.new_param([n_words, n_dim])
            self.reset_parameters()
        else:
            self.has_npvec_init = True
            assert conf.n_words == len(npvec)
            self.reset_with_npvec(npvec)
        if freeze:
            self.rop.add_fixed_value("trainable", False)
            if npvec is None:
                zwarn("Meaningless to freeze random embeddings?")
        # todo(+n): should we use special embedding drop (drop certain word types at each time)?

    def reset_with_npvec(self, npvec: np.ndarray):
        conf: EmbeddingConf = self.conf
        # --
        vec_shape = npvec.shape
        conf.n_words, conf.n_dim = vec_shape
        self.E = BK.new_param(vec_shape)
        zlog(f"Reset Embeddings with npvec.shape={vec_shape}")
        BK.init_param(self.E, npvec, scale=conf.init_scale)  # todo(note): still apply scale!

    def reset_parameters(self):  # random reset!
        conf: EmbeddingConf = self.conf
        BK.init_param(self.E, "glorot", lookup=True, scale=conf.init_scale)
        if self.has_npvec_init:
            zwarn("Reset Embedding to random, maybe need to reassign with pre-trained ones?!")

    # --
    def refresh(self, rop=None):
        super().refresh(rop)
        if self.conf.fix_row0:
            # todo(note): zero for idx 0
            BK.zero_row(self.E, 0)

    # todo(note): override for pretty printing
    def extra_repr(self) -> str:
        return f"Embedding({self.conf.n_words},{self.conf.osize}+{super().extra_repr()})"

    def forward(self, input_idxes, **kwargs):
        # input should be a list of ints or int
        if isinstance(input_idxes, int):
            input_idxes = [input_idxes]
        h0 = BK.lookup(self.E, input_idxes)
        # output
        h1 = super().forward(h0)
        return h1

# ======
# Special PosiEmbedding
# todo(+n): very similar to Embedding, could they share parts?

class PosiEmbeddingConf(FFConf):
    def __init__(self):
        super().__init__()
        # --
        self.max_val: int = 5000
        self.min_val: int = None
        self.init_sincos = True  # init with sincos ones
        self.init_sincos_div = 10000.
        self.freeze = True  # freeze value
        self.zero0 = False  # zero for v=0 (not necessary row0)

    def _do_validate(self):
        # set range
        assert self.max_val >= 0
        if self.min_val is None:  # by default [-K,K]
            self.min_val = -self.max_val

    @classmethod
    def _get_type_hints(cls):
        return {"min_val": int}

@node_reg(PosiEmbeddingConf)
class PosiEmbeddingNode(FFNode):
    def __init__(self, conf: PosiEmbeddingConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: PosiEmbeddingConf = self.conf
        # --
        self.E = BK.new_param([conf.max_val-conf.min_val+1, conf.osize])
        self.reset_parameters()
        if conf.freeze:
            self.rop.add_fixed_value("trainable", False)
            if not conf.init_sincos:
                zwarn("Meaningless to freeze random posi-embeddings?")

    @staticmethod
    def init_sincos_arr(min_val: int, max_val: int, n_dim: int, scale: float, div: float):
        all_size = max_val - min_val + 1  # range
        pe = np.zeros([all_size, n_dim])  # [range, n_dim]
        position = np.arange(min_val, max_val+1).reshape([-1, 1])  # [all_size, 1]
        div_term = np.exp((np.arange(0, n_dim, 2) * -(math.log(div)/n_dim)))  # [d//2]
        div_results = position * div_term
        pe[:, 0::2] = np.sin(div_results)
        pe[:, 1::2] = np.cos(div_results)
        # # scale as embedding, todo(note): not fixed
        # scale = np.sqrt(3.0 / n_dim)
        return pe * scale

    def reset_parameters(self):
        conf: PosiEmbeddingConf = self.conf
        if conf.init_sincos:
            pe = PosiEmbeddingNode.init_sincos_arr(conf.min_val, conf.max_val, conf.osize,
                                                   conf.init_scale, conf.init_sincos_div)  # [R, d]
            BK.init_param(self.E, pe)
        else:
            pe = None
            BK.init_param(self.E, "glorot", lookup=True, scale=conf.init_scale)

    # --
    def refresh(self, rop=None):
        super().refresh(rop)
        conf: PosiEmbeddingConf = self.conf
        # --
        if conf.zero0:  # zero the 0 one
            if conf.min_val <= 0 and conf.max_val >= 0:
                BK.zero_row(self.E, -conf.min_val)

    # todo(note): override for pretty printing
    def extra_repr(self) -> str:
        conf: PosiEmbeddingConf = self.conf
        return f"PosiEmbedding([{conf.min_val},{conf.max_val}]+{super().extra_repr()})"

    def forward(self, input_idxes, **kwargs):
        conf: PosiEmbeddingConf = self.conf
        offset = - conf.min_val
        # input should be a list of ints or int
        if isinstance(input_idxes, int):
            input_idxes = [input_idxes]
        clamped_idx_repr = BK.clamp(BK.input_idx(input_idxes), min=conf.min_val, max=conf.max_val)
        h0 = BK.lookup(self.E, clamped_idx_repr+offset)
        # output
        h1 = super().forward(h0)
        return h1
