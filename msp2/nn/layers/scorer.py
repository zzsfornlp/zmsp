#

# affine and biaffine scorer

__all__ = [
    "RScorerConf", "RScorerNode",
    "PlainScorerConf", "PlainScorerNode", "PairScorerConf", "PairScorerNode",
    "MyScorerConf", "MyScorerNode",
]

from typing import Union, List
from ..backends import BK
from .base import *
from .ff import AffineConf
from .multi import *
from .helper import *
from msp2.utils import Constants, zlog

# =====
# repeat and reduce scorer component

class RScorerConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.out_piece = 1  # number of pieces to produce
        self.out_reduce_f = "max"  # max/avg/... pooling

@node_reg(RScorerConf)
class RScorerNode(BasicNode):
    def __init__(self, conf: RScorerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: RScorerConf = self.conf
        self.out_reduce_f = ActivationHelper.get_pool(conf.out_reduce_f)

    @property
    def piece(self):
        return self.conf.out_piece

    def get_output_dims(self, *input_dims):
        return input_dims[0] // self.piece

    def forward(self, x: BK.Expr):
        conf: RScorerConf = self.conf
        return apply_piece_pooling(x, conf.out_piece, self.out_reduce_f, -1)

# =====
# Plain Scorer
class PlainScorerConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize: Union[int, List[int]] = []
        self.osize: int = -1
        # --
        self.hid_dim = 256  # hidden dim
        self.hid_nlayer = 0  # number of hidden layers
        self.hid_aff = AffineConf().direct_update(out_act='elu')
        self.out_aff = AffineConf().direct_update(no_drop=True)  # usually no drop as scorer's output
        self.rs = RScorerConf()
        # --
        self.mask_value = Constants.REAL_PRAC_MIN

    @classmethod
    def _get_type_hints(cls):
        return {"isize": int}

@node_reg(PlainScorerConf)
class PlainScorerNode(BasicNode):
    def __init__(self, conf: PlainScorerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: PlainScorerConf = self.conf
        # --
        self.rs = RScorerNode(conf.rs)
        self.mlp = get_mlp(conf.isize, conf.osize*self.rs.piece,
                           conf.hid_dim, conf.hid_nlayer, conf.hid_aff, conf.out_aff)

    def get_output_dims(self, *input_dims):
        return (self.conf.osize, )  # still osize!

    # [*, D], [*]
    def forward(self, input_expr, input_mask=None):
        mlp_out = self.mlp(input_expr)
        scores = self.rs(mlp_out)
        if input_mask is not None:
            scores += self.conf.mask_value * ((1.-input_mask).unsqueeze(-1))
        return scores

# =====
# Pairwise Scorer (can be biaffine)

class PairScorerConf(BasicConf):
    def __init__(self):
        super().__init__()
        # sizes
        self.osize = -1
        self.isize0: int = -1
        self.isize1: int = -1
        self.isizeP: int = -1
        self.rs = RScorerConf()
        # general
        self.use_bias = True
        self.mask_value = Constants.REAL_PRAC_MIN
        # what input to use
        self.use_input0 = True
        self.use_input1 = True
        self.use_input_pair = False  # direct pairwise input?
        # biaffine?
        self.use_biaffine = True
        self.biaffine_div = 1.  # <=0 means auto
        self.biaffine_init_ortho = True
        # ff1: separate for each
        self.use_ff1 = True
        self.ff1_hid_layer = 0
        self.ff1_hid_size = 256
        self.ff1_hid_act = "elu"
        # ff2: concat the inputs
        self.use_ff2 = False
        self.ff2_hid_layer = 1  # usually at least 1 hid to combine features
        self.ff2_hid_size = 256
        self.ff2_hid_act = "elu"

@node_reg(PairScorerConf)
class PairScorerNode(BasicNode):
    def __init__(self, conf: PairScorerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: PairScorerConf = self.conf
        # --
        # rs
        self.rs = RScorerNode(conf.rs)
        self.apply_osize = conf.osize * self.rs.piece
        # --
        self.use_input_flags = [conf.use_input0, conf.use_input1, conf.use_input_pair]  # use what in ff1 and ff2
        self.input_sizes = [conf.isize0, conf.isize1, conf.isizeP]
        self.input_sizes_valid = [s for s,f in zip(self.input_sizes, self.use_input_flags) if f]
        # --
        # add components
        assert conf.use_ff1 or conf.use_ff2 or conf.use_biaffine, "No real calculations here!"
        # bias
        if conf.use_bias:
            self.B = BK.new_param([self.apply_osize])
        # ff1
        if conf.use_ff1:
            self.FF1s = []
            for one_size, one_flag in zip(self.input_sizes, self.use_input_flags):
                if one_flag:
                    one_node = get_mlp(one_size, self.apply_osize, conf.ff1_hid_size, conf.ff1_hid_layer,
                                       AffineConf().direct_update(out_act=conf.ff1_hid_act),
                                       AffineConf().direct_update(no_drop=True, use_bias=False))
                    self.add_module(f"FF1_{len(self.FF1s)}", one_node)
                else:
                    one_node = lambda x: 0.
                self.FF1s.append(one_node)
        # ff2
        if conf.use_ff2:
            self.FF2 = get_mlp(self.input_sizes_valid, self.apply_osize, conf.ff2_hid_size, conf.ff2_hid_layer,
                               AffineConf().direct_update(out_act=conf.ff2_hid_act, which_affine=3),
                               AffineConf().direct_update(no_drop=True, use_bias=False))
        # biaffine
        if conf.use_biaffine:
            # this is different than BK.bilinear or layers.BiAffine
            self.BW = BK.new_param([conf.isize0, conf.isize1*self.apply_osize])
            if conf.biaffine_div <= 0.:
                conf.biaffine_div = (conf.isize0 * conf.isize1) ** 0.25  # sqrt(sqrt(in1*in2))
                zlog(f"Adopt biaffine_div of {conf.biaffine_div} for the current PairScorer!")
        # todo(note): no dropout at output, add it if needed!
        # --
        self.reset_parameters()

    def reset_parameters(self):
        conf: PairScorerConf = self.conf
        # --
        if conf.use_bias:
            BK.init_param(self.B, "zero")
        if conf.use_biaffine:
            BK.init_param(self.BW, "ortho" if conf.biaffine_init_ortho else "default")

    # plain call
    # io: [*, isize0], [*, isize1], [*, isizeP] -> [*, out_size]; masks: [*]
    def plain_score(self, input0, input1, inputp=None, mask0=None, mask1=None, maskp=None):
        conf: PairScorerConf = self.conf
        # --
        ret = 0.
        cur_input_list = [input0, input1, inputp]
        if conf.use_ff1:
            for one_node, one_input in zip(self.FF1s, cur_input_list):
                ret = ret + one_node(one_input)
        if conf.use_ff2:
            FF2_inputs = [one_input for one_flag, one_input in zip(self.use_input_flags, cur_input_list) if one_flag]
            ret = ret + self.FF2(FF2_inputs)
        if conf.use_biaffine:
            # [*, in0] * [in0, in1*out] -> [*, in1*out] -> [*, in1, out]
            expr0 = BK.matmul(input0, self.BW).view(BK.get_shape(input0)[:-1]+[conf.isize1, self.apply_osize])
            # [*, 1, in1] * [*, in1, out] -> [*, 1, out] -> [*, out]
            expr1 = BK.matmul(input1.unsqueeze(-2), expr0).squeeze(-2)
            ret = ret + expr1 / conf.biaffine_div
        if conf.use_bias:
            ret += self.B
        ret = self.rs(ret)  # apply rs
        # mask
        has_mask = False
        cur_mask = 1.
        if mask0 is not None:
            has_mask = True
            cur_mask *= mask0
        if mask1 is not None:
            has_mask = True
            cur_mask *= mask1
        if maskp is not None:
            has_mask = True
            cur_mask *= maskp
        if has_mask:
            ret += conf.mask_value*(1.-cur_mask).unsqueeze(-1)
        return ret

    # special call: [len0, len1]
    # io: [*, len0, isize0], [*, len1, isize1] -> [*, len0, len1, out_size], masks: [*, len0], [*, len1], [*, len0, len1]
    def paired_score(self, input0, input1, inputp=None, mask0=None, mask1=None, maskp=None):
        conf: PairScorerConf = self.conf
        # --
        # [*, len0, 1, isize0]
        expand0 = input0.unsqueeze(-2)
        # [*, ?, len1, isize1]
        expand1 = input1.unsqueeze(-3)
        ret = 0.
        cur_input_list = [expand0, expand1, inputp]
        if conf.use_ff1:
            for one_node, one_input in zip(self.FF1s, cur_input_list):
                ret = ret + one_node(one_input)
        if conf.use_ff2:
            FF2_inputs = [one_input for one_flag, one_input in zip(self.use_input_flags, cur_input_list) if one_flag]
            ret = ret + self.FF2(FF2_inputs)
        if conf.use_biaffine:
            # [*, len0, in0] * [in0, in1*out] -> [*, len0, in1*out] -> [*, len0, in1, out]
            expr0 = BK.matmul(input0, self.BW).view(BK.get_shape(input0)[:-1]+[conf.isize1, self.apply_osize])
            # [*, 1, len1, in1] * [*, len0, in1, out]
            expr1 = BK.matmul(expand1, expr0)
            # [*, len0, len1, out]
            ret = ret + expr1 / conf.biaffine_div
        if conf.use_bias:
            ret += self.B
        ret = self.rs(ret)  # apply rs
        # mask
        has_mask = False
        cur_mask = 1.
        if mask0 is not None:
            has_mask = True
            cur_mask *= mask0.unsqueeze(-1)  # [*, len0, 1]
        if mask1 is not None:
            has_mask = True
            cur_mask *= mask1.unsqueeze(-2)  # [*, 1, len1]
        if maskp is not None:
            has_mask = True
            cur_mask *= maskp  # [*, len0, len1]
        if has_mask:
            ret += conf.mask_value*(1.-cur_mask).unsqueeze(-1)
        return ret

    # =====
    def get_output_dims(self, *input_dims):
        return (self.conf.osize, )

    def extra_repr(self) -> str:
        conf: PairScorerConf = self.conf
        return f"PairScorer([{conf.isize0},{conf.isize1},{conf.isizeP}]->{conf.osize})"

    def forward(self, *input, **kwargs):
        raise RuntimeError("Use specific routines for PairwiseScorer!!")

# =====
# the scorer (either plain or pairwise)

class MyScorerConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize: Union[int, List[int]] = []  # main input size
        self.psize: int = -1  # pairwise input size
        self.osize: int = -1
        # --
        # pre-final layer MLP (shared conf for all)
        self.pre_mlp = MLPConf().direct_update(use_out=False)  # only using the hidden layers
        # --
        # final scorer
        self.pas_conf = PairScorerConf()  # pairwise mode
        self.pls_conf = PlainScorerConf()  # plain mode

    @property
    def is_pairwise(self):
        return self.psize > 0

    @classmethod
    def _get_type_hints(cls):
        return {"isize": int}

@node_reg(MyScorerConf)
class MyScorerNode(BasicNode):
    def __init__(self, conf: MyScorerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: MyScorerConf = self.conf
        # --
        osize = conf.osize
        self.is_pairwise = conf.is_pairwise
        self.mlp_main = MLPNode(conf.pre_mlp, isize=conf.isize)
        isize_main = self.mlp_main.get_output_dims((conf.isize,))[0]
        if self.is_pairwise:
            self.mlp_pair = MLPNode(conf.pre_mlp, isize=conf.psize)
            isize_pair = self.mlp_pair.get_output_dims((conf.psize,))[0]
            # note: put main as input1, pair as input0
            self.scorer = PairScorerNode(conf.pas_conf, osize=osize, isize0=isize_pair, isize1=isize_main)
        else:  # no pairwise ones
            self.scorer = PlainScorerNode(conf.pls_conf, osize=osize, isize=isize_main)

    def get_output_dims(self, *input_dims):
        return (self.conf.osize, )

    # [*, Dm], [*, Dp], [*]
    def forward(self, input_main: BK.Expr, input_pair: BK.Expr=None, mask: BK.Expr=None):
        if self.is_pairwise:
            hid_pair = self.mlp_pair(input_pair)
            hid_main = self.mlp_main(input_main)
            scores = self.scorer.plain_score(hid_pair, hid_main, mask1=mask)  # [*, L]
        else:
            hid_main = self.mlp_main(input_main)
            scores = self.scorer(hid_main, mask)
        return scores
