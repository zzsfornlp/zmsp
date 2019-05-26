#

# biaffines
import numpy as np
from collections import Iterable
from ..backends import BK
from . import Affine, get_mlp, BasicNode, ActivationHelper, Dropout, NoDropRop, NoFixRop
from msp.utils import zcheck, zwarn, zlog, Random, Constants

#
# biaffine (inputs are already paired)
# (N,*,in1_features), (N,*,in2_features) -> (N,*,out_features)
class BiAffine(BasicNode):
    def __init__(self, pc, n_ins0, n_ins1, n_out, n_hidden, act="linear", bias=True, name=None, hidden_init_rop=None, init_rop=None):
        super().__init__(pc, name, init_rop)
        #
        if not isinstance(n_ins0, Iterable):
            n_ins0 = [n_ins0]
        if not isinstance(n_ins1, Iterable):
            n_ins1 = [n_ins1]
        #
        self.n_ins0 = n_ins0
        self.n_ins1 = n_ins1
        self.n_hidden = n_hidden
        self.n_out = n_out
        # activations
        self.act = act
        self._act_f = ActivationHelper.get_act(act)
        # params
        self.affine0 = self.add_sub_node("a0", Affine(pc, n_ins0, n_hidden, affine2=False, init_rop=hidden_init_rop))
        self.affine1 = self.add_sub_node("a1", Affine(pc, n_ins1, n_hidden, affine2=False, init_rop=hidden_init_rop))
        self.W = self.add_param(name="W", shape=(n_out, n_hidden, n_hidden))
        if bias:
            self.b = self.add_param(name="B", shape=(n_out, ))
        else:
            self.b = None
        # =====
        # refreshed values
        self.drop_node = self.add_sub_node("drop", Dropout(pc, (self.n_out,)))

    def __repr__(self):
        return "# PairedBiAffine: %s (%s & %s -> %s -> %s [%s])" % (self.name, self.n_ins0, self.n_ins1, self.n_hidden, self.n_out, self.act)

    def __call__(self, input_exp0, input_exp1):
        # input_shape = BK.get_shape(input_exp0)
        # view_shape = [np.prod(input_shape[:-1]), -1]
        # hid0 = self.affine0(input_exp0.view(view_shape))
        # hid1 = self.affine1(input_exp1.view(view_shape))
        hid0 = self.affine0(input_exp0)
        hid1 = self.affine1(input_exp1)
        # [*, hid], [*, hid] -> [*, out]
        h0 = BK.bilinear(hid0, hid1, self.W, self.b)    # bilinear supports high dim
        h1 = self._act_f(h0)
        h2 = self.drop_node(h1)
        # input_shape[-1] = -1
        # h3 = BK.reshape(h2, input_shape)        # back to original dim
        # return h3
        return h2

    def get_output_dims(self, *input_dims):
        return (self.n_out, )

#
# the final layer scorer (no dropout if used as the final scoring layer)
class BiAffineScorer(BasicNode):
    def __init__(self, pc, in_size0, in_size1, out_size, ff_hid_size=-1, ff_hid_layer=0, use_bias=True, use_biaffine=True, use_ff=True, no_final_drop=True, name=None, init_rop=None, mask_value=Constants.REAL_PRAC_MIN):
        super().__init__(pc, name, init_rop)
        #
        self.in_size0 = in_size0
        self.in_size1 = in_size1
        self.out_size = out_size
        self.ff_hid_size = ff_hid_size
        self.use_bias = use_bias
        self.use_biaffine = use_biaffine
        self.use_ff = use_ff
        self.no_final_drop = no_final_drop
        self.mask_value = mask_value        # function practically as -inf
        #
        if self.use_bias:
            self.B = self.add_param(name="B", shape=(out_size, ))
        zcheck(use_ff or use_biaffine, "No real calculations here!")
        if self.use_ff:
            self.A0 = self.add_sub_node("A0", get_mlp(pc, in_size0, out_size, ff_hid_size, n_hidden_layer=ff_hid_layer,
                                                      final_bias=False, final_init_rop=NoDropRop()))
            self.A1 = self.add_sub_node("A1", get_mlp(pc, in_size1, out_size, ff_hid_size, n_hidden_layer=ff_hid_layer,
                                                      final_bias=False, final_init_rop=NoDropRop()))
        if self.use_biaffine:
            # this is different than BK.bilinear or layers.BiAffine
            self.W = self.add_param(name="W", shape=(in_size0, in_size1*out_size))
        # todo(0): meaningless to use fix-drop
        if no_final_drop:
            self.drop_node = lambda x: x
        else:
            self.drop_node = self.add_sub_node("drop", Dropout(pc, (self.out_size,), init_rop=NoFixRop()))

    # plain call
    # io: [*, in_size0], [*, in_size1] -> [*, out_size]; masks: [*]
    def plain_score(self, input0, input1, mask0=None, mask1=None):
        ret = 0
        if self.use_ff:
            ret = self.A0(input0) + self.A1(input1)
        if self.use_biaffine:
            # [*, in0] * [in0, in1*out] -> [*, in1*out] -> [*, in1, out]
            expr0 = BK.matmul(input0, self.W).view(BK.get_shape(input0)[:-1]+[self.in_size1, self.out_size])
            # [*, 1, in1] * [*, in1, out] -> [*, 1, out] -> [*, out]
            expr1 = BK.matmul(input1.unsqueeze(-2), expr0).squeeze(-2)
            ret += expr1
        if self.use_bias:
            ret += self.B
        # mask
        if mask0 is not None:
            ret += self.mask_value*(1.-mask0).unsqueeze(-1)
        if mask1 is not None:
            ret += self.mask_value*(1.-mask1).unsqueeze(-1)
        return self.drop_node(ret)

    # special call
    # io: [*, len0, in_size0], [*, len1, in_size1] -> [*, len0, len1, out_size], masks: [*, len0], [*, len1]
    def paired_score(self, input0, input1, mask0=None, mask1=None):
        ret = 0
        # [*, 1, len1, in_size1]
        expand1 = input1.unsqueeze(-3)
        if self.use_ff:
            # [*, len0, 1, in_size0]
            expand0 = input0.unsqueeze(-2)
            ret = self.A0(expand0) + self.A1(expand1)
        if self.use_biaffine:
            # [*, len0, in0] * [in0, in1*out] -> [*, len0, in1*out] -> [*, len0, in1, out]
            expr0 = BK.matmul(input0, self.W).view(BK.get_shape(input0)[:-1]+[self.in_size1, self.out_size])
            # [*, 1, len1, in1] * [*, len0, in1, out]
            expr1 = BK.matmul(expand1, expr0)
            # [*, len0, len1, out]
            ret += expr1
        if self.use_bias:
            ret += self.B
        # mask
        if mask0 is not None:
            ret += self.mask_value*(1.-mask0).unsqueeze(-1).unsqueeze(-1)
        if mask1 is not None:
            ret += self.mask_value*(1.-mask1).unsqueeze(-2).unsqueeze(-1)
        return self.drop_node(ret)

    # default is paired special version
    def __call__(self, *args, **kwargs):
        return self.paired_score(*args, **kwargs)

    def get_output_dims(self, *input_dims):
        return (self.out_size, )
