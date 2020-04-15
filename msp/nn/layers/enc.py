#

# mostly used for encoders

import numpy as np
from typing import List

from ..backends import BK
from .basic import BasicNode, Dropout, ActivationHelper
from .att import AttentionNode, AttConf
from .ff import LayerNorm
from .multi import AddNormWrapper, AddActWrapper, get_mlp, GatedMixer

# # rnn nodes
# OLD: [inputs] or input + {"H":hidden, "C":(optional)cell} -> {"H":hidden, "C":(optional)cell}
# todo(note): change to new format of tuple (hidden, (optional) cell)!
# no output drops
class RnnNode(BasicNode):
    def __init__(self, pc, n_input, n_hidden, name=None, init_rop=None):
        super(RnnNode, self).__init__(pc, name, init_rop)
        self.n_input = n_input
        self.n_hidden = n_hidden
        #
        self.idrop_node = self.add_sub_node("idrop", Dropout(pc, (self.n_input,), which_drop="idrop"))
        self.gdrop_node = self.add_sub_node("gdrop", Dropout(pc, (self.n_hidden,), which_drop="gdrop"))

    def zero_init_hidden(self, bsize):
        raise NotImplementedError()

    def __call__(self, input_exp, hidden_exp_tuple, mask):
        raise NotImplementedError()

    def __repr__(self):
        return "# RnnNode[%s] (input=%s, hidden=%s)" % (type(self), self.n_input, self.n_hidden)

    @staticmethod
    def get_rnn_node(node_type, pc, n_input, n_hidden, name=None, init_rop=None):
        _RNN_TYPES = {"gru": GruNode, "lstm": LstmNode, "gru2": GruNode2, "lstm2": LstmNode2}
        if isinstance(node_type, str):
            node_c = _RNN_TYPES[node_type]
        else:
            node_c = node_type
        return node_c(pc, n_input, n_hidden, name=name, init_rop=init_rop)

    def _apply_mask(self, mask, new_val, old_val):
        # mask: 0 for pass through, 1 for real value
        # mask at the batch axis
        mask_expr = BK.unsqueeze(BK.input_real(mask), -1)
        # hidden = mask_expr*new_val + (1.-mask_expr)*old_val
        hidden = old_val + mask_expr * (new_val - old_val)
        return hidden

    def get_output_dims(self, *input_dims):
        return (self.n_hidden, )

#
class GruNode(RnnNode):
    def __init__(self, pc, n_input, n_hidden, name=None, init_rop=None):
        super(GruNode, self).__init__(pc, n_input, n_hidden, name, init_rop)
        # params
        self.x2rz = self.add_param("x2rz", (2*self.n_hidden, n_input))
        self.h2rz = self.add_param("h2rz", (2*self.n_hidden, n_hidden), init="ortho")
        self.brz = self.add_param("brz", (2*self.n_hidden, ))
        self.x2h = self.add_param("x2h", (self.n_hidden, n_input))
        self.h2h = self.add_param("h2h", (self.n_hidden, n_hidden), init="ortho")
        self.bh = self.add_param("bh", (self.n_hidden, ))

    # input: value, hidden: dict of value
    def __call__(self, input_exp, hidden_exp_tuple, mask):
        input_exp = self.idrop_node(input_exp)
        # todo(warn): only using [0]
        orig_hidden_exp_h = hidden_exp_tuple[0]
        hidden_exp_h = self.gdrop_node(orig_hidden_exp_h)
        #
        rzt = BK.affine([self.brz, self.x2rz, input_exp, self.h2rz, hidden_exp_h])
        rzt = BK.sigmoid(rzt)
        rt, zt = BK.chunk(rzt, 2)
        h_reset = BK.cmult(rt, hidden_exp_h)
        ht = BK.affine([self.bh, self.x2h, input_exp, self.h2h, h_reset])
        ht = BK.tanh(ht)
        # hidden = BK.cmult(zt, hidden_exp["H"]) + BK.cmult((1. - zt), ht)     # first one use original hh
        hidden = ht + zt * (orig_hidden_exp_h - ht)
        if mask is not None:
            hidden = self._apply_mask(mask, hidden, orig_hidden_exp_h)
        return (hidden, )

    def zero_init_hidden(self, bsize):
        return (BK.zeros((bsize, self.n_hidden)), )

# todo(+N): get strangely large scaled values from this one!
class GruNode2(RnnNode):
    def __init__(self, pc, n_input, n_hidden, name=None, init_rop=None):
        super(GruNode2, self).__init__(pc, n_input, n_hidden, name, init_rop)
        self.x2h = self.add_param("x2h", (3*self.n_hidden, n_input))
        self.h2h = self.add_param("h2h", (3*self.n_hidden, n_hidden), init="ortho")
        self.xb = self.add_param("xb", (3*self.n_hidden, ))
        self.hb = self.add_param("hb", (3*self.n_hidden, ))

    def __call__(self, input_exp, hidden_exp_tuple, mask):
        input_exp = self.idrop_node(input_exp)
        orig_hidden_exp_h = hidden_exp_tuple[0]
        hidden_exp_h = self.gdrop_node(orig_hidden_exp_h)
        #
        hidden = BK.gru_oper(input_exp, hidden_exp_h, self.x2h, self.h2h, self.xb, self.hb)
        if mask is not None:
            hidden = self._apply_mask(mask, hidden, orig_hidden_exp_h)
        return (hidden, )

    def zero_init_hidden(self, bsize):
        return (BK.zeros((bsize, self.n_hidden)), )

class LstmNode(RnnNode):
    def __init__(self, pc, n_input, n_hidden, name=None, init_rop=None):
        super(LstmNode, self).__init__(pc, n_input, n_hidden, name, init_rop)
        # params
        self.xw = self.add_param("xw", (4*self.n_hidden, n_input))
        self.hw = self.add_param("hw", (4*self.n_hidden, n_hidden), init="ortho")
        self.b = self.add_param("b", (4*self.n_hidden, ))

    def __call__(self, input_exp, hidden_exp_tuple, mask):
        input_exp = self.idrop_node(input_exp)
        orig_h, orig_c = hidden_exp_tuple
        hidden_exp_h = self.gdrop_node(orig_h)
        #
        ifco = BK.affine([self.b, self.xw, input_exp, self.hw, hidden_exp_h])
        i_t, f_t, g_t, o_t = BK.chunk(ifco, 4)
        i_t = BK.sigmoid(i_t)
        f_t = BK.sigmoid(f_t)
        g_t = BK.tanh(g_t)
        o_t = BK.sigmoid(o_t)
        c_t = BK.cmult(f_t, orig_c) + BK.cmult(i_t, g_t)
        hidden = BK.cmult(o_t, BK.tanh(c_t))
        if mask is not None:
            hidden = self._apply_mask(mask, hidden, orig_h)
            c_t = self._apply_mask(mask, c_t, orig_c)
        return (hidden, c_t)

    def zero_init_hidden(self, bsize):
        z0 = BK.zeros((bsize, self.n_hidden))
        return (z0, z0)

class LstmNode2(RnnNode):
    def __init__(self, pc, n_input, n_hidden, name=None, init_rop=None):
        super(LstmNode2, self).__init__(pc, n_input, n_hidden, name, init_rop)
        # params
        self.xw = self.add_param("xw", (4*self.n_hidden, n_input))
        self.hw = self.add_param("hw", (4*self.n_hidden, n_hidden), init="ortho")
        self.xb = self.add_param("xb", (4*self.n_hidden, ))
        self.hb = self.add_param("hb", (4*self.n_hidden, ))

    def __call__(self, input_exp, hidden_exp_tuple, mask):
        input_exp = self.idrop_node(input_exp)
        orig_h, orig_c = hidden_exp_tuple
        hidden_exp_h = self.gdrop_node(orig_h)
        #
        hidden, c_t = BK.lstm_oper(input_exp, (hidden_exp_h, orig_c), self.xw, self.hw, self.xb, self.hb)
        if mask is not None:
            hidden = self._apply_mask(mask, hidden, orig_h)
            c_t = self._apply_mask(mask, c_t, orig_c)
        return (hidden, c_t)

    def zero_init_hidden(self, bsize):
        z0 = BK.zeros((bsize, self.n_hidden))
        return (z0, z0)

# stateless encoder
# here only one layer, to join by Seq
# todo(warn): this layer accept special inputs, use RnnLayerBatchFirstWrapper for ordinary ones
class RnnLayer(BasicNode):
    def __init__(self, pc, n_input, n_hidden, n_layers=1, node_type="lstm", node_init_rop=None, init_rop=None,
                 bidirection=False, sep_bidirection=False, name=None, no_output_dropout=True):
        super().__init__(pc, name, init_rop)
        #
        def _get_node(name, n_input, n_hidden):
            return self.add_sub_node(name, RnnNode.get_rnn_node(node_type, pc, n_input, n_hidden, init_rop=node_init_rop))
        #
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.bidirection = bidirection
        self.sep_bidirection = sep_bidirection  # separately model the two directions
        #
        dim_hid = self.n_hidden * (2 if (self.bidirection and not self.sep_bidirection) else 1)
        self.fnodes = [_get_node("f", n_input, n_hidden)] + [_get_node("f", dim_hid, n_hidden) for z in range(n_layers-1)]
        if bidirection:
            self.bnodes = [_get_node("b", n_input, n_hidden)] + [_get_node("b", dim_hid, n_hidden) for z in range(n_layers-1)]
        if n_layers > 0:
            self.output_dim = self.n_hidden * (2 if self.bidirection else 1)
        else:
            self.output_dim = n_input
        # dropout for the output of each layer (also controlled by ``no_output_dropout''
        self.no_output_dropout = no_output_dropout
        if no_output_dropout:
            self.f_drop_nodes = self.b_drop_nodes = None
        else:
            self.f_drop_nodes = [self.add_sub_node("fdrop", Dropout(pc, (self.n_hidden,))) for _ in range(n_layers)]
            self.b_drop_nodes = [self.add_sub_node("bdrop", Dropout(pc, (self.n_hidden,))) for _ in range(n_layers)]

    def __repr__(self):
        return f"# RnnLayer[fb={self.bidirection}, sep={self.sep_bidirection}] (input={self.n_input}, hidden={self.n_hidden})"

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    # embeds: list(step) of {(n_emb, ), batch_size}, using padding for batches
    # masks: list(step) of {batch_size, } or None
    def __call__(self, embeds, masks=None):
        f_outputs = [embeds]
        b_outputs = [embeds]
        # todo(warn), how about disabling masks for speeding up (although might not be critical)?
        if masks is None:
            masks = [None for _ in embeds]
        bsize = BK.get_shape(embeds[0], 0)
        init_hidden = BK.zeros((bsize, self.n_hidden))      # broadcast
        init_states = (init_hidden, init_hidden)            # todo(note): inner side of the RNN-units
        for layer_idx in range(self.n_layers):
            f_node = self.fnodes[layer_idx]
            tmp_f = []  # forward
            tmp_b = []  # backward
            tmp_f_prev = init_states
            for e, m in zip(f_outputs[-1], masks):
                one_output = f_node(e, tmp_f_prev, m)
                tmp_f.append(one_output[0])                 # todo(note): inner side of the RNN-units
                tmp_f_prev = one_output
            if self.bidirection:
                b_node = self.bnodes[layer_idx]
                tmp_b_prev = init_states
                for e, m in zip(reversed(b_outputs[-1]), reversed(masks)):
                    one_output = b_node(e, tmp_b_prev, m)
                    tmp_b.append(one_output[0])             # todo(note): inner side of the RNN-units
                    tmp_b_prev = one_output
                tmp_b.reverse()  # todo(note): always store in l2r order
            # output/middle dropouts
            if not self.no_output_dropout:
                f_drop_node = self.f_drop_nodes[layer_idx]
                tmp_f = [f_drop_node(z) for z in tmp_f]
                b_drop_node = self.b_drop_nodes[layer_idx]
                tmp_b = [b_drop_node(z) for z in tmp_b]
            # concat if not sep
            if not self.sep_bidirection:
                ctx = [BK.concat([f, b]) for f, b in zip(tmp_f, tmp_b)]
                f_outputs.append(ctx)
                b_outputs.append(ctx)
            else:
                f_outputs.append(tmp_f)
                b_outputs.append(tmp_b)
        # -----
        if self.bidirection and self.sep_bidirection:
            final_outputs = [BK.concat([f, b]) for f, b in zip(f_outputs[-1], b_outputs[-1])]
        else:
            final_outputs = f_outputs[-1]
        return final_outputs

#
class RnnLayerBatchFirstWrapper(BasicNode):
    def __init__(self, pc, rnn_node):
        super().__init__(pc, None, None)
        self.rnn_node: RnnLayer = self.add_sub_node("z", rnn_node)

    # mask>0. means valid, mask=0. means padding
    @staticmethod
    def tranform_mask(m):
        return None if all(z>0. for z in m) else m

    # seq helper (step-first <-> batch-first)
    # input should be [batch, length, *], mask is np.array as [batch, length]
    @staticmethod
    def rnn_inputs(embeds_expr, word_mask_arr):
        step_exprs = [BK.squeeze(z, 1) for z in BK.split(embeds_expr, 1, 1)]
        if word_mask_arr is not None:
            masks = [RnnLayerBatchFirstWrapper.tranform_mask(m) for m in np.transpose(word_mask_arr, (1, 0))]
        else:
            masks = None
        return step_exprs, masks

    @staticmethod
    def rnn_outputs(step_encodings):
        return BK.stack(step_encodings, 1)

    def get_output_dims(self, *input_dims):
        return self.rnn_node.get_output_dims(*input_dims)       # only change upper dims

    def __call__(self, embeds, mask_arr=None):
        step_exprs, masks = RnnLayerBatchFirstWrapper.rnn_inputs(embeds, mask_arr)
        step_encodings = self.rnn_node(step_exprs, masks=masks)
        return RnnLayerBatchFirstWrapper.rnn_outputs(step_encodings)

# # cnn nodes
# currently only Conv1D layers
# operations at the last two dimension (*, length, n_input) -> (*, length, n_output) if not pooling else (*, n_output)
class CnnNode(BasicNode):
    def __init__(self, pc, n_input, n_output, n_windows, pooling=None, act="linear", init_rop=None, name=None):
        super().__init__(pc, name, init_rop)
        #
        if isinstance(n_windows, int):
            n_windows = [n_windows]
        self.n_out = n_output * len(n_windows)      # simply concat
        self.conv_opers = [BK.CnnOper(self, n_input, n_output, z) for z in n_windows]      # add params here
        # act -> pooling -> dropout
        self._act_f = ActivationHelper.get_act(act)
        if pooling is not None:
            self._pool_f = ActivationHelper.get_pool(pooling)
        else:
            self._pool_f = lambda x: x
        self.drop_node = self.add_sub_node("drop", Dropout(pc, (self.n_out,)))

    # input should be a single Expr
    # todo(+1): currently no handling mask
    def __call__(self, input_expr, mask_arr=None):
        output_exprs = [oper.conv(input_expr) for oper in self.conv_opers]
        output_expr = BK.concat(output_exprs, -1)
        # act -> pooling -> dropout
        expr_a = self._act_f(output_expr)
        expr_p = self._pool_f(expr_a)
        expr_d = self.drop_node(expr_p)
        return expr_d

    def get_output_dims(self, *input_dims):
        return (self.n_out, )

CnnLayer = CnnNode

# =====
# self attentional encoders

# common helper
def get_selfatt_node(pc, d_model, aconf: AttConf, fix_range_val):
    orig_range_val = aconf._fixed_range_val
    if fix_range_val is not None:
        aconf._fixed_range_val = fix_range_val
    node = AttentionNode.get_att_node(aconf.type, pc, d_model, d_model, d_model, aconf)
    aconf._fixed_range_val = orig_range_val
    return node

# =====
# transformer (self-attention)
# todo(warn): layer-norm at the output rather than input
class TransformerEncoderLayer(BasicNode):
    def __init__(self, pc, d_model, d_ff, add_wrapper, aconf: AttConf, fix_range_val, name=None, init_rop=None):
        super().__init__(pc, name, init_rop)
        # todo(note): residual adding is like preserving 0.5 attention for self
        wrapper_getf = {"addnorm": AddNormWrapper, "addtanh": lambda *args: AddActWrapper(*args, act="tanh")}[add_wrapper]
        #
        self.d_model = d_model
        self.self_att = self.add_sub_node("s", wrapper_getf(get_selfatt_node(pc, d_model, aconf, fix_range_val), [d_model]))
        self.feed_forward = self.add_sub_node("ff", wrapper_getf(
            get_mlp(pc, d_model, d_model, d_ff, n_hidden_layer=1, hidden_act="relu", final_act="linear"), [d_model]))

    def get_output_dims(self, *input_dims):
        return (self.d_model, )

    def __call__(self, input_expr, mask_arr=None):
        mask_expr = None if mask_arr is None else BK.input_real(mask_arr)
        context = self.self_att(input_expr, input_expr, input_expr, mask_expr)
        output = self.feed_forward(context)
        return output

# transformer (multiple-layers)
class TransformerEncoder(BasicNode):
    def __init__(self, pc, n_layers, d_model, d_ff, add_wrapper, aconf: AttConf, fixed_range_vals, final_act="linear", name=None, init_rop=None):
        super().__init__(pc, name, init_rop)
        #
        self.d_model = d_model
        self.n_layers = n_layers
        # todo(note): here do not make std back since there can be zero input problems!
        in_getf = {"addnorm": lambda *args: self.add_sub_node("in", LayerNorm(*args, std_no_grad=True)),
                   "addtanh": lambda *args: ActivationHelper.get_act("tanh")}[add_wrapper]
        self.input_f = in_getf(pc, d_model)
        #
        if fixed_range_vals is None:
            fixed_range_vals = [None] * self.n_layers
        else:
            assert len(fixed_range_vals) == self.n_layers, "Unmatched ranges and number of layers"
        # todo(+N): not elegant here!
        self.layers = [self.add_sub_node("one", TransformerEncoderLayer(pc, d_model, d_ff, add_wrapper, aconf, fix_range_val=fixed_range_vals[i])) for i in range(n_layers)]
        self._act_f = ActivationHelper.get_act(final_act)

    def get_output_dims(self, *input_dims):
        return (self.d_model, )

    def __call__(self, input_expr, mask_arr=None):
        mask_expr = None if mask_arr is None else BK.input_real(mask_arr)
        x = self.input_f(input_expr)
        for layer in self.layers:
            x = layer(x, mask_expr)
        return self._act_f(x)

# ======
# my-transformer: with slight modifications
# -- fat-transformer: combining more horizontally

class Transformer2EncoderLayer(BasicNode):
    def __init__(self, pc, d_model, aconf: AttConf, short_range, long_range, name=None, init_rop=None):
        super().__init__(pc, name, init_rop)
        self.d_model = d_model
        # use range or not depends on the one argument
        self.short_att = self.add_sub_node("sa", get_selfatt_node(pc, d_model, aconf, short_range))
        self.long_att = self.add_sub_node("la", get_selfatt_node(pc, d_model, aconf, long_range))
        # final gated combiner (mix input/short-att/long-att)
        self.combiner = self.add_sub_node("fc", GatedMixer(pc, d_model, 3))

    def get_output_dims(self, *input_dims):
        return (self.d_model, )

    def __call__(self, input_expr, mask_arr=None):
        mask_expr = None if mask_arr is None else BK.input_real(mask_arr)
        short_output = self.short_att(input_expr, input_expr, input_expr, mask_expr)
        long_output = self.long_att(input_expr, input_expr, input_expr, mask_expr)
        final_output = self.combiner([input_expr, short_output, long_output])
        return final_output

# my-transformer (multiple-layers)
class Transformer2Encoder(BasicNode):
    def __init__(self, pc, n_layers, d_model, aconf: AttConf, short_range:int, long_ranges:List[int], name=None, init_rop=None):
        super().__init__(pc, name, init_rop)
        self.d_model = d_model
        self.n_layers = n_layers
        #
        if long_ranges is None:
            long_ranges = [None] * self.n_layers
        else:
            assert len(long_ranges) == self.n_layers, "Unmatched long ranges and number of layers"
        self.layers = [self.add_sub_node("one", Transformer2EncoderLayer(pc, d_model, aconf, short_range=short_range, long_range=long_ranges[i])) for i in range(n_layers)]

    def get_output_dims(self, *input_dims):
        return (self.d_model, )

    def __call__(self, input_expr, mask_arr=None):
        mask_expr = None if mask_arr is None else BK.input_real(mask_arr)
        x = input_expr
        for layer in self.layers:
            x = layer(x, mask_expr)
        return x
