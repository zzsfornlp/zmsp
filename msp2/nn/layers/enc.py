#

__all__ = [
    "VrecSteppingResult", "VrecSteppingState", "MultiLayerEncNode",
    "RnnCellConf", "RnnCellNode", "LstmCellNode", "RnnConf", "RnnNode",
    "CnnConf", "CnnNode", "TransformerConf", "TransformerNode",
]

# some Enc layers

from typing import Union, Type, List, Iterable
import math
from ..backends import BK
from .base import *
from .att import *
from .ff import *
from .multi import *
from .helper import *
from msp2.utils import Constants, zwarn

# =====
# vertical stepping state

class VrecSteppingResult:
    def __init__(self, expr: BK.Expr, description: str = None):
        self.expr = expr
        self.description = description

class VrecSteppingState:
    def __init__(self, input_t: BK.Expr, mask_t: BK.Expr):
        # input_t & current_t
        self.input_t: BK.Expr = input_t  # [*, len, Din]
        self.mask_t: BK.Expr = mask_t  # [*, len]
        # --
        self.current_t: BK.Expr = input_t  # [*, len, Dout]
        # list of steps
        self.steps: List[VrecSteppingResult] = []

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, item):
        return self.steps[item]

    def update(self, expr: BK.Expr, **kwargs):
        # update current_t and add to list
        one = VrecSteppingResult(expr, **kwargs)
        self.current_t = one
        self.steps.append(one)
        return None  # if None, only recording and no change!!

# only an interface
class MultiLayerEncNode(BasicNode):
    def __init__(self, conf: BasicConf, **kwargs):
        super().__init__(conf, **kwargs)

    @property
    def num_layers(self):
        return self.conf.n_layers  # todo(note): specific name!!

    def forward(self, input_expr: BK.Expr, mask_expr: BK.Expr=None, vstate: VrecSteppingState=None, **kwargs):
        raise NotImplementedError()

# =====
# RNN Cell

# todo(note): make it common for all kinds of cells for convenience
class RnnCellConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize = -1
        self.osize = -1
        # --
        self.idrop: float = None
        self.gdrop: float = None

    @classmethod
    def _get_type_hints(cls):
        return {"idrop": float, "gdrop": float}

@node_reg(RnnCellConf)
class RnnCellNode(BasicNode):
    def __init__(self, conf: RnnCellConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: RnnCellConf = self.conf
        # --
        self.idrop = DropoutNode(None, drop_rate=conf.idrop, osize=conf.isize, which_drop="idrop")
        self.gdrop = DropoutNode(None, drop_rate=conf.gdrop, osize=conf.osize, which_drop="gdrop")

    def _apply_mask(self, mask, new_val, old_val):
        # mask: 0 for pass through, 1 for real value
        # mask at the batch axis
        mask_expr = BK.unsqueeze(BK.input_real(mask), -1)
        # hidden = mask_expr*new_val + (1.-mask_expr)*old_val
        hidden = old_val + mask_expr * (new_val - old_val)
        return hidden

    # --
    def make_zero_init(self, pre_sizes: Iterable[int]):
        raise NotImplementedError()

    def make_init(self, init_h: BK.Expr):
        raise NotImplementedError()

    def extra_repr(self) -> str:
        return f"{self.__class__.__name__}(I={self.conf.isize},H={self.conf.osize})"

    @staticmethod
    def get_rnn_node(node_type: Union[str, Type], conf: RnnCellConf, **kwargs):
        _RNN_TYPES = {"lstm": LstmCellNode}
        if isinstance(node_type, str):
            node_c = _RNN_TYPES[node_type]
        else:
            node_c = node_type
        return node_c(conf, **kwargs)

# LSTM
@node_reg(RnnCellConf)
class LstmCellNode(RnnCellNode):
    def __init__(self, conf: RnnCellConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: RnnCellConf = self.conf
        # --
        n_input, n_hidden = conf.isize, conf.osize
        self.xw = BK.new_param([4*n_hidden, n_input])
        self.hw = BK.new_param([4*n_hidden, n_hidden])
        self.xb = BK.new_param([4*n_hidden, ])
        self.hb = BK.new_param([4*n_hidden, ])
        self.reset_parameters()

    def reset_parameters(self):
        nhid = self.conf.osize
        for i in range(4):  # they are four matrix
            BK.init_param(self.xw[i*nhid:(i+1)*nhid], "default")
            BK.init_param(self.hw[i*nhid:(i+1)*nhid], "ortho")
        BK.init_param(self.xb, "zero")
        BK.init_param(self.hb, "zero")

    def forward(self, input_exp, hidden_exp_tuple, mask=None):
        input_exp = self.idrop(input_exp)
        orig_h, orig_c = hidden_exp_tuple
        hidden_exp_h = self.gdrop(orig_h)
        # --
        hidden, c_t = BK.lstm_oper(input_exp, (hidden_exp_h, orig_c), self.xw, self.hw, self.xb, self.hb)
        if mask is not None:
            hidden = self._apply_mask(mask, hidden, orig_h)
            c_t = self._apply_mask(mask, c_t, orig_c)
        return (hidden, c_t)

    # make zero init
    def make_zero_init(self, pre_sizes: Iterable[int]):
        z0 = BK.zeros(list(pre_sizes)+[self.conf.osize])
        return (z0, z0)

    # make init partially from input
    def make_init(self, init_h: BK.Expr):
        hsize = self.conf.osize
        h_shape = BK.get_shape(init_h)  # [*, hsize]
        assert h_shape[-1] == hsize
        z0 = self.make_zero_init(h_shape[:-1])  # still use zero c!
        return (init_h, z0[1])

# =====
# RNN

class RnnConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize = -1
        self.osize = -1  # todo(note): //2 if bidirection
        # --
        self.cell = RnnCellConf()
        self.cell_type = "lstm"  # cell type
        self.step_dim = 1  # which dim as time step
        self.n_layers = 1  # number of layers
        self.bidirection = True  # bidirection?
        self.sep_bidirection = False  # sep the two directions when stacking layers?
        self.no_drop = False
        self.out_drop = DropoutConf()  # final output dropout

@node_reg(RnnConf)
class RnnNode(MultiLayerEncNode):
    def __init__(self, conf: RnnConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: RnnConf = self.conf
        # --
        # actually three modes: 1) uni-d, 2) bi-d + sep, 3) bi-d + non-sep
        n_input, n_hidden = conf.isize, conf.osize
        if not conf.bidirection:  # 1) both full n_hidden
            one_hid_dim, one_inp_dim = n_hidden, n_hidden
        else:
            assert n_hidden % 2 == 0, f"Hidden-dim {n_hidden} not dividable by 2 for bidirection!"
            one_hid_dim = n_hidden // 2  # each hid only get half
            if conf.sep_bidirection:  # 2) bi-d + sep
                one_inp_dim = n_hidden // 2
            else:  # 3) bi-d + non-sep
                one_inp_dim = n_hidden
        # --
        cur_inp_dim = n_input  # start with real input dim
        cur_out_dim = cur_inp_dim
        self.fnodes, self.bnodes = [], []
        self.drop_nodes = []  # especially for output, since RnnCells use idrop
        for i in range(conf.n_layers):
            one_fnode = RnnCellNode.get_rnn_node(conf.cell_type, conf.cell, isize=cur_inp_dim, osize=one_hid_dim)
            self.add_module(f"F{i}", one_fnode)
            self.fnodes.append(one_fnode)
            if conf.bidirection:
                one_bnode = RnnCellNode.get_rnn_node(conf.cell_type, conf.cell, isize=cur_inp_dim, osize=one_hid_dim)
                self.add_module(f"B{i}", one_bnode)
                self.bnodes.append(one_bnode)
            # drop on output combined size
            if conf.no_drop:
                one_dnode = lambda x: x
            else:
                one_dnode = DropoutNode(conf.out_drop, osize=conf.osize)
                self.add_module(f"D{i}", one_dnode)
            self.drop_nodes.append(one_dnode)
            # --
            cur_inp_dim = one_inp_dim
            cur_out_dim = n_hidden
        self.output_dim = cur_out_dim  # final output dim

    # =====
    def extra_repr(self) -> str:
        conf: RnnConf = self.conf
        return f"RNN(I={conf.isize},N={conf.osize},L={conf.n_layers})"

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    # running at step_dim, by default [bsize, step, D]
    def forward(self, input_expr, mask_expr=None, vstate: VrecSteppingState=None, **kwargs):
        conf: RnnConf = self.conf
        step_dim = conf.step_dim
        # --
        step_inputs, step_masks = prepare_step_inputs(input_expr, mask_expr, step_dim, False)
        pre_sizes = BK.get_shape(step_inputs[0])[:-1]
        init_f_hidden = self.fnodes[0].make_zero_init(pre_sizes) if len(self.fnodes)>0 else None
        init_b_hidden = self.bnodes[0].make_zero_init(pre_sizes) if len(self.bnodes)>0 else None
        # --
        f_outputs = [step_inputs]  # List(layer)[List(step)]
        b_outputs = [step_inputs]  # ...
        all_layered_outputs = []  # filled if vstate is not None
        # --
        # for each layer
        has_vstate = (vstate is not None)
        for layer_idx in range(conf.n_layers):
            d_node = self.drop_nodes[layer_idx]  # only used in layer-iteration for outputting for vstate
            f_node = self.fnodes[layer_idx]
            tmp_f = []  # forward
            tmp_b = []  # backward
            tmp_f_prev = init_f_hidden
            for e, m in zip(f_outputs[-1], step_masks):
                one_f = f_node(e, tmp_f_prev, m)
                tmp_f.append(one_f[0])  # 0 is the Hidden
                tmp_f_prev = one_f  # recurrent
            if conf.bidirection:
                b_node = self.bnodes[layer_idx]
                tmp_b_prev = init_b_hidden
                for e, m in zip(reversed(b_outputs[-1]), reversed(step_masks)):
                    one_b = b_node(e, tmp_b_prev, m)
                    tmp_b.append(one_b[0])  # 0 is the hidden
                    tmp_b_prev = one_b  # recurrent
                tmp_b.reverse()  # todo(note): always store in l2r order
            # output or for next layer
            if not conf.bidirection:  # 1) uni-d
                if has_vstate:  # stack, update and then split again
                    stacked_res = d_node(BK.stack(tmp_f, step_dim))  # stack and drop
                    updated_res = vstate.update(stacked_res)
                    if updated_res is None:
                        updated_res = stacked_res
                    all_layered_outputs.append(updated_res)
                    tmp_f = split_at_dim(updated_res, step_dim, False)
                f_outputs.append(tmp_f)
            else:
                if has_vstate:
                    stacked_res_f, stacked_res_b = BK.stack(tmp_f, step_dim), BK.stack(tmp_b, step_dim)
                    stacked_res = d_node(BK.concat([stacked_res_f, stacked_res_b], -1))  # stack and drop
                    updated_res = vstate.update(stacked_res)
                    if updated_res is None:
                        updated_res = stacked_res
                    all_layered_outputs.append(updated_res)
                if conf.sep_bidirection:  # 2) bi-d + sep
                    if has_vstate:
                        all_ctx0, all_ctx1 = BK.chunk(updated_res, 2, -1)  # chunk last dim!
                        tmp_f, tmp_b = split_at_dim(all_ctx0, step_dim, False), split_at_dim(all_ctx1, step_dim, False)
                    f_outputs.append(tmp_f)
                    b_outputs.append(tmp_b)
                else:  # 3) bi-d + non-sep
                    if has_vstate:
                        all_ctx_slices = split_at_dim(updated_res, step_dim, False)
                    else:
                        all_ctx_slices = [BK.concat([f, b]) for f, b in zip(tmp_f, tmp_b)]
                    f_outputs.append(all_ctx_slices)
                    b_outputs.append(all_ctx_slices)
        # finally
        if conf.n_layers == 0:
            return input_expr  # simply return inputs
        else:
            if has_vstate:  # already calculated
                return all_layered_outputs[-1]
            else:  # stack and concat and drop
                if conf.bidirection:
                    stacked_res_f, stacked_res_b = BK.stack(tmp_f, step_dim), BK.stack(tmp_b, step_dim)
                    stacked_res = BK.concat([stacked_res_f, stacked_res_b], -1)  # concat at last dim
                else:
                    stacked_res = BK.stack(tmp_f, step_dim)
                dropped_res = self.drop_nodes[-1](stacked_res)
                return dropped_res

# =====
# CNN node
# currently only Conv1D layers
# operations at the last two dimension (*, length, n_input) -> (*, length, n_output) if not pooling else (*, n_output)

class CnnConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize = -1
        self.osize = -1
        # --
        self.n_layers = 1
        self.win_sizes = [3,5]  # window sizes
        self.out_act = "elu"  # activation
        self.out_pool = "none"
        self.no_drop = False  # no dropout for output
        self.out_drop = DropoutConf()

    @classmethod
    def _get_type_hints(cls):
        return {z: int for z in ["n_input", "n_output", "win_sizes"]}

    def _do_validate(self):
        if isinstance(self.win_sizes, int):
            self.win_sizes = [self.win_sizes]

# one cnn node
class _OneCnnNode(BasicNode):
    def __init__(self, n_input: int, n_output: int, n_window: int):
        super().__init__(None)
        # --
        self.W = BK.new_param([n_output, n_input, n_window])
        self.B = BK.new_param([n_output,])
        assert n_window % 2 == 1, "Currently only support ODD window size!"
        self.conv_padding = n_window // 2
        self.n_output = n_output
        self.reset_parameters()

    def reset_parameters(self):
        BK.init_param(self.W.view([self.n_output, -1]), "default")
        BK.init_param(self.B, "zero")

    def forward(self, input_expr):
        cur_dims = BK.get_shape(input_expr)
        expr1 = input_expr.view([-1, cur_dims[-2], cur_dims[-1]]).transpose(-2, -1)  # [*, dim_in, LEN]
        val0 = BK.F.conv1d(expr1, self.W, self.B, padding=self.conv_padding)
        # reshape
        reshape_dims = cur_dims
        reshape_dims[-1] = -1
        reshape_dims[-2] = self.n_output
        val1 = val0.view(reshape_dims).transpose(-2, -1)  # [*, LEN, dim_out]
        return val1

# one layer of cnn
@node_reg(CnnConf)
class _OneLayerCnnNode(BasicNode):
    def __init__(self, conf: CnnConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: CnnConf = self.conf
        # --
        assert conf.osize%len(conf.win_sizes)==0, f"Output-dim {conf.osize} not dividable by NumWin {len(conf.win_sizes)}!"
        one_n_output = conf.osize // len(conf.win_sizes)
        self.all_cnodes = []
        for wi, wsize in enumerate(conf.win_sizes):
            one_cnode = _OneCnnNode(conf.isize, one_n_output, wsize)
            self.add_module(f"C{wi}", one_cnode)
            self.all_cnodes.append(one_cnode)
        # --
        # act -> pooling -> dropout
        self._act_f = ActivationHelper.get_act(conf.out_act)
        self._pool_f = ActivationHelper.get_pool(conf.out_pool)
        if conf.no_drop:
            self.drop_node = lambda x: x
        else:
            self.drop_node = DropoutNode(conf.out_drop, osize=conf.osize)

    # todo(+n): mask is not included yet!
    def forward(self, input_expr, mask=None, **kwargs):
        output_exprs = [cnode(input_expr) for cnode in self.all_cnodes]
        output_expr = BK.concat(output_exprs, -1)  # [*, len, OUT]
        # act -> pooling -> dropout
        expr_a = self._act_f(output_expr)
        expr_p = self._pool_f(expr_a, -2)  # note: pooling at -2!
        expr_d = self.drop_node(expr_p)
        return expr_d

@node_reg(CnnConf)
class CnnNode(MultiLayerEncNode):
    def __init__(self, conf: CnnConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: CnnConf = self.conf
        # --
        self.nodes = []
        cur_input = conf.isize
        output_dim = cur_input
        for i in range(conf.n_layers):
            if i<conf.n_layers-1:
                # no use pooling at middle layers
                node = _OneLayerCnnNode(conf, isize=output_dim, osize=conf.osize, out_pool="none")
            else:
                node = _OneLayerCnnNode(conf, isize=output_dim, osize=conf.osize)
            self.add_module(f"L{i}", node)
            self.nodes.append(node)
            output_dim = conf.osize
        # --
        self.output_dim = output_dim

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    def extra_repr(self) -> str:
        return f"CNN(I={self.conf.isize},O={self.conf.osize},L={self.conf.n_layers})"

    def forward(self, input_expr, mask_expr=None, vstate: VrecSteppingState=None, **kwargs):
        x = input_expr
        for node in self.nodes:
            x = node(x, mask_expr)
            if vstate is not None:
                x1 = vstate.update(x)
                if x1 is not None:
                    x = x1
        return x

# =====
# Transformer

class TransformerConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize = -1  # input dim
        self.osize = -1  # hidden/output_dim
        # --
        self.n_layers = 4
        self.layered_ranges = []  # cutoff_range for each layer
        self.use_posi = False  # add posi embeddings at input?
        self.pconf = PosiEmbeddingConf().direct_update(min_val=0)
        # for each layer
        self.d_ff = 1024  # dim of FF, 0 for skipping ff
        self.aconf = AttentionConf()
        self.wconf = WrapperConf().direct_update(strategy="addnorm", act="tanh")
        self.ff_act = "relu"

    @property
    def d_model(self):
        assert self.isize == self.osize
        return self.isize

    @classmethod
    def _get_type_hints(cls):
        return {"layered_ranges": int}

    def _do_validate(self):
        len_range = len(self.layered_ranges)
        self.layered_ranges += [None] * (self.n_layers - len_range)  # None means no range!

class _OneTSFNode(BasicNode):
    def __init__(self, conf: TransformerConf):
        super().__init__(None)
        # --
        d_model = conf.d_model
        att_node = AttentionNode(conf.aconf, dim_q=d_model, dim_k=d_model, dim_v=d_model)
        self.self_att = WrapperNode(att_node, conf.wconf, isize=d_model)
        if conf.d_ff > 0:
            ff_node = get_mlp(d_model, d_model, conf.d_ff, 1, AffineConf().direct_update(out_act=conf.ff_act), AffineConf())
            self.feed_forward = WrapperNode(ff_node, conf.wconf, isize=d_model)
        else:
            self.feed_forward = lambda x: x

    def forward(self, input_expr, mask_expr=None, cutoff_range:int=None, **kwargs):
        context = self.self_att(input_expr, input_expr, input_expr, mask_k=mask_expr, cutoff_range=cutoff_range, **kwargs)
        output = self.feed_forward(context)
        return output

@node_reg(TransformerConf)
class TransformerNode(MultiLayerEncNode):
    def __init__(self, conf: TransformerConf, **kwargs):
        super().__init__(conf, **kwargs)
        # --
        conf: TransformerConf = self.conf
        self.tnodes = []
        for i in range(conf.n_layers):
            one_node = _OneTSFNode(conf)
            self.add_module(f"T{i}", one_node)
            self.tnodes.append(one_node)
        # add posi embeddings
        # (note: nope!!) self.scale_when_add_posi = math.sqrt(conf.d_model)
        if conf.use_posi:
            self.PE = PosiEmbeddingNode(conf.pconf, osize=conf.d_model)
        # input f
        if conf.wconf.strategy == "addnorm":
            self.input_f = LayerNormNode(None, osize=conf.d_model)
        elif conf.wconf.strategy == "addact":
            self.input_f = ActivationHelper.get_act(conf.wconf.act)
        else:
            zwarn("No calculations for input in TransformerEncoder!!")
            self.input_f = lambda x: x

    def get_output_dims(self, *input_dims):
        return (self.conf.d_model, )

    def extra_repr(self) -> str:
        return f"Transformer(D={self.conf.d_model},L={self.conf.n_layers})"

    def forward(self, input_expr, mask_expr=None, vstate: VrecSteppingState=None, **kwargs):
        conf: TransformerConf = self.conf
        # --
        if len(self.tnodes) == 0:
            return input_expr  # change nothing if no layers
        # --
        if conf.use_posi:
            ssize = BK.get_shape(input_expr, 1)  # step size
            if mask_expr is None:
                posi_embed = self.PE(BK.arange_idx(ssize)).unsqueeze(0)  # [1, step, D]
            else:
                positions = mask2posi(mask_expr, offset=-1, cmin=0)  # [*, step]
                posi_embed = self.PE(positions)
            # x = self.input_f(input_expr * self.scale_when_add_posi + posi_embed)  # add absolute positional embeddings
            x = self.input_f(input_expr + posi_embed)  # add absolute positional embeddings
        else:
            x = self.input_f(input_expr)  # process input
        # --
        cur_hid = x
        for ti, tnode in enumerate(self.tnodes):
            cur_hid = tnode(cur_hid, mask_expr=mask_expr, cutoff_range=conf.layered_ranges[ti], **kwargs)
            if vstate is not None:
                x1 = vstate.update(cur_hid)
                if x1 is not None:
                    cur_hid = x1
        return cur_hid
