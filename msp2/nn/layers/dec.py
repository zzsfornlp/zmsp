#

# incremental Decs
# support both testing-time incremental building and training-time fast building

__all__ = [
    "DecConf", "DecCache", "DecNode", "RnnDecConf", "RnnDecCache", "RnnDecNode",
    "TransformerDecConf", "TransformerDecCache", "TransformerDecNode",
]

# TODO(W): attentional Decoders?

from typing import List, Tuple, Union, Iterable
from msp2.nn import BK
from .base import *
from .att import *
from .ff import *
from .multi import *
from .enc import *
from .helper import *
from msp2.utils import zwarn

# =====
# basic ones

class DecConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self.isize = -1  # input dim
        self.osize = -1  # hidden/output_dim
        self.step_dim = 1  # at which dim to go for step, usually dim=1

class DecCache:
    def update(self, states: List, mask: BK.Expr = None): raise NotImplementedError()
    def get_last_state(self, layer_idx=-1): raise NotImplementedError()

    # this one is for specific updates
    def _arrange_idxes(self, idxes: BK.Expr): raise NotImplementedError()

    # this one if for general ones
    def arrange_idxes(self, idxes: BK.Expr):
        # general arrange
        for k in list(self.__dict__.keys()):
            d = getattr(self, k, None)
            if isinstance(d, DecCache):
                d.arrange_idxes(idxes)  # recursive call
            elif isinstance(d, BK.Expr):
                d2 = d.index_select(0, idxes)
                setattr(self, k, d2)
            # others should be handled by specific _arrange_idxes
        # specific arrange: (mainly for composed types like List, Tuple, ...)
        self._arrange_idxes(idxes)

@node_reg(DecConf)
class DecNode(BasicNode):
    def __init__(self, conf: DecConf, **kwargs):
        super().__init__(conf, **kwargs)

    @property
    def num_layers(self):
        return self.conf.n_layers  # todo(note): specific name

    # init cache with starting ones or None for empty one!
    def go_init(self, init_hs: Union[List[BK.Expr], BK.Expr], init_mask: BK.Expr = None, **kwargs):
        raise NotImplementedError()

    # continue on cache, the inputs/outputs both have a step dim, even for one step!
    def go_feed(self, cache: DecCache, input_expr: BK.Expr, mask_expr: BK.Expr = None):
        raise NotImplementedError()

# =====
# RNN based ones

class RnnDecConf(DecConf):
    def __init__(self):
        super().__init__()
        # --
        self.cell = RnnCellConf()
        self.cell_type = "lstm"  # cell type
        self.n_layers = 1  # number of layers
        self.no_drop = False
        self.out_drop = DropoutConf()  # final output dropout

class RnnDecCache(DecCache):
    def __init__(self, states: List):
        # list of rnn's hidden states for all layers: List[*[*, hidden]]
        # note: no step dim here and no history stored!!
        self.states: List[Tuple[BK.Expr]] = states

    def update(self, states: List[Tuple[BK.Expr]], mask: BK.Expr = None):
        self.states = states  # simply replace!

    def get_last_state(self, layer_idx=-1):
        ret = self.states[layer_idx]  # simply return since RNN already handles mask!
        if isinstance(ret, (list, tuple)):
            return ret[0]  # first one is h
        else:
            return ret

    def _arrange_idxes(self, idxes: BK.Expr):
        if self.states is not None:
            self.states = [tuple([s.index_select(0, idxes) for s in spair]) for spair in self.states]

@node_reg(RnnDecConf)
class RnnDecNode(DecNode):
    def __init__(self, conf: RnnDecConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: RnnDecConf = self.conf
        # --
        # simpler than RnnEncoder since here is unidirectional
        n_input, n_hidden = conf.isize, conf.osize
        self.rnn_nodes: List[RnnCellNode] = []
        self.drop_nodes: List[DropoutNode] = []  # especially for output, since RnnCells may use idrop
        cur_inp_dim = n_input
        for i in range(conf.n_layers):
            # rnn cell node
            one_rnn_node = RnnCellNode.get_rnn_node(conf.cell_type, conf.cell, isize=cur_inp_dim, osize=n_hidden)
            self.add_module(f"R{i}", one_rnn_node)
            self.rnn_nodes.append(one_rnn_node)
            cur_inp_dim = n_hidden
            # drop on output combined size
            if conf.no_drop:
                one_dnode = lambda x: x
            else:
                one_dnode = DropoutNode(conf.out_drop, osize=n_hidden)
                self.add_module(f"D{i}", one_dnode)
            self.drop_nodes.append(one_dnode)
        self.output_dim = n_hidden if conf.n_layers>0 else n_input

    def extra_repr(self) -> str:
        conf: RnnDecConf = self.conf
        return f"RNNDec(I={conf.isize},N={conf.osize},L={conf.n_layers})"

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    # init a cache with init hidden state
    # input should be [*, D], not extra step dim! ignore init_mask for RNN since RNN will handle mask ...
    def go_init(self, init_hs: Union[List[BK.Expr], BK.Expr] = None, init_mask: BK.Expr = None, **kwargs):
        conf: RnnDecConf = self.conf
        # --
        if init_hs is None:
            return RnnDecCache(None)
        # --
        n_layers = conf.n_layers
        if isinstance(init_hs, BK.Expr):  # simply reuse it for all layers
            init_hiddens = [init_hs] * n_layers
        else:
            init_hiddens = list(init_hs)
            assert len(init_hiddens) == n_layers, "Wrong number of input layer for RnnDec.go_init!"
        # --
        init_states = [node.make_init(h) for node, h in zip(self.rnn_nodes, init_hiddens)]
        cache = RnnDecCache(init_states)
        return cache

    # continue on cache, the inputs/outputs both have a step dim, even for one step!
    # here input should be [*, step, D]
    def go_feed(self, cache: RnnDecCache, input_expr: BK.Expr, mask_expr: BK.Expr = None):
        conf: RnnDecConf = self.conf
        step_dim = conf.step_dim
        # --
        step_inputs, step_masks = prepare_step_inputs(input_expr, mask_expr, step_dim, False)  # List[Expr]
        n_steps, n_layers = len(step_inputs), conf.n_layers
        # prepare
        all_final_outputs = []  # List(step) of Final_layer output
        all_step_states = []
        if cache.states is None:  # todo(note): make an empty start according to current inputs' shape!
            pre_sizes = BK.get_shape(input_expr)[:step_dim]
            prev_step_states = [n.make_zero_init(pre_sizes) for n in self.rnn_nodes]  # make empty ones
        else:
            prev_step_states = cache.states  # previous state
        # for each step
        for step_idx in range(n_steps):
            new_step_states = []
            cur_input = step_inputs[step_idx]  # [*, isize] or later [*, hidden]
            cur_mask = step_masks[step_idx]  # [*]
            # for each layer
            for layer_idx in range(n_layers):
                # d_node = self.drop_nodes[layer_idx]  # only used in layer-iteration for outputting for vstate
                f_node = self.rnn_nodes[layer_idx]
                # call rnn cell
                one_hid = f_node(cur_input, prev_step_states[layer_idx], cur_mask)  # *[*, hid]
                new_step_states.append(one_hid)  # append for states
                cur_input = one_hid[0]  # idx=0 is the hidden!
            all_final_outputs.append(cur_input)
            # update for step
            all_step_states.append(new_step_states)
            prev_step_states = new_step_states
        cache.update(prev_step_states)  # update cache
        # finally
        if n_layers == 0:
            return input_expr  # simply return inputs
        else:
            stacked_res = BK.stack(all_final_outputs, step_dim)
            dropped_res = self.drop_nodes[-1](stacked_res)
            return dropped_res

# =====
# Transformer based ones

class TransformerDecConf(DecConf):
    def __init__(self):
        super().__init__()
        # --
        self.n_layers = 4
        # self.layered_ranges = []  # cutoff_range for each layer
        self.use_posi = False  # add posi embeddings at input?
        self.pconf = PosiEmbeddingConf().direct_update(min_val=0)
        # for each layer
        self.d_ff = 512  # dim of FF, 0 for skipping ff
        self.aconf = AttentionConf()
        self.wconf = WrapperConf().direct_update(strategy="addnorm", act="tanh")
        self.ff_act = "gelu"

    @property
    def d_model(self):
        assert self.isize == self.osize
        return self.isize

class _OneTSFDecNode(BasicNode):
    def __init__(self, conf: TransformerDecConf):
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

    def forward(self, q_expr: BK.Expr, kv_expr: BK.Expr, mask_k: BK.Expr, mask_qk: BK.Expr, **kwargs):
        context = self.self_att(q_expr, kv_expr, kv_expr, mask_k=mask_k, mask_qk=mask_qk, **kwargs)
        output = self.feed_forward(context)
        return output

class TransformerDecCache(DecCache):
    def __init__(self, states: List[BK.Expr], mask: BK.Expr=None, cum_state_layers: Iterable[int]=()):
        # List_layer of [*, accumulated_num_steps, D], mask=[*, step] (same for all layers)
        # todo(note): need to record all hiddens for self-att!
        self.states: List[BK.Expr] = None  # [*, step, D]
        self.steps = 0
        self.mask: BK.Expr = None  # [*, step]
        self.positions: BK.Expr = None  # int[*, step], starting from 0
        # --
        self._cur_layer_idx = -1  # status for adding, -1: close, 0+: expecting layer ?
        self.cum_state_lset = set(cum_state_layers)  # which layers to accumulate state, usually -1 if needed!
        assert all(z>=0 for z in self.cum_state_lset), "Must provide positive layer idx here!!"
        self._arange2_t: BK.Expr = None  # [*, 1]
        self._arange_sel_t: BK.Expr = None  # [*, new_step]
        # --
        if states is not None:
            self.update(states, mask)

    # --
    # need to split into several steps

    # step 0: open one transaction for new steps
    def s0_open_new_steps(self, bsize: int, ssize: int, mask: BK.Expr = None):
        assert ssize > 0
        assert self._cur_layer_idx == -1
        self._cur_layer_idx = 0
        # --
        new_mask = BK.constants([bsize, ssize], 1.) if mask is None else mask  # [*, ssize]
        # --
        # prepare for store_lstate selecting
        if len(self.cum_state_lset) > 0:  # any layer need to accumulat?
            self._arange2_t = BK.arange_idx(bsize).unsqueeze(-1)  # [bsize, 1]
            # note: if no last state, simply clamp 0, otherwise, offset by 1 since we will concat later
            self._arange_sel_t = mask2posi_padded(new_mask, 0, 0) if mask is None else mask2posi_padded(new_mask, 1, 0)
        # prev_steps = self.steps  # previous accumulated steps
        self.steps += ssize
        self.mask = new_mask if self.mask is None else BK.concat([self.mask, new_mask], 1)  # [*, old+new]
        self.positions = mask2posi(self.mask, offset=-1, cmin=0)  # [*, old+new], recalculate!!

    # step 1*: feed new layer
    def ss_add_new_layer(self, layer_idx: int, expr: BK.Expr):
        assert layer_idx == self._cur_layer_idx
        self._cur_layer_idx += 1
        # --
        if self.states is None:
            self.states = []
        _cur_cum_state = (layer_idx in self.cum_state_lset)
        added_expr = expr
        if len(self.states) == layer_idx:  # first group of calls
            if _cur_cum_state:  # direct select!
                added_expr = added_expr[self._arange2_t, self._arange_sel_t]
            self.states.append(added_expr)  # directly add as first-time adding
        else:
            prev_state_all = self.states[layer_idx]  # [bsize, old_step]
            if _cur_cum_state:  # concat last and select
                added_expr = BK.concat([prev_state_all[:, -1].unsqueeze(1), added_expr], 1)[self._arange2_t, self._arange_sel_t]
            self.states[layer_idx] = BK.concat([prev_state_all, added_expr], 1)  # [*, old+new, D]
        return added_expr, self.states[layer_idx]  # q; kv

    # step final: close up, simply checking!
    def sz_close(self):
        # clear things
        assert self._cur_layer_idx == len(self.states)
        self._cur_layer_idx = -1
        self._arange2_t = None  # [*, 1]
        self._arange_sel_t = None  # [*, new_step]

    # finish all in one call!
    def update(self, states: List[BK.Expr], mask: BK.Expr = None):
        bsize, ssize = BK.get_shape(states[0])[:2]
        self.s0_open_new_steps(bsize, ssize, mask)
        for i, s in enumerate(states):
            self.ss_add_new_layer(i, s)
        self.sz_close()

    # need init with store_last_state
    def get_last_state(self, layer_idx=-1):
        if layer_idx < 0:
            layer_idx = len(self.states) + layer_idx
        assert layer_idx in self.cum_state_lset, "LastState can be invalid if not accumulating!"
        return self.states[layer_idx][:, -1]  # return last one, need accu to make it valid!

    # arrange them
    def _arrange_idxes(self, idxes: BK.Expr):
        if self.states is not None:
            self.states = [s.index_select(0, idxes) for s in self.states]
            # note: direct ones are updated by arrange_idxes
            # self.mask = self.mask.index_select(0, idxes)
            # self.positions = self.positions.index_select(0, idxes)

@node_reg(TransformerDecConf)
class TransformerDecNode(DecNode):
    def __init__(self, conf: TransformerDecConf, **kwargs):
        super().__init__(conf, **kwargs)
        # --
        conf: TransformerDecConf = self.conf
        assert conf.step_dim == 1, "Transformer assumes [bsize, len, D]!!"
        self.tnodes = []
        for i in range(conf.n_layers):
            one_node = _OneTSFDecNode(conf)
            self.add_module(f"T{i}", one_node)
            self.tnodes.append(one_node)
        # add posi embeddings
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
        return (self.conf.d_model,)

    def extra_repr(self) -> str:
        return f"Transformer(D={self.conf.d_model},L={self.conf.n_layers})"

    # init a cache with init hidden state
    # input should be [*, D], not extra step dim!
    # note: require 'store_lstate_layers' (usually -1) for incremental decoding!
    def go_init(self, init_hs: Union[List[BK.Expr], BK.Expr] = None, init_mask: BK.Expr = None,
                cum_state_layers: Iterable[int]=(), **kwargs):
        conf: TransformerDecConf = self.conf
        # --
        cache_n_layers = conf.n_layers + 1  # also store the input ones!!
        if init_hs is None:  # make zero starts!
            init_states = None
        else:
            # todo(note): unsqueeze here to make step=1!
            if isinstance(init_hs, BK.Expr):  # simply reuse it for all layers
                init_hiddens = [init_hs.unsqueeze(1)] * cache_n_layers
            else:
                init_hiddens = [h.unsqueeze(1) for h in init_hs]
                assert len(init_hiddens) == cache_n_layers, "Wrong number of input layer for RnnDec.go_init!"
            init_states = init_hiddens
        # --
        cum_state_layers = [(z if z>=0 else (cache_n_layers+z)) for z in cum_state_layers]  # translate NEG ones!
        cache = TransformerDecCache(init_states, init_mask, cum_state_layers=cum_state_layers)
        return cache

    # continue on cache, the inputs/outputs both have a step dim, even for one step!
    # here input should be [*, step, D]
    # todo(+N): mask needs to setup 'cum_state_layers' to the output (unlike RNN)!!
    def go_feed(self, cache: TransformerDecCache, input_expr: BK.Expr, mask_expr: BK.Expr = None):
        conf: TransformerConf = self.conf
        n_layers = conf.n_layers
        # --
        if n_layers == 0:  # only add the input
            cache.update([input_expr], mask_expr)  # one call of update is enough
            return input_expr  # change nothing
        # --
        # first prepare inputs
        input_shape = BK.get_shape(input_expr)  # [bsize, ssize, D]
        bsize, ssize = input_shape[:2]
        cache.s0_open_new_steps(bsize, ssize, mask_expr)  # open one
        # --
        if conf.use_posi:
            positions = cache.positions[:, -ssize:]  # [*, step]
            posi_embed = self.PE(positions)
            input_emb = self.input_f(input_expr+posi_embed)  # add absolute positional embeddings
        else:
            input_emb = self.input_f(input_expr)  # process input, [*, step, D]
        cur_q, cur_kv = cache.ss_add_new_layer(0, input_emb)
        # --
        # prepare rposi and casual-mask
        all_posi = cache.positions  # [*, old+new]
        rposi = all_posi[:, -ssize:].unsqueeze(-1) - all_posi.unsqueeze(-2)  # Q-KV [*, new(query), new+old(kv)]
        mask_qk = (rposi>=0).float()  # q must be later than kv
        # go!
        for ti, tnode in enumerate(self.tnodes):
            cur_q = tnode(cur_q, cur_kv, cache.mask, mask_qk, rposi=rposi)
            cur_q, cur_kv = cache.ss_add_new_layer(ti+1, cur_q)  # kv for next layer
        cache.sz_close()
        return cur_q  # [*, ssize, D]
