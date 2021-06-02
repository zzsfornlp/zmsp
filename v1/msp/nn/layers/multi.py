#

from ..backends import BK
from .basic import BasicNode, ActivationHelper, NoDropRop, Dropout
from .ff import Affine, LayerNorm

from msp.utils import zcheck
import numpy as np

# containers and wrappers

class Sequential(BasicNode):
    def __init__(self, pc, node_iter, name=None, init_rop=None):
        super().__init__(pc, name, init_rop)
        #
        self.ns_ = []
        for node in node_iter:
            x = self.add_sub_node(node.name, node)
            self.ns_.append(x)

    def get_output_dims(self, *input_dims):
        cur_dims = input_dims
        for nn in self.ns_:
            cur_dims = nn.get_output_dims(*cur_dims)
        return cur_dims

    def __repr__(self):
        s = "# Sequential composed of %s nodes:\n" % len(self.ns_)
        s += "\n".join([str(nn) for nn in self.ns_])
        return s

    def __call__(self, input_exp, *args, **kwargs):
        x = input_exp
        for nn in self.ns_:
            x = nn(x, *args, **kwargs)
        return x

# no params
class Summer(BasicNode):
    def __init__(self, pc, name=None, init_rop=None):
        super().__init__(pc, name, init_rop)

    # NO dropouts!!
    def __call__(self, input_list):
        # todo(warn): not that efficient!
        ret = input_list[0]
        for one in input_list[1:]:
            ret = ret + one
        return ret

    def get_output_dims(self, *input_dims):
        xs = input_dims[0]      # -1 dimension
        out = max(xs)
        zcheck(all((one==out or one==1) for one in input_dims), "Should sum with same-dim tensors or broadcastable!")
        return (out, )

class Concater(BasicNode):
    def __init__(self, pc, name=None, init_rop=None):
        super().__init__(pc, name, init_rop)

    # NO dropouts!!
    def __call__(self, input_list):
        return BK.concat(input_list, -1)

    def get_output_dims(self, *input_dims):
        xs = input_dims[0]
        return sum(xs)

# sum the inputs with gates
class GatedMixer(BasicNode):
    def __init__(self, pc, dim, num_input, name=None, init_rop=None):
        super().__init__(pc, name, init_rop)
        self.dim = dim
        self.num_input = num_input
        #
        self.ff = self.add_sub_node("g", Affine(pc, dim*num_input, dim*num_input, act="sigmoid"))
        self.ff2 = self.add_sub_node("f", Affine(pc, dim*num_input, dim, act="tanh"))

    def __call__(self, input_list):
        concat_input_t = BK.concat(input_list, -1)
        gates_t = self.ff(concat_input_t)
        final_output_t = self.ff2(concat_input_t * gates_t)
        # output_shape = BK.get_shape(concat_output_t)[:-1] + [self.num_input, -1]
        # # [*, num, D] ->(avg)-> [*, D]
        # return BK.avg(concat_output_t.view(output_shape), dim=-2)
        return final_output_t

    def get_output_dims(self, *input_dims):
        return (self.dim, )

# input -> various outputs -> join
# -- join mode can be "cat", "add"
class Joiner(BasicNode):
    def __init__(self, pc, node_iter, mode="cat", name=None, init_rop=None):
        super().__init__(pc, name, init_rop)
        #
        self.ns_ = []
        for node in node_iter:
            x = self.add_sub_node(node.name, node)
            self.ns_.append(x)
        #
        self.mode = mode
        self.final_node = self.add_sub_node("f", {"cat": Concater, "add": Summer}[mode]())

    def get_output_dims(self, *input_dims):
        multi_dims = [nn.get_output_dims(*input_dims) for nn in self.ns_]
        # rearrange
        multi_dims2 = [z for z in zip(*multi_dims)]
        return self.final_node.get_output_dims(*multi_dims2)

    def __repr__(self):
        s = "# Joiner composed of %s nodes + final %s :\n" % (len(self.ns_), self.mode)
        s += "\n".join([str(nn) for nn in self.ns_])
        return s

    def __call__(self, input_exp, *args, **kwargs):
        multi_outs = [nn(input_exp, *args, **kwargs) for nn in self.ns_]
        x = self.final_node(multi_outs)
        return x

# =====
# wrappers

class NodeWrapper(BasicNode):
    def __init__(self, node, node_last_dims):
        super().__init__(node.pc, None, None)
        self.node = self.add_sub_node("z", node)
        self.input_ns = node_last_dims
        self.output_ns = self.node.get_output_dims(node_last_dims)      # only supporting last dim

    def get_output_dims(self, *input_dims):
        raise NotImplementedError()

# Adding with the first arg
class AddNormWrapper(NodeWrapper):
    def __init__(self, node, node_last_dims, std_no_grad=False):
        super().__init__(node, node_last_dims)
        self.size = self.input_ns[0]
        zcheck(self.size==self.output_ns[0], "AddNormWrapper meets unequal dims.")
        self.normer = self.add_sub_node("n", LayerNorm(self.pc, self.size, std_no_grad=std_no_grad))

    def get_output_dims(self, *input_dims):
        return (self.size, )

    def __call__(self, *args):
        hid = self.node(*args)
        r = self.normer(hid+args[0])    # the first one should be the input0
        return r

# Adding with the first and Activate
class AddActWrapper(NodeWrapper):
    def __init__(self, node, node_last_dims, act="tanh"):
        super().__init__(node, node_last_dims)
        self.size = self.input_ns[0]
        self.act_f = ActivationHelper.get_act(act)

    def get_output_dims(self, *input_dims):
        return (self.size, )

    def __call__(self, *args):
        hid = self.node(*args)
        r = self.act_f(hid+args[0])    # the first one should be the input0
        return r

# Highway Wrapper: also with the first arg
class HighWayWrapper(NodeWrapper):
    def __init__(self, node, node_last_dims):
        super().__init__(node, node_last_dims)
        self.size = self.input_ns[0]
        zcheck(self.size==self.output_ns[0], "AddNormWrapper meets unequal dims.")
        self.gate = self.add_sub_node("g", Affine(self.pc, self.size, self.size, act="linear"))

    def get_output_dims(self, *input_dims):
        return (self.size, )

    def __call__(self, *args):
        hid = self.node(*args)
        g = BK.sigmoid(self.gate(args[0]))
        r = hid*g + args[0]*(1.-g)
        return r

# shortcuts
# shortcur for MLP
def get_mlp(pc, n_ins, n_out, n_hidden, n_hidden_layer=1, hidden_act="tanh", final_act="linear", hidden_bias=True, final_bias=True, hidden_init_rop=None, final_init_rop=None, hidden_which_affine=2):
    layer_dims = [n_ins] + [n_hidden]*n_hidden_layer
    nodes = []
    for idx in range(n_hidden_layer):
        hidden_one = Affine(pc, layer_dims[idx], layer_dims[idx+1], act=hidden_act, bias=hidden_bias,
                            init_rop=hidden_init_rop, which_affine=hidden_which_affine)
        nodes.append(hidden_one)
    nodes.append(Affine(pc, layer_dims[-1], n_out, act=final_act, bias=final_bias, init_rop=final_init_rop))
    return Sequential(pc, nodes, name="mlp")

# version 2
def get_mlp2(pc, n_ins, n_out, n_hidden, n_hidden_layer, hidden_act="tanh", final_act="linear", hidden_bias=True, final_bias=True, hidden_dropout=0., final_dropout=0., hidden_which_affine=2):
    layer_dims = [n_hidden]*n_hidden_layer + [n_out]
    layer_drops = [hidden_dropout]*n_hidden_layer + [final_dropout]
    layer_acts = [hidden_act]*n_hidden_layer + [final_act]
    layer_biases = [hidden_bias]*n_hidden_layer + [final_bias]
    # -----
    nodes = []
    cur_dim = n_ins
    for idx in range(n_hidden_layer+1):
        hidden_one = Affine(pc, cur_dim, layer_dims[idx], act=layer_acts[idx], bias=layer_biases[idx],
                            init_rop=NoDropRop(), which_affine=hidden_which_affine)
        nodes.append(hidden_one)
        if layer_drops[idx] > 0.:
            dropout_one = Dropout(pc, (layer_dims[idx], ), fix_rate=layer_drops[idx])
            nodes.append(dropout_one)
        cur_dim = layer_dims[idx]
    return Sequential(pc, nodes, name="mlp")
