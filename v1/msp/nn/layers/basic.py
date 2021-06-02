#

from ..backends import BK
from ..backends.common import get_unique_name
import numpy as np

from msp.utils import extract_stack

# with special None-Rule for turning off refresh
class RefreshOptions(object):
    def __init__(self, training=True, hdrop=0., idrop=0., gdrop=0., edrop=0., dropmd=0., fix_drop=False, trainable=True, fix_set=()):
        self.training = training
        self.hdrop = hdrop     # output drop
        self.idrop = idrop     # mainly for recurrent connections' inputs
        self.gdrop = gdrop     # var-drop for recurrent connections (input)
        self.edrop = edrop     # var-drop for embedding (whole input one, should fix_row0)
        self.dropmd = dropmd    # Ld dropout: dropout last multiple dimensions
        self.fix_drop = fix_drop       # fixed dropout for each batch
        self.trainable = trainable
        #
        self._fix_set = set(fix_set)     # name of the fields that are un-changeable

    #
    def add_to_fix_set(self, n):
        self._fix_set.add(n)

    def add_fixed_value(self, n, v):
        assert n in self.__dict__, "Strange fix-name "+n
        self.__dict__[n] = v
        self._fix_set.add(n)

    # update for the non-fixed values
    def update(self, rop):
        for n in rop.__dict__:
            if n[0] != "_" and n not in self._fix_set:
                self.__dict__[n] = rop.__dict__[n]

def NoDropRop(): return RefreshOptions(hdrop=0., idrop=0., gdrop=0., edrop=0., fix_set=("hdrop", "idrop", "gdrop", "edrop"))
def FreezeRop(): return RefreshOptions(trainable=False, fix_set=("trainable", ))
def NoFixRop(): return RefreshOptions(fix_drop=False, fix_set=("fix_drop", ))

# helpers
class ActivationHelper(object):
    ACTIVATIONS = {"tanh": BK.tanh, "softmax": BK.softmax, "relu": BK.relu, "elu": BK.elu, "gelu": BK.gelu,
                   "sigmoid": BK.sigmoid, "linear": lambda x:x}
    # reduction for seq after conv
    POOLINGS = {"max": lambda x: BK.max(x, -2)[0], "avg": lambda x: BK.avg(x, -2)}

    @staticmethod
    def get_act(name):
        return ActivationHelper.ACTIVATIONS[name]

    # dim -= 1
    @staticmethod
    def get_pool(name):
        return ActivationHelper.POOLINGS[name]

#
# three time points: build(__init__), before-run(refresh), real-run(run)
# Node can contains the params or sub-nodes
class BasicNode(object):
    def __init__(self, pc, name, init_rop):
        # basic
        self.pc = pc
        if name is None:
            name = self.__class__.__name__
        self.name = self.pc.get_unique_name(name)
        # params and sub-nodes
        self.params = {}
        self.sub_nodes = {}      # node.parent == self
        self.temp_nodes = {}     # otherwise
        self.name_dict = {}      # str -> idx
        # fixed options: some can be changed by refresh ROP by specific rules
        if init_rop is None:
            init_rop = RefreshOptions()    # default one
        self.rop = init_rop
        #
        self.parent = None      # only one parent who owns this node (controlling refresh)!
        self.pc.nnc_push(self.name)

    def get_unique_name(self, name):
        return get_unique_name(self.name_dict, name)

    def __repr__(self):
        return "# Node(%s): %s" % (self.__class__.__name__, self.name)

    # rop==None means no change
    def refresh(self, rop=None):
        # update self
        if rop is not None:
            self.rop.update(rop)
        # refresh the contents
        # param
        cur_trainable = self.rop.trainable
        for n, v in self.params.items():
            self.pc.param_set_trainable(v, cur_trainable)
        # sub-nodes
        # todo(warn): use self.rop instead of rop
        for n, v in self.sub_nodes.items():
            v.refresh(self.rop)
        # temp-nodes?
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("No calling __call__ from Virtual Nodes: %s." % (str(self),))

    # *input_dims: [-1_dims(Iterable), -2_dims, -3_dims, ...], usually no or last-dim will be fine
    # return iter of (dim-1, dim-2, ...)
    def get_output_dims(self, *input_dims):
        return input_dims

    # create param from PC
    def add_param(self, name, shape, init=None, lookup=False, check_stack=True, out_p4i=1, scale=1.):
        if init is None:
            init = "default"
        # -----
        if isinstance(init, str):
            w = BK.get_params_init(shape, init, lookup, out_p4i, scale)
        else:
            w = init
        name = self.get_unique_name(name)
        combined_name = self.pc.nnc_name(self.name, check_stack) + "/" + name
        ret = self.pc.param_new(combined_name, shape, w, lookup)
        self.params[name] = ret
        return ret

    # only recording the sub-node, which is built previously
    def add_sub_node(self, name, node):
        self.pc.nnc_pop(node.name)
        assert isinstance(node, BasicNode), "Subnodes should be a Node!"
        name = self.get_unique_name(name)
        if node.parent is None:
            node.parent = self
            self.sub_nodes[name] = node
        else:
            self.temp_nodes[name] = node
        return node

    # get params recursively (owned subnodes)
    def get_parameters(self, recursively=True):
        ret = list(self.params.values())
        if recursively:
            for node in self.sub_nodes.values():
                ret.extend(node.get_parameters(recursively))
        return ret

    # count number of parameters
    def count_allsize_parameters(self, recursively=True):
        count = 0
        list_params = self.get_parameters(recursively)
        for p in list_params:
            count += np.prod(BK.get_shape(p))
        return int(count)

# commonly used Nodes
class Dropout(BasicNode):
    def __init__(self, pc, shape, which_drop="hdrop", name=None, init_rop=None, fix_rate=None):
        super().__init__(pc, name, init_rop)
        self.f_ = None
        self.shape = shape
        #
        self.which_drop = which_drop
        # edrop is not performed here!!
        self.drop_getter_ = {"hdrop": lambda x: x.hdrop, "gdrop": lambda x: x.gdrop, "idrop": lambda x: x.idrop}[which_drop]
        if which_drop == "gdrop":
            self.rop.fix_drop = True
            self.rop.add_to_fix_set("fix_drop")
            assert fix_rate is None
        #
        self.fix_rate = fix_rate

    def refresh(self, rop=None):
        super().refresh(rop)
        #
        r = self.rop
        if self.fix_rate is not None:
            drop = self.fix_rate
        else:
            drop = self.drop_getter_(r)
        # todo(+3): another overall switch, not quite elegant!
        if not r.training:
            self.f_ = lambda x: x
        else:
            self.f_ = Dropout._dropout_f_obtain(r.fix_drop, drop, self.shape)

    def __call__(self, val):
        return self.f_(val)

    # useful routines
    @staticmethod
    def _dropout_f_obtain(fix_drop, dropout, shape):
        if dropout <= 0.:
            return lambda x: x
        elif not fix_drop:
            return lambda x: BK.dropout(x, dropout)
        else:
            # fix dropout after each refresh
            cur_mask = BK.random_bernoulli(shape, 1.-dropout, 1./(1.-dropout))
            return lambda x: BK.cmult(x, cur_mask)

# dropout the entire of last-N dim, like in dropout2d/3d/...
# (dropout like Dropout, no fix_drop since that might be not intuitive)
class DropoutLastN(BasicNode):
    def __init__(self, pc, lastn=1, name=None, init_rop=None):
        super().__init__(pc, name, init_rop)
        self.lastn = lastn
        self.rate = 0.

    def refresh(self, rop=None):
        super().refresh(rop)
        #
        self.rate = self.rop.dropmd

    def __call__(self, val):
        dropout = self.rate
        if dropout <= 0.:
            return val
        else:
            val_shape = BK.get_shape(val)
            for i in range(self.lastn):
                val_shape[-i-1] = 1
            cur_mask = BK.random_bernoulli(val_shape, 1.-dropout, 1./(1.-dropout))
            val_drop = val * cur_mask
            return val_drop
