#

# Basic Layer

__all__ = [
    "RefreshOptions", "ActivationHelper", "BasicConf", "BasicNode", "node_reg",
    "ModuleWrapperConf", "ModuleWrapper", "DropoutConf", "DropoutNode"
]

from typing import Tuple, List, Type, Dict
from collections import OrderedDict
import numpy as np
from ..backends import BK
from msp2.utils import Conf, get_class_id, DictHelper
from msp2.proc import ScheduledValue

# =====
# storing dynamic Node status
# -- only as default options, can be overridden by specific layer settings (which are usually fixed)
class RefreshOptions:
    def __init__(self, training=True, hdrop=0., idrop=0., gdrop=0., fix_drop=False, trainable=True):
        self.training = training
        # some of the most common ones
        self.hdrop = hdrop  # output drop (hidden-drop)
        self.idrop = idrop  # mainly for recurrent connections' inputs
        self.gdrop = gdrop  # var-drop for recurrent connections (input)
        self.fix_drop = fix_drop  # fixed dropout for each batch
        self.trainable = trainable  # trainable or not
        # --
        self._fix_set = set()     # name of the fields that are un-changeable

    # =====
    def add_to_fix_set(self, n: str):
        self._fix_set.add(n)

    def add_fixed_value(self, n: str, v):
        assert n in self.__dict__, "Strange fix-name "+n
        self.__dict__[n] = v
        self._fix_set.add(n)

    # update for the non-fixed values
    def update(self, rop: 'RefreshOptions'):
        for n in rop.__dict__:
            if n[0] != "_" and n not in self._fix_set:
                self.__dict__[n] = rop.__dict__[n]

# =====
# helpers

class ActivationHelper(object):
    ACTIVATIONS = {"tanh": BK.tanh, "relu": BK.relu, "elu": BK.elu, "gelu": BK.gelu,
                   "sigmoid": BK.sigmoid, "linear": (lambda x:x), "softmax": (lambda x,d=-1: x.softmax(d))}
    # reduction for seq after conv
    POOLINGS = {"max": (lambda x,d: BK.max(x, d)[0]), "min": (lambda x,d: BK.min(x, d)[0]),
                "avg": (lambda x,d: BK.avg(x, d)), "none": (lambda x,d: x)}

    @staticmethod
    def get_act(name):
        return ActivationHelper.ACTIVATIONS[name]

    @staticmethod
    def get_pool(name):
        return ActivationHelper.POOLINGS[name]

# =====
# todo(note): (conventions) some fields like isize/osize are mainly for ARGS use,
#  should not directly set with conf.update_*
class BasicConf(Conf):
    _CONF_MAP = {}
    _NODE_MAP = {}

    @staticmethod
    def get_conf_type(cls: Type, df):
        _k = get_class_id(cls, use_mod=True)
        return BasicConf._CONF_MAP.get(_k, df)

    @staticmethod
    def get_node_type(cls: Type, df):
        _k = get_class_id(cls, use_mod=True)
        return BasicConf._NODE_MAP.get(_k, df)

    def make_node(self, *args, **kwargs):
        cls = BasicConf.get_node_type(self.__class__, None)
        return cls(self, *args, **kwargs)

    def __init__(self):
        pass

# reg for both conf and node
def node_reg(conf_type: Type):
    def _f(cls: Type):
        # reg for conf
        _m = BasicConf._CONF_MAP
        _k = get_class_id(cls, use_mod=True)
        assert _k not in _m
        _m[_k] = conf_type
        # reg for node
        _m2 = BasicConf._NODE_MAP
        _k2 = get_class_id(conf_type, use_mod=True)
        if _k2 not in _m2:  # note: reg the first one!!
            _m2[_k2] = cls
        # --
        return cls
    return _f

# todo(note): now we give up and simply inherit from nn.Module!!
@node_reg(BasicConf)
class BasicNode(BK.Module):
    def __init__(self, conf: BasicConf, **kwargs):
        super().__init__()
        self.conf = self.setup_conf(conf, **kwargs)
        self.rop = RefreshOptions()  # simply make a local status

    # count number of parameters
    def count_param_number(self, recurse=True):
        count = 0
        list_params = self.parameters(recurse=recurse)
        for p in list_params:
            count += np.prod(BK.get_shape(p))
        return int(count)

    # save and load
    def save(self, path: str): BK.save_model(self, path)
    def load(self, path: str, strict=False): BK.load_model(self, path, strict)

    def setup_conf(self, conf: BasicConf, **kwargs):
        if conf is None:
            conf_type = BasicConf.get_conf_type(self.__class__, BasicConf)  # by default the basic one!
            conf = conf_type()
        else:  # todo(note): always copy to local!
            conf = conf.copy()
        conf.direct_update(_assert_exists=True, **kwargs)
        # conf._do_validate()
        conf.validate()  # note: here we do full validate to setup the private conf!
        return conf

    # override them as refresh!!
    def train(self, mode=True): self.refresh(RefreshOptions(training=mode))
    def eval(self): self.train(mode=False)

    def is_training(self):
        return self.rop.training

    def reset_params_and_modules(self):
        # first reset current params
        self.reset_parameters()
        # then for all sub-modules
        for m in self.children():
            if isinstance(m, BasicNode):
                m.reset_params_and_modules()
            # todo(note): ignore others

    # do not go through Model.__setattr__, thus will not add it!
    def setattr_borrow(self, key: str, value: object, assert_nonexist=True):
        if assert_nonexist:
            assert not hasattr(self, key)
        object.__setattr__(self, key, value)

    # =====
    # these can be overridden

    # todo(note): most need not override this default one!
    # rop==None means no change
    def refresh(self, rop: RefreshOptions = None):
        # update self
        if rop is not None:
            self.rop.update(rop)
        # --
        # todo(note): use self.rop instead of rop
        # also update underlying NN module; todo(+N): only at this level!!
        # refresh the contents
        # param
        cur_trainable = bool(self.rop.trainable)
        for p in self.parameters(recurse=False):  # sub-modules will be refreshed later
            if p.requires_grad != cur_trainable:
                p.requires_grad = cur_trainable
        self.training = self.is_training()  # change the one in pytorch, simply set it
        # sub-modules
        for m in self.children():
            if isinstance(m, BasicNode):  # sometimes the module can be nn.Module
                m.refresh(self.rop)
            elif isinstance(m, BK.Module):  # set training!
                assert isinstance(self, ModuleWrapper), "Error: Module gets included inside None-ModuleWrapper!"
                m.train(self.is_training())
            else:
                raise NotImplementedError()

    # todo(note): some may not have this info, but does not matter
    # *input_dims: [-1_dims(Iterable), -2_dims, -3_dims, ...], usually no or last-dim will be fine
    # return iter of (dim-1, dim-2, ...)
    def get_output_dims(self, *input_dims):
        raise NotImplementedError()

    # the overall one
    def get_scheduled_values(self) -> Dict[str, ScheduledValue]:
        ret = OrderedDict()
        for cname, m in self.named_children():
            if isinstance(m, BasicNode):
                DictHelper.update_dict(ret, m.get_scheduled_values(), key_prefix=cname+".")
        DictHelper.update_dict(ret, self._get_scheduled_values(), key_prefix="")
        return ret

    # note: to be implemented, by default nothing, only for special ones
    def _get_scheduled_values(self) -> Dict[str, ScheduledValue]:
        return OrderedDict()

    # todo(note): only reset for the current layer
    def reset_parameters(self):
        pass  # by default no params to reset!

    # todo(note): override for pretty printing
    def extra_repr(self) -> str:
        return f"{self.__class__.__name__}"

    # todo(note): everyone must do this!
    def forward(self, *input, **kwargs):
        raise NotImplementedError("Not implemented __call__ for BasicNode!")

# ModuleWrapper
class ModuleWrapperConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.no_reg = False  # do not register into the Module
        # self.no_reset = False  # do not reset

@node_reg(ModuleWrapperConf)
class ModuleWrapper(BasicNode):
    def __init__(self, node: BK.Module, conf: ModuleWrapperConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ModuleWrapperConf = self.conf
        # --
        assert isinstance(node, BK.Module) and not isinstance(node, BasicNode), f"Wrong type for ModuleWrapper: {type(node)}"
        if conf.no_reg:  # no register into modules
            self.setattr_borrow("node", node)
        else:
            self.node = node
            pm = BK.get_current_param_manager()
            for p in self.node.parameters():  # record the extra ones
                pm.record_param(p, f"{self}/{node}")

    def forward(self, *input, **kwargs):
        return self.node.forward(*input, **kwargs)

# =====
# commonly used Nodes
class DropoutConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.osize = -1  # output dim: only need when fix_drop
        # --
        # None as fallback to default
        self.drop_rate: float = None  # dropout rate
        self.fix_drop: bool = None  # whether fix drop mask
        self.drop_dim = -1  # dropout for which dim, should be NEG
        self.which_drop = "hdrop"  # which drop to look for when finding default

    @classmethod
    def _get_type_hints(cls):
        return {"drop_rate": float, "fix_drop": bool}

@node_reg(DropoutConf)
class DropoutNode(BasicNode):
    def __init__(self, conf: DropoutConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: DropoutConf = self.conf
        # --
        assert conf.drop_dim < 0
        self._f = None  # actual running function: to be refreshed
        # --
        self.default_drop_getter = {
            "hdrop": lambda x: x.hdrop, "gdrop": lambda x: x.gdrop, "idrop": lambda x: x.idrop}[conf.which_drop]
        self.is_gdrop = (conf.which_drop == "gdrop")

    # useful routines
    @staticmethod
    def _dropout_f_obtain(rate: float, fix_drop: bool, fixed_size: int, dim: int):
        if rate <= 0.:
            return lambda x: x
        # --
        def _get_mask():
            _mask = BK.random_bernoulli((fixed_size,), 1. - rate, 1. / (1. - rate))
            _mask = _mask.reshape(BK.get_shape(_mask) + [1] * (-dim-1))  # [?, 1, ..., 1]
            return _mask
        # --
        if not fix_drop:
            if dim != -1:  # called inside
                return lambda x: x * _get_mask()
            else:  # directly use the plain one!
                return lambda x: BK.dropout(x, rate)
        else:  # called at outside, fix dropout after each refresh
            cur_mask = _get_mask()
            return lambda x: x * cur_mask

    # special refresh!!
    def refresh(self, rop=None):
        super().refresh(rop)
        #
        conf: DropoutConf = self.conf
        r = self.rop
        if not r.training:  # todo(note): an overall switch
            self._f = lambda x: x
            return  # no need for other setting
        # --
        # either from self.conf or fallback to rop
        drop_rate = conf.drop_rate if conf.drop_rate is not None else self.default_drop_getter(r)
        fix_drop = conf.fix_drop if conf.fix_drop is not None else r.fix_drop
        # todo(note): special one, enforce fix_drop for gdrop
        fix_drop = (fix_drop or self.is_gdrop)
        assert conf.drop_dim < 0
        self._f = DropoutNode._dropout_f_obtain(drop_rate, fix_drop, conf.osize, conf.drop_dim)

    def extra_repr(self) -> str:
        conf: DropoutConf = self.conf
        return f"Dropout({conf.drop_rate},{conf.which_drop})"

    def forward(self, val, **kwargs):
        return self._f(val)

    def get_output_dims(self, *input_dims):
        return input_dims  # the same as input_dims
