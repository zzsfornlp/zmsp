#

# basic layers

__all__ = [
    "NnConf", "NnLayer", "ActivationHelper",
]

from typing import Union, Callable, List
from collections import OrderedDict
import numpy as np
from mspx.utils import Conf, ZHelper, zlog, Registrable, Configurable
from ..backends import BK

# --
@Registrable.rd('N')
class NnConf(Conf):
    @classmethod
    def get_base_conf_type(cls): return NnConf
    @classmethod
    def get_base_node_type(cls): return NnLayer

    def get_sv_confs(self):
        from mspx.proc.run import SVConf
        return {k: v for k,v in self.__dict__.items() if isinstance(v, SVConf)}

# note: now NnLayer and nn.Module can be mixed in any way (NnLayer simply slightly add some more functions)
# note: special protocol, NO indirect store/use of comps (for example, [self.nodes]) to allow easier param-sharing
@Registrable.rd('_N')
class NnLayer(Configurable, BK.Module):
    def __init__(self, conf: NnConf, **kwargs):
        super().__init__(conf, **kwargs)
        BK.Module.__init__(self)  # note: extraly init Module!
        conf: NnConf = self.conf
        self.output_dim = None
        # scheduled values
        from mspx.proc.run import ScheduledValue
        self.svs = {k: ScheduledValue(v, name=k) for k, v in conf.get_sv_confs().items()}
        # --

    # count number of parameters
    def count_param_number(self, recurse=True):
        count = 0
        list_params = self.parameters(recurse=recurse)
        for p in list_params:
            count += np.prod(BK.get_shape(p))
        return int(count)

    # do not go through Model.__setattr__, thus will not add it!
    def setattr_borrow(self, key: str, value: object, assert_nonexist=True):
        if assert_nonexist:
            assert not hasattr(self, key)
        object.__setattr__(self, key, value)

    # the overall one
    def get_values(self, fn: Union[str, Callable]):
        if isinstance(fn, str):
            _tmp = fn
            fn = lambda x: getattr(x, _tmp)()
        # --
        ret = OrderedDict()
        for cname, m in self.named_children():
            if isinstance(m, NnLayer):
                ZHelper.update_dict(ret, m.get_values(fn), key_prefix=cname+".")
        ZHelper.update_dict(ret, fn(self), key_prefix="")  # finally self!
        return ret

    # note: override for brief printing
    OVERALL_BRIEF = True
    def brief_repr(self) -> str: return f"{self.__class__.__name__}[{self.count_param_number()}]"
    def __repr__(self): return self.brief_repr() if NnLayer.OVERALL_BRIEF else super().__repr__()

    def full_repr(self):
        old_value = NnLayer.OVERALL_BRIEF
        NnLayer.OVERALL_BRIEF = False   # temporary setting!
        ret = repr(self)
        NnLayer.OVERALL_BRIEF = old_value
        return ret

    def load(self, *args, **kwargs): return BK.load_mod(self, *args, **kwargs)
    def save(self, *args, **kwargs): return BK.save_mod(self, *args, **kwargs)

    def get_scheduled_values(self):
        return self.get_values('_scheduled_values')

    # note: to be implemented, by default nothing, only for special ones
    def _scheduled_values(self):
        ret = {}
        ret.update(self.svs)
        return ret

    # todo(note): some may not have this info, but does not matter
    # *input_dims: [-1_dims(Iterable), -2_dims, -3_dims, ...], usually no or last-dim will be fine
    # return iter of (dim-1, dim-2, ...)
    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    # note: forward!
    def forward(self, *inputs, **kwargs):
        raise NotImplementedError("Not implemented __call__ for BasicNode!")

# --
class ActivationHelper:
    ACTIVATIONS = {"tanh": BK.tanh, "relu": BK.relu, "elu": BK.elu, "gelu": BK.gelu,
                   "sigmoid": BK.sigmoid, "linear": (lambda x:x), "softmax": (lambda x,d=-1: x.softmax(d))}
    # reduction for seq after conv
    POOLINGS = {"max": (lambda x,d: BK.max_d(x, d)[0]), "min": (lambda x,d: BK.min_d(x, d)[0]),
                "avg": (lambda x,d: BK.avg(x, d)), "none": (lambda x,d: x)}

    @staticmethod
    def get_act(name):
        return ActivationHelper.ACTIVATIONS[name]

    @staticmethod
    def get_pool(name):
        return ActivationHelper.POOLINGS[name]
# --

# --
# b mspx/nn/l3/base:
