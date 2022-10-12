#

# Base Layer

__all__ = [
    'ActivationHelper', 'node_reg', 'ZlayerConf', 'Zlayer',
]

from typing import List, Dict
from collections import OrderedDict
import numpy as np
from msp2.utils import Conf, DictHelper
from msp2.proc import ScheduledValue
from ..backends import BK

# simply use those from previous if they are good!
from ..layers import ActivationHelper, node_reg
from ..layers import BasicConf as ZlayerConf

# --
# new base layer!
@node_reg(ZlayerConf)
class Zlayer(BK.Module):
    def __init__(self, conf: ZlayerConf, **kwargs):
        super().__init__()
        self.conf = self.setup_conf(conf, **kwargs)
        # --

    # setup conf
    def setup_conf(self, conf: ZlayerConf, **kwargs):
        if conf is None:  # use a default one!
            conf_type = ZlayerConf.get_conf_type(self.__class__, ZlayerConf)  # by default the basic one!
            conf = conf_type()
        else:  # todo(note): always copy to local!
            conf = conf.copy()
        conf.direct_update(**kwargs)
        # conf._do_validate()
        conf.validate()  # note: here we do full validate to setup the private conf!
        return conf

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
    def get_scheduled_values(self) -> Dict[str, ScheduledValue]:
        ret = OrderedDict()
        for cname, m in self.named_children():
            if isinstance(m, Zlayer):
                DictHelper.update_dict(ret, m.get_scheduled_values(), key_prefix=cname+".")
        DictHelper.update_dict(ret, self._get_scheduled_values(), key_prefix="")
        return ret

    # note: to be implemented, by default nothing, only for special ones
    def _get_scheduled_values(self) -> Dict[str, ScheduledValue]:
        return OrderedDict()

    # todo(note): override for pretty printing
    def extra_repr(self) -> str:
        return f"{self.__class__.__name__}"

    def setup(self, *args, **kwargs):
        pass  # by default, nothing to setup
