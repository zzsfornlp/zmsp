#

# the labeling component, managing loss functions and outputing as middle layer (as in attention)
# this module itself usually does not contain parameters, mostly based on outside scores

from msp.utils import Conf, zfatal, zcheck
from msp.nn import BK, layers
from msp.nn.layers import BasicNode

class LabConf(Conf):
    def __init__(self):
        self._label_dim = -1  # to be filled
        self.binary_mode = False  # does not compare among last dim, but treat each one as its own against 0

# todo(+N): currently only take care of single-class gold, there are also situations where we need multi-labels
class MyLabeler(BasicNode):
    def __init__(self, pc: BK.ParamCollection, lconf: LabConf):
        super().__init__(pc, None, None)
        self.conf = lconf

    # [*, D] or [*]
    def __call__(self, raw_scores, gold_idxes, **kwargs):
        # TODO(!)
        pass
