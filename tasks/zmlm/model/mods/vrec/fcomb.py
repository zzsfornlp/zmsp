#

from typing import List, Union, Dict, Iterable
from msp.utils import Constants, Conf, zwarn, Helper
from msp.nn import BK
from msp.nn.layers import BasicNode, Affine, NoDropRop, PosiEmbedding2, ActivationHelper, Dropout
from .base import AffineHelperNode

# =====
# actually very similar to MattNode.c1_scorer

# conf
class FCombConf(Conf):
    def __init__(self):
        pass

# node
class FCombNode(BasicNode):
    def __init__(self, pc, dim_q, dim_k, dim_v, conf: FCombConf):
        super().__init__(pc, None, None)
        pass

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()
