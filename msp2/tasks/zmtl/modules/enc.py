#

# the base of enc

__all__ = [
    "ZEncoderConf", "ZEncoder",
]

from typing import List
from msp2.nn.layers import *
from msp2.nn import BK

# =====
# the overall encoder

class ZEncoderConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --

@node_reg(ZEncoderConf)
class ZEncoder(BasicNode):
    def __init__(self, conf: ZEncoderConf, **kwargs):
        super().__init__(conf, **kwargs)
        # --

    # info
    def get_enc_dim(self) -> int: raise NotImplementedError()
    def get_head_dim(self) -> int: raise NotImplementedError()
