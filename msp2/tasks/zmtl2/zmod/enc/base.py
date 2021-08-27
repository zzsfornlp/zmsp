#

# the base of enc

__all__ = [
    "ZTaskEncConf", "ZTaskEnc", "ZEncoderConf", "ZEncoder",
]

from typing import List
from msp2.nn.layers import *
from msp2.nn import BK
from ...core import ZMod, ZModConf, ZTask, ZTaskConf

# =====
# the overall encoder

class ZTaskEncConf(ZTaskConf):
    def __init__(self):
        super().__init__()
        self.name = "enc"
        # --

class ZTaskEnc(ZTask):
    def __init__(self, conf: ZTaskConf):
        super().__init__(conf)
        # --

class ZEncoderConf(ZModConf):
    def __init__(self):
        super().__init__()
        # --

@node_reg(ZEncoderConf)
class ZEncoder(ZMod):
    def __init__(self, conf: ZEncoderConf, ztask, **kwargs):
        super().__init__(conf, ztask, **kwargs)
        # --

    # info
    def get_enc_dim(self) -> int: raise NotImplementedError()
    def get_head_dim(self) -> int: raise NotImplementedError()
    def get_embed_w(self): raise NotImplementedError()

    # prepare and restart an ibatch
    def restart(self, ibatch, med):
        raise NotImplementedError()
