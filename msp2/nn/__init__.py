#

# for nn

from .backends import BK
from msp2.utils import zlog
from .backends import NIConf, OptimConf

# --
def init(cc: NIConf = None):
    if cc is not None:
        zlog(f"Updating NIConf with device={cc.device}.", func="config")
    BK.init(cc)

# lib-wise refresh
def refresh():
    # ExprWrapperPool.clear()
    BK.refresh()
