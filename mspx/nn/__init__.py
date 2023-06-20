#

# for nn

from .backends import BK, NIConf, OptimConf
from .layers import *
from .layers2 import *
from .mods import *

# --
def init(cc: NIConf = None, **kwargs):
    BK.init(cc, **kwargs)

# lib-wise refresh
def refresh():
    # ExprWrapperPool.clear()
    BK.refresh()
