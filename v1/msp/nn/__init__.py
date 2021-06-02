#

# dynamic nn module
# from the outside, view seqs as list(step) of (batch-size, dim)
# thus, mainly dealing with tensors with dim<=2

"""
Things to notice:
1. col or row major?
For parameters, use Column-Major (output-size, ..., input-size) will be fine since both pytorch & dynet adopt this,
But for values, they seem to disagree.
=> Nevertheless, use Row-Major: like (batch-size, step-size, ..., embed-size), reverse & special-dealing for dynet
2. value wrapper?
Needed for manually batching, simply as a pointer+batch-idx.
=> in fact, providing a pythonised-extension of the shape for tensor, list & dict can be wrapped at the outside.
"""

from .backends import BK
from .backends.common import COMMON_CONFIG, NIConf
from .expr import ExprWrapperPool, ExprWrapper, SliceManager, SlicedExpr

from msp.utils import zlog

def init(cc=None):
    # set up common cc (cannot be changed later!!)
    if cc is None:
        pass
    elif isinstance(cc, NIConf):
        zlog("Updating NIConf with device=%s." % cc.device, func="config")
        COMMON_CONFIG.update_from_conf(cc)
    elif isinstance(cc, dict):
        COMMON_CONFIG.update_from_v(cc)
    else:
        COMMON_CONFIG.update_from_s(cc)
    #
    BK.init()

# lib-wise refresh
def refresh():
    ExprWrapperPool.clear()
    BK.refresh()
