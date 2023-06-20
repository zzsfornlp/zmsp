#

import os
from .common import NIConf, OptimConf

# decide which backend from ENV
_BK_NAME = os.environ.get("ZMSP_BK", "torch")
if _BK_NAME in {"torch", "pytorch"}:
    from . import bktr as BK
else:
    raise NotImplementedError(f"Unknown Backend {_BK_NAME}")
