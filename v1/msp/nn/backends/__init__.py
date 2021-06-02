#

import os

# decide which backend from ENV
_BK_NAME = os.environ.get("BK", "torch")
if str.lower(_BK_NAME) in {"dy", "dynet", "cnn"}:
    from . import bkdy as BK
else:
    from . import bktr as BK
