#

# especially CRF-styled inference
# -- individual scores + transition scores

__all__ = [
    "BigramConf", "BigramNode",
]

from msp2.nn import BK
from msp2.nn.layers import *

# simple bigram matrix
class BigramConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.osize = -1  # number of entries
        # --
        self.lrank_k = -1  # if >0; then use E^T W E instead of full

@node_reg(BigramConf)
class BigramNode(BasicNode):
    def __init__(self, conf: BigramConf, extra_values: BK.Expr = None, **kwargs):
        super().__init__(conf, **kwargs)
        conf: BigramConf = self.conf
        # --
        self._M = None  # [out_prev, out_next]
        self.use_lrank = (conf.lrank_k > 0)
        self.extra_values = extra_values  # [out, out] extra ones like mask
        if self.use_lrank:
            self.E = BK.new_param([conf.osize, conf.lrank_k])  # [out, K]
            self.W = BK.new_param([conf.lrank_k, conf.lrank_k])  # [K, K]
        else:  # direct
            self.W = BK.new_param([conf.osize, conf.osize])
        self.reset_parameters()

    def reset_parameters(self):
        if self.use_lrank:
            BK.init_param(self.E, "glorot", lookup=True)
            BK.init_param(self.W, "ortho")
        else:
            BK.init_param(self.W, "zero")

    def set_extra_values(self, extra_values: BK.Expr):
        self.extra_values = extra_values

    def refresh(self, rop: RefreshOptions = None):
        super().refresh(rop)
        self._M = None

    def get_matrix(self):
        _extra_values = self.extra_values
        if _extra_values is None:
            _extra_values = 0.
        # --
        if self._M is None:
            if self.use_lrank:
                tmp_v = BK.matmul(self.E, self.W)  # [out, K]
                self._M = BK.matmul(tmp_v, self.E.t()) + _extra_values  # [out, out]
            else:
                self._M = self.W + _extra_values  # [out, out]
        return self._M  # [out_prev, out_next]

    # [*] -> [*, out_next]
    def forward(self, input: BK.Expr, **kwargs):
        M = self.get_matrix()
        next_scores = M[input]
        return next_scores
