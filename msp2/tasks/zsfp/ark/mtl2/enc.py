#

# the base of enc

__all__ = [
    "ZMediator", "ZEncoderConf", "ZEncoder",
]

from typing import List
from msp2.nn.layers import *
from msp2.nn import BK
from .dec import *

# =====
# the mediator: connection between encoder and idec

class ZMediator:
    def __init__(self):
        # --
        # static
        # idec/connectors, note: 100 should be enough!
        self.idecs = []
        self.emb_connectors = [[] for _ in range(100)]  # lidx -> []
        self.att_connectors = [[] for _ in range(100)]  # lidx -> []
        # --
        # caches
        self.lidx = 0
        self.mask_t = None  # [*, slen]
        self.attns = []

    # --
    def start(self, mask_t):
        self.lidx = 0
        self.mask_t = mask_t
        self.attns.clear()

    def next(self):
        self.lidx += 1

    def end(self):  # clear
        self.start(None)

    def is_end(self):
        return False

    # [*, H, lenq, lenk]
    def forw_attn(self, expr_t, scores_t):
        self.attns.append(scores_t)
        assert len(self.attns) == self.lidx  # L0 has no attns!
        conns = self.att_connectors[self.lidx]
        _all_scores_t = None
        for conn in conns:
            if _all_scores_t is None:
                _all_scores_t = BK.stack(self.attns, -1).permute(0,2,3,4,1)  # [*, H, lenq, lenk, NL] -> [*, lenq, lenk, NL, H]
            out = conn.forward(expr_t, self.mask_t, scores_t=_all_scores_t)
            if out is not None:  # [*, lenq, lenk, H]
                scores_t += out.permute(0,3,1,2)
        return scores_t

    # [*, slen, D]
    def forw_emb(self, expr_t, norm_node=None):
        conns = self.emb_connectors[self.lidx]
        feed_flag = False
        for conn in conns:
            out = conn.forward(expr_t, self.mask_t)
            if out is not None:
                expr_t += out  # [*, slen, D]
                feed_flag = True
        if feed_flag and norm_node is not None:
            return norm_node(expr_t)
        else:
            return expr_t

    # --
    # adding connectors
    def add_connector(self, conn: IdecConnectorNode):
        lidx = conn.lidx
        if isinstance(conn, IdecConnectorAttNode):
            self.att_connectors[lidx].append(conn)
        else:  # otherwise all plain ones
            self.emb_connectors[lidx].append(conn)
        # --

    def add_idec(self, idec: IdecNode):
        self.idecs.append(idec)
        for conn in idec.connectors:
            self.add_connector(conn)
        # --

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
    def get_num_layers(self) -> int: raise NotImplementedError()
    def get_num_heads(self) -> int: raise NotImplementedError()
