#

# some graph-related layers

__all__ = [
    "GnnConf", "GnnLayer"
]

from ..backends import BK
from .base import *
from mspx.utils import ConfEntryChoices

# multiple layers
@NnConf.rd('gnn')
class GnnConf(NnConf):
    def __init__(self):
        super().__init__()
        self.dim = -1
        self.layer_num = 2  # how many layers?
        self.layer_type = ConfEntryChoices({'simple': GnnlSimpleConf()}, 'simple')

@GnnConf.conf_rd()
class GnnLayer(NnLayer):
    def __init__(self, conf: GnnConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: GnnConf = self.conf
        # --
        self.node_names = []
        for idx in range(conf.layer_num):
            node = conf.layer_type.make_node(dim=conf.dim)
            self.add_module(f"G{idx}", node)
            self.node_names.append(f"G{idx}")
        # --

    def extra_repr(self) -> str:
        conf: GnnConf = self.conf
        return f"GnnLayer({conf.dim}x{conf.layer_num}x{conf.layer_type})+{super().extra_repr()})"

    def get_output_dims(self, *input_dims):
        conf: GnnConf = self.conf
        return (conf.dim, )

    @property
    def nodes(self):
        return [getattr(self, k) for k in self.node_names]

    def forward(self, t_in, t_att, t_mask):
        cur_expr = t_in
        for n in self.nodes:
            cur_expr = n(cur_expr, t_att, t_mask)
        return cur_expr

# --
# layer ones

@NnConf.rd('gnnl_simple')
class GnnlSimpleConf(NnConf):
    def __init__(self):
        super().__init__()
        self.dim = -1
        # --
        from .seq import AttentionConf
        self.att1 = AttentionConf.direct_conf(head_count=1)  # simply single head!
        self.att2 = AttentionConf.direct_conf(head_count=1)

@GnnlSimpleConf.conf_rd()
class GnnlSimpleLayer(NnLayer):
    def __init__(self, conf: GnnlSimpleConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: GnnlSimpleConf = self.conf
        # --
        dim = conf.dim
        self.att1 = conf.att1.make_node(dim_q=dim, dim_k=dim, dim_v=dim)
        self.att2 = conf.att1.make_node(dim_q=dim, dim_k=dim, dim_v=dim)
        self.ln = BK.nn.LayerNorm(dim)

    def forward(self, t_in, t_att, t_mask):
        t_att = t_att * t_mask.unsqueeze(-1) * t_mask.unsqueeze(-2)
        res1 = self.att1(t_in, t_in, t_in, mask_k=t_mask, external_attn=t_att)
        res2 = self.att1(t_in, t_in, t_in, mask_k=t_mask, external_attn=t_att.transpose(-1, -2).contiguous())  # [*, L, D]
        ret = self.ln(t_in + res1 + res2)
        return ret

# --
# b mspx/nn/layers/graph:70
