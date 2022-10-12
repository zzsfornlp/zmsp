#

# common ones

__all__ = [
    "ZReprConf", "ZReprLayer",
]

from msp2.nn import BK
from msp2.nn.l3 import *
from msp2.tasks.zmtl3.mod.pretrained import *

class ZReprConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        # --
        # bert output
        self.bert_out = BertOuterConf()  # how to get bert output?
        self.sub_pooler = SubPoolerConf()
        # --

@node_reg(ZReprConf)
class ZReprLayer(Zlayer):
    def __init__(self, conf: ZReprConf, base_layer, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZReprConf = self.conf
        self.setattr_borrow('base_layer', base_layer)
        # --
        self.bert_out = conf.bert_out.make_node(bert_dim=base_layer.bmod.get_mdim(), att_num=base_layer.bmod.get_head_num())
        _bert_out_dim = self.bert_out.dim_out_hid()  # output of bert_outer
        self.sub_pooler = conf.sub_pooler.make_node()  # sub pooler
        self.output_dim = _bert_out_dim
        # --

    def get_output_dim(self):
        return self.output_dim

    def forward(self, bert_out, sublens_t):
        hid1_t = self.bert_out.forward_hid(bert_out)  # [bs, len1, D]
        hid0_t = self.sub_pooler.forward_hid(hid1_t, sublens_t)  # [bs, len0, D]
        return hid0_t
