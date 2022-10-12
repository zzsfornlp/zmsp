#

# repr for arg candidate?

__all__ = [
    "ZArgReprConf", "ZArgReprLayer"
]

from msp2.nn import BK
from msp2.nn.l3 import *
from ...pretrained import *

# --

class ZArgReprConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        # --
        # bert output
        self.bert_out = BertOuterConf()  # how to get bert output?
        self.sub_pooler = SubPoolerConf()
        # --
        self.comps = ['self']  # self/att/inner/outer
        self.r_inner = PairReprConf.direct_conf(pair_func='inner', hsize=32*32, pair_piece=32, osize=0)
        self.r_outer = PairReprConf.direct_conf(pair_func='outer', hsize=8*8, pair_piece=8, osize=0)
        self.final_aff = AffineConf.direct_conf(osize=0)  # final output
        # --

@node_reg(ZArgReprConf)
class ZArgReprLayer(Zlayer):
    def __init__(self, conf: ZArgReprConf, arg_layer, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZArgReprConf = self.conf
        self.setattr_borrow('arg_layer', arg_layer)
        # --
        self.bert_out = conf.bert_out.make_node(bert_dim=arg_layer.bmod.get_mdim(), att_num=arg_layer.bmod.get_head_num())
        _bert_out_dim = self.bert_out.dim_out_hid()  # output of bert_outer
        self.sub_pooler = conf.sub_pooler.make_node()  # sub pooler
        # ==
        self.r_inner, self.r_outer = None, None
        self.comp_dims = []
        for comp in conf.comps:
            if comp == 'self':
                self.comp_dims.append(_bert_out_dim)
            elif comp == 'att':
                self.comp_dims.append(self.bert_out.dim_out_att() * 2)  # both directions
            elif comp == 'inner':
                assert self.r_inner is None
                self.r_inner = conf.r_inner.make_node(isize1=_bert_out_dim, isize2=_bert_out_dim)
                self.comp_dims.append(self.r_inner.get_output_dim())
            elif comp == 'outer':
                assert self.r_outer is None
                self.r_outer = conf.r_outer.make_node(isize1=_bert_out_dim, isize2=_bert_out_dim)
                self.comp_dims.append(self.r_outer.get_output_dim())
            else:
                raise NotImplementedError(f"UNK comp-type of {comp}")
        # --
        self.final_aff = AffineLayer(conf.final_aff, isize=self.comp_dims)
        # --

    def forward(self, bert_out, sublens_t, arr_toks, t_ifr):
        conf: ZArgReprConf = self.conf
        # --
        hid1_t = self.bert_out.forward_hid(bert_out)  # [bs, len1, D]
        hid0_t = self.sub_pooler.forward_hid(hid1_t, sublens_t)  # [bs, len0, D]
        # --
        evt_mask = []  # [bs, len0, D]
        evt_hid = []  # [bs, 1, D]
        def _get_evt_mask():
            if len(evt_mask) == 0:
                _mm = (t_ifr > 0).float()
                _mm = _mm / _mm.sum(-2, keepdims=True).clamp(min=1.)  # [bs, len0]
                _mm = _mm.unsqueeze(-1)  # [bs, len0, 1]
                evt_mask.append(_mm)
            return evt_mask[0]
        def _get_evt_hid():
            if len(evt_hid) == 0:
                _mm = _get_evt_mask()
                evt_hid.append((hid0_t * _mm).sum(-2, keepdims=True))  # [bs, 1, D]
            return evt_hid[0]
        # --
        comp_reprs = []
        for comp in conf.comps:
            if comp == 'self':
                rr = hid0_t  # [bs, len0, D]
            elif comp == 'att':
                att1_t = self.bert_out.forward_att(bert_out)  # [bs, len1, len1, D]
                att0_t = self.sub_pooler.forward_att(att1_t, sublens_t)  # [bs, len0, len0, D]
                _mm = _get_evt_mask()  # [bs, len0, 1]
                rr = BK.concat([(att0_t*_mm.unsqueeze(-3)).sum(-2), (att0_t*_mm.unsqueeze(-2)).sum(-3)], -1)
            elif comp == 'inner':
                rr = self.r_inner(_get_evt_hid(), hid0_t).squeeze(-3)
            elif comp == 'outer':
                rr = self.r_outer(_get_evt_hid(), hid0_t).squeeze(-3)
            else:
                raise NotImplementedError(f"UNK comp-type of {comp}")
            # --
            comp_reprs.append(rr)  # [bs, len0, ??]
        # --
        ret = self.final_aff(comp_reprs)
        return ret

# --
# b msp2/tasks/zmtl3/mod/extract/evt_arg/m_repr:
