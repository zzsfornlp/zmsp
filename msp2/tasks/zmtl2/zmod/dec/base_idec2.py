#

# a specific idec node for pairwise score

__all__ = [
    "Idec2Conf", "Idec2Node",
]

from typing import List
from msp2.nn import BK
from msp2.nn.layers import *
from ..common import ZMediator
from .base_idec import DSelector

# --

class Idec2Conf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        # general sizes from outside
        self._isize = -1  # input size (emb)
        self._nhead = -1  # input size (num_head)
        self._csize = -1  # number of classes
        # apply layers
        self.app_layers = []  # must be one integer >=1
        # how to get things from att
        self.gatt_name = "attn_probs"  # attn_logits/attn_probs
        self.gatt_num_layer = -1  # how many layers to get, -1 means all
        self.gatt_rh = 'max'  # how to reduce subword->word for head/pred dim: [*,H,h,m]
        self.gatt_rm = 'sum'  # how to reduce subword->word for mod/children dim: [*,H,h,m]
        self.gatt_init_scale_in = 1.  # init scale for gatt->hid
        self.gatt_drop = DropoutConf()
        # combine h/m hid layer repr?
        self.ghid_h = True
        self.ghid_m = True
        # hidden layer for combining things
        self.hid_dim = 128  # if 0, directly to output size!
        self.hid_act = 'elu'
        # --

@node_reg(Idec2Conf)
class Idec2Node(BasicNode):
    def __init__(self, conf: Idec2Conf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: Idec2Conf = self.conf
        # --
        self.app_lidxes = [int(z) for z in conf.app_layers]  # app_idx -> lidx
        assert len(self.app_lidxes)==1 and self.app_lidxes[0]>=1, "This mode only allow one layer!!"
        self.app_lidxes_set = set(self.app_lidxes)
        self.max_app_lidx = max(self.app_lidxes) if len(self.app_lidxes) > 0 else -1
        # --
        # get params (basically one/two affine layers)
        _flayer = self.max_app_lidx
        _has_hid = (conf.hid_dim>0)
        _hid_dim = conf.hid_dim if _has_hid else conf._csize
        _hid_act = conf.hid_act if _has_hid else 'linear'
        # note: layer=0 does not have att!
        _gatt_input_nlayers = _flayer if conf.gatt_num_layer<=0 else min(conf.gatt_num_layer, _flayer)
        _gatt_input_dim = conf._nhead * 2 * _gatt_input_nlayers  # get both directions!
        _aff_inputs = [_gatt_input_dim] + ([conf._isize] if conf.ghid_h else []) + ([conf._isize] if conf.ghid_m else [])
        self.aff_hid = AffineNode(None, isize=_aff_inputs, osize=_hid_dim, out_act=_hid_act,
                                  no_drop=(not _has_hid), which_affine=3)
        self.min_gatt_lidx = self.max_app_lidx - _gatt_input_nlayers
        with BK.no_grad_env():
            _w_gatt = self.aff_hid.get_ws()[0]
            _w_gatt *= conf.gatt_init_scale_in  # only scale this one!
        self.aff_final = None
        if _has_hid:
            self.aff_final = AffineNode(None, isize=_hid_dim, osize=conf._csize, no_drop=True)
        self.gatt_drop = DropoutNode(conf.gatt_drop)
        # --
        # dsel
        self.dsel_rm = DSelector(None, dsel_method=conf.gatt_rm)
        self.dsel_rh = DSelector(None, dsel_method=conf.gatt_rh)
        self.gatt_key = (conf.gatt_name, self.min_gatt_lidx, self.max_app_lidx, conf.gatt_rm, conf.gatt_rh)
        self.dsel_hid = DSelector(None, dsel_method='first')  # simply use first!
        # --

    @property
    def num_app_layers(self):
        return len(self.app_lidxes)

    @property
    def app_layers(self):
        return self.app_lidxes

    def has_layer(self, lidx: int):
        return lidx in self.app_lidxes_set

    def has_feed(self, lidx: int):
        return False  # currently no support!

    def forward(self, med: ZMediator):
        conf: Idec2Conf = self.conf
        cur_lidx = med.lidx
        assert cur_lidx == self.max_app_lidx
        seq_info = med.ibatch.seq_info
        # --
        # get att values
        # todo(+N): modify med to make (cached) things more flexible!
        v_att_final = med.get_cache(self.gatt_key)
        if v_att_final is None:
            v_att = BK.concat(med.get_enc_cache(conf.gatt_name).vals[self.min_gatt_lidx:cur_lidx], 1)  # [*, L*H, h, m]
            v_att_rm = self.dsel_rm(v_att.permute(0,3,2,1), seq_info)  # first reduce m: [*,m',h,L*H]
            v_att_rh = self.dsel_rh(v_att_rm.transpose(1,2), seq_info)  # then reduce h: [*,h',m',L*H]
            v_att_final = BK.concat([v_att_rh, v_att_rh.transpose(1,2)], -1)  # final concat: [*,h',m',L*H*2]
            med.set_cache(self.gatt_key, v_att_final)
        hid_inputs = [self.gatt_drop(v_att_final)]
        # --
        # get hid values
        if conf.ghid_m or conf.ghid_h:
            _dsel = self.dsel_hid
            v_hid = med.get_enc_cache_val(  # [*, len', D]
                "hid", signature=_dsel.signature, function=(lambda x: _dsel.forward(x, seq_info)))
            if conf.ghid_h:
                hid_inputs.append(v_hid.unsqueeze(-2))  # [*, h, 1, D]
            if conf.ghid_m:
                hid_inputs.append(v_hid.unsqueeze(-3))  # [*, 1, m, D]
        # --
        # go
        ret = self.aff_hid(hid_inputs)
        if self.aff_final is not None:
            ret = self.aff_final(ret)
        return ret, None  # currently no feed!

# --
# b msp2/tasks/zmtl2/zmod/dev/base_idec2:
