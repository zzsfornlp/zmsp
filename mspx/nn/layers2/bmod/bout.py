#

__all__ = [
    "BoutConf", "BoutLayer",
]

from mspx.data.inst import DataPadder
from mspx.utils import ConfEntryChoices, zwarn, zlog, ZObject
from mspx.nn import BK, NnConf, NnLayer, CombinerConf

@NnConf.rd('zout')
class BoutConf(NnConf):
    def __init__(self):
        super().__init__()
        # --
        # output
        # for hidden layer
        self.bert_dim = -1  # dimension of bert
        self.bert_lidx = [-1]  # output
        self.bert_comb = CombinerConf.direct_conf(comb_method='concat')  # combining method
        # extra mixings
        self.extra_names = []
        self.extra_comb = CombinerConf.direct_conf(comb_method='gate')
        # for att
        self.att_num = -1  # number of attention heads (att_dim)
        self.att_lidx = []  # which attention layer?
        self.att_comb = CombinerConf.direct_conf(comb_method='concat')  # combining the attention heads
        # --
        # pooler
        self.pool_hid_f = 'first'  # cls/gmean/gmax /// first/last/mean2/max2
        self.pool_att_f = 'max4'  # first/last/mean4/max4
        # --

@BoutConf.conf_rd()
class BoutLayer(NnLayer):
    def __init__(self, conf: BoutConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: BoutConf = self.conf
        # --
        self.bert_comb = conf.bert_comb.make_node(isizes=[conf.bert_dim]*len(conf.bert_lidx))
        if len(conf.extra_names) > 0:
            self.extra_comb = conf.extra_comb.make_node(isizes=[conf.bert_dim]*(len(conf.extra_names)+1))
        else:
            self.extra_comb = None
        self.att_lidx = [int(z) for z in conf.att_lidx]
        if len(self.att_lidx) > 0:
            self.att_comb = conf.att_comb.make_node(isizes=[conf.att_num]*len(self.att_lidx))
        else:
            self.att_comb = None
        # --

    def dim_out_hid(self): return self.bert_comb.get_output_dims()[0]
    def dim_out_att(self): return self.att_comb.get_output_dims()[0]

    def comb_hid(self, bout):
        conf: BoutConf = self.conf
        # --
        hids = [bout.hidden_states[z] for z in conf.bert_lidx]  # *[bs, len, D]
        ret = self.bert_comb(hids)  # [bs, len, ??]
        return ret

    def comb_att(self, bout):
        conf: BoutConf = self.conf
        # --
        atts = [bout.attentions[int(z)].transpose(-2,-3).transpose(-1,-2) for z in conf.att_lidx]  # *[bs, Q, K, H]
        ret = self.att_comb(atts)  # [bs, Q, K, ??]
        return ret

    def forward(self, bout, rc=None, sublen_t=None, ext_sidx=None):
        conf: BoutConf = self.conf
        # --
        # enc
        retE = self.comb_hid(bout)
        # extra
        if len(conf.extra_names) > 0:
            extra_ts = [rc.get_cache(z) for z in conf.extra_names]
            if ext_sidx is not None:  # special re-arrange idxes!
                extra_ts = [z[ext_sidx] for z in extra_ts]
            retE = self.extra_comb([retE] + extra_ts)
        # att
        retA = None
        if self.att_comb is not None:
            retA = self.comb_att(bout)
        ret = {'E': retE, 'A': retA}
        # pool for token-level?
        if sublen_t is not None:
            ret['ET'] = self.pool_hid(retE, sublen_t, conf.pool_hid_f)
            if retA is not None:
                ret['AT'] = self.pool_att(retA, sublen_t, conf.pool_att_f)
        return ret

    # [*, L, D], [*, ??]
    @staticmethod
    def pool_hid(hid_t, sublen_t, pool_f: str):
        _arange_t = BK.arange_idx(len(hid_t)).unsqueeze(-1)  # [bs, 1]
        _idx1_t = sublen_t.cumsum(-1) - 1  # [bs, ??]
        _idx0_t = _idx1_t - (sublen_t-1).clamp(min=0)  # [bs, ??]
        # --
        if pool_f == 'first':
            ret = hid_t[_arange_t, _idx0_t]  # [bs, ??, D]
        elif pool_f == 'last':
            ret = hid_t[_arange_t, _idx1_t]  # [bs, ??, D]
        elif pool_f == 'mean2':
            ret = (hid_t[_arange_t, _idx0_t] + hid_t[_arange_t, _idx1_t]) / 2
        elif pool_f == 'max2':
            ret = BK.max_elem(hid_t[_arange_t, _idx0_t], hid_t[_arange_t, _idx1_t])
        else:
            raise NotImplementedError(f"UNK pool_f: {pool_f}")
        return ret

    # [*, Q, K, D], [*, Q], [*, K]
    @staticmethod
    def pool_att(att_t, sublen_qt, pool_f: str, sublen_kt=None):
        _arange_t = BK.arange_idx(len(att_t)).unsqueeze(-1).unsqueeze(-1)  # [bs, 1, 1]
        _idx1_qt = (sublen_qt.cumsum(-1) - 1).unsqueeze(-1)  # [bs, ??, 1]
        _idx0_qt = _idx1_qt - (sublen_qt - 1).clamp(min=0).unsqueeze(-1)  # [bs, ??, 1]
        if sublen_kt is None:  # same as Q
            _idx0_kt, _idx1_kt = _idx0_qt.squeeze(-1).unsqueeze(-2), _idx1_qt.squeeze(-1).unsqueeze(-2)  # [bs, 1, ??]
            sublen_kt = sublen_qt
        else:
            _idx1_kt = (sublen_kt.cumsum(-1) - 1).unsqueeze(-2)  # [bs, 1, ??]
            _idx0_kt = _idx1_kt - (sublen_kt - 1).clamp(min=0).unsqueeze(-2)  # [bs, 1, ??]
        # --
        # => [bs, Q?, K?, D]
        if pool_f == 'first':
            ret = att_t[_arange_t, _idx0_qt, _idx0_kt]
        elif pool_f == 'last':
            ret = att_t[_arange_t, _idx1_qt, _idx1_kt]  # [bs, Q?, K?, D]
        else:
            all4 = [att_t[_arange_t, a, b] for a in [_idx0_qt, _idx1_qt] for b in [_idx0_kt, _idx1_kt]]
            if pool_f == 'mean4':
                ret = BK.stack(all4, -1).mean(-1)
            elif pool_f == 'max4':
                ret = BK.stack(all4, -1).max(-1)[0]
            else:
                raise NotImplementedError(f"UNK pool_f: {pool_f}")
        # mask out invalid ones (probably for easier checking)
        t_mask = (sublen_qt > 0).unsqueeze(-1).to(BK.DEFAULT_FLOAT) * (sublen_kt > 0).unsqueeze(-2).to(BK.DEFAULT_FLOAT)  # [*, Q, K]
        ret = ret * t_mask.unsqueeze(-1)
        return ret
