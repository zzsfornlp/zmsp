#

# gcn

__all__ = [
    "ZGcn0Layer", "ZGcn0Node", "ZGcn0Conf",
]

import numpy as np
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.utils import Constants, zwarn
from .med import ZMediator

# --

class ZGcn0Conf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self._isize = -1
        self.n_layer = 1
        self.type_num = 50  # this should be enough?
        self.add_type_emb = False  # whether add syntax type_emb to each input
        self.out_drop = DropoutConf()
        self.use_addnorm = False  # add & norm
        # --

class ZGcn0Layer(BasicNode):
    def __init__(self, conf: ZGcn0Conf):
        super().__init__(conf)
        _isize = conf._isize
        _ntype = conf.type_num
        self.W_hid = BK.new_param([_isize, _isize*3])  # neg, self, pos
        self.b_hid = BK.new_param([2*_ntype+1, _isize])
        self.W_gate = BK.new_param([_isize, 3])
        self.b_gate = BK.new_param([2*_ntype+1])
        self.reset_parameters()
        self.drop_node = DropoutNode(conf.out_drop, osize=_isize)
        if conf.use_addnorm:
            self.ln = LayerNormNode(None, osize=_isize)
        else:
            self.ln = None
        # --

    def reset_parameters(self):
        _isize = self.conf._isize
        for ii in range(3):
            BK.init_param(self.W_hid[:,ii*_isize:(ii+1)*_isize], "glorot")
        BK.init_param(self.W_gate, "glorot")
        BK.init_param(self.b_hid, "zero")
        BK.init_param(self.b_gate, "zero")

    # [*, L, D], [*, L, L](-?,0,+?), [*, L]
    def forward(self, input_t: BK.Expr, edges: BK.Expr, mask_t: BK.Expr):
        _isize = self.conf._isize
        _ntype = self.conf.type_num
        _slen = BK.get_shape(edges, -1)
        # --
        edges3 = edges.clamp(min=-1, max=1) + 1
        edgesF = edges + _ntype  # offset to positive!
        # get hid
        hid0 = BK.matmul(input_t, self.W_hid).view(BK.get_shape(input_t)[:-1]+[3,_isize])  # [*, L, 3, D]
        hid1 = hid0.unsqueeze(-4).expand(-1, _slen, -1, -1, -1)  # [*, L, L, 3, D]
        hid2 = BK.gather_first_dims(hid1.contiguous(), edges3.unsqueeze(-1), -2).squeeze(-2)  # [*, L, L, D]
        hidB = self.b_hid[edgesF]  # [*, L, L, D]
        _hid = hid2 + hidB
        # get gate
        gate0 = BK.matmul(input_t, self.W_gate)  # [*, L, 3]
        gate1 = gate0.unsqueeze(-3).expand(-1, _slen, -1, -1)  # [*, L, L, 3]
        gate2 = gate1.gather(-1, edges3.unsqueeze(-1))  # [*, L, L, 1]
        gateB = self.b_gate[edgesF].unsqueeze(-1)  # [*, L, L, 1]
        _gate0 = BK.sigmoid(gate2 + gateB)
        _gmask0 = ((edges != 0) | (BK.eye(_slen)>0)).float() * mask_t.unsqueeze(-2)  # [*,L,L]
        _gate = _gate0 * _gmask0.unsqueeze(-1)  # [*,L,L,1]
        # combine
        h0 = BK.relu((_hid * _gate).sum(-2))  # [*, L, D]
        h1 = self.drop_node(h0)
        # add & norm?
        if self.ln is not None:
            h1 = self.ln(h1+input_t)
        return h1

@node_reg(ZGcn0Conf)
class ZGcn0Node(BasicNode):
    def __init__(self, conf: ZGcn0Conf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZGcn0Conf = self.conf
        # --
        self.nodes = []
        for ii in range(conf.n_layer):
            _node = ZGcn0Layer(conf)
            self.add_module(f'G{ii}', _node)
            self.nodes.append(_node)
        # --
        if conf.add_type_emb:
            self.type_emb = EmbeddingNode(None, osize=conf._isize, n_words=conf.type_num, fix_row0=False)
        else:
            self.type_emb = None
        # --

    def forward(self, med: ZMediator):
        # --
        # get hid_t
        hid_t0 = med.get_enc_cache_val("hid")
        sinfo = med.ibatch.seq_info
        _arange_t, _sel_t = sinfo.arange2_t, sinfo.dec_sel_idxes
        hid_t = hid_t0[_arange_t, _sel_t]  # [*, dlen, D]
        # --
        # prepare relations
        bsize, dlen = BK.get_shape(sinfo.dec_sel_masks)
        arr_rels = np.full([bsize, dlen, dlen], 0, dtype=np.int)  # by default 0
        arr_labs = np.full([bsize, dlen], 0, dtype=np.int)
        for bidx, item in enumerate(med.ibatch.items):  # for each item in the batch
            _dec_offsets = item.seq_info.dec_offsets
            for sidx, sent in enumerate(item.sents):  # for each sent in the msent item
                tree = sent.tree_dep
                _start = _dec_offsets[sidx]
                _slen = len(sent)
                _arr_ms = np.asarray(range(_slen)) + _start  # [??]
                _arr_hs = np.asarray(tree.seq_head.vals) + (_start - 1)  # note(+N): need to do more if msent!!
                _arr_labs = np.asarray(tree.seq_label.idxes)  # [??]
                arr_labs[bidx, _start:_start+_slen] = _arr_labs
                arr_rels[bidx, _arr_hs, _arr_ms] = _arr_labs
                arr_rels[bidx, _arr_ms, _arr_hs] = - _arr_labs
        expr_labs = BK.input_idx(arr_rels)  # [*, dlen, dlen]
        # --
        # go through
        res_t = hid_t
        if self.type_emb is not None:
            expr_seq_labs = BK.input_idx(arr_labs)  # [*, dlen]
            lab_t = self.type_emb(expr_seq_labs)
            res_t = res_t + lab_t
        for node in self.nodes:
            res_t = node.forward(res_t, expr_labs, sinfo.dec_sel_masks)
            med.layer_end({'hid': res_t})  # step once!
        return res_t

# --
# b msp2/tasks/zmtl2/zmod/common/gcn:??
