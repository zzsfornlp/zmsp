#

# label matcher (labeler)

from msp2.data.vocab import SimpleVocab
from msp2.utils import ZObject, zlog, zwarn, Constants
from msp2.nn import BK
from msp2.nn.l3 import *

# --
# labeling layer
"""
# note: how to change the weights
1) before training: init embs or not? (create_and_assign*)
2) training: use my_emb or create? use my_proj or create? (do_train)
3) before testing: finalize the weights (create_and_assign*)
"""

class LabConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        self.isize = -1  # input dim
        # --
        self.detach_input = False  # whether detach input?
        self.proj_dim = -1  # whether adding a projection layer?
        self.sim = SimConf()
        self.sim_dropout = 0.  # whether dropout for final reprs?
        self.pos_number = 1  # how many positive reprs (must be >0)
        self.neg_number = 0  # how many negative reprs (if zero, then simply 0)
        self.neg_delta = ScalarConf.direct_conf(init=0., fixed=True)  # score delta added to neg score
        # --
        # specific for training
        self.train_create_embs = False  # create proto embs for pos/neg targets
        self.train_create_proj = False  # create proj for projection
        self.label_smoothing = 0.
        self.train_neg_sample = -1.  # negative down-sampling (relative to positives)
        self.train_neg_ratio = 2.  # negative down-weighting (relative to positives)
        self.train_neg_no_exclude = True  # when 'train_create_embs', no exclude_self to avoid OOM
        # --
        # different losses
        self.loss_plain = 1.  # plain classification
        self.loss_proto = 0.  # proto contrast
        self.loss_pair = 0.  # pairwise contrast
        # --

@node_reg(LabConf)
class LabLayer(Zlayer):
    def __init__(self, conf: LabConf, voc: SimpleVocab, **kwargs):
        super().__init__(conf, **kwargs)
        conf: LabConf = self.conf
        # --
        self.voc = voc
        self.trg_range = voc.non_special_range()
        self.trg_num = self.trg_range[1] - self.trg_range[0]
        # --
        # note: currently we simply assume NIL=0
        assert self.trg_range[0] == 1
        # --
        self.sim = conf.sim.make_node()
        self.sim_drop = BK.nn.Dropout(conf.sim_dropout)  # dropout node
        if conf.proj_dim > 0:
            self.proj = BK.nn.Linear(conf.isize, conf.proj_dim, bias=False)
        else:
            self.proj = None
        self.pos_emb = BK.nn.Embedding(self.trg_num * conf.pos_number, conf.isize)  # [Lp*Np, D]
        if conf.neg_number > 0:
            self.neg_emb = BK.nn.Embedding(conf.neg_number, conf.isize)  # [Nn, D]
        else:
            self.neg_emb = None
        self.neg_delta = conf.neg_delta.make_node()  # []
        # --

    @property
    def pos_weight(self):  # [Np, D]
        return self.pos_emb.weight

    @property
    def neg_weight(self):  # [Nn, D]
        return self.neg_emb.weight if self.neg_emb is not None else None

    @property
    def proj_weight(self):  # [D, D']
        return self.proj.weight.t() if self.proj is not None else None

    def do_input(self, t):
        if self.conf.detach_input:
            t = t.detach()
        return t

    def do_proj(self, t, proj_w):
        if proj_w is None or t is None:
            ret0 = t
        else:
            ret0 = BK.matmul(t, proj_w)
        return ret0

    def do_sim(self, input1, input2, **kwargs):
        in1, in2 = self.sim_drop(input1), self.sim_drop(input2)
        ret = self.sim(in1, in2, **kwargs)
        return ret

    # do score; [*, D] -> [*, 1+Lp]
    def do_score(self, input_t, w_proj, w_pos, w_neg, is_pair=True, no_neg_delta=False):
        conf: LabConf = self.conf
        zero_input = BK.is_zero_shape(input_t)
        # --
        in_t = self.do_input(input_t)  # [*, D]
        p_input = self.do_proj(in_t, w_proj)  # [*, D']
        p_pos, p_neg = self.do_proj(w_pos, w_proj), self.do_proj(w_neg, w_proj)  # [??, D']
        s_pos0 = self.do_sim(p_input, p_pos, is_pair=is_pair)  # [*, pos*np]
        if conf.pos_number > 1:
            _tmp_shape = BK.get_shape(s_pos0)
            _tmp_shape = _tmp_shape[:-1] + [_tmp_shape[-1]//conf.pos_number, conf.pos_number]  # [*, Lp, np]
            _tmp_pos = s_pos0.view(_tmp_shape)
            s_pos = _tmp_pos.sum(-1) if zero_input else _tmp_pos.max(-1)[0]  # [*, Lp]
        else:
            s_pos = s_pos0
        if p_neg is None or zero_input:  # no neg embs or zero input
            s_neg1 = BK.zeros(BK.get_shape(s_pos)[:-1] + [1])  # [*, 1]
        else:
            s_neg0 = self.do_sim(p_input, p_neg, is_pair=is_pair)  # [*, neg]
            s_neg1, _ = s_neg0.max(-1, keepdims=True)  # [*, 1]
        if not no_neg_delta:
            s_neg = s_neg1 + self.neg_delta()  # [*, 1], add negative delta
        else:
            s_neg = s_neg1
        ret = BK.concat([s_neg, s_pos], -1)  # [*, 1+Lp]
        return ret

    # do score with weights inside this layer
    def do_my_score(self, input_t):
        return self.do_score(input_t, self.proj_weight, self.pos_weight, self.neg_weight)

    # create proto embs (with simple average): [bs, D], [bs]
    def create_embs(self, input_t, label_t, exclude_self: bool, neg_no_exclude=False, neg_k=None, assign=False, **kwargs):
        conf: LabConf = self.conf
        in_t = self.do_input(input_t)  # [bs, D]
        _tr0, _tr1 = self.trg_range
        # --
        # positive ones
        _trglab_t = BK.arange_idx(_tr1)  # [T]
        _pos_hit_t = (label_t == _trglab_t.unsqueeze(-1)) & (_trglab_t.unsqueeze(-1) >= _tr0)  # [T, bs]
        _pos_count = _pos_hit_t.long().sum(-1)  # [T]
        # filter
        _valid = (_pos_count > 0)  # [T]
        pos_lab_t = _trglab_t[_valid]  # [T']
        pos_count_t = _pos_count[_valid]  # [T']
        pos_hit_t = _pos_hit_t[_valid]  # [T', bs]
        # --
        _pk = conf.pos_number
        assert _pk >= 1, "Must have >=1 pk"
        if _pk > 0 and len(pos_hit_t)>0:
            # note: loop over each hit type here!
            all_clu, all_ex_clu = [], []
            for tmpi in range(len(pos_hit_t)):
                _curr_t = in_t[pos_hit_t[tmpi]]  # [??, D], filtered to the current type!
                _curr_clu = self.sim.run_kmeans(_curr_t, None, _pk)  # [pk, D] first get plain ones
                all_clu.append(_curr_clu)
                if exclude_self:
                    _tmpn = _curr_t.shape[0]  # [??]
                    _mm = 1. - BK.eye(_tmpn)  # [??, ??]
                    _expand_t = _curr_t.unsqueeze(0).expand(_tmpn, -1, -1)  # [??, ??, D]
                    _curr_ex_clu = self.sim.run_kmeans(_expand_t, _mm, _pk)  # [??, pk, D]
                    all_ex_clu.append(_curr_ex_clu)
            plain_pos_proto_t = BK.concat(all_clu, 0)  # [T'*pk, D]
            if exclude_self:
                ex_clu_t = BK.concat(all_ex_clu, 0)  # [sum(pos), pk, D]
                _offset0 = len(ex_clu_t)+1  # sum(pos)+1
                _aug_clu_t = BK.concat([ex_clu_t, BK.stack(all_clu, 0)], 0)  # [sum(pos)+T', pk, D]
                _pos_hit_t1 = pos_hit_t.long()  # [T', bs]
                _pos_hit_idx = _pos_hit_t1.view(-1).cumsum(-1).long().view_as(_pos_hit_t1) * _pos_hit_t1 \
                               + (1-_pos_hit_t1) * (BK.arange_idx(len(_pos_hit_t1)).unsqueeze(-1) + _offset0)  # [T', bs]
                ex_pos_proto_t = _aug_clu_t[(_pos_hit_idx-1).transpose(-1,-2)]  # [bs, T', pk, D]
                ex_pos_proto_t = ex_pos_proto_t.view([BK.get_shape(pos_hit_t, -1)]+BK.get_shape(plain_pos_proto_t))  # [bs,T'*pk,D]
                pos_proto_t = ex_pos_proto_t
            else:
                pos_proto_t = plain_pos_proto_t  # [T'*pk, D]
        else:
            pos_proto_t = None
        # breakpoint()
        # --
        # if _pk == 1:
        #     # simply do average
        #     pos_sum_t = BK.matmul(pos_hit_t.float(), in_t)  # [T', D]
        #     pos_proto_t = pos_sum_t / (pos_count_t.float().unsqueeze(-1))  # [T', D], average
        #     if exclude_self:
        #         _div_t = (pos_count_t.float().unsqueeze(-1) - 1).clamp(min=1)  # [T', 1], minus 1
        #         _avg_t = (pos_sum_t - in_t.unsqueeze(-2)) / _div_t  # [bs, T', D], m1 average
        #         _mt = pos_hit_t.transpose(-1, -2).unsqueeze(-1).float()  # [bs, T', 1], _avg_t if pos else full-avg
        #         pos_proto_t = _avg_t * _mt + pos_proto_t * (1.-_mt)  # [bs, T', D]
        # if _pk > 1:
        #     _tmp_pk = _pk
        #     assert _tmp_pk > 1 and not exclude_self, "Currently does not support this mode!"
        #     all_clu = []
        #     for tmpi in range(len(pos_hit_t)):
        #         clu = self.sim.run_kmeans(in_t[pos_hit_t[tmpi]], None, _tmp_pk)  # [k, D]
        #         all_clu.append(clu)
        #     pos_proto_t = BK.concat(all_clu, 0)  # [T'*k, D]
        #     # breakpoint()
        # --
        # negative ones
        _nk = conf.neg_number if neg_k is None else neg_k
        if _nk > 0:
            _neg_hit_t = (label_t == 0)  # [bs]
            _neg_in_t = in_t[_neg_hit_t]  # [Nbs, D]
            if exclude_self and not neg_no_exclude:  # everyone differs, but does not matter!
                _bs = _neg_hit_t.shape[0]
                _mm = 1. - BK.eye(_bs)  # [bs, bs]
                _mm = _mm[:, _neg_hit_t]  # [bs, Nbs]
                _expand_t = _neg_in_t.unsqueeze(0).expand(_bs, -1, -1)  # [bs, Nbs, D]
                neg_proto_t = self.sim.run_kmeans(_expand_t, _mm, _nk)  # [bs, k, D]
            else:
                neg_proto_t = self.sim.run_kmeans(_neg_in_t, None, _nk)  # [k, D]
                if exclude_self:
                    neg_proto_t = neg_proto_t.unsqueeze(0)  # [1, k, D]
        else:  # no neg embs!
            neg_proto_t = None
        # --
        # label map: from T(orig) -> T'(current)
        label_map = BK.constants_idx([_tr1], value=-1)  # [T], by default -1:error
        label_map[0] = 0  # 0 -> 0
        label_map[_valid] = BK.arange_idx(BK.get_shape(pos_lab_t, 0)) + 1  # offset by idx0
        # --
        if assign:
            neg_info = None
            if _nk > 0:  # negative
                BK.set_value(self.neg_weight, neg_proto_t)
                neg_info = neg_proto_t.shape
            # positive
            pos_info = (pos_proto_t.shape, _valid.nonzero().flatten().tolist())
            with BK.no_grad_env():
                _tmp_m = BK.simple_repeat_interleave(_valid[_tr0:_tr1], _pk, 0)  # [L*pk]
                self.pos_weight.zero_()  # by default zero things!
                self.pos_weight[_tmp_m] = pos_proto_t
            zlog(f"Assign proto-embs to {self}: neg={neg_info}, pos={pos_info}")
            if (label_map<0).any():
                zwarn(f"Assigning missed sth: {label_map.tolist()}")
        # --
        return pos_proto_t, neg_proto_t, label_map

    # create proj??
    def create_proj(self, input_t, label_t, assign=False, **kwargs):
        raise NotImplementedError()  # TODO(+N): currently no immediate plan for this ...

    # training: [*, D], [*], [*]
    def do_loss(self, emb_t, lab_t, mask_t):
        conf: LabConf = self.conf
        # --
        # first do down-sampling
        if conf.train_neg_sample >= 0.:
            mask_t = down_neg(mask_t, (lab_t>0).float(), conf.train_neg_sample, do_sample=True)  # [*]
        # utilize mask_t to put things compact
        valid_t = (mask_t > 0.)  # [*]
        input_t, label_t = emb_t[valid_t], lab_t[valid_t]  # [bs, D], [bs]
        # re-weight for pos/neg
        weight_t = BK.constants(label_t.shape, 1.)  # [bs]
        if conf.train_neg_ratio > 0:
            weight_t = down_neg(weight_t, (label_t>0).float(), conf.train_neg_ratio, do_sample=False)  # [bs]
        # --
        ret_items = []
        if conf.loss_plain > 0.:  # add plain loss
            scores_t = self.do_my_score(input_t)  # [bs, 1+?]
            loss_t = BK.loss_nll(scores_t, label_t, label_smoothing=conf.label_smoothing)  # [bs]
            _loss_item = LossHelper.compile_leaf_loss(
                'labPl', (loss_t * weight_t).sum(), weight_t.sum(), loss_lambda=conf.loss_plain)
            ret_items.append(_loss_item)
        if conf.loss_proto > 0.:  # add proto-contrast loss
            assert int(conf.train_create_embs)+int(conf.train_create_proj)==1  # xor
            # create trg weights?
            if conf.train_create_embs:
                is_pair = False
                w_pos, w_neg, label_map = self.create_embs(
                    input_t, label_t, exclude_self=True, neg_no_exclude=conf.train_neg_no_exclude)  # [bs, T', D], [T]
                pr_input_t = input_t.unsqueeze(-2)  # [bs, 1, D]
            else:
                is_pair = True
                w_pos, w_neg = self.pos_weight, self.neg_weight  # [bs, T]
                label_map = None
                pr_input_t = input_t
            # create proj weights?
            if conf.train_create_proj:
                w_proj = self.create_proj(input_t, label_t)
            else:
                w_proj = self.proj_weight  # [D, D']
            # forward scores
            scores_t = self.do_score(pr_input_t, w_proj, w_pos, w_neg, is_pair=is_pair)  # [bs, 1+?]
            label2_t = label_map[label_t] if label_map is not None else label_t  # [bs], transform label if needed
            loss_t = BK.loss_nll(scores_t, label2_t, label_smoothing=conf.label_smoothing)  # [bs]
            _loss_item = LossHelper.compile_leaf_loss(
                'labPr', (loss_t * weight_t).sum(), weight_t.sum(), loss_lambda=conf.loss_proto)
            ret_items.append(_loss_item)
        if conf.loss_pair > 0:  # add pairwise ones!
            diag_mask = BK.eye(len(input_t))  # [bs, bs]
            p_input = self.do_proj(self.do_input(input_t), self.proj_weight)  # [bs, D']
            scores_t = self.do_sim(p_input, p_input)  # [bs, bs]
            scores_t = scores_t + diag_mask * Constants.REAL_PRAC_MIN
            trg_t = (label_t.unsqueeze(-1) == label_t.unsqueeze(-2)).float() * (1.-diag_mask)  # [bs, bs]
            nll_t = - (scores_t.log_softmax(-1) * trg_t).sum(-1) / trg_t.sum(-1).clamp(min=1)  # [bs]
            _loss_item = LossHelper.compile_leaf_loss(
                'labPa', (nll_t * weight_t).sum(), weight_t.sum(), loss_lambda=conf.loss_pair)
            ret_items.append(_loss_item)
        # --
        return ret_items

# --
# b msp2/tasks/zmtl3/mod/extract/layer_lab:??
