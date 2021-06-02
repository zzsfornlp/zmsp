#

# the decoders

from typing import List
import numpy as np
from msp.utils import Conf, Helper, Constants
from msp.nn import BK
from msp.nn.layers import BasicNode, Affine, NoDropRop, PairScorerConf, PairScorer
from msp.zext.seq_helper import DataPadder

from ..common.data import ParseInstance
from ..algo import nmst_unproj, nmarginal_unproj

# =====
# scorers

class FpScorerConf(Conf):
    def __init__(self):
        self._input_dim = -1  # enc's (input) last dimension
        self._num_label = -1  # number of labels
        # space transferring (0 means no this layer)
        self.arc_space = 512
        self.lab_space = 128
        self.transform_act = "elu"
        # pairwise conf
        self.ps_conf = PairScorerConf()

class FpPairedScorer(BasicNode):
    def __init__(self, pc: BK.ParamCollection, sconf: FpScorerConf):
        super().__init__(pc, None, None)
        # options
        input_dim = sconf._input_dim
        arc_space = sconf.arc_space
        lab_space = sconf.lab_space
        transform_act = sconf.transform_act
        #
        self.input_dim = input_dim
        self.num_label = sconf._num_label
        # attach/arc
        if arc_space>0:
            self.arc_m = self.add_sub_node("am", Affine(pc, input_dim, arc_space, act=transform_act))
            self.arc_h = self.add_sub_node("ah", Affine(pc, input_dim, arc_space, act=transform_act))
        else:
            self.arc_h = self.arc_m = None
            arc_space = input_dim
        self.arc_scorer = self.add_sub_node("as", PairScorer(pc, arc_space, arc_space, 1, sconf.ps_conf))
        # labeling
        if lab_space>0:
            self.lab_m = self.add_sub_node("lm", Affine(pc, input_dim, lab_space, act=transform_act))
            self.lab_h = self.add_sub_node("lh", Affine(pc, input_dim, lab_space, act=transform_act))
        else:
            self.lab_h = self.lab_m = None
            lab_space = input_dim
        self.lab_scorer = self.add_sub_node("ls", PairScorer(pc, lab_space, lab_space, self.num_label, sconf.ps_conf))

    # =====
    # score

    def transform_and_arc_score(self, senc_expr, mask_expr=None):
        ah_expr = self.arc_h(senc_expr) if self.arc_h else senc_expr
        am_expr = self.arc_m(senc_expr) if self.arc_m else senc_expr
        arc_full_score = self.arc_scorer.paired_score(am_expr, ah_expr, mask_expr, mask_expr)
        return arc_full_score  # [*, m, h, 1]

    def transform_and_lab_score(self, senc_expr, mask_expr=None):
        lh_expr = self.lab_h(senc_expr) if self.lab_h else senc_expr
        lm_expr = self.lab_m(senc_expr) if self.lab_m else senc_expr
        lab_full_score = self.lab_scorer.paired_score(lm_expr, lh_expr, mask_expr, mask_expr)
        return lab_full_score  # [*, m, h, Lab]

    def transform_and_arc_score_plain(self, mod_srepr, head_srepr, mask_expr=None):
        ah_expr = self.arc_h(head_srepr) if self.arc_h else head_srepr
        am_expr = self.arc_m(mod_srepr) if self.arc_m else mod_srepr
        arc_full_score = self.arc_scorer.plain_score(am_expr, ah_expr, mask_expr, mask_expr)
        return arc_full_score

    def transform_and_lab_score_plain(self, mod_srepr, head_srepr, mask_expr=None):
        lh_expr = self.lab_h(head_srepr) if self.lab_h else head_srepr
        lm_expr = self.lab_m(mod_srepr) if self.lab_m else mod_srepr
        lab_full_score = self.lab_scorer.plain_score(lm_expr, lh_expr, mask_expr, mask_expr)
        return lab_full_score

class FpSingleScorer(BasicNode):
    def __init__(self, pc: BK.ParamCollection, sconf: FpScorerConf):
        super().__init__(pc, None, None)
        # options
        input_dim = sconf._input_dim
        arc_space = sconf.arc_space
        lab_space = sconf.lab_space
        transform_act = sconf.transform_act
        #
        self.input_dim = input_dim
        self.num_label = sconf._num_label
        self.mask_value = Constants.REAL_PRAC_MIN
        # attach/arc
        if arc_space>0:
            self.arc_f = self.add_sub_node("af", Affine(pc, input_dim, arc_space, act=transform_act))
        else:
            self.arc_f = None
            arc_space = input_dim
        self.arc_scorer = self.add_sub_node("as", Affine(pc, arc_space, 1, init_rop=NoDropRop()))
        # labeling
        if lab_space>0:
            self.lab_f = self.add_sub_node("lf", Affine(pc, input_dim, lab_space, act=transform_act))
        else:
            self.lab_f = None
            lab_space = input_dim
        self.lab_scorer = self.add_sub_node("ls", Affine(pc, lab_space, self.num_label, init_rop=NoDropRop()))

    # =====
    # score

    def transform_and_arc_score(self, senc_expr, mask_expr=None):
        a_expr = self.arc_f(senc_expr) if self.arc_f else senc_expr
        arc_full_score = self.arc_scorer(a_expr)
        if mask_expr is not None:
            arc_full_score += self.mask_value * (1. - mask_expr).unsqueeze(-1)
        return arc_full_score  # [*, 1]

    def transform_and_lab_score(self, senc_expr, mask_expr=None):
        l_expr = self.lab_f(senc_expr) if self.lab_f else senc_expr
        lab_full_score = self.lab_scorer(l_expr)
        if mask_expr is not None:
            lab_full_score += self.mask_value * (1. - mask_expr).unsqueeze(-1)
        return lab_full_score  # [*, Lab]

# =====
# decoders

class FpDecConf(Conf):
    def __init__(self):
        self.sconf = FpScorerConf()
        self.use_ablpair = False  # special mode?
        # loss weight
        self.lambda_arc = 1.
        self.lambda_lab = 1.

class FpDecoder(BasicNode):
    def __init__(self, pc, conf: FpDecConf, label_vocab):
        super().__init__(pc, None, None)
        self.conf = conf
        self.label_vocab = label_vocab
        self.predict_padder = DataPadder(2, pad_vals=0)
        # the scorer
        self.use_ablpair = conf.use_ablpair
        conf.sconf._input_dim = conf._input_dim
        conf.sconf._num_label = conf._num_label
        if self.use_ablpair:
            self.scorer = self.add_sub_node("s", FpSingleScorer(pc, conf.sconf))
        else:
            self.scorer = self.add_sub_node("s", FpPairedScorer(pc, conf.sconf))

    # -----
    # scoring
    def _score(self, enc_expr, mask_expr):
        # -----
        def _special_score(one_score):  # specially change ablpair scores into [bs,m,h,*]
            root_score = one_score[:,:,0].unsqueeze(2)  # [bs, rlen, 1, *]
            tmp_shape = BK.get_shape(root_score)
            tmp_shape[1] = 1  # [bs, 1, 1, *]
            padded_root_score = BK.concat([BK.zeros(tmp_shape), root_score], dim=1)  # [bs, rlen+1, 1, *]
            final_score = BK.concat([padded_root_score, one_score.transpose(1,2)], dim=2)  # [bs, rlen+1[m], rlen+1[h], *]
            return final_score
        # -----
        if self.use_ablpair:
            input_mask_expr = (mask_expr.unsqueeze(-1) * mask_expr.unsqueeze(-2))[:, 1:]  # [bs, rlen, rlen+1]
            arc_score = self.scorer.transform_and_arc_score(enc_expr, input_mask_expr)  # [bs, rlen, rlen+1, 1]
            lab_score = self.scorer.transform_and_lab_score(enc_expr, input_mask_expr)  # [bs, rlen, rlen+1, Lab]
            # put root-scores for both directions
            arc_score = _special_score(arc_score)
            lab_score = _special_score(lab_score)
        else:
            # todo(+2): for training, we can simply select and lab-score
            arc_score = self.scorer.transform_and_arc_score(enc_expr, mask_expr)  # [bs, m, h, 1]
            lab_score = self.scorer.transform_and_lab_score(enc_expr, mask_expr)  # [bs, m, h, Lab]
        # mask out diag scores
        diag_mask = BK.eye(BK.get_shape(arc_score, 1))
        diag_mask[0,0] = 0.
        diag_add = Constants.REAL_PRAC_MIN * (diag_mask.unsqueeze(-1).unsqueeze(0))  # [1, m, h, 1]
        arc_score += diag_add
        lab_score += diag_add
        return arc_score, lab_score

    # loss
    # todo(note): no margins here, simply using target-selection cross-entropy
    def loss(self, insts: List[ParseInstance], enc_expr, mask_expr, **kwargs):
        conf = self.conf
        # scoring
        arc_score, lab_score = self._score(enc_expr, mask_expr)  # [bs, m, h, *]
        # loss
        bsize, max_len = BK.get_shape(mask_expr)
        # gold heads and labels
        gold_heads_arr, _ = self.predict_padder.pad([z.heads.vals for z in insts])
        # todo(note): here use the original idx of label, no shift!
        gold_labels_arr, _ = self.predict_padder.pad([z.labels.idxes for z in insts])
        gold_heads_expr = BK.input_idx(gold_heads_arr)  # [bs, Len]
        gold_labels_expr = BK.input_idx(gold_labels_arr)  # [bs, Len]
        # collect the losses
        arange_bs_expr = BK.arange_idx(bsize).unsqueeze(-1)  # [bs, 1]
        arange_m_expr = BK.arange_idx(max_len).unsqueeze(0)  # [1, Len]
        # logsoftmax and losses
        arc_logsoftmaxs = BK.log_softmax(arc_score.squeeze(-1), -1)  # [bs, m, h]
        lab_logsoftmaxs = BK.log_softmax(lab_score, -1)  # [bs, m, h, Lab]
        arc_sel_ls = arc_logsoftmaxs[arange_bs_expr, arange_m_expr, gold_heads_expr]  # [bs, Len]
        lab_sel_ls = lab_logsoftmaxs[arange_bs_expr, arange_m_expr, gold_heads_expr, gold_labels_expr]  # [bs, Len]
        # head selection (no root)
        arc_loss_sum = (- arc_sel_ls * mask_expr)[:, 1:].sum()
        lab_loss_sum = (- lab_sel_ls * mask_expr)[:, 1:].sum()
        final_loss = conf.lambda_arc * arc_loss_sum + conf.lambda_lab * lab_loss_sum
        final_loss_count = mask_expr[:, 1:].sum()
        return [[final_loss, final_loss_count]]

    # decode
    def predict(self, insts: List[ParseInstance], enc_expr, mask_expr, **kwargs):
        conf = self.conf
        # scoring
        arc_score, lab_score = self._score(enc_expr, mask_expr)  # [bs, m, h, *]
        full_score = BK.log_softmax(arc_score, -2) + BK.log_softmax(lab_score, -1)  # [bs, m, h, Lab]
        # decode
        mst_lengths = [len(z) + 1 for z in insts]  # +1 to include ROOT for mst decoding
        mst_lengths_arr = np.asarray(mst_lengths, dtype=np.int32)
        mst_heads_arr, mst_labels_arr, mst_scores_arr = \
            nmst_unproj(full_score, mask_expr, mst_lengths_arr, labeled=True, ret_arr=True)
        # ===== assign, todo(warn): here, the labels are directly original idx, no need to change
        misc_prefix = "g"
        for one_idx, one_inst in enumerate(insts):
            cur_length = mst_lengths[one_idx]
            one_inst.pred_heads.set_vals(mst_heads_arr[one_idx][:cur_length])  # directly int-val for heads
            one_inst.pred_labels.build_vals(mst_labels_arr[one_idx][:cur_length], self.label_vocab)
            one_scores = mst_scores_arr[one_idx][:cur_length]
            one_inst.pred_par_scores.set_vals(one_scores)
            # extra output
            one_inst.extra_pred_misc[misc_prefix+"_score"] = one_scores.tolist()
