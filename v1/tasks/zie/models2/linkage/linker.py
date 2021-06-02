#

# simple pairwise linker

from typing import List
import numpy as np
from msp.utils import Conf, Constants
from msp.nn import BK
from msp.nn.layers import BasicNode, Affine, BiAffineScorer, Embedding
from msp.zext.ie import HLabelVocab

#
class LinkerConf(Conf):
    def __init__(self):
        self._input_dim = -1
        # todo(note): this value should be enough
        self._num_ef_label = 200
        self._num_evt_label = 200
        # input labels
        self.dim_label = 128
        self.zero_unk_lemb = True
        # space transferring
        self.use_arc_score = True
        self.arc_space = 512
        self.lab_space = 128
        # -----
        # final biaffine scoring
        self.transform_act = "elu"
        self.ff_hid_size = 0
        self.ff_hid_layer = 0
        self.use_biaffine = True
        self.use_ff = False
        self.use_ff2 = True
        self.biaffine_div = 1.
        self.biaffine_init_ortho = False
        #
        self.nil_score0 = True
        self.train_min_rate = 0.5  # a min selecting rate for neg pairs
        self.train_drop_ef_lab = 0.5
        self.train_drop_evt_lab = 0.5
        #
        self.linker_ef_detach = False
        self.linker_evt_detach = False

# todo(note): similar to biaffine parser
class Linker(BasicNode):
    def __init__(self, pc, conf: LinkerConf, vocab: HLabelVocab):
        super().__init__(pc, None, None)
        self.conf = conf
        self.vocab = vocab
        assert vocab.nil_as_zero
        assert len(vocab.layered_hlidx)==1, "Currently we only allow one layer role"
        self.hl_output_size = len(vocab.layered_hlidx[0])  # num of output labels
        # -----
        # models
        sconf = conf
        input_dim = sconf._input_dim
        dim_label = sconf.dim_label
        arc_space = sconf.arc_space
        lab_space = sconf.lab_space
        ff_hid_size = sconf.ff_hid_size
        ff_hid_layer = sconf.ff_hid_layer
        use_biaffine = sconf.use_biaffine
        use_ff = sconf.use_ff
        use_ff2 = sconf.use_ff2
        biaffine_div = sconf.biaffine_div
        biaffine_init_ortho = sconf.biaffine_init_ortho
        transform_act = sconf.transform_act
        #
        self.input_dim = input_dim
        self.num_label = self.hl_output_size
        # label embeddings
        self.emb_ef = self.add_sub_node("eef", Embedding(pc, conf._num_ef_label, dim_label, fix_row0=sconf.zero_unk_lemb))
        self.emb_evt = self.add_sub_node("eevt", Embedding(pc, conf._num_evt_label, dim_label, fix_row0=sconf.zero_unk_lemb))
        # attach/arc
        self.arc_m = self.add_sub_node("am", Affine(pc, [input_dim, dim_label], arc_space, act=transform_act))
        self.arc_h = self.add_sub_node("ah", Affine(pc, [input_dim, dim_label], arc_space, act=transform_act))
        self.arc_scorer = self.add_sub_node(
            "as", BiAffineScorer(pc, arc_space, arc_space, 1, ff_hid_size, ff_hid_layer=ff_hid_layer,
                                 use_biaffine=use_biaffine, use_ff=use_ff, use_ff2=use_ff2,
                                 biaffine_div=biaffine_div, biaffine_init_ortho=biaffine_init_ortho))
        # labeling
        self.lab_m = self.add_sub_node("lm", Affine(pc, [input_dim, dim_label], lab_space, act=transform_act))
        self.lab_h = self.add_sub_node("lh", Affine(pc, [input_dim, dim_label], lab_space, act=transform_act))
        self.lab_scorer = self.add_sub_node(
            "ls", BiAffineScorer(pc, lab_space, lab_space, self.num_label, ff_hid_size, ff_hid_layer=ff_hid_layer,
                                 use_biaffine=use_biaffine, use_ff=use_ff, use_ff2=use_ff2,
                                 biaffine_div=biaffine_div, biaffine_init_ortho=biaffine_init_ortho))
        #
        self.nil_mask = None

    # to outside
    @property
    def UNK_IDX(self):
        return -1

    def hlidx2idx(self, hlidx) -> int:
        return hlidx.get_idx(0)

    def idx2hlidx(self, idx: int):
        return self.vocab.get_hlidx(idx, 1)

    #
    def refresh(self, rop=None):
        super().refresh(rop)
        self.nil_mask = BK.input_real([0.] + [1.] * (self.hl_output_size-1))  # used for later masking

    # [*, len-ef, D], [*, len-evt, D]
    def _score(self, repr_ef, repr_evt, lab_ef, lab_evt):
        # =====
        emb_lab_ef = self.emb_ef(lab_ef)  # [*, len-ef, D']
        emb_lab_evt = self.emb_evt(lab_evt)  # [*, len-evt, D']
        # label score: [*, len-ef, len-evt, Lab]
        lh_expr = self.lab_h([repr_evt, emb_lab_evt])
        lm_expr = self.lab_m([repr_ef, emb_lab_ef])
        lab_full_score = self.lab_scorer.paired_score(lm_expr, lh_expr)
        if self.conf.use_arc_score:
            # arc score: [*, len-ef, len-evt, 1]
            ah_expr = self.arc_h([repr_evt, emb_lab_evt])
            am_expr = self.arc_m([repr_ef, emb_lab_ef])
            arc_full_score = self.arc_scorer.paired_score(am_expr, ah_expr)
            full_score = lab_full_score + arc_full_score.expand([-1, -1, -1, self.hl_output_size]) * self.nil_mask
        else:
            full_score = lab_full_score
        # if zero nil dim
        if self.conf.nil_score0:
            full_score *= self.nil_mask
        return full_score

    # dropout idxes
    def _dropout_idxes(self, idxes, rate):
        zero_mask = (BK.rand(BK.get_shape(idxes)) < rate).long()
        return zero_mask * idxes

    # [*, len-ef, D], [*, len-evt, D], [*, len-ef], [*, len-evt]
    def predict(self, repr_ef, repr_evt, lab_ef, lab_evt, mask_ef=None, mask_evt=None, ret_full_logprobs=False):
        # -----
        ret_shape = BK.get_shape(lab_ef)[:-1] + [BK.get_shape(lab_ef, -1), BK.get_shape(lab_evt, -1)]
        if np.prod(ret_shape) == 0:
            if ret_full_logprobs:
                return BK.zeros(ret_shape+[self.num_label])
            else:
                return BK.zeros(ret_shape), BK.zeros(ret_shape).long()
        # -----
        # todo(note): +1 for space of DROPED(UNK)
        full_score = self._score(repr_ef, repr_evt, lab_ef+1, lab_evt+1)  # [*, len-ef, len-evt, D]
        full_logprobs = BK.log_softmax(full_score, -1)
        if ret_full_logprobs:
            return full_logprobs
        else:
            # greedy maximum decode
            ret_logprobs, ret_idxes = full_logprobs.max(-1)  # [*, len-ef, len-evt]
            # mask non-valid ones
            if mask_ef is not None:
                ret_idxes *= (mask_ef.unsqueeze(-1)).long()
            if mask_evt is not None:
                ret_idxes *= (mask_evt.unsqueeze(-2)).long()
            return ret_logprobs, ret_idxes

    # [*, len-ef, D], [*, len-evt, D], [*, len-ef], [*, len-evt]; [*, len-ef, len-evt]
    def loss(self, repr_ef, repr_evt, lab_ef, lab_evt, mask_ef, mask_evt, gold_idxes, margin=0.):
        conf = self.conf
        # -----
        if np.prod(BK.get_shape(gold_idxes)) == 0:
            return [[BK.zeros([]), BK.zeros([])]]
        # -----
        # todo(note): +1 for space of DROPED(UNK)
        lab_ef = self._dropout_idxes(lab_ef+1, conf.train_drop_ef_lab)
        lab_evt = self._dropout_idxes(lab_evt+1, conf.train_drop_evt_lab)
        if conf.linker_ef_detach:
            repr_ef = repr_ef.detach()
        if conf.linker_evt_detach:
            repr_evt = repr_evt.detach()
        full_score = self._score(repr_ef, repr_evt, lab_ef, lab_evt)  # [*, len-ef, len-evt, D]
        if margin>0.:
            aug_score = BK.zeros(BK.get_shape(full_score)) + margin
            aug_score.scatter_(-1, gold_idxes.unsqueeze(-1), 0.)
            full_score += aug_score
        full_logprobs = BK.log_softmax(full_score, -1)
        gold_logprobs = full_logprobs.gather(-1, gold_idxes.unsqueeze(-1)).squeeze(-1)  # [*, len-ef, len-evt]
        # sampling and mask
        loss_mask = mask_ef.unsqueeze(-1) * mask_evt.unsqueeze(-2)
        # ====
        # first select examples (randomly)
        sel_mask = (BK.rand(BK.get_shape(loss_mask)) < conf.train_min_rate).float()  # [*, len-ef, len-evt]
        # add gold and exclude pad
        sel_mask += (gold_idxes>0).float()
        sel_mask.clamp_(max=1.)
        loss_mask *= sel_mask
        # =====
        loss_sum = - (gold_logprobs * loss_mask).sum()
        loss_count = loss_mask.sum()
        ret_losses = [[loss_sum, loss_count]]
        return ret_losses

    #
    def lookup(self, **kwargs):
        raise NotImplementedError("Currently no need for this!!")
