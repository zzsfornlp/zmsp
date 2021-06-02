#

# dependency parser (currently simply a first-order graph one)

from typing import List, Dict
import numpy as np
from msp.data import Vocab
from msp.utils import Conf, Random, Constants
from msp.nn import BK
from msp.nn.layers import PairScorerConf, PairScorer, Affine
from ..base import BaseModuleConf, BaseModule, LossHelper, SVConf, ScheduledValue
from .embedder import Inputter
from ...data.insts import GeneralSentence, DepTreeField
from tasks.zdpar.algo import nmst_unproj

# conf
class DparG1DecoderConf(BaseModuleConf):
    def __init__(self):
        super().__init__()
        # --
        # space transferring for individual inputs (0 means no this layer)
        self.pre_dp_space = 0
        self.pre_dp_act = "elu"
        # dependency relation pairwise conf
        self.dps_conf = PairScorerConf().init_from_kwargs(use_input_pair=True, use_biaffine=False, use_ff1=False, use_ff2=True)
        # fix idx=0 score?
        self.fix_s0 = True  # whether fix idx=0(NOPE) scores
        self.fix_s0_val = 0.  # if fix, use what fix val?
        # minus idx=0 score?
        self.minus_s0 = True  # whether minus idx=0(NOPE) scores
        # loss lambdas
        self.lambda_label = 1.  # on the last dim
        self.label_neg_rate = 0.1  # sample how much neg examples (label=0=nope)
        self.lambda_head = 1.  # simply max on last dim and on -2 dim
        # detach input? (for example, warmup for dec?)
        # "dpar_conf.no_detach_input.mode:linear dpar_conf.no_detach_input.start_bias:2"
        self.no_detach_input = SVConf().init_from_kwargs(val=1.)

# module
class DparG1Decoder(BaseModule):
    def __init__(self, pc, input_dim: int, inputp_dim: int, conf: DparG1DecoderConf, inputter: Inputter):
        super().__init__(pc, conf, name="dp")
        self.conf = conf
        self.inputter = inputter
        self.input_dim = input_dim
        self.inputp_dim = inputp_dim
        # checkout and assign vocab
        self._check_vocab()
        # -----
        # this step is performed at the embedder, thus still does not influence the inputter
        self.add_root_token = self.inputter.embedder.add_root_token
        assert self.add_root_token, "Currently assert this one!!"  # todo(+N)
        # -----
        # transform dp space
        if conf.pre_dp_space > 0:
            dp_space = conf.pre_dp_space
            self.pre_aff_m = self.add_sub_node("pm", Affine(pc, input_dim, dp_space, act=conf.pre_dp_act))
            self.pre_aff_h = self.add_sub_node("ph", Affine(pc, input_dim, dp_space, act=conf.pre_dp_act))
        else:
            dp_space = input_dim
            self.pre_aff_m = self.pre_aff_h = lambda x: x
        # dep pairwise scorer: output includes [0, r1) -> [non]+valid_words
        self.dps_node = self.add_sub_node("dps", PairScorer(pc, dp_space, dp_space, self.dlab_r1,
                                                            conf=conf.dps_conf, in_size_pair=inputp_dim))
        self.dps_s0_mask = np.array([1.]+[0.]*(self.dlab_r1-1))  # [0, 1, ..., 1]
        # whether detach input?
        self.no_detach_input = ScheduledValue(f"dpar:no_detach", conf.no_detach_input)

    def get_scheduled_values(self):
        return super().get_scheduled_values() + [self.no_detach_input]

    # check vocab at init
    def _check_vocab(self):
        # check range
        _l_vocab: Vocab = self.inputter.vpack.get_voc("deplabel")
        r0, r1 = _l_vocab.nonspecial_idx_range()  # [r0, r1) is the valid range
        assert _l_vocab.non==0 and r0 == 1, "Should have preserve only idx=0 as NO-Relation"
        # assign
        self.dlab_vocab: Vocab = _l_vocab
        self.dlab_r0, self.dlab_r1 = r0, r1  # no unk or others

    # -----
    # scoring
    # [bs, slen, D], [bs, len_q, len_k, D'], [bs, slen]
    def _score(self, repr_t, attn_t, mask_t):
        conf = self.conf
        # -----
        repr_m = self.pre_aff_m(repr_t)  # [bs, slen, S]
        repr_h = self.pre_aff_h(repr_t)  # [bs, slen, S]
        scores0 = self.dps_node.paired_score(repr_m, repr_h, inputp=attn_t)  # [bs, len_q, len_k, 1+N]
        # mask at outside
        slen = BK.get_shape(mask_t, -1)
        score_mask = BK.constants(BK.get_shape(scores0)[:-1], 1.)  # [bs, len_q, len_k]
        score_mask *= (1.-BK.eye(slen))  # no diag
        score_mask *= mask_t.unsqueeze(-1)  # input mask at len_k
        score_mask *= mask_t.unsqueeze(-2)  # input mask at len_q
        NEG = Constants.REAL_PRAC_MIN
        scores1 = scores0 + NEG * (1.-score_mask.unsqueeze(-1))  # [bs, len_q, len_k, 1+N]
        # add fixed idx0 scores if set
        if conf.fix_s0:
            fix_s0_mask_t = BK.input_real(self.dps_s0_mask)  # [1+N]
            scores1 = (1.-fix_s0_mask_t) * scores1 + fix_s0_mask_t * conf.fix_s0_val  # [bs, len_q, len_k, 1+N]
        # minus s0
        if conf.minus_s0:
            scores1 = scores1 - scores1.narrow(-1, 0, 1)  # minus idx=0 scores
        return scores1, score_mask

    # get ranged label scores for head
    def _ranged_label_scores(self, label_scores):
        return label_scores.narrow(-1, self.dlab_r0, self.dlab_r1-self.dlab_r0)

    # loss
    def loss(self, insts: List[GeneralSentence], repr_t, attn_t, mask_t, **kwargs):
        conf = self.conf
        # detach input?
        if self.no_detach_input.value <= 0.:
            repr_t = repr_t.detach()  # no grad back if no_detach_input<=0.
        # scoring
        label_scores, score_masks = self._score(repr_t, attn_t, mask_t)  # [bs, len_q, len_k, 1+N], [bs, len_q, len_k]
        # -----
        # get golds
        bsize, max_len = BK.get_shape(mask_t)
        shape_lidxes = [bsize, max_len, max_len]
        gold_lidxes = np.zeros(shape_lidxes, dtype=np.long)  # [bs, mlen, mlen]
        gold_heads = np.zeros(shape_lidxes[:-1], dtype=np.long)  # [bs, mlen]
        for bidx, inst in enumerate(insts):
            cur_dep_tree = inst.dep_tree
            cur_len = len(cur_dep_tree)
            gold_lidxes[bidx, :cur_len, :cur_len] = cur_dep_tree.label_matrix
            gold_heads[bidx, :cur_len] = cur_dep_tree.heads
        # -----
        margin = self.margin.value
        all_losses = []
        # first is loss_labels
        lambda_label = conf.lambda_label
        if lambda_label > 0.:
            gold_lidxes_t = BK.input_idx(gold_lidxes)  # [bs, len_q, len_k]
            label_losses = BK.loss_nll(label_scores, gold_lidxes_t, margin=margin)  # [bs, mlen, mlen]
            positive_mask_t = (gold_lidxes_t>0).float()  # [bs, mlen, mlen]
            negative_mask_t = (BK.rand(shape_lidxes) < conf.label_neg_rate).float()  # [bs, mlen, mlen]
            loss_mask_t = score_masks * (positive_mask_t + negative_mask_t)  # [bs, mlen, mlen]
            loss_mask_t.clamp_(max=1.)
            masked_label_losses = label_losses * loss_mask_t
            # compile loss
            final_label_loss = LossHelper.compile_leaf_info(f"label", masked_label_losses.sum(), loss_mask_t.sum(),
                                                            loss_lambda=lambda_label, npos=positive_mask_t.sum())
            all_losses.append(final_label_loss)
        # then head loss
        lambda_head = conf.lambda_head
        if lambda_head > 0.:
            # get head score simply by argmax on ranges
            head_scores, _ = self._ranged_label_scores(label_scores).max(-1)  # [bs, mlen, mlen]
            gold_heads_t = BK.input_idx(gold_heads)
            head_losses = BK.loss_nll(head_scores, gold_heads_t, margin=margin)  # [bs, mlen]
            # mask
            head_mask_t = BK.copy(mask_t)
            head_mask_t[:, 0] = 0  # not for ARTI_ROOT
            masked_head_losses = head_losses * head_mask_t
            # compile loss
            final_head_loss = LossHelper.compile_leaf_info(f"head", masked_head_losses.sum(), head_mask_t.sum(),
                                                           loss_lambda=lambda_label)
            all_losses.append(final_head_loss)
        # --
        return self._compile_component_loss("dp", all_losses)

    # decode
    def predict(self, insts: List[GeneralSentence], repr_t, attn_t, mask_t, **kwargs):
        conf = self.conf
        # scoring
        label_scores, score_masks = self._score(repr_t, attn_t, mask_t)  # [bs, len_q, len_k, 1+N], [bs, len_q, len_k]
        # get ranged label scores
        ranged_label_scores = self._ranged_label_scores(label_scores)  # [bs, m, h, RealLab]
        # decode
        mst_lengths = [len(z) + 1 for z in insts]  # +1 to include ROOT for mst decoding
        mst_lengths_arr = np.asarray(mst_lengths, dtype=np.int32)
        mst_heads_arr, mst_labels_arr, mst_scores_arr = \
            nmst_unproj(ranged_label_scores, mask_t, mst_lengths_arr, labeled=True, ret_arr=True)
        mst_labels_arr += self.dlab_r0  # add back offset
        # assign; todo(+2): record scores?
        for one_idx, one_inst in enumerate(insts):
            cur_length = mst_lengths[one_idx]
            cur_heads = [0] + mst_heads_arr[one_idx][1:cur_length].tolist()
            cur_labels = DepTreeField.build_label_vals(mst_labels_arr[one_idx][:cur_length], self.dlab_vocab)
            pred_dep_tree = DepTreeField(cur_heads, cur_labels, mst_labels_arr[one_idx][:cur_length].tolist())
            one_inst.add_item("pred_dep_tree", pred_dep_tree, assert_non_exist=False)
        return

# b tasks/zmlm/model/mods/dpar.py:?
