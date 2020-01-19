#

# simplified two-step parsing
# -- based on structures by marginal masking/pruning

from typing import List
import numpy as np

from msp.utils import Constants, zlog, Helper, Conf
from msp.data import VocabPackage
from msp.nn import BK
from msp.nn.layers import BasicNode, MultiHeadAttention, AttConf, Affine
from msp.zext.seq_helper import DataPadder

from ...common.data import ParseInstance
from ...common.model import BaseParserConf, BaseInferenceConf, BaseTrainingConf, BaseParser
from ..scorer import Scorer, ScorerConf, SL1Layer, SL1Conf

from .g1p import G1Parser, G1ParserConf, PreG1Conf, PruneG1Conf

# overall
class S2ParserConf(G1ParserConf):
    def __init__(self):
        super().__init__()
        self.sc_conf = ScorerConf()
        self.sl_conf = SL1Conf()
        self.pre_g1_conf = PreG1Conf()
        #
        self.sl_as_enc = False  # put sl node into the encoder part (for optimization)
        # two pruning confs
        self.iconf.pruning_conf = None
        self.dprune = PruneG1Conf()  # decoding pruning (for scoring)
        self.sprune = PruneG1Conf()  # structured pruning (for structured layer)

# =====
# model

# the model
# todo(note): for simplicity, aggregate the features (sib, grandparent) at the head-node's srepr
class S2Parser(G1Parser):
    def __init__(self, conf: S2ParserConf, vpack: VocabPackage):
        super().__init__(conf, vpack)
        # ===== basic G1 Parser's loading
        self.g1parser = G1Parser.pre_g1_init(self, conf.pre_g1_conf)
        self.lambda_g1_arc_training = conf.pre_g1_conf.lambda_g1_arc_training
        self.lambda_g1_arc_testing = conf.pre_g1_conf.lambda_g1_arc_testing
        self.lambda_g1_lab_training = conf.pre_g1_conf.lambda_g1_lab_training
        self.lambda_g1_lab_testing = conf.pre_g1_conf.lambda_g1_lab_testing
        # ===== build extra inner part of scorer
        self.add_slayer()

    def build_slayer(self):
        conf = self.conf
        # ===== Structured Layer =====
        conf.sl_conf._input_dim = self.enc_output_dim
        conf.sl_conf._num_label = self.label_vocab.trg_len(True)  # todo(WARN): use the original idx
        return SL1Layer(self.pc, conf.sl_conf)

    # =====
    # similar to g2p for preparing steps

    # get g1 prunings and extra-scores
    # -> if no g1parser provided, then use aux scores; otherwise, it depends on g1_use_aux_scores
    def _get_g1_pack(self, insts: List[ParseInstance], mask_expr, score_arc_lambda: float, score_lab_lambda: float):
        if self.g1parser is None or self.g1parser.g1_use_aux_scores:
            arc_score, lab_score, _ = G1Parser.collect_aux_scores(insts, self.num_label)
        else:
            arc_score, lab_score = self.g1parser.score_on_batch(insts)
            arc_score = arc_score.unsqueeze(-1)
        # pruner1 (decoding pruning)
        valid_mask_d, arc_marginals = G1Parser.prune_with_scores(arc_score, lab_score, mask_expr, self.conf.dprune)
        # pruner2 (structured one) -- no need to recalculate arc_marginals
        valid_mask_s, _ = G1Parser.prune_with_scores(arc_score, lab_score, mask_expr, self.conf.sprune, arc_marginals)
        #
        if score_arc_lambda <= 0. and score_lab_lambda <= 0.:
            go1_pack = None
        else:
            arc_score *= score_arc_lambda
            lab_score *= score_lab_lambda
            go1_pack = (arc_score, lab_score)
        # [*, slen, slen], ([*, slen, slen], [*, slen, slen, Lab]), [*, m, h]
        assert arc_marginals is not None
        return valid_mask_d, valid_mask_s, go1_pack, arc_marginals

    # make real valid masks (inplaced): valid(byte): [bs, len-m, len-h]; mask(float): [bs, len]
    def _make_final_valid(self, valid_expr, mask_expr):
        maxlen = BK.get_shape(mask_expr, -1)
        # first apply masks
        mask_expr_byte = mask_expr.byte()
        valid_expr &= mask_expr_byte.unsqueeze(-1)
        valid_expr &= mask_expr_byte.unsqueeze(-2)
        # then diag
        mask_diag = 1-BK.eye(maxlen).byte()
        valid_expr &= mask_diag
        # root not as mod, todo(note): here no [0,0] since no need
        valid_expr[:, 0] = 0
        return valid_expr.float()

    # common calculations for both decoding and training
    def _score(self, insts: List[ParseInstance], training: bool, lambda_g1_arc: float, lambda_g1_lab: float):
        # encode
        input_repr, enc_repr, jpos_pack, mask_arr = self.bter.run(insts, training)
        mask_expr = BK.input_real(mask_arr)
        # pruning and scores from g1
        valid_mask_d, valid_mask_s, go1_pack, arc_marginals = self._get_g1_pack(insts, mask_expr, lambda_g1_arc, lambda_g1_lab)
        # s-encode (using s-mask)
        final_valid_expr_s = self._make_final_valid(valid_mask_s, mask_expr)
        senc_repr = self.slayer(enc_repr, final_valid_expr_s, arc_marginals)
        # decode
        arc_score = self.scorer_helper.score_arc(senc_repr)
        lab_score = self.scorer_helper.score_label(senc_repr)
        full_score = arc_score + lab_score
        # add go1 scores and apply pruning (using d-mask)
        final_valid_expr_d = valid_mask_d.float()  # no need to mask out others here!
        mask_value = Constants.REAL_PRAC_MIN
        if go1_pack is not None:
            go1_arc_score, go1_label_score = go1_pack
            full_score += go1_arc_score.unsqueeze(-1) + go1_label_score
        full_score += (mask_value * (1. - final_valid_expr_d)).unsqueeze(-1)
        # [*, m, h, lab], (original-scores), [*, m], [*, m, h]
        return full_score, (arc_score, lab_score), jpos_pack, mask_expr, final_valid_expr_d, final_valid_expr_s

    # decoding
    def inference_on_batch(self, insts: List[ParseInstance], **kwargs):
        # iconf = self.conf.iconf
        with BK.no_grad_env():
            self.refresh_batch(False)
            full_score, _, jpos_pack, mask_expr, _, _ = \
                self._score(insts, False, self.lambda_g1_arc_testing, self.lambda_g1_lab_testing)
            # collect the results together
            # =====
            self._decode(insts, full_score, mask_expr, "s2")
            # put jpos result (possibly)
            self.jpos_decode(insts, jpos_pack)
            # -----
            info = {"sent": len(insts), "tok": sum(map(len, insts))}
            return info

    # training
    def fb_on_batch(self, annotated_insts: List[ParseInstance], training=True, loss_factor=1., **kwargs):
        self.refresh_batch(training)
        # todo(note): here always using training lambdas
        full_score, original_scores, jpos_pack, mask_expr, valid_mask_d, _ = \
            self._score(annotated_insts, False, self.lambda_g1_arc_training, self.lambda_g1_lab_training)
        parsing_loss, info = self._loss(annotated_insts, full_score, mask_expr, valid_mask_d)
        # other loss?
        jpos_loss = self.jpos_loss(jpos_pack, mask_expr)
        reg_loss = self.reg_scores_loss(*original_scores)
        #
        info["loss_parse"] = BK.get_value(parsing_loss).item()
        final_loss = parsing_loss
        if jpos_loss is not None:
            info["loss_jpos"] = BK.get_value(jpos_loss).item()
            final_loss = parsing_loss + jpos_loss
        if reg_loss is not None:
            final_loss = final_loss + reg_loss
        info["fb"] = 1
        if training:
            BK.backward(final_loss, loss_factor)
        return info

