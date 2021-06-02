#

# the ef parser

from typing import List

from msp.utils import FileHelper, zlog
from msp.data import VocabPackage
from msp.nn import BK
from msp.zext.process_train import SVConf, ScheduledValue

from ...common.data import ParseInstance
from ...common.model import BaseParserConf, BaseInferenceConf, BaseTrainingConf, BaseParser, DepLabelHelper
from ..systems import EfSearcher, EfSearchConf, EfState, EfAction, ScorerHelper
from ..scorer import Scorer, ScorerConf, SL0Layer, SL0Conf

from .g1p import G1Parser, PreG1Conf

# =====
# confs

# decoding conf
class EfInferenceConf(BaseInferenceConf):
    def __init__(self):
        super().__init__()
        # by default beam search (no training-related settings)
        # self.search_conf = EfSearchConf().init_from_kwargs(plain_k_arc=5, oracle_mode="", plain_beam_size=5, cost_type="")
        self.search_conf = EfSearchConf().init_from_kwargs(plain_k_arc=1, oracle_mode="", plain_beam_size=1, cost_type="")

# training conf
class EfTrainingConf(BaseTrainingConf):
    def __init__(self):
        super().__init__()
        # by default beam+oracle search
        self.search_conf = EfSearchConf()
        self.loss_function = "hinge"
        self.loss_div_step = True  # loss /= steps or sents?
        self.loss_div_weights = False  # a further mode in loss_div_step mode to div by actual weights
        self.loss_div_fullbatch = True  # whether also count in no-error sentences when div
        # special weight for loss0?
        self.cost0_weight = SVConf().init_from_kwargs(val=1.0)
        # include the action even if it is hit in the predicted ones (does not treat differently with cost0_weight)
        self.include_hit_oracle = False
        # no blame for label if the arc is wrong
        self.noblame_label_on_arc_err = False
        #
        # feature dropout (to be convenient, only in final forward/backward run)
        self.fdrop_chs = 0.
        self.fdrop_par = 0.

# parser conf
class EfParserConf(BaseParserConf):
    def __init__(self):
        super().__init__(EfInferenceConf(), EfTrainingConf())
        self.sc_conf = ScorerConf()
        self.sl_conf = SL0Conf()
        self.pre_g1_conf = PreG1Conf()
        self.system_labeled = True
        # ignore certain children
        self.ef_ignore_chs = ""  # none, punct, func  # todo(+2): ignore parent labels?
        # adjustable arc beam size (<1 means not adjustable!!)
        self.aabs = SVConf().init_from_kwargs(val=0., max_val=5., mode="linear")

    def do_validate(self):
        # specific setting
        self.iconf.search_conf._system_labeled = self.system_labeled
        self.tconf.search_conf._system_labeled = self.system_labeled
        if not self.system_labeled:
            # do not use label-related features or strategies
            self.ef_ignore_chs = ""
            self.sl_conf.use_label_feat = False
            self.iconf.search_conf.cost_weight_label = 0.
            self.tconf.search_conf.cost_weight_label = 0.

# =====
# model

# the model
class EfParser(BaseParser):
    def __init__(self, conf: EfParserConf, vpack: VocabPackage):
        super().__init__(conf, vpack)
        # ===== basic G1 Parser's loading (also possibly load g1's params)
        self.g1parser = G1Parser.pre_g1_init(self, conf.pre_g1_conf)
        self.lambda_g1_arc_training = conf.pre_g1_conf.lambda_g1_arc_training
        self.lambda_g1_arc_testing = conf.pre_g1_conf.lambda_g1_arc_testing
        self.lambda_g1_lab_training = conf.pre_g1_conf.lambda_g1_lab_training
        self.lambda_g1_lab_testing = conf.pre_g1_conf.lambda_g1_lab_testing
        #
        self.add_slayer()
        # True if ignored
        ignore_chs_label_mask = DepLabelHelper.select_label_idxes(conf.ef_ignore_chs, self.label_vocab.keys(), True, True)
        # ===== For decoding =====
        self.inferencer = EfInferencer(self.scorer, self.slayer, conf.iconf, ignore_chs_label_mask)
        # ===== For training =====
        self.cost0_weight = ScheduledValue("c0w", conf.tconf.cost0_weight)
        self.add_scheduled_values(self.cost0_weight)
        self.losser = EfLosser(self.scorer, self.slayer, conf.tconf, self.margin, self.cost0_weight, ignore_chs_label_mask)
        # ===== adjustable beam size
        self.arc_abs = ScheduledValue("aabs", conf.aabs)
        self.add_scheduled_values(self.arc_abs)
        #
        self.num_label = self.label_vocab.trg_len(True)  # todo(WARN): use the original idx

    # todo(+3): ugly changing beam size, here!
    def refresh_batch(self, training: bool):
        super().refresh_batch(training)
        cur_arc_abs = int(self.arc_abs)
        if cur_arc_abs >= 1:  # only set if >=1
            self.inferencer.searcher.set_plain_arc_beam_size(cur_arc_abs)
            self.losser.searcher.set_plain_arc_beam_size(cur_arc_abs)

    def build_decoder(self):
        conf = self.conf
        # ===== Decoding Scorer =====
        conf.sc_conf._input_dim = self.enc_output_dim
        conf.sc_conf._num_label = self.label_vocab.trg_len(True)  # todo(WARN): use the original idx
        return Scorer(self.pc, conf.sc_conf)

    def build_slayer(self):
        conf = self.conf
        # ===== Structured Layer =====
        conf.sl_conf._input_dim = self.enc_output_dim
        conf.sl_conf._num_label = self.label_vocab.trg_len(True)  # todo(WARN): use the original idx
        return SL0Layer(self.pc, conf.sl_conf)

    # =====
    # main procedures

    # get g1 scores as extra ones
    def _get_g1_pack(self, insts: List[ParseInstance], score_arc_lambda: float, score_lab_lambda: float):
        if score_arc_lambda <= 0. and score_lab_lambda <= 0.:
            return None
        else:
            # only need scores
            if self.g1parser is None or self.g1parser.g1_use_aux_scores:
                arc_score, lab_score, _ = G1Parser.collect_aux_scores(insts, self.num_label)
                a, b = (arc_score.suqeeze(-1), lab_score)
            else:
                a, b = self.g1parser.score_on_batch(insts)
            # multiply by lambda here!
            a *= score_arc_lambda
            b *= score_lab_lambda
            return a, b

    # decoding
    def inference_on_batch(self, insts: List[ParseInstance], **kwargs):
        with BK.no_grad_env():
            self.refresh_batch(False)
            # encode
            input_repr, enc_repr, jpos_pack, mask_arr = self.bter.run(insts, False)
            # g1 score
            g1_pack = self._get_g1_pack(insts, self.lambda_g1_arc_testing, self.lambda_g1_lab_testing)
            # decode for parsing
            self.inferencer.decode(insts, enc_repr, mask_arr, g1_pack, self.label_vocab)
            # put jpos result (possibly)
            self.jpos_decode(insts, jpos_pack)
            # -----
            info = {"sent": len(insts), "tok": sum(map(len, insts))}
            return info

    # training
    def fb_on_batch(self, annotated_insts: List[ParseInstance], training=True, loss_factor=1., **kwargs):
        self.refresh_batch(training)
        # encode
        input_repr, enc_repr, jpos_pack, mask_arr = self.bter.run(annotated_insts, training)
        mask_expr = BK.input_real(mask_arr)
        # g1 score
        g1_pack = self._get_g1_pack(annotated_insts, self.lambda_g1_arc_training, self.lambda_g1_lab_training)
        # the parsing loss
        parsing_loss, parsing_scores, info = self.losser.loss(annotated_insts, enc_repr, mask_arr, g1_pack)
        # whether add jpos loss?
        jpos_loss = self.jpos_loss(jpos_pack, mask_expr)
        #
        no_loss = True
        final_loss = 0.
        if parsing_loss is None:
            info["loss_parse"] = 0.
        else:
            final_loss = final_loss + parsing_loss
            info["loss_parse"] = BK.get_value(parsing_loss).item()
            no_loss = False
        if jpos_loss is None:
            info["loss_jpos"] = 0.
        else:
            final_loss = final_loss + jpos_loss
            info["loss_jpos"] = BK.get_value(jpos_loss).item()
            no_loss = False
        if parsing_scores is not None:
            arc_scores, lab_scores = parsing_scores
            reg_loss = self.reg_scores_loss(arc_scores, lab_scores)
            if reg_loss is not None:
                final_loss = final_loss + reg_loss
        info["fb"] = 1
        if training and not no_loss:
            info["fb_back"] = 1
            BK.backward(final_loss, loss_factor)
        return info

# =====
# other components: decoder and trainer

class EfInferencer:
    def __init__(self, scorer: Scorer, slayer: SL0Layer, iconf: EfInferenceConf, ignore_chs_label_mask):
        self.searcher = EfSearcher.build(iconf.search_conf, scorer, slayer)
        self.hm_feature_getter0 = ScorerHelper.HmFeatureGetter(ignore_chs_label_mask, 0., 0.)  # nodrop for search

    # inplaced writing results
    def decode(self, insts: List[ParseInstance], enc_repr, mask_arr, g1_pack, label_vocab):
        # =====
        def _set_ef_extra(inst: ParseInstance, end_state: EfState):
            path_states = end_state.get_path()
            mods = [x.action.mod for x in path_states]
            inst.extra_pred_misc["ef_order"] = mods
            scores = [x.score for x in path_states]
            inst.extra_pred_misc["ef_score"] = scores
        # =====
        # todo(warn): may need sg if require k-best
        ags = self.searcher.start(insts, self.hm_feature_getter0, enc_repr, mask_arr, g1_pack, margin=0.)
        self.searcher.go(ags)
        # put the results inplaced
        for one_inst, one_ag in zip(insts, ags):
            best_state = max(one_ag.ends[0], key=lambda s: s.score_accu)
            one_inst.pred_heads.set_vals(best_state.list_arc)  # directly int-val for heads
            # todo(warn): already the correct labels, no need to transform
            one_inst.pred_labels.build_vals(best_state.list_label, label_vocab)
            # auxiliary info for ef action order
            _set_ef_extra(one_inst, best_state)
        return ags

class EfLosser:
    def __init__(self, scorer: Scorer, slayer: SL0Layer, tconf: EfTrainingConf, margin: ScheduledValue,
                 cost0_weight: ScheduledValue, ignore_chs_label_mask):
        self.searcher = EfSearcher.build(tconf.search_conf, scorer, slayer)
        self.scorer = scorer
        self.slayer = slayer
        self.margin = margin
        self.cost0_weight = cost0_weight
        assert tconf.loss_function == "hinge", "Currently only support max-margin"
        self.system_labeled = self.searcher.system_labeled
        #
        self.loss_div_fullbatch = tconf.loss_div_fullbatch
        self.loss_div_step = tconf.loss_div_step
        self.loss_div_weights = tconf.loss_div_weights
        self.include_hit_oracle = tconf.include_hit_oracle
        self.noblame_label_on_arc_err = tconf.noblame_label_on_arc_err
        #
        self.hm_feature_getter0 = ScorerHelper.HmFeatureGetter(ignore_chs_label_mask, 0., 0.)  # nodrop for search
        self.hm_feature_getter = ScorerHelper.HmFeatureGetter(ignore_chs_label_mask, tconf.fdrop_chs, tconf.fdrop_par)

    # helper function for adding the lists, mainly for adjusting cost-aware weights, collect for one piece
    # todo(note): in one piece, there cannot be actions with the same mod
    def _add_lists(self, plain_states: List[EfState], oracle_states: List[EfState], action_list: List,
                   arc_weight_list: List, label_weight_list: List, bidxes_list: List, bidx, cost0_weight):
        cur_size = len(plain_states)
        plain_actions, oracle_actions = [x.action for x in plain_states], [x.action for x in oracle_states]
        action_list.extend(plain_actions)
        action_list.extend(oracle_actions)
        bidxes_list.extend([bidx] * (cur_size * 2))
        include_hit_oracle = self.include_hit_oracle
        noblame_label_on_arc_err = self.noblame_label_on_arc_err
        # assign weights for the loss function
        if cost0_weight == 1.:
            cur_weight_list = [1.] * cur_size + [-1.] * cur_size
            arc_weight_list.extend(cur_weight_list)
            label_weight_list.extend(cur_weight_list)
            arc_valid_weights, label_valid_weights = cur_size, cur_size
        else:
            arc_valid_weights, label_valid_weights = 0, 0  # accumulated weights
            arc_cost0_mod_set, label_cost0_mod_set = set(), set()  # cost==0 mods' set
            # always based on the predicted ones
            for cur_s in plain_states:
                cur_mod = cur_s.action.mod
                cur_wrong_arc, cur_wrong_label = cur_s.wrong_al
                cur_w_arc = cur_w_label = 1.
                if cur_wrong_arc == 0:
                    cur_w_arc = cost0_weight
                    arc_cost0_mod_set.add(cur_mod)
                    if cur_wrong_label == 0:
                        cur_w_label = cost0_weight
                        label_cost0_mod_set.add(cur_mod)
                elif noblame_label_on_arc_err:
                    # in this mode, mainly blame the arc error
                    cur_w_label = cost0_weight
                    label_cost0_mod_set.add(cur_mod)
                # record them
                arc_valid_weights += cur_w_arc
                label_valid_weights += cur_w_label
                arc_weight_list.append(cur_w_arc)
                label_weight_list.append(cur_w_label)
            # for the oracles, depending on the strategy
            if include_hit_oracle:
                # average over all oracles
                oracle_w_arc, oracle_w_label = -arc_valid_weights/cur_size, -label_valid_weights/cur_size
                arc_weight_list.extend([oracle_w_arc]*cur_size)
                label_weight_list.extend([oracle_w_label]*cur_size)
            else:
                # special treatment for the hits
                # arc
                arc_cost0_mods = [int(a.mod in arc_cost0_mod_set) for a in oracle_actions]
                arc_cost0_count = sum(arc_cost0_mods)
                arc_hascost_count = cur_size - arc_cost0_count
                arc_avg_w = -(arc_valid_weights-cost0_weight*arc_cost0_count)/arc_hascost_count \
                    if arc_hascost_count>0 else 0.
                arc_weight_list.extend([-cost0_weight if z>0 else arc_avg_w for z in arc_cost0_mods])
                # label
                label_cost0_mods = [int(a.mod in label_cost0_mod_set) for a in oracle_actions]
                label_cost0_count = sum(label_cost0_mods)
                label_hascost_count = cur_size - label_cost0_count
                label_avg_w = -(label_valid_weights-cost0_weight*label_cost0_count)/label_hascost_count \
                    if label_hascost_count>0 else 0.
                label_weight_list.extend([-cost0_weight if z>0 else label_avg_w for z in label_cost0_mods])
        return arc_valid_weights, label_valid_weights

    # todo(warn): first do a cost-augmented search and then do forward-backward
    # todo(+N): still a slight diff in search and fb because of non-fix dropout of Att-module!
    def loss(self, insts: List[ParseInstance], enc_repr, mask_arr, g1_pack):
        # todo(WARN): may need sg if using other loss functions
        # first-round search
        cur_margin = self.margin.value
        cur_cost0_weight = self.cost0_weight.value
        with BK.no_grad_env():
            ags = self.searcher.start(insts, self.hm_feature_getter0, enc_repr, mask_arr, g1_pack, margin=cur_margin)
            self.searcher.go(ags)
        # then forward and backward
        # collect only loss-related actions
        toks_all, sent_all = 0, len(insts)
        pieces_all, pieces_no_cost, pieces_serr, pieces_valid = 0, 0, 0, 0
        toks_valid = 0
        arc_valid_weights, label_valid_weights = 0., 0.
        action_list, arc_weight_list, label_weight_list, bidxes_list = [], [], [], []
        bidx = 0
        # =====
        score_getter = self.searcher.ender.plain_ranker
        oracler_ranker = self.searcher.ender.oracle_ranker
        # =====
        for one_inst, one_ag in zip(insts, ags):
            cur_size = len(one_inst)  # excluding ROOT
            toks_all += cur_size
            # for all the pieces
            for sp in one_ag.special_points:
                plain_finals, oracle_finals = sp.plain_finals, sp.oracle_finals
                best_plain = max(plain_finals, key=score_getter)
                best_oracle = max(plain_finals+oracle_finals, key=oracler_ranker)
                cost_plain, cost_oracle = best_plain.cost_accu, best_oracle.cost_accu
                score_plain, score_oracle = score_getter(best_plain), score_getter(best_oracle)
                # if cost_oracle > 0.:  # the gold one cannot be searched?
                #     sent_oracle_has_cost += 1
                # add them
                pieces_all += 1
                if cost_plain <= cost_oracle:
                    pieces_no_cost += 1
                elif score_plain < score_oracle:
                    pieces_serr += 1  # search error
                else:
                    pieces_valid += 1
                    plain_states, oracle_states = best_plain.get_path(), best_oracle.get_path()
                    toks_valid += len(plain_states)
                    # Loss = score(best_plain) - score(best_oracle)
                    cur_aw, cur_lw = self._add_lists(plain_states, oracle_states, action_list,
                                    arc_weight_list, label_weight_list, bidxes_list, bidx, cur_cost0_weight)
                    arc_valid_weights += cur_aw
                    label_valid_weights += cur_lw
            bidx += 1
        # collect the losses
        info = {"sent": sent_all, "tok": toks_all, "tok_valid": toks_valid, "vw_arc": arc_valid_weights, "vw_lab": label_valid_weights,
                "pieces_all": pieces_all, "pieces_no_cost": pieces_no_cost, "pieces_serr": pieces_serr, "pieces_valid": pieces_valid}
        if toks_valid == 0:
            return None, None, info
        final_arc_loss_sum, final_label_loss_sum, arc_scores, label_scores = \
            self._loss(enc_repr, action_list, arc_weight_list, label_weight_list, bidxes_list)
        # todo(+1): other indicators?
        info["loss_sum_arc"] = BK.get_value(final_arc_loss_sum).item()
        info["loss_sum_lab"] = BK.get_value(final_label_loss_sum).item()
        # how to div
        if self.loss_div_step:
            if self.loss_div_weights:
                cur_div_arc = max(arc_valid_weights, 1.)
                cur_div_lab = max(label_valid_weights, 1.)
            else:
                cur_div_arc = cur_div_lab = (toks_all if self.loss_div_fullbatch else toks_valid)
        else:
            # todo(warn): here use pieces rather than sentences
            cur_div_arc = cur_div_lab = (pieces_all if self.loss_div_fullbatch else pieces_valid)
        final_loss = final_arc_loss_sum/cur_div_arc + final_label_loss_sum/cur_div_lab
        return final_loss, (arc_scores, label_scores), info

    # the weighted sum
    def _loss(self, enc_repr, action_list: List[EfAction], arc_weight_list: List[float], label_weight_list: List[float],
              bidxes_list: List[int]):
        # 1. collect (batched) features; todo(note): use prev state for scoring
        hm_features = self.hm_feature_getter.get_hm_features(action_list, [a.state_from for a in action_list])
        # 2. get new sreprs
        scorer = self.scorer
        s_enc = self.slayer
        bsize_range_t = BK.input_idx(bidxes_list)
        node_h_idxes_t, node_h_srepr = ScorerHelper.calc_repr(s_enc, hm_features[0], enc_repr, bsize_range_t)
        node_m_idxes_t, node_m_srepr = ScorerHelper.calc_repr(s_enc, hm_features[1], enc_repr, bsize_range_t)
        # label loss
        if self.system_labeled:
            node_lh_expr, _ = scorer.transform_space_label(node_h_srepr, True, False)
            _, node_lm_pack = scorer.transform_space_label(node_m_srepr, False, True)
            label_scores_full = scorer.score_label(node_lm_pack, node_lh_expr)  # [*, Lab]
            label_scores = BK.gather_one_lastdim(label_scores_full, [a.label for a in action_list]).squeeze(-1)
            final_label_loss_sum = (label_scores * BK.input_real(label_weight_list)).sum()
        else:
            label_scores = final_label_loss_sum = BK.zeros([])
        # arc loss
        node_ah_expr, _ = scorer.transform_space_arc(node_h_srepr, True, False)
        _, node_am_pack = scorer.transform_space_arc(node_m_srepr, False, True)
        arc_scores = scorer.score_arc(node_am_pack, node_ah_expr).squeeze(-1)
        final_arc_loss_sum = (arc_scores * BK.input_real(arc_weight_list)).sum()
        # score reg
        return final_arc_loss_sum, final_label_loss_sum, arc_scores, label_scores

# todo(WARN): very interesting! it seems that with the decoder fixed, ef mode can be trained good enough (almost similar to g1), but letting it trainable actually cannot reach that high, hyper-parameter problems or sth else?
# --> another point is that with fix-decoder g1 model, direct ef-decoding can get similar results, but now worse. (seems natural, previously we acutally have a weak/general? decoder that is not that strong)

# b tasks/zdpar/ef/parser/efp.py:154
