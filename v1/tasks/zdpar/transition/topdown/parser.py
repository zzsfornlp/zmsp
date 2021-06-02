#

# the top-down transition-based parser

import numpy as np
from typing import List

from msp.utils import Conf, zlog, JsonRW, zwarn, zcheck
from msp.data import VocabPackage
from msp.nn import BK
from msp.zext.process_train import SVConf, ScheduledValue

from ...common.data import ParseInstance
from ...common.model import BaseParserConf, BaseInferenceConf, BaseTrainingConf, BaseParser
from .decoder import TdScorerConf, TdScorer, TdState
from .runner import TdInferencer, TdFber

# =====
# confs

# decoding conf
class InferenceConf(BaseInferenceConf):
    def __init__(self):
        super().__init__()
        # self.dec_algorithm = "TD"
        # expand
        self.expand_strategy = "free"       # constrains for expanding candidates
        self.expand_projective = False      # projective expanding?
        # beam sizes
        self.global_beam_size = 5           # beam size (global_beam_size) or sampling size (global-selector size)
        self.local_arc_beam_size = 5
        self.local_label_beam_size = 2
        # for sampling
        self.topk_sample = False            # constraining topk for sampling mode?
        # slightly advanced beam
        self.merge_sig_type = "none"        # none, plain, ...
        self.attach_num_beam_size = 5       # secondary beam of how many attaching edges that current state has
        # todo(warn): special mode, check search error
        self.check_serr = False

# training conf
class TraningConf(BaseTrainingConf):
    def __init__(self):
        super().__init__()
        # oracle
        self.oracle_strategy = "free"       # constrains for expanding candidates
        self.oracle_projective = False      # projective expanding?
        self.oracle_free_dist_alpha = 0.    # for free oracle, distribution \alpha for prob <= e^(|dist|*alpha)
        self.oracle_label_order = "freq"    # core/freq
        # training strategy
        self.train_strategy = "force"       # force(teacher-forcing), ss(scheduled-sampling), rl, of(oracle-force/sample)
        self.loss_div_step = True           # loss /= steps or sents?
        self.oracle_log_prob_sum = False        # use log sum prob of all oracles for of/force-mode
        # loss functions & general
        # TODO(+N)
        # depth scheduling (100 levels should be enough, set 'mode=linear' and 'init_val=1.' to turn on!
        self.sched_depth = SVConf().init_from_kwargs(val=100., which_idx="eidx", mode="none", k=1., b=1., scale=1.)
        # for scheduled sampling
        # the oracler only returns one oracle, but the current sampling can be anther zero-loss good operation,
        #  whether strcitly follow oracle or accept this one rather than possible random oracle as local loss
        self.ss_strict_oracle = False
        # include rate of correct states in loss (effective ss_strict_oracle==False)
        self.ss_include_correct_rate = 0.1
        # alpha for local normalization
        self.local_norm_alpha = 1.0

# overall parser conf
class TdParserConf(BaseParserConf):
    def __init__(self):
        super().__init__(InferenceConf(), TraningConf())
        self.sc_conf = TdScorerConf()
        #
        self.is_bfs = False
        # output
        # self.sc_conf.output_local_norm = ?
        # self.iconf.dec_algorithm = "?"
        # self.tconf.loss_function = "?"

# =====
# model

# the model
class TdParser(BaseParser):
    def __init__(self, conf: TdParserConf, vpack: VocabPackage):
        super().__init__(conf, vpack)
        # ===== For decoding =====
        self.inferencer = TdInferencer(self.scorer, conf.iconf)
        # ===== For training =====
        sched_depth = ScheduledValue("depth", conf.tconf.sched_depth)
        self.add_scheduled_values(sched_depth)
        self.fber = TdFber(self.scorer, conf.iconf, conf.tconf, self.margin, self.sched_sampling, sched_depth)
        # todo(warn): not elegant, global flag!
        TdState.is_bfs = conf.is_bfs
        # =====
        zcheck(not self.bter.jpos_multitask_enabled(), "Not implemented for joint pos in this mode!!")
        zwarn("WARN: This topdown mode is deprecated!!")

    def build_decoder(self):
        conf = self.conf
        conf.sc_conf._input_dim = self.enc_output_dim
        conf.sc_conf._num_label = self.label_vocab.trg_len(True)  # todo(warn): plus 0 for reduce-label
        return TdScorer(self.pc, conf.sc_conf)

    # called before each mini-batch
    def refresh_batch(self, training):
        super().refresh_batch(training)
        # todo(+N): not elegant to put it here!
        if training:
            self.scorer.set_local_norm_alpha(self.conf.tconf.local_norm_alpha)
        else:
            self.scorer.set_local_norm_alpha(1.)

    # obtaining the instance preparer for training
    def get_inst_preper(self, training, **kwargs):
        if training:
            return self.fber.oracle_manager.init_inst           # method
        else:
            return None

    # =====
    # main procedures
    def _prepare_score(self, insts, training):
        input_repr, enc_repr, jpos_pack, mask_arr = self.bter.run(insts, training)
        # assert jpos_pack[0] is None, "not allowing jpos in this mode yet!"
        # mask_expr = BK.input_real(mask_arr)           # do not need this as in Graph-parser
        mask_expr = None
        scoring_expr_pack = self.scorer.transform_space(enc_repr)
        return scoring_expr_pack, mask_expr

    # get children order info from state
    def _get_chs_ordering(self, state: TdState):
        assert state.is_end()
        # collect all last reduce operations
        remain = state.num_tok
        ret = [""] * remain
        while remain>0 and state is not None:
            idx_cur = state.idx_cur
            if len(ret[idx_cur]) == 0:
                act_head, act_attach, act_label = state.action
                if act_head == act_attach:
                    ret[idx_cur] = f"CORDER={','.join([str(z) for z in state.idxes_chs])}"
                remain -= 1
            state = state.prev
        return ret

    # decoding
    def inference_on_batch(self, insts: List[ParseInstance], **kwargs):
        _LEQ_MAP = lambda x, m: [max(m, z) for z in x]  # larger or equal map
        #
        with BK.no_grad_env():
            self.refresh_batch(False)
            scoring_prep_f = lambda ones: self._prepare_score(ones, False)[0]
            get_best_state_f = lambda ag: sorted(ag.ends, key=lambda s: s.score_accu, reverse=True)[0]
            ags, info = self.inferencer.decode(insts, scoring_prep_f, False)
            # put the results inplaced
            for one_inst, one_ag in zip(insts, ags):
                best_state = get_best_state_f(one_ag)
                one_inst.pred_heads.set_vals(_LEQ_MAP(best_state.list_arc, 0))  # directly int-val for heads
                # todo(warn): already the correct labels, no need to transform
                # -- one_inst.pred_labels.build_vals(self.pred2real_labels(best_state.list_label), self.label_vocab)
                one_inst.pred_labels.build_vals(_LEQ_MAP(best_state.list_label, 1), self.label_vocab)
                # todo(warn): add children ordering in MISC field
                one_inst.pred_miscs.set_vals(self._get_chs_ordering(best_state))
            # check search err
            if self.conf.iconf.check_serr:
                ags_fo, _ = self.fber.force_decode(insts, scoring_prep_f, False)
                serr_sent = 0
                serr_tok = 0
                for ag, ag_fo in zip(ags, ags_fo):
                    best_state = get_best_state_f(ag)
                    best_fo_state = get_best_state_f(ag_fo)
                    # for this one, only care about UAS
                    if best_state.score_accu < best_fo_state.score_accu:
                        cur_serr_tok = sum((1 if a!=b else 0) for a,b in zip(best_state.list_arc[1:], best_fo_state.list_arc[1:]))
                        if cur_serr_tok > 0:
                            serr_sent += 1
                            serr_tok += cur_serr_tok
                info["serr_sent"] = serr_sent
                info["serr_tok"] = serr_tok
            return info

    # training (forward/backward)
    def fb_on_batch(self, annotated_insts: List[ParseInstance], training=True, loss_factor=1.):
        self.refresh_batch(training)
        scoring_expr_pack, mask_expr = self._prepare_score(annotated_insts, training)
        info = self.fber.fb(annotated_insts, scoring_expr_pack, training, loss_factor)
        return info

# b tasks/zdpar/transition/topdown/parser:150
