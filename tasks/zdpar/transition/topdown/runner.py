#
import copy

from msp.nn import BK, SliceManager
from msp.utils import Helper
from msp.zext.process_train import ScheduledValue
from .decoder import TdSearcher, TdOracleManager
from .decoder import BfsLinearAgenda, TdState

# =====
# decoder
class TdInferencer:
    def __init__(self, scorer, iconf, oracle_manager=None, force_oracle=False):
        self.searcher = TdSearcher.create_beam_searcher(scorer, oracle_manager, iconf, force_oracle)
        # todo(warn): specify the non-failing backoff one
        bo_iconf = copy.copy(iconf)
        bo_iconf.expand_strategy = "free"
        self.backoff_searcher = TdSearcher.create_beam_searcher(scorer, oracle_manager, bo_iconf, force_oracle)
        self.require_sg = False         # whether need the recordings of SearchGraph

    # return final states
    def decode(self, insts, scoring_prep_f, backoff):
        cur_searcher = self.backoff_searcher if backoff else self.searcher
        # decoding
        scoring_expr_pack = scoring_prep_f(insts)
        ags = [BfsLinearAgenda.init_agenda(TdState, z, self.require_sg) for z in insts]
        cur_searcher.refresh(scoring_expr_pack)
        cur_searcher.go(ags)
        # backoff
        failed_insts = []
        failed_idxes = []
        idx = 0
        for one_inst, one_ag in zip(insts, ags):
            if len(one_ag.ends) == 0:
                failed_insts.append(one_inst)
                failed_idxes.append(idx)
            idx += 1
        # -----
        if len(failed_insts) > 0:
            if backoff:
                assert len(failed_insts) == 0, "Cannot fail in backoff mode!"
            else:
                # inplaced output results
                bo_ags, _ = self.decode(failed_insts, scoring_prep_f, True)
                assert len(bo_ags) == len(failed_idxes)
                for bo_idx, bo_ag in zip(failed_idxes, bo_ags):
                    ags[bo_idx] = bo_ag
        # -----
        info = {"sent": len(insts), "tok": sum(map(len, insts)), "failed": len(failed_insts)}
        return ags, info

# =====
# fber(learner)
class TdFber:
    def __init__(self, scorer, iconf, tconf, margin: ScheduledValue, sched_sampling: ScheduledValue, sched_depth: ScheduledValue):
        self.tconf = tconf
        self.oracle_manager = TdOracleManager(tconf.oracle_strategy, tconf.oracle_projective, tconf.oracle_free_dist_alpha, tconf.oracle_label_order)
        self.require_sg = False
        # forced decoding
        self.oracled_searcher = TdInferencer(scorer, iconf, self.oracle_manager, True)
        # training strategy
        self.train_force, self.train_ss, self.train_rl, self.train_of = \
            [tconf.train_strategy==z for z in ["force", "ss", "rl", "of"]]
        if self.train_force:
            self.searcher = TdSearcher.create_oracle_follower(scorer, self.oracle_manager, iconf, tconf.oracle_log_prob_sum)
        elif self.train_ss:
            self.searcher = TdSearcher.create_scheduled_sampler(scorer, self.oracle_manager, iconf, sched_sampling, iconf.topk_sample, tconf.ss_strict_oracle, tconf.ss_include_correct_rate)
        elif self.train_rl:
            self.searcher = TdSearcher.create_rl_sampler(scorer, self.oracle_manager, iconf, iconf.topk_sample)
        elif self.train_of:
            self.searcher = TdSearcher.create_of_sampler(scorer, self.oracle_manager, iconf, tconf.oracle_log_prob_sum)
        else:
            raise NotImplementedError(f"Err: UNK training strategy {tconf.train_strategy}")
        self.sched_depth = sched_depth
        # TODO(+N): margin, local/global?
        self.margin = margin

    def force_decode(self, insts, scoring_prep_f, backoff):
        insts = [self.oracle_manager.init_inst(z) for z in insts]
        self.oracle_manager.refresh_insts(insts)
        return self.oracled_searcher.decode(insts, scoring_prep_f, backoff)

    def fb(self, annotated_insts, scoring_expr_pack, training: bool, loss_factor: float):
        # depth constrain: <= sched_depth
        cur_depth_constrain = int(self.sched_depth.value)
        # run
        ags = [BfsLinearAgenda.init_agenda(TdState, z, self.require_sg) for z in annotated_insts]
        self.oracle_manager.refresh_insts(annotated_insts)
        self.searcher.refresh(scoring_expr_pack)
        self.searcher.go(ags)
        # collect local loss: credit assignment
        if self.train_force or self.train_ss:
            states = []
            for ag in ags:
                for final_state in ag.local_golds:
                    # todo(warn): remember to use depth_eff rather than depth
                    # todo(warn): deprecated
                    # if final_state.depth_eff > cur_depth_constrain:
                    #     continue
                    states.append(final_state)
            logprobs_arc = [s.arc_score_slice for s in states]
            # no labeling scores for reduce operations
            logprobs_label = [s.label_score_slice for s in states if s.label_score_slice is not None]
            credits_arc, credits_label = None, None
        elif self.train_of:
            states = []
            for ag in ags:
                for final_state in ag.ends:
                    for s in final_state.get_path(True):
                        states.append(s)
            logprobs_arc = [s.arc_score_slice for s in states]
            # no labeling scores for reduce operations
            logprobs_label = [s.label_score_slice for s in states if s.label_score_slice is not None]
            credits_arc, credits_label = None, None
        elif self.train_rl:
            logprobs_arc, logprobs_label, credits_arc, credits_label = [], [], [], []
            for ag in ags:
                # todo(+2): need to check search failure?
                # todo(+2): ignoring labels when reducing or wrong-arc
                for final_state in ag.ends:
                    # todo(warn): deprecated
                    # if final_state.depth_eff > cur_depth_constrain:
                    #     continue
                    one_credits_arc = []
                    one_credits_label = []
                    self.oracle_manager.set_losses(final_state)
                    for s in final_state.get_path(True):
                        _, _, delta_arc, delta_label = s.oracle_loss_cache
                        logprobs_arc.append(s.arc_score_slice)
                        if delta_arc > 0:
                            # only blame arc
                            one_credits_arc.append(-delta_arc)
                        else:
                            one_credits_arc.append(0)
                            if delta_label > 0:
                                logprobs_label.append(s.label_score_slice)
                                one_credits_label.append(-delta_label)
                            elif s.label_score_slice is not None:
                                # not bad labeling
                                logprobs_label.append(s.label_score_slice)
                                one_credits_label.append(0)
                    # TODO(+N): minus average may encourage bad moves?
                    # balance
                    # avg_arc = sum(one_credits_arc) / len(one_credits_arc)
                    # avg_label = 0. if len(one_credits_label)==0 else sum(one_credits_label) / len(one_credits_label)
                    baseline_arc = baseline_label = -0.5
                    credits_arc.extend(z-baseline_arc for z in one_credits_arc)
                    credits_label.extend(z-baseline_label for z in one_credits_label)
        else:
            raise NotImplementedError("CANNOT get here!")
        # sum all local losses
        loss_zero = BK.zeros([])
        if len(logprobs_arc) > 0:
            batched_logprobs_arc = SliceManager.combine_slices(logprobs_arc, None)
            loss_arc = (-BK.sum(batched_logprobs_arc)) if (credits_arc is None) \
                else (-BK.sum(batched_logprobs_arc * BK.input_real(credits_arc)))
        else:
            loss_arc = loss_zero
        if len(logprobs_label) > 0:
            batched_logprobs_label = SliceManager.combine_slices(logprobs_label, None)
            loss_label = (-BK.sum(batched_logprobs_label)) if (credits_label is None) \
                else (-BK.sum(batched_logprobs_label*BK.input_real(credits_label)))
        else:
            loss_label = loss_zero
        final_loss_sum = loss_arc + loss_label
        # divide loss by what?
        num_sent = len(annotated_insts)
        num_valid_arcs, num_valid_labels = len(logprobs_arc), len(logprobs_label)
        # num_valid_steps = len(states)
        if self.tconf.loss_div_step:
            final_loss = loss_arc/max(1,num_valid_arcs) + loss_label/max(1,num_valid_labels)
        else:
            final_loss = final_loss_sum/num_sent
        #
        val_loss_arc = BK.get_value(loss_arc).item()
        val_loss_label = BK.get_value(loss_label).item()
        val_loss_sum = val_loss_arc + val_loss_label
        #
        cur_has_loss = 1 if ((num_valid_arcs+num_valid_labels)>0) else 0
        if training and cur_has_loss:
            BK.backward(final_loss, loss_factor)
        # todo(warn): make tok==steps for dividing in common.run
        info = {"sent": num_sent, "tok": num_valid_arcs, "valid_arc": num_valid_arcs, "valid_label": num_valid_labels,
                "loss_sum": val_loss_sum, "loss_arc": val_loss_arc, "loss_label": val_loss_label,
                "fb_all": 1, "fb_valid": cur_has_loss}
        return info

# b tasks/zdpar/transition/topdown/runner:46
