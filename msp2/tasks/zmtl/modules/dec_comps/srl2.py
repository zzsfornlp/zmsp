#

# SRL2: (assume given predicate) + arg

__all__ = [
    "ZDecoderSRL2Conf", "ZDecoderSRL2Node", "ZDecoderSRL2Helper",
]

from typing import List
from collections import OrderedDict
import numpy as np
import time
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.nn.modules import LossHelper
from msp2.data.inst import Sent, Frame, Mention, DataPadder
from msp2.data.vocab import SimpleVocab, SeqVocab
from msp2.data.rw import WriterGetterConf
from msp2.utils import Constants, zlog, ZObject, zwarn, ConfEntryChoices, Timer
from msp2.tasks.common.models.seqlab import BigramInferenceHelper
from msp2.proc import SVConf, ScheduledValue
from ..common import *
from ..enc import *
from ..dec import *
from ..dec_help import *

# =====

class ZDecoderSRL2Conf(ZDecoderConf):
    def __init__(self):
        super().__init__()
        # --
        # general specifications
        self.arg_ftag = "ef"  # special target for args
        self.arg_span_mode = "span"
        self.arg_only_rank1 = True  # only consider rank1 arg
        # --
        # for predicate
        self.evt_conf = IdecSingleConf()  # predicate
        self.loss_evt = 0.
        self.pred_evt = False  # whether predict evt?
        self.evt_label_smoothing = 0.
        self.evt_lookup_init_scale = 1.  # predicate input (no need to be large since bert's is not quite large!)
        # --
        # for argument
        self.arg_conf = IdecSingleConf()  # argument
        self.loss_arg = 1.0
        self.arg_label_smoothing = 0.
        self.arg_loss_sample_neg = 1.0  # how much of neg to include
        self.arg_distill_ratio = SVConf().direct_update(val=0., which_idx="uidx", mode="none", min_val=0.)
        self.arg_adaptive_helper = ConfEntryChoices({"adaptive": AdaptiveTrainingHelperConf(), "none": None}, "none")
        self.arg_do_aug_distill = False
        # special for arguments (BIO tagging)
        self.arg_use_bio = True  # use BIO mode!
        self.arg_pred_use_seq_cons = False  # use bio constraints in prediction
        self.arg_beam_k = 20
        # --
        # for arg2: aug arg loss
        self.arg2_conf: IdecConf = IdecSingleConf()  # argument
        self.loss_arg2 = 0.0
        self.arg2_label_smoothing = 0.
        self.arg2_loss_sample_neg = 1.0  # how much of neg to include
        # --
        # for cf
        self.cf_conf: IdecConf = IdecSingleConf()  # seq-level confident scoring
        # note: to be specified externally!! "srl_conf.do_seq_mode:sel srl_conf.seq_sel_key:evt_idx" or "srl_conf.do_seq_mode:pool srl_conf.seq_pool:idx0"
        self.loss_cf = 0.
        # --
        # for early exit at predicting
        self.pred_exit_by_arg = ConfEntryChoices({"yes": SeqExitHelperConf(), "no": None}, "no")  # exit by arg scores
        self.pred_exit_by_arg_mode = "plain"  # plain/bs1/index
        # --
        # output logits of the scorers (for external usage such as calibration)
        self.pred_decode_all_layers = True
        self.pred_logits_output_file = ""  # '' means nope
        # --
        # predict by max over all layers
        self.pred_max_all_layers = False
        # --
        # special for dpath
        self.dpath_conf = ConfEntryChoices({"yes": DPathConf(), "no": None}, "no")  # dpath features
        # --
        # special for score
        self.score_sent_topk = -1  # from token-v to sent-v, -1 means average all
        self.score_helper = ScoreHelperConf()

@node_reg(ZDecoderSRL2Conf)
class ZDecoderSRL2Node(ZDecoder):
    def __init__(self, conf: ZDecoderSRL2Conf, name: str,
                 vocab_evt: SimpleVocab, vocab_arg: SimpleVocab, ref_enc: ZEncoder, **kwargs):
        super().__init__(conf, name, **kwargs)
        conf: ZDecoderSRL2Conf = self.conf
        self.vocab_evt = vocab_evt
        self.vocab_arg = vocab_arg
        _enc_dim, _head_dim = ref_enc.get_enc_dim(), ref_enc.get_head_dim()
        # --
        self.vocab_bio_arg = None
        self.pred_cons_mat = None
        if conf.arg_use_bio:
            self.vocab_bio_arg = SeqVocab(vocab_arg)  # simply BIO vocab
            zlog(f"Use BIO vocab for srl: {self.vocab_bio_arg}")
            if conf.arg_pred_use_seq_cons:
                _m = self.vocab_bio_arg.get_allowed_transitions()
                self.pred_cons_mat = (1. - BK.input_real(_m)) * Constants.REAL_PRAC_MIN  # [L, L]
                zlog(f"Further use BIO constraints for decoding: {self.pred_cons_mat.shape}")
            helper_vocab_arg = self.vocab_bio_arg
        else:
            helper_vocab_arg = self.vocab_arg
        # --
        self.helper = ZDecoderSRL2Helper(conf, self.vocab_evt, helper_vocab_arg, self.vocab_arg)
        # --
        # nodes
        self.evt_node: IdecNode = conf.evt_conf.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=len(vocab_evt))
        self.arg_node: IdecNode = conf.arg_conf.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=len(helper_vocab_arg))
        self.arg2_node: IdecNode = conf.arg2_conf.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=len(self.vocab_arg))
        self.indicator_embed = EmbeddingNode(None, osize=_enc_dim, n_words=2, init_scale=conf.evt_lookup_init_scale)
        self.cf_node: IdecNode = conf.cf_conf.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=1)
        # --
        # special for score
        self.score_helper = ScoreHelperNode(conf.score_helper, nl=len(self.arg_node.app_lidxes))
        # special nodes for dpath input/output
        self.dpath_node = DPathNode(conf.dpath_conf, _isize=_enc_dim) if conf.dpath_conf is not None else None
        # --
        # specials
        self.arg_adaptive_helper = AdaptiveTrainingHelper(conf.arg_adaptive_helper) if conf.arg_adaptive_helper is not None else None
        self.arg_distill_ratio = ScheduledValue("ArgDistill", conf.arg_distill_ratio)
        if conf.pred_exit_by_arg is not None:
            self.pred_exit_by_arg = SeqExitHelperNode(conf.pred_exit_by_arg)
        else:
            self.pred_exit_by_arg = None
        self.pred_exit_by_arg_modes = [conf.pred_exit_by_arg_mode==z for z in ['plain', 'bs1', 'index']]
        assert sum(self.pred_exit_by_arg_modes)==1  # only one mode
        # --
        if conf.pred_logits_output_file:
            self.pred_logits_outputter = OutputHelper(conf.pred_logits_output_file)
        else:
            self.pred_logits_outputter = None
        # --
        # cache
        self._evt_mask = None

    def _get_scheduled_values(self):
        return OrderedDict([("_ArgDistill", self.arg_distill_ratio)])

    def get_idec_nodes(self):
        return [self.evt_node, self.arg_node, self.arg2_node, self.cf_node]

    # --
    def set_evt_mask(self, med, insts, mask_t):
        tmp_inds = BK.zeros(mask_t.shape)  # [*, slen]
        pred_idxes = BK.input_idx([z.mention.shead_widx for z in insts]).unsqueeze(-1)  # [*, 1]
        tmp_inds.scatter_(-1, pred_idxes, 1.)
        # --
        final_evt_idx = pred_idxes.squeeze(-1)  # [*]
        final_evt_mask = (tmp_inds > 0)  # [*, slen]
        med.set_cache("evt_idx", final_evt_idx)
        med.set_cache("evt_mask", final_evt_mask)
        # --

    def get_evt_mask(self, med):
        return med.get_cache("evt_mask")
    # --

    # return List[tensor], bool
    def layer_end(self, med: ZMediator):
        # first simply forward them
        _name = self.name
        lidx = med.lidx
        scores = {}
        rets = []
        do_early_exit = False
        # --
        # special adding indicators!
        if lidx == 0:
            self.set_evt_mask(med, med.insts, med.get_mask_t())  # set it!!
            if self.dpath_node is not None and self.dpath_node.conf.input_dpath:
                exp_ind = self.dpath_node.forward_input(med.insts, med.get_mask_t())  # [*, slen, D]
            else:
                exp_ind = self.indicator_embed(self.get_evt_mask(med).long())  # [*, slen, D]
            rets.append(exp_ind)
        # --
        if self.cf_node.need_app_layer(lidx):  # confident score used for early exit!
            cf_scores, cf_rets = self.cf_node.forward(med)
            assert cf_rets is None, "Cf node should not add back things!"
            scores[(_name, 'cf')] = cf_scores
        else:
            cf_scores = None
        if self.evt_node.need_app_layer(lidx):
            evt_scores, evt_rets = self.evt_node.forward(med)
            scores[(_name, 'evt')] = evt_scores
            rets.append(evt_rets)
        if self.arg_node.need_app_layer(lidx):
            arg_scores, arg_rets = self.arg_node.forward(med)
            scores[(_name, 'arg')] = arg_scores
            rets.append(arg_rets)
            # --
            # check early-exit by arg-scores!
            if not self.is_training() and self.pred_exit_by_arg:
                is_last_decoder = (lidx == self.arg_node.app_lidxes[-1])  # we can simply force ee at last decoder
                # check mode
                _mode_plain, _mode_bs1, _mode_index = self.pred_exit_by_arg_modes
                if _mode_plain:  # the old one, simply forward until all_exit!
                    last_scores = arg_scores  # [*, slen, L]
                    # --
                    # if cf_scores is not None:
                    #     cf_cur_probs = cf_scores.squeeze(-1).sigmoid()  # [*]
                    #     cf_prev_accu = med.get_cache("ea_cf", 0.)  # prev-accu
                    #     cf_probs = cf_prev_accu + cf_cur_probs * (1.-cf_prev_accu)  # [*]
                    #     med.set_cache("ea_cf", cf_probs, assert_no_exist=False)  # set current accu!
                    # else:
                    #     cf_probs = None
                    # --
                    cur_judges = [True]*len(med.insts) if is_last_decoder \
                        else BK.get_value(self.pred_exit_by_arg.judge(last_scores, cf_scores, med.get_mask_t()))  # [*]
                    _med_exit_lidxes = med.exit_lidxes
                    _all_exit = True
                    for _i, _e in enumerate(_med_exit_lidxes):
                        if _e < 0:  # still not exit
                            if cur_judges[_i]:
                                _med_exit_lidxes[_i] = lidx  # yep, exit
                            else:
                                _all_exit = False
                    do_early_exit = _all_exit
                elif _mode_bs1:  # also no need to store the idx, since always the last one!
                    if is_last_decoder:
                        do_early_exit = True
                    else:
                        last_scores = arg_scores  # [1, slen, L]
                        cur_judges = BK.get_value(self.pred_exit_by_arg.judge(last_scores, cf_scores, med.get_mask_t()))  # [1]
                        assert len(cur_judges)==1
                        do_early_exit = cur_judges[0]
                elif _mode_index:  # need to store various indexes
                    _len_bs = len(med.insts)
                    # get kept-mask
                    _cur_kept = med.get_cache('_cur_kept', None)  # [bs]
                    if _cur_kept is None:  # at the start, all True
                        _cur_kept = (BK.constants([_len_bs], 1.)>0)  # [bs]
                    # judge current
                    _judge_mask = med.get_mask_t()[_cur_kept]  # [??]
                    _len_cur = len(_judge_mask)
                    last_scores = arg_scores  # [??, slen, L]
                    cur_judges = (BK.constants([_len_cur], 1.)>0) if is_last_decoder \
                        else self.pred_exit_by_arg.judge(last_scores, cf_scores, _judge_mask)  # [??]
                    cur_judges_long = cur_judges.long()
                    # get and set stored index
                    _cur_idx_base = med.get_cache('_cur_idx_base', 0)  # base-idx for concatenated [bs]
                    _arange_t = BK.arange_idx(_len_cur) + _cur_idx_base  # [??]
                    _dec_idx = med.get_cache('_dec_idx', None)  # [bs]
                    if _dec_idx is None:  # by default, out-of-range error!
                        _dec_idx = BK.constants_idx([_len_bs], Constants.INT_PRAC_MAX)  # [bs]
                    _dec_idx[_cur_kept] = _arange_t * cur_judges_long + Constants.INT_PRAC_MAX * (1-cur_judges_long)  # [bs]
                    # store them all
                    _exit_lidx = med.get_cache('_exit_lidx', None)
                    if _exit_lidx is None:
                        _exit_lidx = BK.constants_idx([_len_bs], -1)  # [bs]
                    # cur_judge * lidx + (1-cur_judge) * -1
                    _exit_lidx[_cur_kept] = (lidx+1) * cur_judges_long - 1  # [bs]
                    med.set_cache('_exit_lidx', _exit_lidx, assert_no_exist=False)
                    # --
                    _next_select_mask = (~ cur_judges)  # kept ones, [??]
                    med.set_cache('_next_select_mask', _next_select_mask, assert_no_exist=False)
                    _cur_kept[_cur_kept] = _next_select_mask  # assign new judges, [bs]
                    med.set_cache('_cur_kept', _cur_kept, assert_no_exist=False)
                    med.set_cache('_cur_idx_base', _cur_idx_base+_len_cur, assert_no_exist=False)  # new base_idx
                    med.set_cache('_dec_idx', _dec_idx, assert_no_exist=False)
                    # --
                    do_early_exit = all(BK.get_value(cur_judges))  # early exit if all ok!
                else:
                    raise NotImplementedError()
            # --
        if self.is_training():  # note: only need aug in training!
            if self.dpath_node is not None:  # for the special node!!
                dpath_node = self.dpath_node.dpath_node
                if dpath_node.need_app_layer(lidx):
                    dp_scores, dp_rets = dpath_node.forward(med)
                    scores[(_name, 'dpath')] = dp_scores
                    assert dp_rets is None, "Aug node should not add back things!"
            if self.arg2_node.need_app_layer(lidx):
                arg2_scores, arg2_rets = self.arg2_node.forward(med)
                scores[(_name, 'arg2')] = arg2_scores
                assert arg2_rets is None, "Aug node should not add back things!"
        return scores, rets, (lidx >= self.max_app_lidx) or do_early_exit

    # ----
    # helper for loss
    def _prepare_loss_weights(self, mask_expr: BK.Expr, must_include_t: BK.Expr, neg_rate: float, extra_weights=None):
        if neg_rate <= 0.:  # only must_include
            ret_weights = (must_include_t.float()) * mask_expr
        elif neg_rate < 1.:  # must_include + sample
            ret_weights = ((BK.rand(mask_expr.shape) < neg_rate) | must_include_t).float() * mask_expr
        else:  # all in
            ret_weights = mask_expr  # simply as it is
        if extra_weights is not None:
            ret_weights *= extra_weights
        return ret_weights

    def _loss_evt(self, med: ZMediator, evt_mask, expr_evt_labels):
        conf: ZDecoderSRL2Conf = self.conf
        # --
        assert med.force_lidx is None, "Not implemented for this one!!"
        loss_items = []
        # -- get losses
        all_evt_scores = med.main_scores.get((self.name, "evt"))
        all_evt_losses = []
        for one_evt_scores in all_evt_scores:
            one_losses = BK.loss_nll(one_evt_scores[evt_mask], expr_evt_labels, label_smoothing=conf.evt_label_smoothing)
            all_evt_losses.append(one_losses)
        evt_loss_results = self.evt_node.helper.loss(all_losses=all_evt_losses)
        for loss_t, loss_alpha, loss_name in evt_loss_results:
            one_evt_item = LossHelper.compile_leaf_loss(
                "evt" + loss_name, loss_t.sum(), evt_mask.sum(),
                loss_lambda=conf.loss_evt * loss_alpha, gold=evt_mask.sum())
            loss_items.append(one_evt_item)
        return loss_items

    def _loss_arg(self, med: ZMediator, mask_expr, expr_arg_labels, expr_arg2_labels):
        conf: ZDecoderSRL2Conf = self.conf
        _name = self.name
        # --
        loss_items = []
        # -- get losses
        for aname, anode, glabels, sneg_rate, loss_mul, lsmooth, arg_distill_ratio in \
                zip(["arg", "arg2"], [self.arg_node, self.arg2_node], [expr_arg_labels, expr_arg2_labels],
                    [conf.arg_loss_sample_neg, conf.arg2_loss_sample_neg], [conf.loss_arg, conf.loss_arg2],
                    [conf.arg_label_smoothing, conf.arg2_label_smoothing], [self.arg_distill_ratio, 0.]):
            # --
            if loss_mul <= 0.:
                continue
            # --
            # prepare
            flat_arg_labels = glabels  # [*, slen]
            flat_arg_not_nil = (flat_arg_labels > 0)  # [??, slen]
            flat_arg_weights = self._prepare_loss_weights(mask_expr, flat_arg_not_nil, sneg_rate)
            # scores and losses
            all_arg_scores = med.main_scores.get((self.name, aname))  # NLx[*, slen, L]
            all_augarg_scores = med.aug_scores.get((self.name, aname))  # SAMPLExNLx[*, slen, L]
            # --
            if aname == "arg" and conf.arg_do_aug_distill:  # special distill loss
                assert len(all_arg_scores) == len(all_augarg_scores)
                all_arg_losses = [-(cur.log_softmax(-1)*trg.softmax(-1)).sum() for cur, trg in zip(all_arg_scores, all_augarg_scores)]
            else:
                all_arg_losses = IdecHelper.gather_losses(
                    all_arg_scores, flat_arg_labels, lsmooth=lsmooth, distill_ratio=float(arg_distill_ratio))
            # --
            if aname == "arg" and self.arg_adaptive_helper is not None:
                # try to get aug losses
                if all_augarg_scores is not None:
                    # also repeat labels
                    repeat_arg_labels = BK.concat(
                        [flat_arg_labels]*(BK.get_shape(all_augarg_scores[0],0)//BK.get_shape(flat_arg_labels,0)), 0)
                    all_augarg_losses = IdecHelper.gather_losses(all_augarg_scores, repeat_arg_labels, lsmooth=lsmooth, distill_ratio=0)
                else:
                    all_augarg_losses = None
                # reweight!!
                final_all_arg_losses = self.arg_adaptive_helper.forward(all_arg_losses, all_augarg_losses, flat_arg_weights, insts=med.insts)
            else:
                final_all_arg_losses = all_arg_losses
            # --
            # check forced
            _forced_lidx = med.force_lidx
            if _forced_lidx is not None:  # force for one layer
                _forced_aidx = anode.app_lidxes_map[_forced_lidx]  # mapped to idx in the list
                _extra_loss_kwargs = {f"L{_forced_lidx}": (all_arg_losses[_forced_aidx]*flat_arg_weights).sum()}
                _forced_final_all_arg_losses = [None] * len(final_all_arg_losses)
                _forced_final_all_arg_losses[_forced_aidx] = final_all_arg_losses[_forced_aidx]
                arg_loss_results = anode.helper.loss(all_losses=_forced_final_all_arg_losses)
            else:  # put all
                _extra_loss_kwargs = {f"L{anode.app_lidxes[_i]}": (_lo * flat_arg_weights).sum() for _i, _lo in enumerate(all_arg_losses)}
                arg_loss_results = anode.helper.loss(all_losses=final_all_arg_losses)
            # collect losses
            for loss_t, loss_alpha, loss_name in arg_loss_results:
                one_arg_item = LossHelper.compile_leaf_loss(
                    aname + loss_name, (loss_t * flat_arg_weights).sum(), flat_arg_weights.sum(),
                    loss_lambda=(loss_mul*loss_alpha), gold=flat_arg_not_nil.float().sum(), **_extra_loss_kwargs)
                _extra_loss_kwargs = {}  # no repeat recording!!
                loss_items.append(one_arg_item)
        return loss_items

    # get loss finally
    def loss(self, med: ZMediator):
        conf: ZDecoderSRL2Conf = self.conf
        insts, mask_expr = med.insts, med.get_mask_t()
        # --
        # first prepare golds
        arr_items, expr_evt_labels, expr_arg_labels, expr_arg2_labels = self.helper.prepare(insts, True)
        # --
        loss_items = []
        if conf.loss_evt > 0.:
            evt_mask = self.get_evt_mask(med)
            loss_items.extend(self._loss_evt(med, evt_mask, expr_evt_labels))
        if conf.loss_arg > 0. or conf.loss_arg2 > 0.:
            loss_items.extend(self._loss_arg(med, mask_expr, expr_arg_labels, expr_arg2_labels))
        if self.dpath_node is not None:
            loss_items.extend(self.dpath_node.loss_dpath(med.main_scores.get((self.name, "dpath")), insts, mask_expr))
        if conf.loss_cf > 0.:
            loss_items.extend(self.pred_exit_by_arg.loss_cf(med.main_scores.get((self.name, "cf")), insts, conf.loss_cf))
        # =====
        # return loss
        ret_loss = LossHelper.combine_multiple_losses(loss_items)
        # self.invalid_caches()
        return ret_loss

    # --
    def _pred_evt(self, med: ZMediator, evt_mask):
        conf: ZDecoderSRL2Conf = self.conf
        # --
        all_evt_raw_scores = med.main_scores.get((self.name, "evt"))  # [*, slen, Le]
        all_evt_logprobs = [z.log_softmax(-1) for z in all_evt_raw_scores]
        final_evt_logprobs = self.evt_node.helper.pred(all_logprobs=all_evt_logprobs)  # [*, slen, Le]
        selected_evt_logprobs = final_evt_logprobs[evt_mask>0]  # [*, Le]
        pred_evt_scores, pred_evt_labels = selected_evt_logprobs.max(-1)  # [*]
        return pred_evt_labels, pred_evt_scores  # [*]

    def _pred_arg(self, med: ZMediator, mask_expr, exit_lidxes, insts):
        conf: ZDecoderSRL2Conf = self.conf
        # --
        all_arg_raw_score = med.main_scores.get((self.name, "arg"))  # [*, slen, La]
        all_arg_logprobs = [z.log_softmax(-1) for z in all_arg_raw_score]
        if self.pred_exit_by_arg:
            # check mode
            _mode_plain, _mode_bs1, _mode_index = self.pred_exit_by_arg_modes
            if _mode_plain:
                # select!
                _stack_logprobs = BK.stack(all_arg_logprobs, 1)  # [*, NL, slen, La]
                _arange_t = BK.arange_idx(len(insts))  # [*]
                _app_lidxes_list = self.arg_node.app_lidxes
                _app_lidxes_map = self.arg_node.app_lidxes_map
                _sel_list = [_app_lidxes_map[z] for z in exit_lidxes]
                _sel_t = BK.input_idx(_sel_list)  # [*]
                final_arg_logprobs = _stack_logprobs[_arange_t, _sel_t]  # [*, slen, La]
                for _li, _ins in zip(_sel_list, insts):
                    _ins.info["exit_lidx"] = _app_lidxes_list[_li]  # directly assign it!
            elif _mode_bs1:
                assert len(insts) == 1
                final_arg_logprobs = all_arg_logprobs[-1]  # [*, slen, La], simply last one
                insts[0].info["exit_lidx"] = self.arg_node.app_lidxes[len(all_arg_logprobs)-1]
            elif _mode_index:
                _cat_logprobs = BK.concat(all_arg_logprobs, 0)  # [sum(*), slen, La]
                final_arg_logprobs = _cat_logprobs[med.get_cache('_dec_idx')]  # [*, slen, La], simple index select
                _exit_lidx = BK.get_value(med.get_cache('_exit_lidx')).tolist()  # [*]
                for _li, _ins in zip(_exit_lidx, insts):
                    _ins.info["exit_lidx"] = _li  # directly assign it!
            else:
                raise NotImplementedError()
        elif conf.pred_max_all_layers:
            stack_logprobs = BK.stack(all_arg_logprobs, -2)  # [*, slen, NL, La]
            stack_lmax, _ = stack_logprobs.max(-1)  # [*, slen, NL]
            stack_ssel, stack_lsel = stack_lmax.max(-1)  # [*, slen]
            final_arg_logprobs = BK.gather_first_dims(stack_logprobs, stack_lsel.unsqueeze(-1), 2).squeeze(-2)  # [*, slen, La]
        else:
            final_arg_logprobs = self.arg_node.helper.pred(all_logprobs=all_arg_logprobs)  # [*, slen, La]
        # --
        if self.pred_cons_mat is not None:
            pred_arg_labels, pred_arg_scores = BigramInferenceHelper.inference_search(
                final_arg_logprobs, self.pred_cons_mat, mask_expr, conf.arg_beam_k)  # [*, slen]
        else:
            pred_arg_scores, pred_arg_labels = final_arg_logprobs.max(-1)  # [*, slen]
        return pred_arg_labels, pred_arg_scores  # [*, slen, La]

    # predict finally
    def predict(self, med: ZMediator):
        conf: ZDecoderSRL2Conf = self.conf
        insts, mask_expr = med.insts, med.get_mask_t()
        # --
        # note: especially write logits
        if self.pred_logits_outputter is not None:
            _, _, expr_arg_labels, _ = self.helper.prepare(insts, True)  # [*, slen]
            all_arg_logits = BK.stack(med.main_scores.get((self.name, "arg")), -1)  # [*, slen, La, NL]
            for iidx in range(len(insts)):
                _one_mask = (mask_expr[iidx]>0)
                # [len, L, NL], [len]
                one_obj = {
                    "logits": BK.get_value(all_arg_logits[iidx][_one_mask]),
                    "labels": BK.get_value(expr_arg_labels[iidx][_one_mask]),
                }
                self.pred_logits_outputter.write(one_obj)
        # --
        if conf.pred_evt:
            pred_evt_labels, pred_evt_scores = self._pred_evt(med, self.get_evt_mask(med))
        else:
            pred_evt_labels, pred_evt_scores = None, None
        pred_arg_labels, pred_arg_scores = self._pred_arg(med, mask_expr, med.exit_lidxes, insts)
        # transfer data from gpu also counts (also make sure gpu calculations are done)!
        all_arrs = [(BK.get_value(z) if z is not None else None)
                    for z in [pred_evt_labels, pred_evt_scores, pred_arg_labels, pred_arg_scores]]
        # =====
        # assign; also record post-processing (non-computing) time
        time0 = time.time()
        self.helper.put_results(insts, all_arrs)
        # special setting for output all
        if conf.pred_decode_all_layers:
            stacked_scores = BK.stack(med.main_scores.get((self.name, "arg")), 0)  # [NL, *, slen, L]
            _, argmax_label_idxes = stacked_scores.max(-1)  # [NL, *, slen]
            self.helper.put_results_all_layers(insts, BK.get_value(argmax_label_idxes))
        time1 = time.time()
        # --
        # self.invalid_caches()
        return {f"{self.name}_posttime": time1-time0}

    # score: put score
    def score(self, med: ZMediator, orig_insts: List):
        conf: ZDecoderSRL2Conf = self.conf
        # --
        # note: use orig_insts instead!!
        arr_items, _, expr_arg_labels, _ = self.helper.prepare(orig_insts, True)
        # --
        # note: only for arg!!
        all_arg_scores = med.main_scores.get((self.name, "arg"))  # (NL)x[S*bs, slen, L]
        s_size = BK.get_shape(all_arg_scores[0],0) // len(orig_insts)  # sample size
        repeat_arg_labels = BK.concat([expr_arg_labels] * s_size, 0)  # [S*bs, slen]
        # gather all losses: (NL)x[S*bs, slen] => [S, NL, bs, slen]
        all_arg_losses = IdecHelper.gather_losses(all_arg_scores, repeat_arg_labels, lsmooth=0, distill_ratio=0)
        _shape1 = [s_size, len(orig_insts)] + BK.get_shape(all_arg_losses[0])[1:]  # [S, bs, slen]
        stack_loss = BK.stack([z.view(_shape1) for z in all_arg_losses], 2)  # [S, bs, NL, slen]
        # gather seq-level info
        if s_size == 1:
            _mask_expr2 = med.get_mask_t().unsqueeze(-2)  # [bs, 1, slen]
            stack_loss *= _mask_expr2  # 0. for invalid ones!
            loss_avg = topk_avg(stack_loss[0], _mask_expr2, conf.score_sent_topk, dim=-1, largest=True)  # [bs, NL]
            loss_std = loss_avg * 0.  # [NL, bs], no std for single sample!
        else:
            _mask_expr2 = BK.input_real(DataPadder.lengths2mask([len(z.sent) for z in orig_insts])).unsqueeze(-2)  # [bs, 1, slen]
            stack_loss *= _mask_expr2  # 0. for invalid ones!
            loss_avg = topk_avg(stack_loss.mean(0), _mask_expr2, conf.score_sent_topk, dim=-1, largest=True)  # [bs, NL]
            loss_std = topk_avg(stack_loss.std(0), _mask_expr2, conf.score_sent_topk, dim=-1, largest=True)  # [bs, NL]
        # --
        # gather err-rate info
        layered_tag_err, layered_span_err = [], []  # NL*[bs]
        layered_tag_corr_seq = []  # NL*[bs, len]
        _orig_mask_t = BK.input_real(DataPadder.lengths2mask([len(z.sent) for z in orig_insts]))  # [bs, slen]
        for layer_idx, layer_scores in enumerate(all_arg_scores):
            _, argmax_label_idxes = layer_scores.max(-1)  # [S*bs, slen]
            argmax_label_idxes = argmax_label_idxes.view(_shape1)  # [S, bs, slen]
            # first tag err
            tag_err_t = ((argmax_label_idxes != expr_arg_labels).float() * _orig_mask_t)  # [S, bs, slen]
            tag_err_t = (tag_err_t.sum(-1) / _orig_mask_t.sum(-1)).mean(0)  # [bs]
            layered_tag_err.append(BK.get_value(tag_err_t).tolist())
            # tag correctness for all tokens
            tag_corr_seq_t = ((argmax_label_idxes == expr_arg_labels) | (_orig_mask_t==0.)).float().mean(0)  # [bs, slen]
            layered_tag_corr_seq.append(BK.get_value(tag_corr_seq_t))
            # then span err
            argmax_label_arrs = BK.get_value(argmax_label_idxes)
            span_err_arrs = [self.helper.fast_eval_results(orig_insts, vv) for vv in argmax_label_arrs]  # S*[bs]
            span_err_arrs = np.mean(span_err_arrs, 0).tolist()  # [bs]
            layered_span_err.append(span_err_arrs)
        # --
        # put vals
        arr_loss_avg, arr_loss_std = BK.get_value(loss_avg), BK.get_value(loss_std)
        for ii, inst in enumerate(orig_insts):
            # add container
            for name in ["loss_avg", "loss_std", "err_tag", "err_span", "corr_seq"]:
                if name not in inst.info:
                    inst.info[name] = []
            # --
            _slen = len(inst.sent)
            _list_loss_avg, _list_loss_std = arr_loss_avg[ii].tolist(), arr_loss_std[ii].tolist()  # List[NL]
            _list_err_tag, _list_err_span = [z[ii] for z in layered_tag_err], [z[ii] for z in layered_span_err]
            inst.info["loss_avg"].append(_list_loss_avg)
            inst.info["loss_std"].append(_list_loss_std)
            inst.info["err_tag"].append(_list_err_tag)
            inst.info["err_span"].append(_list_err_span)
            inst.info["corr_seq"].append([z[ii][:_slen].tolist() for z in layered_tag_corr_seq])
            # --
        # --
        return {}

# --
# special scorer helper!
class ScoreHelperConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        # basics
        self.cidx_start = 10000  # starting from which cidx
        self.cidx_freq = 10000  # how many cidx (interval) to do this
        self.need_forward_model = False  # whether need to forward with current model?
        # how to get the ranking key?
        self.score_f = "lambda ff, li: 1."  # for example: "lambda ff, li: ff.info['err_tag'][-1][li]"
        self.rank_f = "lambda s,r: r"  # for example: "lambda s,r: s"
        # how to count err dpath
        self.err_dpath_clamp = [3, 3]
        # store path
        self.insts_store_path = ""  # to store these insts?

@node_reg(ScoreHelperConf)
class ScoreHelperNode(BasicNode):
    def __init__(self, conf: ScoreHelperConf, nl, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ScoreHelperConf = self.conf
        # --
        self.need_forward_model = conf.need_forward_model
        self.nl = nl
        self.voc_word = None  # to be set!!
        # get self's method or simply get a lambda
        self.score_f = getattr(self, conf.score_f) if hasattr(self, conf.score_f) else eval(conf.score_f)
        self.rank_f = eval(conf.rank_f)
        # --

    # --
    # different static modes
    def _get_static_slen(self, f: Frame, *args, **kwargs):
        return len(f.sent)

    def _get_static_rarity(self, f: Frame, *args, **kwargs):
        voc_word = self.voc_word
        # --
        all_count = voc_word.get_all_counts()
        word_counts = [voc_word.word2count(w) for w in f.sent.seq_word.vals]
        probs = (np.array(word_counts)+1)/(all_count+1.)  # add-1 smoothing
        rarity = - np.log(probs).sum().item()
        return rarity

    def _get_static_dpath_avg(self, f: Frame, *args, **kwargs):
        return self._get_static_dpath(f, do_avg=True, **kwargs)

    def _get_static_dpath(self, f: Frame, *args, do_avg=False, **kwargs):
        arr_dp = np.array(f.info["dpaths"])  # [2, slen, 2]
        _clamp0, _clamp1 = self.conf.err_dpath_clamp
        arr_err = ((arr_dp[0].clip(max=_clamp0) != arr_dp[1].clip(max=_clamp1)).sum(-1) > 0)  # [slen]
        if do_avg:
            return arr_err.mean().item()
        else:
            return arr_err.sum().item()
    # --

    def score_and_rank(self, batch_stream, model, cidx: int):
        conf: ScoreHelperConf = self.conf
        # --
        self.voc_word = model.vpack.get_voc("word")  # used for rarity
        # --
        _nl = self.nl
        _c_start, _c_freq = conf.cidx_start, conf.cidx_freq
        if cidx >= _c_start and (cidx - _c_start) % _c_freq == 0:
            # yep, time to go!!
            with Timer(info=f"ScoreRank@cidx={cidx}", print_date=True):
                all_sents = []
                all_frames = []
                # first forward model to get scores
                for one_sents in batch_stream:
                    if self.need_forward_model:
                        model.score_on_batch(one_sents)
                    all_sents.extend(one_sents)
                    for s in one_sents:
                        all_frames.extend(s.events)
                # then rank them all!
                # all_scores = np.zeros([len(all_frames), _nl], dtype=np.float32)
                all_scores = np.asarray([[self.score_f(ff, li) for li in range(_nl)] for ff in all_frames])  # sometimes may range [0,1]
                all_ranks = all_scores.argsort(axis=0).argsort(axis=0) / float(len(all_frames))  # ranking from 0. to ->1.
                for ii, ff in enumerate(all_frames):  # assign
                    _list_scores = all_scores[ii].tolist()
                    _list_cl_ranks = all_ranks[ii].tolist()
                    rs = [self.rank_f(s, r) for s, r in zip(_list_scores, _list_cl_ranks)]
                    if "cl_ranks" not in ff.info:
                        ff.info["cl_ranks"] = [rs]
                    else:
                        ff.info["cl_ranks"].append(rs)
                # write?
                if conf.insts_store_path:
                    all_sents.sort(key=lambda x: x.read_idx)
                    with WriterGetterConf().get_writer(output_path=conf.insts_store_path) as writer:
                        writer.write_insts(all_sents)
        # --

# --
# adaptive training helper
class AdaptiveTrainingHelperConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self.weight_scheme = "curr"  # aug/upper/curr/rank
        # softmax(factor/this); note: if do not want to select at certain level, simply set temp to a high value!
        self.at_sent_temperature = SVConf().direct_update(val=1000., which_idx="uidx", mode="none", min_val=0.)
        self.at_tok_temperature = SVConf().direct_update(val=1000., which_idx="uidx", mode="none", min_val=0.)
        self.at_rescale = 0.  # scale to certain range? 0. means Nope!
        # keeps are list of pairs (flattened): [start, end]
        self.at_sent_keeps = []  # by default [1., 1.] (higher bounds)
        self.at_sent_keeps_low = []  # by default [0., 0.] (lower bounds); todo(+N): currently only used for rank mode!
        self.at_tok_keeps = []  # by default [1., 1.]
        # schedule from start to end, <0. means nope! (keep=1.!!)
        self.at_keep_sched = SVConf().direct_update(val=1., which_idx="cidx", mode="none", b=0., k=1., max_val=1.)
        # self.at_keep_scale_loss = True  # scale the loss by keep-value (no-scale_back to full 1.)
        # from loss_tok to loss_sent
        self.at_sent_topk = -1  # topk average for sent-level summary of losses, <0 means avg all
        # special for upper
        self.upper_rolling_max = True  # whether do rolling max?
        # special for rank
        self.rank_take_idxes = []  # by default, 0,1,2,...
        # --

@node_reg(AdaptiveTrainingHelperConf)
class AdaptiveTrainingHelper(BasicNode):
    def __init__(self, conf: AdaptiveTrainingHelperConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: AdaptiveTrainingHelperConf = self.conf
        # --
        _MAX_NL = 100  # note: this should be enough
        # --
        self._weight_f = {'aug': self._weight_aug, 'upper': self._weight_upper,
                          'curr': self._weight_curr, 'rank': self._weight_rank}[conf.weight_scheme]
        self._at_sent_keeps = [float(z) for z in conf.at_sent_keeps] + [1.]*(_MAX_NL*2)
        self._at_sent_keeps_low = [float(z) for z in conf.at_sent_keeps_low] + [0.]*(_MAX_NL*2)
        self._at_tok_keeps = [float(z) for z in conf.at_tok_keeps] + [1.]*(_MAX_NL*2)
        self.at_sent_temperature = ScheduledValue("AtSentTemp", conf.at_sent_temperature)
        self.at_tok_temperature = ScheduledValue("AtTokTemp", conf.at_tok_temperature)
        self.at_keep_sched = ScheduledValue("AtKeepSched", conf.at_keep_sched)
        self._rank_take_idxes = list(range(_MAX_NL))
        self._rank_take_idxes[:len(conf.rank_take_idxes)] = [int(z) for z in conf.rank_take_idxes]
        # --

    def _get_scheduled_values(self):
        return OrderedDict([("_AtSentTemp", self.at_sent_temperature), ("_AtTokTemp", self.at_tok_temperature),
                            ("_AtKeepSched", self.at_keep_sched)])

    def _get_keeps(self, length: int):
        _rate = self.at_keep_sched.value
        if _rate < 0.:  # not started yet!!
            return [1.]*length, [0.]*length, [1.]*length,
        else:
            rets = []
            for _keeps in [self._at_sent_keeps, self._at_sent_keeps_low, self._at_tok_keeps]:
                _ret = []
                for ii in range(length):
                    a, b = _keeps[2*ii], _keeps[2*ii+1]
                    _ret.append(a+(b-a)*_rate)
                rets.append(_ret)
            return rets[0], rets[1], rets[2]  # sent(_high), sent_low, tok

    # =====
    # helper
    # [*, V]
    def _rescale(self, v, _scale: float):
        if _scale <= 0.:
            return v
        else:  # note: v should be >0
            ss = (v + 1e-10) / (v.max(-1, keepdim=True)[0] + 1e-10)  # simple smooth
            return _scale * ss

    # [*, slen, ??], [*, slen]/None;;
    def _get_factor(self, v, mask, _scale: float, _temp: float, _keep_ratio: float):
        # first rescale
        _NEG = Constants.REAL_PRAC_MIN
        rescaled_v = - self._rescale(v, _scale) / _temp  # [NL?, *, slen, ??], note: remember negative!
        if mask is not None:
            rescaled_v += (_NEG * (1.-mask))
        # then discard
        if not BK.is_zero_shape(rescaled_v) and _keep_ratio < 1.:
            _LEN = BK.get_shape(rescaled_v, -1)
            keep_num = max(1, int(_LEN * _keep_ratio))  # at least keep one!!
            # note: here no pass of mask since they will always be _NEG; here select_topk since we already neg it!
            topk_mask = select_topk(rescaled_v, keep_num, dim=-1)
            rescaled_v += (_NEG * (1.-topk_mask))
        # then softmax
        ret_factor = rescaled_v.softmax(-1)
        return ret_factor

    # --
    def _common_weight_f(self, all_values, arg_mask):
        conf: AdaptiveTrainingHelperConf = self.conf
        _sent_temp = self.at_sent_temperature.value
        _tok_temp = self.at_tok_temperature.value
        _rescale = conf.at_rescale
        _sent_keeps, _, _tok_keeps = self._get_keeps(len(all_values))
        # --
        # for each one
        rets = []
        arg_mask_sum = arg_mask.sum()
        for ii, vv in enumerate(all_values):
            # token level
            token_v = vv
            token_factor = self._get_factor(token_v, arg_mask, _rescale, _tok_temp, _tok_keeps[ii])
            # sent level
            sent_v = topk_avg(token_v, arg_mask, conf.at_sent_topk, dim=-1, largest=True)  # [*]
            sent_facotr = self._get_factor(sent_v, None, _rescale, _sent_temp, _sent_keeps[ii])
            # get one weight
            # TODO(+N): can we directly suppress instead of softmax reweight?
            one_weight = token_factor * sent_facotr.unsqueeze(-1)  # [*, slen]
            one_weight *= arg_mask_sum  # multiply back to scales!
            rets.append(one_weight)
        return rets

    # =====
    # weight by var of aug!
    def _weight_aug(self, all_arg_losses: List, all_augarg_losses: List, arg_mask, *args, **kwargs):
        # --
        # input is List(NL)[S*bs, slen]
        _bs = BK.get_shape(all_arg_losses[0], 0)
        _aug_shape = BK.get_shape(all_augarg_losses[0])
        _reshape_aug_shape = [_aug_shape[0]//_bs, _bs] + _aug_shape[1:]  # [S, bs, slen]
        stack_aug_losses = BK.stack([z.view(_reshape_aug_shape) for z in all_augarg_losses], 1)  # [S, NL, bs, slen]
        # --
        all_vars = split_at_dim(stack_aug_losses.var(0)*arg_mask, 0, False)  # List(NL)[*, slen]
        rets = self._common_weight_f(all_vars, arg_mask)
        return rets

    # weight by current losses
    def _weight_curr(self, all_arg_losses: List, all_augarg_losses: List, arg_mask, *args, **kwargs):
        rets = self._common_weight_f([z*arg_mask for z in all_arg_losses], arg_mask)
        return rets

    # weight by upper (note: no weights for last layer)
    def _weight_upper(self, all_arg_losses: List, all_augarg_losses: List, arg_mask, *args, **kwargs):
        conf: AdaptiveTrainingHelperConf = self.conf
        if len(all_arg_losses) <= 1:
            return [None] * len(all_arg_losses)  # no upper ones!
        # --
        # stack higher losses
        if conf.upper_rolling_max:
            feed_arg_losses = [all_arg_losses[-1]]
            for one_losses in reversed(all_arg_losses[1:-1]):
                feed_arg_losses.append(BK.max_elem(one_losses, feed_arg_losses[-1]))
            feed_arg_losses.reverse()
        else:
            feed_arg_losses = all_arg_losses[1:]
        # --
        rets = self._common_weight_f([z*arg_mask for z in feed_arg_losses], arg_mask)
        return rets + [None]  # no for last layer

    # weight by external ranking
    def _weight_rank(self, all_arg_losses: List, all_augarg_losses: List, arg_mask, insts, *args, **kwargs):
        conf: AdaptiveTrainingHelperConf = self.conf
        _NL = len(all_arg_losses)
        # get ranks
        ranks_t = BK.input_real([z.info["cl_ranks"][-1] for z in insts])  # [bs, NL], rank for cl prepared externally
        sel_ranks_t = ranks_t[:, self._rank_take_idxes[:_NL]]  # [bs, NL]
        # compare with current pace
        _sent_keeps_high, _sent_keeps_low, _ = self._get_keeps(_NL)
        rets = []
        # todo(+N): currently only at inst level!!
        for ii in range(_NL):
            _hi, _lo = _sent_keeps_high[ii], _sent_keeps_low[ii]
            assert _hi > _lo
            _vt = sel_ranks_t[:,ii]
            one_weight = ((_vt<=_hi) & (_vt>=_lo)).float()  # [bs, ]
            one_weight *= (arg_mask.sum() / ((one_weight.unsqueeze(-1)*arg_mask).sum()+1e-5))  # norm back here!
            rets.append(one_weight.unsqueeze(-1))  # [bs,1]
        return rets

    def forward(self, all_arg_losses: List, all_augarg_losses: List, arg_masks_t, *args, **kwargs):
        # --
        _rate = self.at_keep_sched.value
        if _rate < 0.:  # note: no weighting if we have not yet started!!
            return all_arg_losses
        # --
        # get weights
        with BK.no_grad_env():
            all_weights = self._weight_f(all_arg_losses, all_augarg_losses, arg_masks_t, *args, **kwargs)  # List(NL)[*, slen]
        # apply weights
        ret_arg_losses = []
        for one_loss, one_weight in zip(all_arg_losses, all_weights):
            if one_weight is not None:
                one_loss = one_loss * one_weight
            ret_arg_losses.append(one_loss)
        return ret_arg_losses

# =====
# input dpath features
class DPathConf(BasicConf):
    def __init__(self):
        super().__init__()
        self._isize = -1
        # --
        # general
        self.dpath_ceils = [2,2]  # [word_path, pred_path]
        self.train_dpath_gratio = 0.5  # gold ratio
        self.test_dpath_idx = 1  # 0:gold, 1:pred
        # as input
        self.input_dpath = True
        self.input_train_zero_rate = 0.  # zero rate at training time!!
        self.input_train_zero_incorrect = True  # only zero incorrect ones!
        # as aux loss
        self.loss_dpath2 = 0.
        self.dpath2_conf = IdecSingleConf()
        self.dpath2_label_smoothing = 0.

@node_reg(DPathConf)
class DPathNode(BasicNode):
    def __init__(self, conf: DPathConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: DPathConf = self.conf
        # --
        _NWORDS = 100  # note: this should be enough!
        self.dpath_embed = EmbeddingNode(None, osize=conf._isize, n_words=_NWORDS, init_scale=1., fix_row0=True)
        self.dpath_node = conf.dpath2_conf.make_node(_isize=conf._isize, _nhead=None, _csize=_NWORDS)
        # --
        # check
        if conf.input_dpath:
            assert conf.loss_dpath2==0. and self.dpath_node.app_lidxes==[]
        # --

    def forward_input(self, insts, mask_t):
        labs = self.get_labels(insts, mask_t)
        return self.dpath_embed.forward(labs)  # [*, slen, D]

    def loss_dpath(self, all_dpath_scores, insts, mask_t):
        conf: DPathConf = self.conf
        if conf.loss_dpath2 <= 0.:
            return []
        # --
        labs = self.get_labels(insts, mask_t)
        all_dpath_losses = []
        for one_dpath_scores in all_dpath_scores:
            one_losses = BK.loss_nll(one_dpath_scores, labs, label_smoothing=conf.dpath2_label_smoothing)
            all_dpath_losses.append(one_losses)
        dpath_loss_results = self.dpath_node.helper.loss(all_losses=all_dpath_losses)
        # --
        loss_items = []
        for loss_t, loss_alpha, loss_name in dpath_loss_results:
            one_dpath_item = LossHelper.compile_leaf_loss(
                "dpath" + loss_name, (loss_t*mask_t).sum(), mask_t.sum(), loss_lambda=conf.loss_dpath2 * loss_alpha)
            loss_items.append(one_dpath_item)
        return loss_items

    def get_labels(self, insts, mask_t):
        conf: DPathConf = self.conf
        # --
        # prepare
        _bs, _mlen = BK.get_shape(mask_t)  # [*, slen]
        _paths_arr = np.zeros([_bs, 2, _mlen, 2], dtype=np.int)  # [*, g/p, slen, 2]
        for ii, ff in enumerate(insts):
            _slen = len(ff.sent)
            _paths_arr[ii, :, :_slen] = ff.info["dpaths"]
        # input
        _paths_t = BK.input_idx(_paths_arr)  # [*, 2, slen, 2]
        if self.is_training():
            _arange_t = BK.arange_idx(_bs)  # [*]
            _rand_t = (BK.rand(_bs)<conf.train_dpath_gratio).long()  # [*]
            # note: gold-idx is 0!! once a bug, get reversed!!
            _paths_t2 = _paths_t[_arange_t, 1-_rand_t]  # [*, slen, 2]
        else:  # all select one!!
            _paths_t2 = _paths_t[:, conf.test_dpath_idx]  # [*, slen, 2]
        # combine
        _c0, _c1 = conf.dpath_ceils
        _paths_t2[:,:,0].clamp_(max=_c0)
        _paths_t2[:,:,1].clamp_(max=_c1)
        _final_paths_t = 1 + (_c1+1) * _paths_t2[:,:,1] + _paths_t2[:,:,0]  # [*,slen], offset by 1 for padding!
        # --
        if self.is_training() and conf.input_train_zero_rate>0.:
            # no mask for 1: (0,0) which is the indicator!!
            _zero_mask = ((BK.rand(_final_paths_t.shape) < conf.input_train_zero_rate) & (_final_paths_t>1))
            if conf.input_train_zero_incorrect:  # only zero incorrect ones
                _incorrect_mask = ((_paths_t[:,0] != _paths_t[:,1]).long().sum(-1) > 0)  # [*, slen]
                _zero_mask = (_zero_mask & _incorrect_mask)  # [*, slen]
            _final_paths_t *= (1 - _zero_mask.long())  # [*, slen]
        # --
        _final_paths_t *= mask_t.long()  # zero for paddings
        return _final_paths_t

# --
# helper
class ZDecoderSRL2Helper(ZDecoderHelper):
    def __init__(self, conf: ZDecoderSRL2Conf, vocab_evt: SimpleVocab, vocab_arg: SimpleVocab, vocab_arg2: SimpleVocab):
        self.conf = conf
        self.vocab_evt = vocab_evt
        self.vocab_arg = vocab_arg
        self.vocab_arg2 = vocab_arg2
        # --
        self.arg_span_getter = Mention.create_span_getter(conf.arg_span_mode)
        self.arg_span_setter = Mention.create_span_setter(conf.arg_span_mode)

    def _prep_frame(self, frame: Frame):
        conf: ZDecoderSRL2Conf = self.conf
        # --
        slen = len(frame.sent)
        # note: for simplicity, assume no loss_weight_non for args
        arg_arr = np.full([slen], 0, dtype=np.int)  # [arg]
        arg2_arr = np.full([slen], 0, dtype=np.int)  # [arg]
        # arguments
        if conf.arg_only_rank1:
            cur_args = [a for a in frame.args if a.info.get("rank", 1) == 1]
        else:
            cur_args = frame.args
        # bio or not
        if conf.arg_use_bio:  # special
            arg_spans = [self.arg_span_getter(a.mention) + (a.label_idx,) for a in cur_args]
            tag_layers = self.vocab_arg.spans2tags_idx(arg_spans, slen)
            if len(tag_layers) > 1:
                zwarn(f"Warning: 'Full args require multiple layers with {arg_spans}")
            arg_arr[:] = tag_layers[0][0]  # directly assign it!
        else:  # plain ones
            for a in cur_args:
                arg_role = a.label_idx
                arg_widx, arg_wlen = self.arg_span_getter(a.mention)
                arg_arr[arg_widx:arg_widx+arg_wlen] = arg_role
        # arg2, only single ones!
        for a2 in cur_args:
            arg2_arr[a2.mention.shead_widx] = a2.label_idx
        # --
        return ZObject(frame=frame, slen=slen, evt_lab=frame.label_idx, arg_arr=arg_arr, arg2_arr=arg2_arr)

    # prepare inputs
    def prepare(self, insts: List[Sent], use_cache: bool):
        # get info
        zobjs = ZDecoderHelper.get_zobjs(insts, self._prep_frame, use_cache, f"_cache_srl")
        # batch things
        bsize, mlen = len(insts), max(z.slen for z in zobjs) if len(zobjs)>0 else 1  # at least put one as padding
        batched_shape = (bsize, mlen)
        arr_items = np.full(bsize, None, dtype=object)
        arr_evt_labels = np.full(bsize, 0, dtype=np.int)
        arr_arg_labels = np.full(batched_shape, 0, dtype=np.int)
        arr_arg2_labels = np.full(batched_shape, 0, dtype=np.int)
        for zidx, zobj in enumerate(zobjs):
            zlen = zobj.slen
            arr_items[zidx] = zobj.frame
            arr_evt_labels[zidx] = zobj.evt_lab
            arr_arg_labels[zidx, :zlen] = zobj.arg_arr
            arr_arg2_labels[zidx, :zlen] = zobj.arg2_arr
        expr_evt_labels = BK.input_idx(arr_evt_labels)  # [*]
        expr_arg_labels = BK.input_idx(arr_arg_labels)  # [*, slen]
        expr_arg2_labels = BK.input_idx(arr_arg2_labels)  # [*, slen]
        return arr_items, expr_evt_labels, expr_arg_labels, expr_arg2_labels

    # simple arg decode
    def simple_arg_decode(self, tag_idxes: List[int]):
        spans = []
        prev_start, prev_t = 0, 0
        # --
        def _close_prev(_start: int, _end: int, _t, _spans: List):
            if _t > 0:
                assert _end > _start
                _spans.append((_start, _end-_start, _t))
            return _end, 0
        # --
        for idx, tidx in enumerate(tag_idxes):
            if tidx == 0 or tidx != prev_t:
                prev_start, prev_t = _close_prev(prev_start, idx, prev_t, spans)
            prev_t = tidx
        _close_prev(prev_start, len(tag_idxes), prev_t, spans)
        return spans

    # decode arg
    def decode_arg(self, evt: Frame, arg_lidxes, arg_scores, vocab, real_vocab):
        conf: ZDecoderSRL2Conf = self.conf
        # --
        if conf.arg_use_bio:
            new_arg_results = vocab.tags2spans_idx(arg_lidxes)
        else:
            new_arg_results = self.simple_arg_decode(arg_lidxes)
        # --
        new_arg_labels = [vocab.idx2word(z) if z>0 else 'O' for z in arg_lidxes]
        evt.info["arg_lab"] = new_arg_labels
        for a_widx, a_wlen, a_lab in new_arg_results:
            a_lab = int(a_lab)
            assert a_lab > 0, "Error: should not extract 'O'!?"
            new_ef = evt.sent.make_frame(a_widx, a_wlen, self.conf.arg_ftag)  # make an ef as mention
            a_role = real_vocab.idx2word(a_lab)
            alink = evt.add_arg(new_ef, a_role, score=np.mean(arg_scores[a_widx:a_widx+a_wlen]).item())
            alink.set_label_idx(a_lab)  # set idx
            self.arg_span_setter(alink.mention, a_widx, a_wlen)
        # --

    # *[*], *[*, slen]
    def put_results(self, insts: List[Frame], all_arrs):
        conf: ZDecoderSRL2Conf = self.conf
        vocab_evt = self.vocab_evt
        vocab_arg = self.vocab_arg
        if conf.arg_use_bio:
            real_vocab_arg = vocab_arg.base_vocab
        else:
            real_vocab_arg = vocab_arg
        pred_evt_labels, pred_evt_scores, pred_arg_labels, pred_arg_scores = all_arrs
        for bidx, inst in enumerate(insts):
            # --
            # delete old ones
            for arg in list(inst.args):
                # inst.sent.delete_frame(arg.arg, conf.arg_ftag)
                arg.arg.sent.delete_frame(arg.arg, conf.arg_ftag)
            assert len(inst.args) == 0
            # --
            cur_len = len(inst.sent)
            cur_arg_labs, cur_arg_scores = [z[bidx][:cur_len] for z in all_arrs[-2:]]
            # --
            # reuse posi but re-assign label!
            if pred_evt_labels is not None:
                one_lab, one_score = pred_evt_labels[bidx].item(), pred_evt_scores[bidx].item()
                inst.set_label(vocab_evt.idx2word(one_lab))
                inst.set_label_idx(one_lab)
                inst.score = one_score
            # args
            new_arg_scores = cur_arg_scores[:cur_len]
            new_arg_label_idxes = cur_arg_labs[:cur_len]
            self.decode_arg(inst, new_arg_label_idxes, new_arg_scores, vocab_arg, real_vocab_arg)
            # --

    # decode for all layers: [bs], [NL, bs, slen]
    def put_results_all_layers(self, insts, arr_labels):
        # --
        assert self.conf.arg_use_bio
        vocab_arg = self.vocab_arg
        real_vocab_arg = vocab_arg.base_vocab
        # --
        NL = len(arr_labels)  # number of predictions
        for ii, inst in enumerate(insts):
            all_preds = []
            for li in range(NL):
                arr_lidxes = arr_labels[li, ii, :len(inst.sent)].tolist()
                new_arg_labels = [vocab_arg.idx2word(z) if z>0 else 'O' for z in arr_lidxes]
                new_arg_results = vocab_arg.tags2spans_str(new_arg_labels)
                all_preds.append(new_arg_results)
            # --
            inst.info["all_srl2_preds"] = all_preds
        # --

    # fast eval: [bs], [bs, slen]
    def fast_eval_results(self, insts, arr_labels):
        # --
        assert self.conf.arg_use_bio
        vocab_arg = self.vocab_arg
        # --
        err_spans = []
        for ii, inst in enumerate(insts):
            # gold ones
            gold_args = [a.mention.get_span() + (a.label,) for a in inst.args if (a.label not in ["V", "C-V"])]
            gold_args_set = set(gold_args)
            # pred ones
            pred_lidxes = arr_labels[ii, :len(inst.sent)].tolist()
            pred_tags = [vocab_arg.idx2word(z) if z>0 else 'O' for z in pred_lidxes]
            pred_results = vocab_arg.tags2spans_str(pred_tags)
            pred_args_set = set([z for z in pred_results if (z[-1] not in ["V", "C-V"])])
            # get micro result!!
            num_sum = len(gold_args_set) + len(pred_args_set)
            if num_sum == 0:
                res = 1.  # 0/0 is one!!
            else:
                num_hit = 2*len(gold_args_set.intersection(pred_args_set))
                res = num_hit / num_sum
            err_spans.append(1.-res)
        return err_spans

# --
# b msp2/tasks/zmtl/modules/dec_comps/srl2:?
