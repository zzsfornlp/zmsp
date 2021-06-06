#

# helper (loss & pred) for idec

__all__ = [
    "IdecHelperConf", "IdecHelper",
    "IdecHelperSimpleConf", "IdecHelperSimple", "IdecHelperSimple2Conf", "IdecHelperSimple2",
    "SeqExitHelperConf", "SeqExitHelperNode", "OutputHelper",
]

from typing import List, Union
from collections import OrderedDict
import math
import numpy as np
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.nn.modules import LossHelper
from msp2.data.inst import DataPadder
from msp2.proc import SVConf, ScheduledValue
from msp2.utils import zlog, zopen, default_pickle_serializer

# =====

class IdecHelperConf(BasicConf):
    pass

@node_reg(IdecHelperConf)
class IdecHelper(BasicNode):
    # --
    # helper

    # List[*,D], [*,]
    @staticmethod
    def gather_losses(all_scores: List, gold_idxes: BK.Expr, lsmooth: float, distill_ratio: float):
        all_scores = list(all_scores)
        last_scores_t = all_scores[-1]  # [*,D]
        last_probs_t = last_scores_t.softmax(-1).detach() if distill_ratio>0. else None  # [*,D]
        _last_idx = len(all_scores)-1
        # --
        ret_losses = []
        for one_idx, one_scores in enumerate(all_scores):
            if distill_ratio <= 0. or one_idx == _last_idx:
                one_loss = BK.loss_nll(one_scores, gold_idxes, label_smoothing=lsmooth)  # [*]
            elif distill_ratio >= 1.:
                one_loss = - (last_probs_t * one_scores.log_softmax(-1)).sum(-1)  # [*]
            else:  # mix
                one_loss = (1-distill_ratio) * BK.loss_nll(one_scores, gold_idxes, label_smoothing=lsmooth) \
                           - distill_ratio * (last_probs_t * one_scores.log_softmax(-1)).sum(-1)
            ret_losses.append(one_loss)
        return ret_losses
        # --

# 1. simple weighted
class IdecHelperSimpleConf(IdecHelperConf):
    def __init__(self):
        super().__init__()
        # note: pad weights of 0. to the left, thus by default only use last layer!
        self.loss_weights = [1.]
        self.pred_weights = [1.]
        self.pred_mix_probs = False  # weight probs rather than logprobs
        # --
        # special ones
        self.loss_groups = []  # for example, 4-group:1,1,1,1, 2/1-group:2,1, ...
        self.loss_weights_logs2 = []  # for auto loss-weight!
        self.loss_logs2_scale = 1.

@node_reg(IdecHelperSimpleConf)
class IdecHelperSimple(BasicNode):
    def __init__(self, conf: IdecHelperSimpleConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecHelperSimpleConf = self.conf
        # --
        self.loss_weights = IdecHelperSimple.get_weights(conf.loss_weights)  # [?]
        self.pred_weights = IdecHelperSimple.get_weights(conf.pred_weights)  # [?]
        self.loss_groups = [int(z) for z in conf.loss_groups]  # [?]
        self.loss_weights_logs2 = IdecHelperSimple.get_weights(conf.loss_weights_logs2, pad=1.)  # [?]
        self.logs2 = BK.new_param([len(conf.loss_weights_logs2)]) if (len(conf.loss_weights_logs2)>0) else None  # [?]
        self.reset_parameters()

    def reset_parameters(self):
        if self.logs2 is not None:
            BK.init_param(self.logs2, "zero")

    def print_logs2(self):  # mainly for printing
        values = BK.get_value(self.logs2) * self.conf.loss_logs2_scale
        zlog(f"IdecHelper: logs2={values}, exp(-logs2)={np.exp(-values)}")

    @staticmethod
    def get_weights(weights, pad: float=0.):
        final_weights = [pad] * 100  # note: this should be enough!
        final_weights[-len(weights):] = [float(z) for z in weights]  # assign the last ones!
        ret = BK.input_real(final_weights)
        # ret /= ret.sum(-1)  # normalize!, note: not doing this!!
        return ret

    def loss(self, all_losses: Union[List[BK.Expr], BK.Expr], **kwargs):
        if isinstance(all_losses, list):  # stack it
            _nl = len(all_losses)
            stack_t = BK.stack(all_losses, -1)  # [*, NL]
        else:  # already stacked!!
            _nl = BK.get_shape(all_losses, -1)  # NL
            stack_t = all_losses
        _weights = self.loss_weights[-_nl:]  # [NL]
        _weights_stack_t = stack_t*_weights  # [*, NL]
        # --
        # special
        if self.logs2 is not None:  # (1/s^2)*L+log(s)
            _logs2 = self.logs2 * self.conf.loss_logs2_scale
            _weights_stack_t = BK.exp(-_logs2) * _weights_stack_t + 0.5 * _logs2 * self.loss_weights_logs2[-_nl:]
        # --
        loss_list = self._split_loss(_weights_stack_t)  # [*]
        rets = [(z, 1., "" if i==0 else f"_{i}") for i,z in enumerate(loss_list)]
        return rets

    def _split_loss(self, stack_loss: BK.Expr):
        _budget = BK.get_shape(stack_loss, -1)  # NL
        _splits = []
        for _g in self.loss_groups + [_budget]:
            if _g < _budget:  # still have ones
                _splits.append(_g)
                _budget -= _g
            else:  # directly put all remaining budgets
                _splits.append(_budget)
                break
        # --
        loss_list = stack_loss.split(_splits, dim=-1)  # *[*, ??]
        return [z.sum(-1) for z in loss_list]

    def pred(self, all_logprobs: List[BK.Expr], **kwargs):
        _nl = len(all_logprobs)
        _weights = self.pred_weights[-_nl:]  # [NL]
        # --
        stack_t = BK.stack(all_logprobs, -1)  # [*, NL]
        if self.conf.pred_mix_probs:
            prob_t = stack_t.exp()  # [*, NL]
            prob_t2 = (prob_t*_weights).sum(-1)  # [*]
            ret_t = (prob_t2 + 1e-6).log()  # [*]
        else:
            ret_t = (stack_t*_weights).sum(-1)  # [*]
        return ret_t

# 2. simple weighted (simplified and left-align version)
class IdecHelperSimple2Conf(IdecHelperConf):
    def __init__(self):
        super().__init__()
        # --
        self.loss_weights = []
        self.pred_weights = []  # note: not used anymore!!
        self.pred_sel_idx = -1  # simply select one (by default the last)
        # schedule for weighting!!
        self.use_lw_num = 0  # make it brief!
        self.lw0 = SVConf().direct_update(val=1., which_idx="cidx", mode="none", b=0., k=0.02, max_val=1.)
        self.lw1 = SVConf().direct_update(val=1., which_idx="cidx", mode="none", b=0., k=0.02, max_val=1.)
        self.lw2 = SVConf().direct_update(val=1., which_idx="cidx", mode="none", b=0., k=0.02, max_val=1.)
        self.lw3 = SVConf().direct_update(val=1., which_idx="cidx", mode="none", b=0., k=0.02, max_val=1.)
        self.lw4 = SVConf().direct_update(val=1., which_idx="cidx", mode="none", b=0., k=0.02, max_val=1.)
        # --

@node_reg(IdecHelperSimple2Conf)
class IdecHelperSimple2(BasicNode):
    def __init__(self, conf: IdecHelperSimple2Conf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecHelperSimple2Conf = self.conf
        # --
        self.loss_weights = [float(z) for z in conf.loss_weights]  # [?]
        self.lws = [ScheduledValue(f"lw{ii}", cc) for ii,cc in
                    enumerate([conf.lw0, conf.lw1, conf.lw2, conf.lw3, conf.lw4][:conf.use_lw_num])]
        # --

    def _get_scheduled_values(self):
        return OrderedDict([(f"_LW{ii}", cc) for ii,cc in enumerate(self.lws)])

    def _get_lw_values(self, length: int):
        rets = [z.value for z in self.lws[:length]]
        if len(rets) < length:
            rets.extend([1.] * (length - len(rets)))  # by default 1.
        return rets

    def loss(self, all_losses: List[BK.Expr], add_all=True, **kwargs):
        rets = []
        _lws = self._get_lw_values(len(all_losses))
        for i, loss in enumerate(all_losses):
            if loss is not None:
                rets.append((loss, self.loss_weights[i]*_lws[i], f"_{i}"))
        if add_all:
            rets1 = [(BK.stack([a*b for a,b,c in rets], 0).sum(0), 1., "")]
            return rets1
        else:
            return rets

    def pred(self, all_logprobs: List[BK.Expr], **kwargs):
        return all_logprobs[self.conf.pred_sel_idx]

# =====
# early exit helper

class SeqExitHelperConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self.exit_thresh = 1.
        self.exit_crit = "prob"  # prob/cert/cf
        self.exit_min_k = 1.  # avg. of if >=1 min_k, or if <1, as fraction
        # special for cf mode
        self.cf_use_seq = False  # token-level or seq-level
        # for example: "lambda ff: ff.info['corr_seq'][-1]"  # [NL, slen]
        # for example: "lambda ff: [(1.-z) for z in ff.info['err_span'][-1]]"  # [NL]
        self.cf_oracle_f = "lambda ff: [[1.]*len(ff.sent) for _ in range(3)]"
        self.cf_loss_discard = 0.  # discard by rate of (discard*(oracle**curve)), 0. means no discard
        self.cf_loss_discard_curve = 3.
        self.cf_scale = 1.  # scale for cf_scores

@node_reg(SeqExitHelperConf)
class SeqExitHelperNode(BasicNode):
    def __init__(self, conf: SeqExitHelperConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: SeqExitHelperConf = self.conf
        # --
        self._cri_f = {"prob": self._cri_prob, "cert": self._cri_cert, "cf": None}[conf.exit_crit]
        self.is_cf = (conf.exit_crit == "cf")
        self.cf_oracle_f = eval(conf.cf_oracle_f)
        # --

    # criteria
    def _cri_prob(self, scores: BK.Expr):  # max prob
        return scores.softmax(-1).max(-1)[0]  # [*, slen]

    def _cri_cert(self, scores: BK.Expr):  # 1.-uncertainty
        uncertainty = (scores.softmax(-1) * scores.log_softmax(-1)).sum(-1) / (-math.log(BK.get_shape(scores, -1)))
        return 1. - uncertainty

    def _getk(self, slen: int, k: float):
        if k >= 1.:
            return min(slen, int(k))
        else:
            return max(1, int(slen*k))

    # [*, slen, L], [*, slen]
    def judge(self, scores: BK.Expr, cf_scores: BK.Expr, mask_t: BK.Expr):
        conf: SeqExitHelperConf = self.conf
        # --
        if self.is_cf and conf.cf_use_seq:  # in this mode, already aggr_metrics
            return (cf_scores.squeeze(-1)/conf.cf_scale) >= conf.exit_thresh
        # --
        if self.is_cf:  # Geometric-like qt
            # aggr_metrics = (cf_scores.squeeze(-1)).sigmoid()  # [*]
            seq_metrics = cf_scores.squeeze(-1) / conf.cf_scale  # [*, slen]
        else:
            seq_metrics = self._cri_f(scores)  # [*, slen]
        # --
        seq_metrics = (1.-mask_t) + mask_t * seq_metrics  # put 1. at mask place!
        slen = BK.get_shape(seq_metrics, -1)
        K = self._getk(slen, conf.exit_min_k)
        aggr_metrics = topk_avg(seq_metrics, mask_t, K, dim=-1, largest=False)  # [*]
        return aggr_metrics >= conf.exit_thresh

    # # get loss for cf mode
    # def loss_cf(self, cf_scores: List[BK.Expr], insts, loss_cf: float):
    #     assert self.is_cf
    #     # Geometric-like qt
    #     qts = []
    #     _prev = 1.
    #     for one_cf_scores in cf_scores:
    #         cur_p = one_cf_scores.squeeze(-1).sigmoid()  # [*]
    #         qts.append(_prev * cur_p)
    #         _prev = _prev * (1.-cur_p)
    #     qts[-1] = qts[-1] + _prev
    #     stacked_qt = BK.stack(qts, -1)  # [*,NL]
    #     # --
    #     _bs = len(insts)
    #     _nl = len(cf_scores)
    #     oracle_scores = BK.input_real([[self.oracle_cf_f(ff,ii) for ii in range(_nl)] for ff in insts])  # [*,NL]
    #     _, oracle_idx = oracle_scores.min(-1)
    #     loss_t = -(stacked_qt[BK.arange_idx(_bs), oracle_idx] + 1e-10).log()  # [*], simply nll
    #     cf_loss_item = LossHelper.compile_leaf_loss(
    #         "cf", loss_t.sum(), BK.input_real(len(loss_t)), loss_lambda=loss_cf)
    #     return [cf_loss_item]

    # get loss for cf mode
    def loss_cf(self, cf_scores: List[BK.Expr], insts, loss_cf: float):
        conf: SeqExitHelperConf = self.conf
        # --
        assert self.is_cf
        # get oracle
        oracles = [self.cf_oracle_f(ff) for ff in insts]  # bs*[NL, slen] or bs*[NL]
        rets = []
        mask_t = BK.input_real(DataPadder.lengths2mask([len(z.sent) for z in insts]))  # [bs, slen]
        for one_li, one_scores in enumerate(cf_scores):
            if conf.cf_use_seq:
                one_oracle_t = BK.input_real([z[one_li] for z in oracles])  # [bs]
                one_oracle_t *= conf.cf_scale
                one_mask_t = BK.zeros([len(one_oracle_t)]) + 1
            else:
                one_oracle_t = BK.input_real(DataPadder.go_batch_2d([z[one_li] for z in oracles], 1.))  # [bs, slen]
                one_mask_t = (BK.rand(one_oracle_t.shape) >= (
                        (one_oracle_t**conf.cf_loss_discard_curve) * conf.cf_loss_discard)) * mask_t
                one_oracle_t *= conf.cf_scale
            # simple L2 loss
            one_loss_t = (one_scores.squeeze(-1) - one_oracle_t) ** 2
            one_loss_item = LossHelper.compile_leaf_loss(
                f"cf{one_li}", (one_loss_t * one_mask_t).sum(), one_mask_t.sum(), loss_lambda=loss_cf)
            rets.append(one_loss_item)
        return rets

# --
class OutputHelper:
    def __init__(self, file: str):
        self.file = file
        self.fd = zopen(file, 'wb')
        zlog(f"Open file for OutputHelper writing: {file}({self.fd})")

    def __del__(self):
        self.fd.close()
        zlog(f"Close file for OutputHelper writing: {self.file}({self.fd})")

    def write(self, x):
        default_pickle_serializer.to_file(x, self.fd)

# --
# b msp2/tasks/zmtl/modules/dec_help.py