#

# i-decoder

__all__ = [
    "IdecConf", "IdecNode", "IdecManager", "IdecVstate",
    "SingleIdecConf", "SingleIdecNode", "PairwiseIdecConf", "PairwiseIdecNode",
]

from typing import List
from collections import OrderedDict
from msp2.nn import BK
from msp2.nn.layers import *
from .block import *
from msp2.utils import ConfEntryChoices
from msp2.proc import SVConf, ScheduledValue

# =====
class IdecConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        # basic dims
        self.ndim = -1
        self.nlab = -1
        # --
        # apply layers
        self.app_layers = []  # which layers to apply this idec
        self.feed_layers = []  # which layers to further allow feed (first should have 'app')
        # --
        # loss & pred
        self.hconf: IdecHelperConf = ConfEntryChoices(
            {"simple": IdecHelperSimpleConf(), "cw": IdecHelperCWConf(), "cw2": IdecHelperCW2Conf()}, "simple")

@node_reg(IdecConf)
class IdecNode(BasicNode):
    def __init__(self, conf: IdecConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecConf = self.conf
        # --
        self.app_lidxes = set([int(z) for z in conf.app_layers])
        self.feed_lidxes = set([int(z) for z in conf.feed_layers])
        self.buffer_scores = OrderedDict()  # lidx -> scores
        self.buffer_inputs = OrderedDict()  # lidx -> inputs
        self.buffer_cfs = OrderedDict()  # lidx -> confidences
        # special
        self.fixed_score_t = None
        # for loss/pred
        self.helper = conf.hconf.helper_type()(len(self.app_lidxes), conf.hconf)

    def refresh(self, rop: RefreshOptions = None):
        super().refresh(rop)
        # --
        # clear buffer
        self.buffer_scores.clear()
        self.buffer_inputs.clear()
        self.buffer_cfs.clear()
        self.fixed_score_t = None
        # --

    # set special value!
    def set_fixed_score_t(self, t: BK.Expr):
        self.fixed_score_t = t

    # get all scores
    def get_all_values(self):
        rets = []
        for z in [self.buffer_inputs, self.buffer_cfs, self.buffer_scores]:
            ones = list(z.values())
            assert len(ones) == len(self.app_lidxes)
            rets.append(ones)
        return rets

    def forward(self, input_expr: BK.Expr, input_mask: BK.Expr, cur_lidx: int):
        ret = None
        if cur_lidx in self.app_lidxes:
            cur_feed_output = (cur_lidx in self.feed_lidxes)
            # do forward
            score_t, cf_t, final_t = self._forw(input_expr, input_mask, feed_output=cur_feed_output)
            if cur_feed_output:
                ret = final_t
            # store info
            self.buffer_scores[cur_lidx] = score_t
            self.buffer_inputs[cur_lidx] = input_expr
            self.buffer_cfs[cur_lidx] = cf_t
        return ret

    def _forw(self, input_expr: BK.Expr, input_mask: BK.Expr, feed_output: bool):
        raise NotImplementedError()

class SingleIdecConf(IdecConf):
    def __init__(self):
        super().__init__()
        self.sconf = SingleBlockConf()

@node_reg(SingleIdecConf)
class SingleIdecNode(IdecNode):
    def __init__(self, conf: SingleIdecConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: SingleIdecConf = self.conf
        # --
        self.snode = SingleBlockNode(conf.sconf, ndim=conf.ndim, nlab=conf.nlab)

    def _forw(self, input_expr: BK.Expr, input_mask: BK.Expr, feed_output: bool):
        # note: not using input_mask here since masks will be handled later in loss/pred
        return self.snode.forward(input_expr, fixed_scores_t=self.fixed_score_t, feed_output=feed_output)

class PairwiseIdecConf(IdecConf):
    def __init__(self):
        super().__init__()
        self.pconf = PairwiseBlockConf()

@node_reg(PairwiseIdecConf)
class PairwiseIdecNode(IdecNode):
    def __init__(self, conf: PairwiseIdecConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: PairwiseIdecConf = self.conf
        # --
        self.pnode = PairwiseBlockNode(conf.pconf, ndim=conf.ndim, nlab=conf.nlab)

    def _forw(self, input_expr: BK.Expr, input_mask: BK.Expr, feed_output: bool):
        # note: only use input_mask as mask_k!
        return self.pnode.forward(input_expr, fixed_scores_t=self.fixed_score_t, feed_output=feed_output, mask_k=input_mask)

# =====
# manage the idec
class IdecManager:
    def __init__(self):
        self.nodes = []  # note: we follow the overall adding order!!

    def add_node(self, node: IdecNode):
        assert node not in self.nodes, "Cannot add one node twice!"
        self.nodes.append(node)

    def add_nodes(self, nodes):
        for n in nodes:
            self.add_node(n)

    def new_vstate(self, input_t: BK.Expr, mask_t: BK.Expr):
        return IdecVstate(self, input_t, mask_t)

class IdecVstate(VrecSteppingState):
    def __init__(self, manager: IdecManager, input_t: BK.Expr, mask_t: BK.Expr):
        super().__init__(input_t, mask_t)
        # --
        self.manager = manager
        self.cur_lidx = 0  # current layer idx (how many layers pass?)

    def update(self, input_expr: BK.Expr, force_lidx=None, **kwargs):
        if force_lidx is None:
            self.cur_lidx += 1  # starting from 1
        else:
            assert self.cur_lidx == force_lidx  # usually for Layer0!
        # --
        cur_expr = None  # None means input_expr
        for node in self.manager.nodes:
            cur_res = node.forward(input_expr if cur_expr is None else cur_expr, self.mask_t, self.cur_lidx)
            if cur_res is not None:
                cur_expr = cur_res  # at least get replaced!
        return cur_expr  # if None, only recording and no change!!

# =====
# helper (loss & pred) for idec
class IdecHelperConf(BasicConf):
    def helper_type(self):
        raise NotImplementedError()

@node_reg(IdecHelperConf)
class IdecHelper(BasicNode):
    pass

# 1. simply weighted
class IdecHelperSimpleConf(IdecHelperConf):
    def __init__(self):
        super().__init__()
        self.loss_weights = [1.]
        self.pred_weights = [1.]
        self.pred_mix_probs = False  # weight probs rather than logprobs

    def helper_type(self):
        return IdecHelperSimple

@node_reg(IdecHelperSimpleConf)
class IdecHelperSimple(BasicNode):
    def __init__(self, nlayer: int, conf: IdecHelperSimpleConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecHelperSimpleConf = self.conf
        # --
        self.loss_weights = IdecHelperSimple.get_weights(nlayer, conf.loss_weights)  # [NL]
        self.pred_weights = IdecHelperSimple.get_weights(nlayer, conf.pred_weights)  # [NL]

    @staticmethod
    def get_weights(nl: int, weights):
        final_weights = [weights[-1]] * nl
        mlen = min(nl, len(weights))
        final_weights[:mlen] = weights[:mlen]
        ret = BK.input_real(final_weights)
        ret /= ret.sum(-1)  # normalize!
        return ret

    def loss(self, all_losses: List[BK.Expr], **kwargs):
        stack_t = BK.stack(all_losses, -1)  # [*, NL]
        ret_t = (stack_t*self.loss_weights).sum(-1)  # [*]
        return [(ret_t, 1., "")]

    def pred(self, all_logprobs: List[BK.Expr], **kwargs):
        stack_t = BK.stack(all_logprobs, -1)  # [*, NL]
        if self.conf.pred_mix_probs:
            prob_t = stack_t.exp()  # [*, NL]
            prob_t2 = (prob_t*self.pred_weights).sum(-1)  # [*]
            ret_t = (prob_t2 + 1e-6).log()  # [*]
        else:
            ret_t = (stack_t*self.pred_weights).sum(-1)  # [*]
        return ret_t

# 2. confidence weighted
class IdecHelperCWConf(IdecHelperConf):
    def __init__(self):
        super().__init__()
        self.temperature = SVConf().direct_update(val=1., which_idx="eidx", mode="none", min_val=0.)
        self.cf_merge_mode = "Multinomial"  # Multinomial/Geometric
        self.pred_argmax = True  # select the argmax one!
        self.pred_mix_probs = False  # weight probs rather than logprobs

    def helper_type(self):
        return IdecHelperCW

@node_reg(IdecHelperCWConf)
class IdecHelperCW(BasicNode):
    def __init__(self, nlayer: int, conf: IdecHelperCWConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecHelperCWConf = self.conf
        # --
        self._merge_cf_f = {"Multinomial": self._merge_cf_multi, "Geometric": self._merge_cf_geo}[conf.cf_merge_mode]
        self.temperature = ScheduledValue("T", conf.temperature)

    def _get_scheduled_values(self):
        return OrderedDict([("_temp", self.temperature)])

    @property
    def current_temperature(self):
        if self.is_training():  # only != 1. in training!!
            return self.temperature.value
        else:
            return 1.

    def _merge_cf_multi(self, all_cfs: List[BK.Expr]):
        _temp = self.current_temperature
        stack_cf = BK.stack(all_cfs, -1)
        ret = stack_cf.softmax(-1) if _temp==1. else (stack_cf/_temp).softmax(-1)  # [*, NL]
        return ret

    def _merge_cf_geo(self, all_cfs: List[BK.Expr]):
        _temp = self.current_temperature
        accu_cfs = []
        remainings = None
        for cf in all_cfs:
            cf_prob = cf.sigmoid() if _temp==1. else (cf/_temp).sigmoid()  # [*]
            if remainings is None:
                accu_cfs.append(cf_prob)
                remainings = 1.-cf_prob
            else:
                accu_cfs.append(cf_prob * remainings)
                remainings = remainings * (1.-cf_prob)
        # add back to the final one
        accu_cfs[-1] += remainings
        return BK.stack(accu_cfs, -1)  # [*, NL]

    def loss(self, all_losses: List[BK.Expr], all_cfs: List[BK.Expr], **kwargs):
        stack_t = BK.stack(all_losses, -1)  # [*, NL]
        cf_t = self._merge_cf_f(all_cfs)  # [*, NL]
        ret_t = (stack_t*cf_t).sum(-1)  # [*]
        return [(ret_t, 1., "")]

    def pred(self, all_logprobs: List[BK.Expr], all_cfs: List[BK.Expr], **kwargs):
        conf: IdecHelperCWConf = self.conf
        # --
        stack_t = BK.stack(all_logprobs, -2)  # [*, NL, L]
        cf_t = self._merge_cf_f(all_cfs)  # [*, NL]
        if conf.pred_argmax:
            _, _lidxes = cf_t.max(-1, keepdim=True)  # [*, 1]
            ret_t = BK.gather_first_dims(stack_t, _lidxes, -2).squeeze(-2)  # [*, L]
        else:
            if conf.pred_mix_probs:
                prob_t = stack_t.exp()  # [*, NL, L]
                prob_t2 = (prob_t*cf_t.unsqueeze(-1)).sum(-1)  # [*, L]
                ret_t = (prob_t2 + 1e-6).log()  # [*, L]
            else:
                ret_t = (stack_t*cf_t.unsqueeze(-1)).sum(-1)  # [*, L]
        return ret_t

# 3. entropy-styled weighted
class IdecHelperCW2Conf(IdecHelperConf):
    def __init__(self):
        super().__init__()
        # --
        self.temperature = SVConf().direct_update(val=1., which_idx="eidx", mode="none", min_val=0.)
        self.detach_weights = True  # detach weights?
        self.max_cf = 0.999  # act like label-smoothing
        self.cf_trg_rel = False  # if so, target at relative (exp(s-max(s))); otherwise, absolute gold prob as trg
        self.loss_cf = 0.5

    def helper_type(self):
        return IdecHelperCW2

@node_reg(IdecHelperCW2Conf)
class IdecHelperCW2(BasicNode):
    def __init__(self, nlayer: int, conf: IdecHelperCW2Conf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: IdecHelperCW2Conf = self.conf
        # --
        self.temperature = ScheduledValue("T", conf.temperature)

    def _get_scheduled_values(self):
        return OrderedDict([("_temp", self.temperature)])

    def loss(self, all_losses: List[BK.Expr], all_cfs: List[BK.Expr], **kwargs):
        conf: IdecHelperCW2Conf = self.conf
        _temp = self.temperature.value
        # --
        stack_t = BK.stack(all_losses, -1)  # [*, NL]
        w_t = (- stack_t / _temp)  # [*, NL], smaller loss is better!
        w_t_detach = w_t.detach()
        # main loss
        apply_w_t = w_t_detach if conf.detach_weights else w_t
        ret_t = (stack_t * apply_w_t.softmax(-1)).sum(-1)  # [*]
        # cf loss
        cf_t = BK.stack(all_cfs, -1).sigmoid()  # [*, NL]
        if conf.cf_trg_rel:  # relative prob proportion?
            _max_t = w_t_detach.sum(-1, keepdim=True) if BK.is_zero_shape(w_t_detach) else w_t_detach.max(-1, keepdim=True)[0]  # [*, 1]
            _trg_t = (w_t_detach - _max_t).exp() * conf.max_cf  # [*, NL]
        else:
            _trg_t = w_t_detach.exp() * conf.max_cf
        loss_cf_t = BK.loss_binary(cf_t, _trg_t).mean(-1)  # [*]
        return [(ret_t, 1., ""), (loss_cf_t, conf.loss_cf, "_cf")]

    def pred(self, all_logprobs: List[BK.Expr], all_cfs: List[BK.Expr], **kwargs):
        conf: IdecHelperCWConf = self.conf
        # --
        stack_t = BK.stack(all_logprobs, -2)  # [*, NL, L]
        cf_t = BK.stack(all_cfs, -1).sigmoid()  # [*, NL]
        _, _lidxes = cf_t.max(-1, keepdim=True)  # [*, 1]
        ret_t = BK.gather_first_dims(stack_t, _lidxes, -2).squeeze(-2)  # [*, L]
        return ret_t

# --
# b msp2/tasks/common/models/iter/idec:270
