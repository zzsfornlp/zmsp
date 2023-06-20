#

# plain labeler

__all__ = [
    "NilModConf", "NilModLayer", "LabelerConf", "LabelerLayer",
]

from mspx.utils import ConfEntryChoices
from ...backends import BK
from ...layers import *

# modify nil scores
@NnConf.rd('nil_mod')
class NilModConf(NnConf):
    def __init__(self):
        super().__init__()
        self.fixed_nil_val = ConfEntryChoices({'yes': ScalarConf()})

@NilModConf.conf_rd()
class NilModLayer(NnLayer):
    def __init__(self, conf: NilModConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: NilModConf = self.conf
        # --
        if conf.fixed_nil_val is not None:
            self.fixed_nil_val = conf.fixed_nil_val.make_node()
        else:
            self.fixed_nil_val = None
        # --

    def forward_scores(self, score_t: BK.Expr, nil_add_score: float = 0., return_slice0=False):
        # get original NIL score
        orig_nil_t = score_t[..., 0:1]  # [..., 1]
        # fix NIL(idx=0)
        if self.fixed_nil_val is not None:
            _v = self.fixed_nil_val() + nil_add_score
            _slice = orig_nil_t * 0. + _v
            score_t = BK.concat([_slice, score_t[..., 1:]], -1)  # [..., 1+(V-1)]
        elif nil_add_score != 0.:
            _tmp = BK.zeros([BK.get_shape(score_t, -1)])
            _tmp[0] = nil_add_score
            score_t = score_t + _tmp
        # --
        if return_slice0:
            return score_t, orig_nil_t
        else:
            return score_t

# plain labeler
@NnConf.rd('out_lab')
class LabelerConf(NnConf):
    def __init__(self):
        super().__init__()
        # --
        self.csize = -1  # number of classes (final output labels)
        self.isize = -1  # by default, the same as csize
        # --
        self.mlp = MlpConf().direct_update(_rm_names=['isize', 'osize'])  # final MLP scorer
        self.nil_mod = NilModConf()
        # --

@LabelerConf.conf_rd()
class LabelerLayer(NnLayer):
    def __init__(self, conf: LabelerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: LabelerConf = self.conf
        # --
        self.mlp = conf.mlp.make_node(isize=conf.isize, osize=conf.csize)
        self.nil_mod = conf.nil_mod.make_node()

    def forward_scores(self, expr_t: BK.Expr, nil_add_score: float = 0.):
        score_t = self.mlp(expr_t)
        rets = self.nil_mod.forward_scores(score_t, nil_add_score)
        return rets

    # [*, V], [*], ([*] or [*, V])
    def forward_loss(self, t_scores: BK.Expr, t_weight: BK.Expr, t_gold: BK.Expr):
        _shape = BK.get_shape(t_scores)  # [*, V]
        _shape_gold = BK.get_shape(t_gold)
        if len(_shape_gold) < len(_shape):  # gather it!
            t_loss = BK.loss_nll(t_scores, t_gold)  # [*]
        else:  # calculate the cross-entropy!
            _logprob = t_scores.log_softmax(-1)  # [*, V]
            t_loss = - (t_gold * _logprob).sum(-1)  # [*]
        ret = ((t_loss*t_weight).sum(), t_weight.sum())  # (loss-sum, div-count)
        return ret

    def forward_pred(self, t_scores: BK.Expr, t_mask: BK.Expr):
        # simply argmax!
        p_scores, p_idxes = t_scores.max(-1)
        p_idxes = p_idxes * t_mask.to(BK.DEFAULT_INT)
        return p_idxes  # [*]
