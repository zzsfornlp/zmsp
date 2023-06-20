#

# pair-wise dpar scoring

__all__ = [
    "DparLabelerConf", "DparLabelerLayer",
]

from ...backends import BK
from ...layers import *
from .plain import *
from mspx.tools.algo.struct import DparHelperConf

@NnConf.rd('out_dpar')
class DparLabelerConf(NnConf):
    def __init__(self):
        self.csize = -1  # number of classes (final output labels)
        self.isize = -1  # by default, the same as csize
        # --
        self.out = DparHelperConf()
        self.scorer_edge = PairScoreConf().direct_update(aff_dim=500, use_biaff=True, _rm_names=['isize', 'osize'])
        self.scorer_lab = PairScoreConf().direct_update(aff_dim=100, use_biaff=True, _rm_names=['isize', 'osize'])
        self.dist_clip = 0  # explicit dist feature: 0 for off, >0 for abs, <0 for use_neg
        self.debug_drop = 0.  # for debugging

@DparLabelerConf.conf_rd()
class DparLabelerLayer(NnLayer):
    def __init__(self, conf: DparLabelerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: DparLabelerConf = self.conf
        # --
        self.out = conf.out.make_node()
        self.scorer_edge = conf.scorer_edge.make_node(isize=conf.isize, osize=1)
        self.scorer_lab = conf.scorer_lab.make_node(isize=conf.isize, osize=conf.csize)
        _clip = conf.dist_clip
        _clip0, _clip1 = [0, _clip] if _clip >= 0 else [_clip, -_clip]  # dist range
        if _clip1 > _clip0:
            self.dist_aff = AffineLayer(None, isize=conf.isize, osize=(_clip1-_clip0+1))
        else:
            self.dist_aff = None
        self.dist_range = (_clip0, _clip1)
        # --

    # [*, L, D]
    def forward_scores(self, expr_t: BK.Expr):
        s_edge = self.scorer_edge(expr_t, expr_t)  # [*, Lm, Lh, 1]
        # --
        # add dist score
        if self.dist_aff is not None:
            _tmp = BK.arange_idx(BK.get_shape(expr_t, -2))  # [L]
            _dist = _tmp.unsqueeze(0) - _tmp.unsqueeze(1)  # [L, L]
            _clip0, _clip1 = self.dist_range
            if _clip0 >= 0:  # no-neg mode
                _dist = _dist.abs()
            _dist.clamp_(min=_clip0, max=_clip1)  # [L, L]
            td = self.dist_aff(expr_t)  # [*, L, VD]
            t_score = td.gather(-1, unsqueeze_expand((_dist - _clip0), 0, len(expr_t)))  # [*, L, L]
            s_edge = s_edge + t_score.unsqueeze(-1)
        # --
        s_lab = self.scorer_lab(expr_t, expr_t)  # [*, Lm, Lh, V]
        s_full = s_edge + s_lab  # [*, Lm, Lh, V]
        return s_edge, s_full

    # [*, Lm, Lh, V], [*, L], ([*, L, 2] or [*, Lm, Lh, V])
    def forward_loss(self, t_score: BK.Expr, t_weight: BK.Expr, t_gold: BK.Expr):
        _debug_drop = self.conf.debug_drop
        # --
        t_mask = (t_weight > 0.).to(t_weight)
        _shape = BK.get_shape(t_score)  # [*, Lm, Lh, V]
        _shape_gold = BK.get_shape(t_gold)
        if len(_shape_gold) < len(_shape):
            _t1 = extend_idxes(t_gold[..., 0], _shape[-2])  # [*, Lm, Lh]
            _t2 = extend_idxes(t_gold[..., 1], _shape[-1])  # [*, Lm, V]
            t_gold = _t1.unsqueeze(-1) * _t2.unsqueeze(-2)  # [*, Lm, Lh, V]
        with BK.no_grad_env():
            mg = self.out.get_marginals(t_score, t_mask)  # [*, Lm, Lh, V]
        dpar_mask = self.out.make_dpar_mask(t_mask)  # [*, L, L]
        _grad = ((mg - t_gold) * dpar_mask.unsqueeze(-1)).detach()
        # -- for debugging!
        if _debug_drop > 0:
            _dm = BK.random_bernoulli(BK.get_shape(_grad)[:2], (1.-_debug_drop), 1.).unsqueeze(-1).unsqueeze(-1)
            _grad = _dm * _grad
        # --
        t_loss = _grad * t_score  # [*, Lm, Lh, V]
        # --
        t_weight2 = t_weight.detach().clone()
        t_weight2[..., 0] = 0.  # no loss for AROOT!
        ret = ((t_loss.sum(-1).sum(-1)*t_weight2).sum(), t_weight2.sum())  # (loss-sum, div-count)
        return ret

    def forward_pred(self, t_score: BK.Expr, t_mask: BK.Expr):
        ret = self.out.get_argmax(t_score, t_mask)
        return ret  # [*, Lm], [*, Lm]

# --
# b mspx/nn/layers2/zout/dpar:
