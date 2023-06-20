#

# for sequence labeling

__all__ = [
    "SeqLabelerConf", "SeqLabelerLayer",
]

from ...backends import BK
from ...layers import *
from .plain import *
from mspx.tools.algo.struct import SeqlabHelperConf

@NnConf.rd('out_seqlab')
class SeqLabelerConf(NnConf):
    def __init__(self):
        super().__init__()
        self.csize = -1  # number of classes (final output labels)
        self.isize = -1  # by default, the same as csize
        # --
        self.out = SeqlabHelperConf()
        self.unary = LabelerConf().direct_update(_rm_names=['csize', 'isize'])
        self.binary = BigramConf().direct_update(_rm_names=['osize'])

@SeqLabelerConf.conf_rd()
class SeqLabelerLayer(NnLayer):
    def __init__(self, conf: SeqLabelerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: SeqLabelerConf = self.conf
        # --
        self.out = conf.out.make_node()
        self.unary = conf.unary.make_node(csize=conf.csize, isize=conf.isize)
        if self.out.requires_binary:
            self.binary = conf.binary.make_node(osize=conf.csize)
        else:
            self.binary = None
        # --

    def _sum_until(self, t: BK.Expr, until_dims: int, sum_dim: int):
        while len(BK.get_shape(t)) > until_dims:
            t = t.sum(sum_dim)
        return t

    # [*, L, D]
    def forward_scores(self, expr_t: BK.Expr, nil_add_score: float = 0.):
        unary_scores = self.unary.forward_scores(expr_t, nil_add_score=nil_add_score)
        binary_mat = self.binary.M if (self.binary is not None) else None
        return unary_scores, binary_mat

    # [*, L, V], [V, V], [*, L], ([*, L] or [*, L, V]), [..., V, V]
    def forward_loss(self, t_unary: BK.Expr, m_binary: BK.Expr, t_weight: BK.Expr,
                     t_gold: BK.Expr, t_gold_m: BK.Expr):
        t_mask = (t_weight > 0.).to(t_weight)  # [*, L]
        _shape = BK.get_shape(t_unary)  # [*, L, V]
        _shape_gold = BK.get_shape(t_gold)
        if len(_shape_gold) < len(_shape):  # L = logsumexp - score(gold)
            t_weight = t_mask  # note: no weighting in this case!
            t_partition = self.out.get_partition(t_unary, m_binary, t_mask)  # [*]
            g_unary = t_unary.gather(-1, t_gold.unsqueeze(-1)).squeeze(-1) * t_mask  # [*, L]
            if m_binary is not None:  # [*, L-1]
                g_binary = m_binary[t_gold[..., :-1], t_gold[..., 1:]] * t_mask[..., 1:] * t_mask[..., :-1]
                t_loss = t_partition - (g_unary.sum(-1) + g_binary.sum(-1))
            else:
                t_loss = t_partition - g_unary.sum(-1)
        else:  # use marginals to fake the loss!
            with BK.no_grad_env():
                mg_unary, mg_binary = self.out.get_marginals(t_unary, m_binary, t_mask)
            t_loss = ((mg_unary - t_gold).detach() * t_unary).sum(-1) * t_weight  # fake a loss!
            if m_binary is not None:
                # note: no applying mask here since they should ALL come from "get_marginals"!
                _grad = (self._sum_until(mg_binary, 2, 0) - self._sum_until(t_gold_m, 2, 0)).detach()
                t_loss = t_loss.sum() + (_grad * m_binary).sum()  # note: no weighting here!
        ret = (t_loss.sum(), t_weight.sum())  # (loss-sum, div-count)
        return ret

    def forward_pred(self, t_unary: BK.Expr, m_binary: BK.Expr, t_mask: BK.Expr):
        ret = self.out.get_argmax(t_unary, m_binary, t_mask)
        return ret  # [*, L]
