#

# span expander

__all__ = [
    "SpanExpanderConf", "SpanExpanderNode",
]

from msp2.nn import BK
from msp2.nn.layers import *
from msp2.utils import Constants

class SpanExpanderConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize = -1  # main input size
        self.psize = -1  # other psize, used for pairwise scoring mode
        # --
        self.sconf = MyScorerConf()
        self.sconf.pas_conf.direct_update(use_biaffine=False, use_ff2=True)  # no need for biaffine for these

@node_reg(SpanExpanderConf)
class SpanExpanderNode(BasicNode):
    def __init__(self, conf: SpanExpanderConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: SpanExpanderConf = self.conf
        # --
        # boundary pointers
        self.s_left = MyScorerNode(conf.sconf, isize=conf.isize, psize=conf.psize, osize=1)
        self.s_right = MyScorerNode(conf.sconf, isize=conf.isize, psize=conf.psize, osize=1)

    # [*, slen, Dm], [*, Dp], [*, slen]; [*], [*]
    # outputs should: left_pred<=left_constraints and right_pred>=right_constraints
    def score(self, input_main: BK.Expr, input_pair: BK.Expr, input_mask: BK.Expr,
              left_constraints: BK.Expr = None, right_constraints: BK.Expr = None):
        conf: SpanExpanderConf = self.conf
        # --
        # left & right
        rets = []
        seq_shape = BK.get_shape(input_mask)
        cur_mask = input_mask
        arange_t = BK.arange_idx(seq_shape[-1]).view([1]*(len(seq_shape)-1) + [-1])  # [*, slen]
        for scorer, cons_t in zip([self.s_left, self.s_right], [left_constraints, right_constraints]):
            mm = cur_mask if cons_t is None else (cur_mask * (arange_t<=cons_t).float())  # [*, slen]
            ss = scorer(input_main, None if input_pair is None else input_pair.unsqueeze(-2), mm).squeeze(-1)  # [*, slen]
            rets.append(ss)
        return rets[0], rets[1]  # [*, slen] (already masked)
    # --

    # [*, slen], [*, slen]
    @staticmethod
    def decode_with_scores(left_scores: BK.Expr, right_scores: BK.Expr, normalize: bool):
        if normalize:
            left_scores = BK.log_softmax(left_scores, -1)
            right_scores = BK.log_softmax(right_scores, -1)
        # pairwise adding
        score_shape = BK.get_shape(left_scores)
        pair_scores = left_scores.unsqueeze(-1) + right_scores.unsqueeze(-2)  # [*, slen_L, slen_R]
        flt_pair_scores = pair_scores.view(score_shape[:-1] + [-1])  # [*, slen*slen]
        # LR mask
        slen = score_shape[-1]
        arange_t = BK.arange_idx(slen)
        lr_mask = (arange_t.unsqueeze(-1)<=arange_t.unsqueeze(-2)).float().view(-1)  # [slen_L*slen_R]
        max_scores, max_idxes = (flt_pair_scores + (1.-lr_mask)*Constants.REAL_PRAC_MIN).max(-1)  # [*]
        left_idxes, right_idxes = max_idxes // slen, max_idxes % slen  # [*]
        return max_scores, left_idxes, right_idxes
