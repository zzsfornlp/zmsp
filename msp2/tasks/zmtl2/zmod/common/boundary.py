#

# expander / boundary pointer

__all__ = [
    "ZBoundaryPointerConf", "ZBoundaryPointer",
]

from typing import List
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.utils import Constants

# --

class ZBoundaryPointerConf(BasicConf):
    def __init__(self):
        super().__init__()
        self._isize = -1  # main input size
        # --
        # extra embeddings
        self.num_indicators = 2  # evt, ef, ...
        # final encoder
        self.bp_enc = TransformerConf().direct_update(n_layers=1)  # extra layer
        self.bp_enc.aconf.clip_dist = 16  # use relative posi!
        # --

@node_reg(ZBoundaryPointerConf)
class ZBoundaryPointer(BasicNode):
    def __init__(self, conf: ZBoundaryPointerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZBoundaryPointerConf = self.conf
        _isize = conf._isize
        # --
        self.indicator_embeds = []
        for i in range(conf.num_indicators):
            _node = EmbeddingNode(None, osize=_isize, n_words=2)
            self.add_module(f"_MInd{i}", _node)  # add parameters
            self.indicator_embeds.append(_node)
        self.bp_enc = TransformerNode(conf.bp_enc, isize=_isize, osize=_isize)
        self.s_left = AffineNode(None, isize=_isize, osize=1, no_drop=True)
        self.s_right = AffineNode(None, isize=_isize, osize=1, no_drop=True)
        # --

    # [*, len, D], [*, len], List[*, len]
    def score(self, hid_t: BK.Expr, mask_t: BK.Expr, indicators_t: List[BK.Expr]):
        # first add indicators
        for ii, ind in enumerate(indicators_t):
            extra_t = self.indicator_embeds[ii](ind)  # [*, len, D]
            hid_t += extra_t
        # encoding
        enc_t = self.bp_enc.forward(hid_t, mask_expr=mask_t)
        # scoring
        invalid_penalty = (1.-mask_t) * Constants.REAL_PRAC_MIN  # [*, len]
        score_left = self.s_left(enc_t).squeeze(-1) + invalid_penalty
        score_right = self.s_right(enc_t).squeeze(-1) + invalid_penalty
        return score_left, score_right  # [*, len]

    # [*, len, D], [*, len], List[*, len], [*, 2]
    def gather_losses(self, hid_t: BK.Expr, mask_t: BK.Expr, indicators_t: List[BK.Expr], boundaries_t: BK.Expr):
        score_left, score_right = self.score(hid_t, mask_t, indicators_t)  # [*, len]
        idx_left, idx_right = BK.split(boundaries_t, 1, dim=-1)  # [*, 1]
        loss_left_t, loss_right_t = \
            BK.loss_nll(score_left, idx_left.squeeze(-1)), BK.loss_nll(score_right, idx_right.squeeze(-1))  # [*]
        ret = (loss_left_t+loss_right_t) / 2  # simply average these two
        return ret

    # [*, len, D], [*, len], List[*, len]
    def decode(self, hid_t: BK.Expr, mask_t: BK.Expr, indicators_t: List[BK.Expr]):
        score_left, score_right = self.score(hid_t, mask_t, indicators_t)  # [*, len]
        return ZBoundaryPointer.decode_with_scores(score_left, score_right)  # [*]

    # [*, len], [*, len] -> 3x[*]
    @staticmethod
    def decode_with_scores(left_scores: BK.Expr, right_scores: BK.Expr, normalize=True):
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

    # helper: prepare indicators (2d)
    def prepare_indicators(self, flat_idxes: List, shape):
        bs, dlen = shape
        _arange_t = BK.arange_idx(bs)  # [*]
        rets = []
        for one_idxes in flat_idxes:
            one_indicator = BK.constants_idx(shape, 0)  # [*, dlen]
            one_indicator[_arange_t, one_idxes] = 1
            rets.append(one_indicator)
        return rets
