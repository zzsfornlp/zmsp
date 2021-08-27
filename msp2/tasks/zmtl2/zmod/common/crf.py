#

# a component for a final (linear-)CRF layer

__all__ = [
    'ZLinearCrfNode', 'ZLinearCrfConf',
]

from typing import List
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.utils import Constants
from msp2.tasks.common.models.seqlab import BigramConf, BigramNode, BigramInferenceHelper

class ZLinearCrfConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self._csize = -1  # num of classes
        self.crf_beam = 0  # <=0 means all!
        self.loss_by_tok = True  # divide by tok or sent
        self.bigram_conf = BigramConf()

@node_reg(ZLinearCrfConf)
class ZLinearCrfNode(BasicNode):
    def __init__(self, conf: ZLinearCrfConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZLinearCrfConf = self.conf
        # --
        self.bigram = BigramNode(conf.bigram_conf, osize=conf._csize)

    # [*, slen, L], [*, slen], [*, slen]
    # todo(+N): currently cannot handle well (?) non-ending masks!
    def loss(self, unary_scores: BK.Expr, input_mask: BK.Expr, gold_idxes: BK.Expr):
        mat_t = self.bigram.get_matrix()  # [L, L]
        if BK.is_zero_shape(unary_scores):  # note: avoid empty
            potential_t = BK.zeros(BK.get_shape(unary_scores)[:-2])  # [*]
        else:
            potential_t = BigramInferenceHelper.inference_forward(unary_scores, mat_t, input_mask, self.conf.crf_beam)  # [*]
        gold_single_scores_t = unary_scores.gather(-1, gold_idxes.unsqueeze(-1)).squeeze(-1) * input_mask  # [*, slen]
        gold_bigram_scores_t = mat_t[gold_idxes[:, :-1], gold_idxes[:, 1:]] * input_mask[:, 1:]  # [*, slen-1]
        all_losses_t = (potential_t - (gold_single_scores_t.sum(-1) + gold_bigram_scores_t.sum(-1)))  # [*]
        if self.conf.loss_by_tok:
            ret_count = input_mask.sum()  # []
        else:
            ret_count = (input_mask.sum(-1)>0).float()  # [*]
        return all_losses_t, ret_count

    # [*, slen, L], [*, slen]
    def predict(self, unary_scores: BK.Expr, input_mask: BK.Expr):
        mat_t = self.bigram.get_matrix()  # [L, L]
        best_labs, best_scores = BigramInferenceHelper.inference_search(
            unary_scores, mat_t, input_mask, self.conf.crf_beam)  # [*, slen]
        return best_labs, best_scores

# --
# b msp2/tasks/zmtl2/zmod/common/crf:?
