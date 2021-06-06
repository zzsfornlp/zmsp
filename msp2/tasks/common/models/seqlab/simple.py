#

# simple labeling (scoring)
# -> corresponding to simple_vocab

__all__ = [
    "SimpleLabelerConf", "SimpleLabelerNode",
]

from msp2.data.vocab import *
from msp2.nn import BK
from msp2.nn.layers import *

class SimpleLabelerConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize = -1  # main input size
        self.psize = -1  # other psize, used for pairwise scoring mode
        # no need to specify output size since it will be read from vocab
        # --
        # scorer
        self.sconf = MyScorerConf()
        # special
        self.fix_non = False  # fix idx-zero's score
        self.fix_non_score = 0.  # the score if fix_zero
        # extra for predicting
        self.pred_addition_non_score = 0.  # score addition for zero when pred
        # embeddings
        self.e_tie_weights = False  # tie embeddings to scorer's output W (also inference e_dim)
        self.e_dim = 300
        # --
        # local normalization
        self.local_normalize = True  # whether (by default) do local normalization for score
        self.label_smoothing = 0.

@node_reg(SimpleLabelerConf)
class SimpleLabelerNode(BasicNode):
    def __init__(self, vocab: SimpleVocab, conf: SimpleLabelerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: SimpleLabelerConf = self.conf
        # --
        # set up with vocab
        assert vocab.get('O')==0 or vocab.non==0, "Here we simply assume idx=0 is NON/'O'"
        self.vocab = vocab
        self.output_dim = len(self.vocab)  # output the length
        # --
        # scorer
        self.scorer = MyScorerNode(conf.sconf, isize=conf.isize, psize=conf.psize, osize=self.output_dim)
        # embeddings
        if conf.e_tie_weights:
            assert not self.scorer.is_pairwise
            # stole last W out
            last_affine = self.scorer.scorer.mlp.nodes[-1]
            last_affine_ws = last_affine.get_ws()
            assert len(last_affine_ws)==1
            tmp_W = last_affine_ws[0]  # [dim, number]
            e_num, e_dim = BK.get_shape(tmp_W)
            assert e_num == self.output_dim
            conf.e_dim = e_dim  # directly overwrite!!
            self.W_getf = lambda: tmp_W  # [number, dim]
        else:
            self.W = BK.new_param([self.output_dim, conf.e_dim])  # [number, dim]
            self.reset_parameters()
            self.W_getf = lambda: self.W
        # --
        self.embed_dim = conf.e_dim
        # temp values [L]
        self._tmp_mask = BK.zeros([self.output_dim])  # [L]
        self._tmp_mask[0] = 1.
        self._tmp_mask_minus = 1. - self._tmp_mask

    def reset_parameters(self):
        if not self.conf.e_tie_weights:
            BK.init_param(self.W, "glorot", lookup=True)

    def extra_repr(self) -> str:
        conf: SimpleLabelerConf = self.conf
        return f"SimpleLabeler({conf.isize},{conf.psize}->{self.output_dim})"

    @property
    def special_mask_non1(self):  # [1, 0,...,0]
        return self._tmp_mask

    @property
    def speical_mask_non0(self):  # [0, 1,...,1]
        return self._tmp_mask_minus

    # =====
    # as decoder

    def output_score(self, score: BK.Expr, local_normalize: bool):
        if local_normalize is None:
            local_normalize = self.conf.local_normalize
        if local_normalize:
            score = score.log_softmax(-1)
        return score

    # [*, Dm], [*, Dp], [*], [*, L] -> [*, L]
    def score(self, input_main: BK.Expr, input_pair: BK.Expr, input_mask: BK.Expr,
              extra_score: BK.Expr = None, local_normalize: bool = None):
        conf: SimpleLabelerConf = self.conf
        scores = self.scorer(input_main, input_pair, input_mask)  # [*, L]
        # --
        # special cases
        if extra_score is not None:  # extra score from outside
            scores = scores + extra_score
        if conf.fix_non:  # fix_non(idx=0) scores
            scores = self._tmp_mask * conf.fix_non_score + self._tmp_mask_minus * scores  # [*, L]
        # todo(+N): only when predicting (not training)!!
        if not self.is_training() and conf.pred_addition_non_score != 0.:
            scores = scores + self._tmp_mask * conf.pred_addition_non_score
        # --
        # local normalization?
        scores = self.output_score(scores, local_normalize)
        return scores

    # =====
    # loss and pred

    # [*, Dm], [*, Dp], [*], [*, L] ;; [*], [*]
    def loss(self, input_main: BK.Expr, input_pair: BK.Expr, input_mask: BK.Expr, gold_idxes: BK.Expr,
             loss_weight_expr: BK.Expr = None, extra_score: BK.Expr=None):
        # not normalize here!
        scores_t = self.score(input_main, input_pair, input_mask, local_normalize=False, extra_score=extra_score)  # [*, L]
        # negative log likelihood
        # all_losses_t = - scores_t.gather(-1, gold_idxes.unsqueeze(-1)).squeeze(-1) * input_mask  # [*]
        all_losses_t = BK.loss_nll(scores_t, gold_idxes, label_smoothing=self.conf.label_smoothing)  # [*]
        all_losses_t *= input_mask
        if loss_weight_expr is not None:
            all_losses_t *= loss_weight_expr  # [*]
        ret_loss = all_losses_t.sum()  # []
        ret_div = input_mask.sum()
        return (ret_loss, ret_div)

    # [*, Dm], [*, Dp], [*]
    def predict(self, input_main: BK.Expr, input_pair: BK.Expr, input_mask: BK.Expr, extra_score: BK.Expr=None):
        scores_t = self.score(input_main, input_pair, input_mask, extra_score=extra_score)  # [*, L]
        best_scores, best_labs = scores_t.max(-1)  # [*]
        return (best_labs, best_scores)  # [*]

    # =====
    # as encoder

    # [*]
    def lookup(self, idxes: BK.Expr):
        W = self.W_getf()  # [L, D]
        return BK.lookup(W, idxes)

    # [*, L]
    def lookup_soft(self, scores: BK.Expr, use_softmax: bool):
        W = self.W_getf()  # [L, D]
        if use_softmax:
            probs = BK.softmax(scores, -1)  # [*, L]
        else:
            probs = scores  # no need
        return BK.matmul(probs, W)  # [*, D]

    @property
    def lookup_dim(self):
        return self.embed_dim
