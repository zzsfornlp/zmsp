#

# the labeling scoring node
# note: assume idx=0 means NIL!!

__all__ = [
    "ZlabelNode", "ZLabelConf",
]

from typing import List, Tuple
import numpy as np
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.utils import ConfEntryChoices
from .crf import *

# --

class ZLabelConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self._csize = -1  # number of classes (final output labels)
        # todo(+N): not good design, but use this for the indirection ...
        self.emb_size = -1  # if >0, we further stack an affine layer for final csize, otherwise directly take inputs!
        self.input_act = 'linear'  # activation for the input
        # fixed_nil
        self.fixed_nil_val: float = None  # fix NIL as what?
        # loss
        self.loss_do_sel = False  # sel then loss maybe mainly to save some space
        self.loss_neg_sample = 1.  # how much of neg(NIL) to include in loss, >0 means percentage, <0 means ratio
        self.loss_full_alpha = 1.  # ordinarily against ALL
        # extra binary mode (must set fixed_nil_val=True)
        self.use_nil_as_binary = False  # use NIL score as binary score
        self.loss_binary_alpha = 0.  # gold against NIL
        # extra all-binary mode (must set nil=0)
        self.loss_allbinary_alpha = 0.
        # special mode: CRF!
        self.crf = ConfEntryChoices({'yes': ZLinearCrfConf(), 'no': None}, 'no')
        self.pred_crf_add = 100.  # todo(+N): simply increase unary scores after decoding
        # --

    @classmethod
    def _get_type_hints(cls):
        return {"fixed_nil_val": float}

@node_reg(ZLabelConf)
class ZlabelNode(BasicNode):
    def __init__(self, conf: ZLabelConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZLabelConf = self.conf
        # --
        _csize = conf._csize
        # final affine layer?
        self.input_act = ActivationHelper.get_act(conf.input_act)
        if conf.emb_size > 0:
            self.aff_final = AffineNode(None, isize=conf.emb_size, osize=_csize, no_drop=True)
        else:
            self.aff_final = None
        # fixed_nil_mask?
        self.fixed_nil_mask, self.fixed_nil_val = None, None
        if conf.fixed_nil_val is not None:
            self.fixed_nil_val = float(conf.fixed_nil_val)
            self.fixed_nil_mask = BK.input_real([0.] + [1.] * (_csize-1))
        # binary mode
        if conf.use_nil_as_binary or conf.loss_binary_alpha!=0:
            assert conf.use_nil_as_binary and conf.loss_binary_alpha!=0 and self.fixed_nil_mask is not None
        if conf.loss_allbinary_alpha != 0.:
            assert conf.fixed_nil_val == 0.
        # crf mode?
        self.crf = None
        if conf.crf is not None:
            assert not (conf.use_nil_as_binary or conf.loss_binary_alpha!=0)  # not binary mode!
            self.crf = ZLinearCrfNode(conf.crf, _csize=_csize)
        # --

    # get csize for the idec node!
    def get_core_csize(self):
        conf: ZLabelConf = self.conf
        return conf.emb_size if conf.emb_size>0 else conf._csize

    # [bs, ..., D], *[bs, ...],
    def gather_losses(self, scores: List[BK.Expr], label_t: BK.Expr, valid_t: BK.Expr, loss_neg_sample: float = None):
        conf: ZLabelConf = self.conf
        _loss_do_sel = conf.loss_do_sel
        _alpha_binary, _alpha_full = conf.loss_binary_alpha, conf.loss_full_alpha
        _alpha_all_binary = conf.loss_allbinary_alpha
        # --
        if self.crf is not None:  # CRF mode!
            assert _alpha_binary <= 0. and _alpha_all_binary <= 0.
            # reshape them into 3d
            valid_premask = (valid_t.sum(-1) > 0.)  # [bs, ...]
            # note: simply collect them all
            rets = []
            _pm_mask, _pm_label = valid_t[valid_premask], label_t[valid_premask]  # [??, slen]
            for score_t in scores:
                _one_pm_score = score_t[valid_premask]  # [??, slen, D]
                _one_fscore_t, _ = self._get_score(_one_pm_score)  # [??, slen, L]
                # --
                # todo(+N): hacky fix, make it a leading NIL
                _pm_mask2 = _pm_mask.clone()
                _pm_mask2[:,0] = 1.
                # --
                _one_loss, _one_count = self.crf.loss(_one_fscore_t, _pm_mask2, _pm_label)  # ??
                rets.append((_one_loss*_alpha_full, _one_count))
        else:
            pos_t = (label_t>0).float()  # 0 as NIL!!
            loss_mask_t = self._get_loss_mask(pos_t, valid_t, loss_neg_sample=loss_neg_sample)  # [bs, ...]
            if _loss_do_sel:
                _sel_mask = (loss_mask_t > 0.)  # [bs, ...]
                _sel_label = label_t[_sel_mask]  # [??]
                _sel_mask2 = BK.constants([len(_sel_label)], 1.)  # [??]
            # note: simply collect them all
            rets = []
            for score_t in scores:
                if _loss_do_sel:  # [??, ]
                    one_score_t, one_mask_t, one_label_t = score_t[_sel_mask], _sel_mask2, _sel_label
                else:  # [bs, ..., D]
                    one_score_t, one_mask_t, one_label_t = score_t, loss_mask_t, label_t
                one_fscore_t, one_nilscore_t = self._get_score(one_score_t)
                # full loss
                one_loss_t = BK.loss_nll(one_fscore_t, one_label_t) * _alpha_full  # [????]
                # binary loss
                if _alpha_binary > 0.:  # plus ...
                    _binary_loss = BK.loss_binary(one_nilscore_t.squeeze(-1), (one_label_t>0).float()) * _alpha_binary  # [???]
                    one_loss_t = one_loss_t + _binary_loss
                # all binary
                if _alpha_all_binary > 0.:  # plus ...
                    _tmp_label_t = BK.zeros(BK.get_shape(one_fscore_t))  # [???, L]
                    _tmp_label_t.scatter_(-1, one_label_t.unsqueeze(-1), 1.)
                    _ab_loss = BK.loss_binary(one_fscore_t, _tmp_label_t) * _alpha_all_binary  # [???, L]
                    one_loss_t = one_loss_t + _ab_loss[...,1:].sum(-1)
                # --
                one_loss_t = one_loss_t * one_mask_t
                rets.append((one_loss_t, one_mask_t))  # tuple(loss, mask)
        return rets

    # note: actually this is for decoding!!
    # [bs, ..., D], [bs, ...],
    def score_labels(self, scores: List[BK.Expr], seq_mask_t: BK.Expr = None,
                     premask_t: BK.Expr = None, preidx_t: Tuple[BK.Expr] = None, nil_add_score: float = None):
        # note: currently only pick the last one
        score_t = scores[-1]
        if premask_t is None and preidx_t is None:  # if no premask provided, simply score them all
            ret_t, _ = self._get_score(score_t, nil_add_score=nil_add_score)  # [bs, ..., L]
        else:
            if premask_t is None:
                sel_score_t = score_t[preidx_t]  # [??, ..., D]
            else:
                sel_score_t = score_t[premask_t>0.]  # [??, ..., D]
            ret_t, _ = self._get_score(sel_score_t, nil_add_score=nil_add_score)  # [??, ..., L]
        # handle searching with bigram (crf mode!)
        if self.crf is not None:
            if premask_t is None and preidx_t is None:
                _seq_mask_t = seq_mask_t
            else:
                if premask_t is None:
                    _seq_mask_t = seq_mask_t[preidx_t]  # [??, ..., D]
                else:
                    _seq_mask_t = seq_mask_t[premask_t>0.]  # [??, ..., D]
            # --
            # todo(+N): hacky fix, make it a leading NIL
            _seq_mask_t2 = _seq_mask_t.clone()
            _seq_mask_t2[:,0] = 1.
            # --
            assert len(BK.get_shape(ret_t)) == 3  # for simplicity, only use this in 3d: [*, slen, L]
            best_labs, _ = self.crf.predict(ret_t, _seq_mask_t2)  # [*, slen]
            _add_t = BK.zeros(BK.get_shape(ret_t))
            _add_t.scatter_(-1, best_labs.unsqueeze(-1), self.conf.pred_crf_add)  # [*, slen, L]
            ret_t = ret_t + _add_t  # [*, slen, L]
        # --
        return ret_t

    # --
    def _get_core_score(self, expr_t: BK.Expr, nil_add_score: float = None):
        # aff?
        act_t = self.input_act(expr_t)
        if self.aff_final is not None:
            score_t = self.aff_final(act_t)  # [*, ..., L]
        else:
            score_t = act_t
        return score_t

    def _get_score(self, expr_t: BK.Expr, nil_add_score: float = None):
        score_t = self._get_core_score(expr_t)
        # get original NIL score
        orig_nil_t = score_t.narrow(-1, 0, 1)  # [*, ..., 1]
        # fixed_val for nil
        _fixed_nil_mask = self.fixed_nil_mask
        if _fixed_nil_mask is not None:  # fixed val at idx0
            _v0 = self.fixed_nil_val
            if nil_add_score is not None:
                _v0 += nil_add_score
            _fixed_nil_mask_N = (1. - _fixed_nil_mask)
            score_t = score_t * _fixed_nil_mask + _v0 * _fixed_nil_mask_N
            if self.conf.use_nil_as_binary:  # todo(+N): should we detach here? currently nope!
                score_t = score_t - orig_nil_t * _fixed_nil_mask_N  # minus NIL with binary-score!
        else:
            assert nil_add_score is None, "Not supported when no nil-mask"
        # --
        return score_t, orig_nil_t  # [*, ..., L], [*, ..., 1]

    def _get_loss_mask(self, pos_t: BK.Expr, valid_t: BK.Expr, loss_neg_sample: float = None):
        conf: ZLabelConf = self.conf
        # use default config if not from outside!
        _loss_neg_sample = conf.loss_neg_sample if loss_neg_sample is None else loss_neg_sample
        # --
        if _loss_neg_sample >= 1.:  # all valid is ok!
            return valid_t
        # --
        pos_t = pos_t * valid_t  # should also filter pos here!
        # first get sample rate
        if _loss_neg_sample >= 0.:  # percentage to valid
            _rate = _loss_neg_sample  # directly it!!
        else:  # ratio to pos
            _count_pos = pos_t.sum()
            _count_valid = valid_t.sum()
            _rate = (-_loss_neg_sample) * ((_count_pos+1) / (_count_valid-_count_pos+1))  # add-1 to make it >0
        # random select!
        ret_t = (BK.rand(valid_t.shape) <= _rate).float() * valid_t
        ret_t += pos_t  # also include pos ones!
        ret_t.clamp_(max=1.)
        return ret_t

# --
# b msp2/tasks/zmtl2/zmod/common/lab:??
