#

# seq-labeling styled decoder

__all__ = [
    "SeqLabelerConf", "SeqLabelerNode", "SeqLabelerInferencer",
]

from typing import List
import numpy as np
from msp2.data.vocab import *
from msp2.utils import zlog, ZObject, Constants, default_pickle_serializer
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.nn.modules import PlainDecoderConf, PlainDecoder
from .simple import *
from .bigram import *
from .inference import *

# -----
# simple sequence labeler:
"""
# current supported modes, roughly three: {local_simple, local_seqdec, crf}
1. local model -> local_normalize=T
1.1. simple one -> use_bigram=F,use_seqdec=F
1.2. only bigram -> use_bigram=T,use_seqdec=F (todo(+W))
1.3. more -> others
2. global_model with CRF -> use_bigram=T,use_seqdec=F,local_normalize=F,loss_no_bigram=False,loss_mode=crf,beam_k=?
"""

class SeqLabelerConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize = -1  # main input size
        self.psize = -1  # other psize, used for pairwise scoring mode
        # --
        # 0. first transformations for the inputs
        self.main_mlp = MLPConf().direct_update(use_out=False, dim_hid=512, n_hid_layer=0)  # input_main(isize) -> ?
        self.pair_mlp = MLPConf().direct_update(use_out=False, dim_hid=512, n_hid_layer=0)  # input_pair(psize) -> ?
        # 1. labeler: is_pairwise?
        self.labeler_conf = SimpleLabelerConf()
        # 2. decoder: whether using seq-decoder?
        self.use_seqdec = False  # use seqdec?
        self.seqdec_conf = PlainDecoderConf()
        self.seqdec_conf.dec_rnn.n_layers = 1  # by default one layer of rnn
        # init state for decoder: aff(pair) if pairwise else aff(main)
        self.sd_init_aff = AffineConf().direct_update(out_act='elu')  # init starting hidden
        self.sd_init_pool = "max"  # if no pair, then use pool(main) for init's input
        self.sd_input_aff = AffineConf().direct_update(out_act='elu')  # input to decoder
        self.sd_output_aff = AffineConf().direct_update(out_act='elu')  # output from decoder + cur to labeler
        self.sd_skip_non = True  # ignore non labels (idx=0)
        # 3. decoder: whether using bigram transition matrix; note: no score for the starting one or extra eos
        self.use_bigram = False  # use bigram transition scores?
        self.bigram_conf = BigramConf()
        # --
        # loss
        self.loss_mode = "mle"  # mle: max-likelihood; crf: crf
        self.loss_by_tok = True  # divide by tok or sent
        self.loss_by_tok_weighted = True  # further use weight in div if provided
        # search
        self.beam_k = -1  # beam_k for beam_search and inference_search
        self.pred_use_seq_cons = False  # only for mle mode!
        self.pred_use_seq_cons_from_file = ""  # if not null, use loaded weights!
        self.pred_use_seq_cons_alpha = 1.  # multiplier!!
        # --

@node_reg(SeqLabelerConf)
class SeqLabelerNode(BasicNode):
    def __init__(self, vocab: SimpleVocab, conf: SeqLabelerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: SeqLabelerConf = self.conf
        is_pairwise = (conf.psize > 0)
        self.is_pairwise = is_pairwise
        # --
        # 0. pre mlp
        isize, psize = conf.isize, conf.psize
        self.main_mlp = MLPNode(conf.main_mlp, isize=isize, osize=-1, use_out=False)
        isize = self.main_mlp.get_output_dims()[0]
        if is_pairwise:
            self.pair_mlp = MLPNode(conf.pair_mlp, isize=psize, osize=-1, use_out=False)
            psize = self.pair_mlp.get_output_dims()[0]
        else:
            self.pair_mlp = lambda x: x
        # 1/2. decoder & laber
        if conf.use_seqdec:
            # extra for seq-decoder
            dec_hid = conf.seqdec_conf.dec_hidden
            # setup labeler to get embedding dim
            self.laber = SimpleLabelerNode(vocab, conf.labeler_conf, isize=dec_hid, psize=psize)
            laber_embed_dim = self.laber.lookup_dim
            # init starting hidden; note: choose different according to 'is_pairwise'
            self.sd_init_aff = AffineNode(conf.sd_init_aff, isize=(psize if is_pairwise else isize), osize=dec_hid)
            self.sd_init_pool_f = ActivationHelper.get_pool(conf.sd_init_pool)
            # sd input: one_repr + one_idx_embed
            self.sd_input_aff = AffineNode(conf.sd_init_aff, isize=[isize, laber_embed_dim], osize=dec_hid)
            # sd output: cur_expr + hidden
            self.sd_output_aff = AffineNode(conf.sd_output_aff, isize=[isize, dec_hid], osize=dec_hid)
            # sd itself
            self.seqdec = PlainDecoder(conf.seqdec_conf, input_dim=dec_hid)
        else:
            # directly using the scorer (overwrite some values)
            self.laber = SimpleLabelerNode(vocab, conf.labeler_conf, isize=isize, psize=psize)
        # 3. bigram
        # todo(note): bigram does not consider skip_non
        if conf.use_bigram:
            self.bigram = BigramNode(conf.bigram_conf, osize=self.laber.output_dim)
        else:
            self.bigram = None
        # special decoding
        if conf.pred_use_seq_cons_from_file:
            assert not conf.pred_use_seq_cons
            _m = default_pickle_serializer.from_file(conf.pred_use_seq_cons_from_file)
            zlog(f"Load weights from {conf.pred_use_seq_cons_from_file}")
            self.pred_cons_mat = BK.input_real(_m)
        elif conf.pred_use_seq_cons:
            _m = vocab.get_allowed_transitions()
            self.pred_cons_mat = (1.-BK.input_real(_m)) * Constants.REAL_PRAC_MIN
        else:
            self.pred_cons_mat = None
        # =====
        # loss
        self.loss_mle, self.loss_crf = [conf.loss_mode==z for z in ["mle", "crf"]]
        if self.loss_mle:
            if conf.use_seqdec or conf.use_bigram:
                zlog("Setup SeqLabelerNode with Local complex mode!")
            else:
                zlog("Setup SeqLabelerNode with Local simple mode!")
        elif self.loss_crf:
            assert conf.use_bigram and (not conf.use_seqdec), "Wrong mode for crf"
            zlog("Setup SeqLabelerNode with CRF mode!")
        else:
            raise NotImplementedError(f"UNK loss mode: {conf.loss_mode}")

    def extra_repr(self) -> str:
        conf: SeqLabelerConf = self.conf
        return f"SeqLabeler({conf.isize},{conf.psize}=>{self.laber.extra_repr()},DEC={conf.use_seqdec},BIGRAM={conf.use_bigram})"

    # =====
    # helpers

    # Dm, Dp -> Dm', Dp'
    def transform_expr(self, input_main: BK.Expr, input_pair: BK.Expr):
        expr_main = self.main_mlp(input_main)  # [*, slen, Dm']
        expr_pair = self.pair_mlp(input_pair)  # [*, Dp'] or None
        return expr_main, expr_pair

    def prepare_sd_init(self, expr_main: BK.Expr, expr_pair: BK.Expr):
        if self.is_pairwise:
            sd_init_t = self.sd_init_aff(expr_pair)  # [*, hid]
        else:
            if BK.is_zero_shape(expr_main):
                sd_init_t0 = expr_main.sum(-2)  # simply make the shape!
            else:
                sd_init_t0 = self.sd_init_pool_f(expr_main, -2)  # pooling at -2: [*, Dm']
            sd_init_t = self.sd_init_aff(sd_init_t0)  # [*, hid]
        return sd_init_t

    # =====
    # scoring all at once, teacher-forcing mode if use_seqdec
    # [*, slen, Dm'], [*, Dp'], [*, slen], ..., [*, slen, L] -> [*, slen]
    def score_all(self, expr_main: BK.Expr, expr_pair: BK.Expr, input_mask: BK.Expr, gold_idxes: BK.Expr,
                  local_normalize: bool = None, use_bigram: bool = True, extra_score: BK.Expr=None):
        conf: SeqLabelerConf = self.conf
        # first collect basic scores
        if conf.use_seqdec:
            # first prepare init hidden
            sd_init_t = self.prepare_sd_init(expr_main, expr_pair)  # [*, hid]
            # init cache: no mask at batch level
            sd_cache = self.seqdec.go_init(sd_init_t, init_mask=None)  # and no need to cum_state here!
            # prepare inputs at once
            if conf.sd_skip_non:
                gold_valid_mask = (gold_idxes>0).float() * input_mask  # [*, slen], todo(note): fix 0 as non here!
                gv_idxes, gv_masks = BK.mask2idx(gold_valid_mask)  # [*, ?]
                bsize = BK.get_shape(gold_idxes, 0)
                arange_t = BK.arange_idx(bsize).unsqueeze(-1)  # [*, 1]
                # select and forward
                gv_embeds = self.laber.lookup(gold_idxes[arange_t, gv_idxes])  # [*, ?, E]
                gv_input_t = self.sd_input_aff([expr_main[arange_t, gv_idxes], gv_embeds])  # [*, ?, hid]
                gv_hid_t = self.seqdec.go_feed(sd_cache, gv_input_t, gv_masks)  # [*, ?, hid]
                # select back and output_aff
                aug_hid_t = BK.concat([sd_init_t.unsqueeze(-2), gv_hid_t], -2)  # [*, 1+?, hid]
                sel_t = BK.pad(gold_valid_mask[:,:-1].cumsum(-1), (1,0), value=0.).long()  # [*, 1+(slen-1)]
                shifted_hid_t = aug_hid_t[arange_t, sel_t]  # [*, slen, hid]
            else:
                gold_idx_embeds = self.laber.lookup(gold_idxes)  # [*, slen, E]
                all_input_t = self.sd_input_aff([expr_main, gold_idx_embeds])  # inputs to dec, [*, slen, hid]
                all_hid_t = self.seqdec.go_feed(sd_cache, all_input_t, input_mask)  # output-hids, [*, slen, hid]
                shifted_hid_t = BK.concat([sd_init_t.unsqueeze(-2), all_hid_t[:,:-1]], -2)  # [*, slen, hid]
            # scorer
            pre_labeler_t = self.sd_output_aff([expr_main, shifted_hid_t])  # [*, slen, hid]
        else:
            pre_labeler_t = expr_main  # [*, slen, Dm']
        # score with labeler (no norm here since we may need to add other scores)
        scores_t = self.laber.score(pre_labeler_t, None if expr_pair is None else expr_pair.unsqueeze(-2), input_mask,
                                    extra_score=extra_score, local_normalize=False)  # [*, slen, L]
        # bigram score addition
        if conf.use_bigram and use_bigram:
            bigram_scores_t = self.bigram.get_matrix()[gold_idxes[:, :-1]]  # [*, slen-1, L]
            score_shape = BK.get_shape(bigram_scores_t)
            score_shape[1] = 1
            slice_t = BK.constants(score_shape, 0.)  # fix 0., no transition from BOS (and EOS) for simplicity!
            bigram_scores_t = BK.concat([slice_t, bigram_scores_t], 1)  # [*, slen, L]
            scores_t += bigram_scores_t  # [*, slen]
        # local normalization?
        scores_t = self.laber.output_score(scores_t, local_normalize)
        return scores_t

    # =====
    # predict (by step)

    # for expander: slice_main [*, Dm'], slice_mask [*], ...
    def step_score(self, cache: DecCache, slice_main: BK.Expr, slice_mask: BK.Expr, expr_main: BK.Expr, expr_pair: BK.Expr,
                   local_normalize: bool = None, use_bigram: bool = True, extra_score: BK.Expr = None):
        conf: SeqLabelerConf = self.conf
        # --
        if conf.use_seqdec:
            if cache is None:  # the start
                sd_init_t = self.prepare_sd_init(expr_main, expr_pair)  # [*, hid]
                # here we need to set "cum_state_layers" since there can be in-middle masks as predictions!
                cache = self.seqdec.go_init(sd_init_t, init_mask=None,
                                            cum_state_layers=((-1,) if conf.sd_skip_non else ()))  # init cache
                cache.last_idxes = None  # no init one
            # scorer
            hid_t = cache.get_last_state(-1)  # [*, hid]
            pre_labeler_t = self.sd_output_aff([slice_main, hid_t])  # [*, hid]
        else:
            if cache is None:
                cache = DecCache()  # only an object holder
                cache.last_idxes = None  # no init one
            pre_labeler_t = slice_main  # [*, 1, Dm']
        # score with labeler
        scores_t = self.laber.score(pre_labeler_t, expr_pair, slice_mask,
                                    extra_score=extra_score, local_normalize=False)  # [*, L]
        # bigram score addition
        if conf.use_bigram and use_bigram and cache.last_idxes is not None:
            bigram_scores = self.bigram.get_matrix()[cache.last_idxes]  # [*, L]
            scores_t += bigram_scores
        # local normalization?
        scores_t = self.laber.output_score(scores_t, local_normalize)
        return cache, scores_t

    # for ender: slice_main [*, Dm'], slice_mask [*], pred_idxes: [*]
    def step_end(self, cache: DecCache, slice_main: BK.Expr, slice_mask: BK.Expr, pred_idxes: BK.Expr):
        # we do possible decoder step here
        conf: SeqLabelerConf = self.conf
        # --
        if conf.use_seqdec:
            embed_t = self.laber.lookup(pred_idxes)  # [*, E]
            input_t = self.sd_input_aff([slice_main, embed_t])  # [*, hid]
            if conf.sd_skip_non:  # further mask, todo(note): fixed non as 0!
                slice_mask = slice_mask * (pred_idxes>0).float()
            hid_t = self.seqdec.go_feed(cache, input_t.unsqueeze(-2), slice_mask.unsqueeze(-1))  # [*, 1, hid]
        # add here for possible bigram usage
        cache.last_idxes = pred_idxes  # [*]
        return cache  # cache modified inplace

    # =====
    # loss and pred

    # [*, slen, Dm'], [*, Dp'], [*, slen], [*, slen] ;; [*, slen], [*, slen, L]
    def loss(self, input_main: BK.Expr, input_pair: BK.Expr, input_mask: BK.Expr, gold_idxes: BK.Expr,
             loss_weight_expr: BK.Expr = None, extra_score: BK.Expr=None):
        conf: SeqLabelerConf = self.conf
        # --
        expr_main, expr_pair = self.transform_expr(input_main, input_pair)
        if self.loss_mle:
            # simply collect them all (not normalize here!)
            all_scores_t = self.score_all(expr_main, expr_pair, input_mask, gold_idxes, local_normalize=False,
                                          extra_score=extra_score)  # [*, slen, L]
            # negative log likelihood; todo(+1): repeat log-softmax here
            # all_losses_t = - all_scores_t.gather(-1, gold_idxes.unsqueeze(-1)).squeeze(-1) * input_mask  # [*, slen]
            all_losses_t = BK.loss_nll(all_scores_t, gold_idxes,
                                       label_smoothing=self.conf.labeler_conf.label_smoothing)  # [*]
            all_losses_t *= input_mask
            if loss_weight_expr is not None:
                all_losses_t *= loss_weight_expr
            ret_loss = all_losses_t.sum()  # []
        elif self.loss_crf:
            # no normalization & no bigram
            single_scores_t = self.score_all(expr_main, expr_pair, input_mask, None,
                                             use_bigram=False, extra_score=extra_score)  # [*, slen, L]
            mat_t = self.bigram.get_matrix()  # [L, L]
            if BK.is_zero_shape(single_scores_t):  # note: avoid empty
                potential_t = BK.zeros(BK.get_shape(single_scores_t)[:-2])  # [*]
            else:
                potential_t = BigramInferenceHelper.inference_forward(single_scores_t, mat_t, input_mask, conf.beam_k)  # [*]
            gold_single_scores_t = single_scores_t.gather(-1, gold_idxes.unsqueeze(-1)).squeeze(-1) * input_mask  # [*, slen]
            gold_bigram_scores_t = mat_t[gold_idxes[:,:-1], gold_idxes[:,1:]] * input_mask[:,1:]  # [*, slen-1]
            all_losses_t = (potential_t - (gold_single_scores_t.sum(-1) + gold_bigram_scores_t.sum(-1)))  # [*]
            # todo(+N): also no label_smoothing for crf
            # todo(+N): for now, ignore loss_weight for crf mode!!
            # if loss_weight_expr is not None:
            #     assert BK.get_shape(loss_weight_expr, -1) == 1, "Currently CRF loss requires seq level loss_weight!!"
            #     all_losses_t *= loss_weight_expr
            ret_loss = all_losses_t.sum()  # []
        else:
            raise NotImplementedError()
        # ret_count
        if conf.loss_by_tok:  # sum all valid toks
            if conf.loss_by_tok_weighted and loss_weight_expr is not None:
                ret_count = (input_mask * loss_weight_expr).sum()
            else:
                ret_count = input_mask.sum()
        else:  # sum all valid batch items
            ret_count = input_mask.prod(-1).sum()
        return (ret_loss, ret_count)

    # [*, slen, Dm'], [*, Dp'], [*, slen], ..., [*, slen, L]
    def predict(self, input_main: BK.Expr, input_pair: BK.Expr, input_mask: BK.Expr, extra_score: BK.Expr=None):
        conf: SeqLabelerConf = self.conf
        # --
        expr_main, expr_pair = self.transform_expr(input_main, input_pair)
        if self.loss_mle:
            if conf.use_seqdec:  # need to use a searcher
                searcher = SeqLabelerInferencer(self, expr_main, expr_pair, input_mask, extra_score)
                best_labs, best_scores = searcher.beam_search(BK.get_shape(input_main, 0), conf.beam_k)  # [*, slen]
            # todo(+W): to incorporate 'pred_use_seq_cons'?
            else:
                all_scores_t = self.score_all(expr_main, expr_pair, input_mask, None,
                                              use_bigram=False, extra_score=extra_score)  # [*, slen, L]
                # --
                mat_t = self.bigram.get_matrix() if conf.use_bigram else None
                if self.pred_cons_mat is not None:
                    mat_t = self.pred_cons_mat if mat_t is None else (mat_t+self.pred_cons_mat)
                # --
                if mat_t is not None:
                    mat_t = mat_t * conf.pred_use_seq_cons_alpha
                    best_labs, best_scores = BigramInferenceHelper.inference_search(
                        all_scores_t, mat_t, input_mask, conf.beam_k)
                else:
                    best_scores, best_labs = all_scores_t.max(-1)  # [*, slen]
        elif self.loss_crf:
            # no normalization & no bigram
            single_scores_t = self.score_all(expr_main, expr_pair, input_mask, None,
                                             use_bigram=False, extra_score=extra_score)  # [*, slen, L]
            mat_t = self.bigram.get_matrix()  # [L, L]
            if conf.pred_use_seq_cons:  # further use constrain!!
                mat_t += self.pred_cons_mat
            best_labs, best_scores = BigramInferenceHelper.inference_search(
                single_scores_t, mat_t, input_mask, conf.beam_k)  # [*, slen]
        else:
            raise NotImplementedError()
        return (best_labs, best_scores)  # [*, slen]

# =====
# specific searcher

class SeqLabelerInferencer(SimpleInferencer):
    # Model, [*, slen, Dm], [*, Dp], [*, slen]
    def __init__(self, model: SeqLabelerNode, expr_main: BK.Expr, expr_pair: BK.Expr, input_mask: BK.Expr, extra_score: BK.Expr):
        self.model = model
        self.all_steps = BK.get_shape(expr_main, -2)  # slen
        # --
        # store them
        self.expr_main, self.expr_pair, self.input_mask, self.extra_score = expr_main, expr_pair, input_mask, extra_score
        # currently we only need repeat 1 & k
        self.contents: List[ZObject] = [None] * 1000  # this should be enough!

    def get_contents(self, repeat: int):
        ret = self.contents[repeat]
        if ret is None:
            ret = self._make_repeats(repeat)
            self.contents[repeat] = ret
        return ret

    def _make_repeats(self, repeat: int):
        assert repeat >= 1
        expr_main, expr_pair, input_mask, extra_score = self.expr_main, self.expr_pair, self.input_mask, self.extra_score
        if repeat != 1:  # need actual repeat: [*xR, ...]
            expr_main = BK.simple_repeat_interleave(expr_main, repeat, 0)
            expr_pair = None if expr_pair is None else BK.simple_repeat_interleave(expr_pair, repeat, 0)
            input_mask = BK.simple_repeat_interleave(input_mask, repeat, 0)
            extra_score = None if extra_score is None else BK.simple_repeat_interleave(extra_score, repeat, 0)
        z = ZObject()
        z.expr_main, z.expr_pair = expr_main, expr_pair  # [*, slen, Dm'], [*, Dp']
        z.main_slices = split_at_dim(z.expr_main, -2, False)  # List of [*, Dm']
        z.mask_slices = split_at_dim(input_mask, -1, False)  # List of [*]
        if extra_score is None:
            z.extra_score_slices = [None] * len(z.main_slices)
        else:
            z.extra_score_slices = split_at_dim(extra_score, -2, False)  # List of [*, L]
            if len(z.extra_score_slices) < len(z.main_slices):  # broadcast!
                assert len(z.extra_score_slices)==1
                z.extra_score_slices = z.extra_score_slices * len(z.main_slices)
        return z

    def step_score(self, cur_step: int, cur_repeat: int, cache: DecCache):
        z = self.get_contents(cur_repeat)
        # --
        cur_mask_slice = z.mask_slices[cur_step]
        cache, scores_t = self.model.step_score(cache, z.main_slices[cur_step], cur_mask_slice, z.expr_main, z.expr_pair,
                                                extra_score=z.extra_score_slices[cur_step])
        return cache, scores_t, cur_mask_slice

    def step_end(self, cur_step: int, cur_repeat: int, cache: DecCache, preds: BK.Expr):
        z = self.get_contents(cur_repeat)
        # --
        self.model.step_end(cache, z.main_slices[cur_step], z.mask_slices[cur_step], preds)

    def is_end(self, cur_step: int):
        return cur_step >= self.all_steps
