#

# SRL: pred + arg

__all__ = [
    "ZDecoderSRLConf", "ZDecoderSRLNode", "ZDecoderSRLHelper",
]

from typing import List
import numpy as np
import time
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.nn.modules import LossHelper
from msp2.data.inst import Sent, Frame, Mention, DataPadder
from msp2.data.vocab import SimpleVocab, SeqVocab
from msp2.utils import Constants, zlog, ZObject, zwarn, ConfEntryChoices
from msp2.tasks.common.models.seqlab import BigramInferenceHelper
from ..common import *
from ..enc import *
from ..dec import *

# =====

class ZDecoderSRLConf(ZDecoderConf):
    def __init__(self):
        super().__init__()
        # --
        # general specifications
        self.evt_ftag = "evt"
        self.evt_span_mode = "shead"
        self.arg_ftag = "ef"  # special target for args
        self.arg_span_mode = "span"
        self.arg_only_rank1 = True  # only consider rank1 arg
        # --
        # for predicate
        self.binary_evt = True  # no prediction of detailed lemma+sense?
        self.evt_conf = IdecSingleConf()  # predicate
        self.loss_evt = 1.0
        self.evt_label_smoothing = 0.
        self.evt_loss_weight_non = 1.0
        self.evt_loss_sample_neg = 1.0  # how much of neg to include
        # --
        # for argument
        self.arg_conf: IdecConf = \
            ConfEntryChoices({"pair_emb": IdecPairwiseConf(), "pair_att": IdecAttConf()}, "pair_emb")  # argument
        self.loss_arg = 1.0
        self.arg_label_smoothing = 0.
        self.arg_loss_sample_neg = 1.0  # how much of neg to include
        self.arg_loss_inc_neg_evt = 0.  # need to include bad evts? (no args!): a small one is ok
        # special for arguments (BIO tagging)
        self.arg_use_bio = True  # use BIO mode!
        self.arg_pred_use_seq_cons = False  # use bio constraints in prediction
        self.arg_beam_k = 20
        # --
        # for arg2: aug arg loss
        self.arg2_conf: IdecConf = \
            ConfEntryChoices({"pair_emb": IdecPairwiseConf(), "pair_att": IdecAttConf()}, "pair_emb")  # argument
        self.loss_arg2 = 0.0
        self.arg2_label_smoothing = 0.
        self.arg2_loss_sample_neg = 1.0  # how much of neg to include
        # --
        # special decoding!
        self.evt_pred_use_posi = False  # special mode: assume gold posi
        self.evt_pred_use_all = False  # special debug mode: assume all as targets!

@node_reg(ZDecoderSRLConf)
class ZDecoderSRLNode(ZDecoder):
    def __init__(self, conf: ZDecoderSRLConf, name: str,
                 vocab_evt: SimpleVocab, vocab_arg: SimpleVocab, ref_enc: ZEncoder, **kwargs):
        super().__init__(conf, name, **kwargs)
        conf: ZDecoderSRLConf = self.conf
        self.vocab_evt = vocab_evt
        self.vocab_arg = vocab_arg
        _enc_dim, _head_dim = ref_enc.get_enc_dim(), ref_enc.get_head_dim()
        # --
        self.vocab_bio_arg = None
        self.pred_cons_mat = None
        if conf.arg_use_bio:
            self.vocab_bio_arg = SeqVocab(vocab_arg)  # simply BIO vocab
            zlog(f"Use BIO vocab for srl: {self.vocab_bio_arg}")
            if conf.arg_pred_use_seq_cons:
                _m = self.vocab_bio_arg.get_allowed_transitions()
                self.pred_cons_mat = (1. - BK.input_real(_m)) * Constants.REAL_PRAC_MIN  # [L, L]
                zlog(f"Further use BIO constraints for decoding: {self.pred_cons_mat.shape}")
            helper_vocab_arg = self.vocab_bio_arg
        else:
            helper_vocab_arg = self.vocab_arg
        # --
        self.helper = ZDecoderSRLHelper(conf, self.vocab_evt, helper_vocab_arg, self.vocab_arg)
        # --
        # nodes
        self.evt_node: IdecNode = conf.evt_conf.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=(2 if conf.binary_evt else len(vocab_evt)))
        self.arg_node: IdecNode = conf.arg_conf.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=len(helper_vocab_arg))
        self.arg2_node: IdecNode = conf.arg2_conf.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=len(self.vocab_arg))
        # --
        raise RuntimeError("Deprecated after MED's collecting of scores!!")

    def get_idec_nodes(self):
        return [self.evt_node, self.arg_node, self.arg2_node]

    # return List[tensor], bool
    def layer_end(self, med: ZMediator):
        # first simply forward them
        lidx = med.lidx
        rets = []
        if self.evt_node.need_app_layer(lidx):
            rets.append(self.evt_node.forward(med))
        if self.arg_node.need_app_layer(lidx):
            rets.append(self.arg_node.forward(med))
        if self.is_training():  # note: only need aug in training!
            if self.arg2_node.need_app_layer(lidx):
                rr = self.arg2_node.forward(med)
                assert rr is None, "Aug node should not add back things!"
        # todo(+N): early exit for pred
        return rets, (lidx >= self.max_app_lidx)

    # ----
    # helper for loss
    def _prepare_loss_weights(self, mask_expr: BK.Expr, must_include_t: BK.Expr, neg_rate: float, extra_weights=None):
        if neg_rate <= 0.:  # only must_include
            ret_weights = (must_include_t.float()) * mask_expr
        elif neg_rate < 1.:  # must_include + sample
            ret_weights = ((BK.rand(mask_expr.shape) < neg_rate) | must_include_t).float() * mask_expr
        else:  # all in
            ret_weights = mask_expr  # simply as it is
        if extra_weights is not None:
            ret_weights = ret_weights * extra_weights
        return ret_weights

    def _loss_evt(self, mask_expr, expr_evt_labels, expr_loss_weight_non):
        conf: ZDecoderSRLConf = self.conf
        # --
        loss_items = []
        # -- prepare weights and masks
        evt_not_nil = (expr_evt_labels > 0)  # [*, slen]
        evt_extra_weights = BK.where(evt_not_nil, mask_expr, expr_loss_weight_non.unsqueeze(-1) * conf.evt_loss_weight_non)
        evt_weights = self._prepare_loss_weights(mask_expr, evt_not_nil, conf.evt_loss_sample_neg, evt_extra_weights)
        # -- get losses
        all_evt_scores = self.evt_node.buffer_scores.values()  # List([*,slen])
        all_evt_losses = []
        for one_evt_scores in all_evt_scores:
            one_losses = BK.loss_nll(one_evt_scores, expr_evt_labels, label_smoothing=conf.evt_label_smoothing)
            all_evt_losses.append(one_losses)
        evt_loss_results = self.evt_node.helper.loss(all_losses=all_evt_losses)
        for loss_t, loss_alpha, loss_name in evt_loss_results:
            one_evt_item = LossHelper.compile_leaf_loss(
                "evt" + loss_name, (loss_t * evt_weights).sum(), evt_weights.sum(),
                loss_lambda=conf.loss_evt * loss_alpha, gold=evt_not_nil.float().sum())
            loss_items.append(one_evt_item)
        return loss_items

    def _loss_arg(self, mask_expr, expr_evt_labels, expr_arg_labels, expr_arg2_labels):
        conf: ZDecoderSRLConf = self.conf
        # --
        loss_items = []
        slen = BK.get_shape(mask_expr, -1)
        evt_not_nil = (expr_evt_labels > 0)  # [*, slen]
        # --
        # first prepare evts to focus at
        arg_evt_masks = self._prepare_loss_weights(mask_expr, evt_not_nil, conf.arg_loss_inc_neg_evt)  # [*, slen]
        # expand/flat the dims
        arg_flat_mask = (arg_evt_masks > 0)  # [*, slen]
        flat_mask_expr = mask_expr.unsqueeze(-2).expand(-1, slen, slen)[arg_flat_mask]  # [*, 1->slen, slen] => [??, slen]
        # -- get losses
        for aname, anode, glabels, sneg_rate, loss_mul, lsmooth in \
                zip(["arg", "arg2"], [self.arg_node, self.arg2_node], [expr_arg_labels, expr_arg2_labels],
                    [conf.arg_loss_sample_neg, conf.arg2_loss_sample_neg], [conf.loss_arg, conf.loss_arg2],
                    [conf.arg_label_smoothing, conf.arg2_label_smoothing]):
            # --
            if loss_mul <= 0.:
                continue
            # --
            # prepare
            flat_arg_labels = glabels[arg_flat_mask]  # [??, slen]
            flat_arg_not_nil = (flat_arg_labels > 0)  # [??, slen]
            flat_arg_weights = self._prepare_loss_weights(flat_mask_expr, flat_arg_not_nil, sneg_rate)
            # scores and losses
            all_arg_scores = anode.buffer_scores.values()  # [*, slen, slen]
            all_arg_losses = []
            for one_arg_scores in all_arg_scores:
                one_flat_arg_scores = one_arg_scores[arg_flat_mask]  # [??, slen]
                one_losses = BK.loss_nll(one_flat_arg_scores, flat_arg_labels, label_smoothing=lsmooth)
                all_arg_losses.append(one_losses)
            arg_loss_results = anode.helper.loss(all_losses=all_arg_losses)
            # collect losses
            for loss_t, loss_alpha, loss_name in arg_loss_results:
                one_arg_item = LossHelper.compile_leaf_loss(
                    aname + loss_name, (loss_t * flat_arg_weights).sum(), flat_arg_weights.sum(),
                    loss_lambda=(loss_mul*loss_alpha), gold=flat_arg_not_nil.float().sum())
                loss_items.append(one_arg_item)
        return loss_items

    # get loss finally
    def loss(self, med: ZMediator):
        conf: ZDecoderSRLConf = self.conf
        insts, mask_expr = med.insts, med.get_mask_t()
        # --
        # first prepare golds
        arr_items, expr_evt_labels, expr_arg_labels, expr_arg2_labels, expr_loss_weight_non = self.helper.prepare(insts, True)
        if conf.binary_evt:
            expr_evt_labels = (expr_evt_labels>0).long()  # either 0 or 1
        # --
        loss_items = []
        if conf.loss_evt > 0.:
            loss_items.extend(self._loss_evt(mask_expr, expr_evt_labels, expr_loss_weight_non))
        if conf.loss_arg > 0. or conf.loss_arg2 > 0.:
            loss_items.extend(self._loss_arg(mask_expr, expr_evt_labels, expr_arg_labels, expr_arg2_labels))
        # =====
        # return loss
        ret_loss = LossHelper.combine_multiple_losses(loss_items)
        return ret_loss

    # --
    def _pred_evt(self):
        conf: ZDecoderSRLConf = self.conf
        # --
        all_evt_raw_scores = self.evt_node.buffer_scores.values()  # [*, slen, Le]
        all_evt_logprobs = [z.log_softmax(-1) for z in all_evt_raw_scores]
        final_evt_logprobs = self.evt_node.helper.pred(all_logprobs=all_evt_logprobs)  # [*, slen, Le]
        if conf.evt_pred_use_all or conf.evt_pred_use_posi:  # todo(+W): not an elegant way...
            final_evt_logprobs[:,:,0] += Constants.REAL_PRAC_MIN  # all pred sth!!
        pred_evt_scores, pred_evt_labels = final_evt_logprobs.max(-1)  # [*, slen]
        return pred_evt_labels, pred_evt_scores

    def _pred_arg(self, mask_expr, pred_evt_labels):
        conf: ZDecoderSRLConf = self.conf
        slen = BK.get_shape(mask_expr, -1)
        # --
        all_arg_raw_score = self.arg_node.buffer_scores.values()  # [*, slen, slen, La]
        all_arg_logprobs = [z.log_softmax(-1) for z in all_arg_raw_score]
        final_arg_logprobs = self.arg_node.helper.pred(all_logprobs=all_arg_logprobs)  # [*, slen, slen, La]
        # slightly more efficient by masking valid evts??
        full_pred_shape = BK.get_shape(final_arg_logprobs)[:-1]  # [*, slen, slen]
        pred_arg_scores, pred_arg_labels = BK.zeros(full_pred_shape), BK.zeros(full_pred_shape).long()
        # mask
        arg_flat_mask = (pred_evt_labels > 0)  # [*, slen]
        flat_arg_logprobs = final_arg_logprobs[arg_flat_mask]  # [??, slen, La]
        if not BK.is_zero_shape(flat_arg_logprobs):  # at least one predicate
            if self.pred_cons_mat is not None:
                flat_mask_expr = mask_expr.unsqueeze(-2).expand(-1, slen, slen)[arg_flat_mask]  # [*, 1->slen, slen] => [??, slen]
                flat_pred_arg_labels, flat_pred_arg_scores = BigramInferenceHelper.inference_search(
                    flat_arg_logprobs, self.pred_cons_mat, flat_mask_expr, conf.arg_beam_k)  # [??, slen]
            else:
                flat_pred_arg_scores, flat_pred_arg_labels = flat_arg_logprobs.max(-1)  # [??, slen]
            pred_arg_scores[arg_flat_mask] = flat_pred_arg_scores
            pred_arg_labels[arg_flat_mask] = flat_pred_arg_labels
        return pred_arg_labels, pred_arg_scores  # [*, slen, slen, La]

    # predict finally
    def predict(self, med: ZMediator):
        conf: ZDecoderSRLConf = self.conf
        insts, mask_expr = med.insts, med.get_mask_t()
        # --
        pred_evt_labels, pred_evt_scores = self._pred_evt()
        pred_arg_labels, pred_arg_scores = self._pred_arg(mask_expr, pred_evt_labels)
        # transfer data from gpu also counts (also make sure gpu calculations are done)!
        all_arrs = [BK.get_value(z) for z in [pred_evt_labels, pred_evt_scores, pred_arg_labels, pred_arg_scores]]
        # =====
        # assign; also record post-processing (non-computing) time
        time0 = time.time()
        self.helper.put_results(insts, all_arrs)
        time1 = time.time()
        # --
        return {f"{self.name}_posttime": time1-time0}

# helper
class ZDecoderSRLHelper(ZDecoderHelper):
    def __init__(self, conf: ZDecoderSRLConf, vocab_evt: SimpleVocab, vocab_arg: SimpleVocab, vocab_arg2: SimpleVocab):
        self.conf = conf
        self.vocab_evt = vocab_evt
        self.vocab_arg = vocab_arg
        self.vocab_arg2 = vocab_arg2
        # --
        self.evt_span_getter = Mention.create_span_getter(conf.evt_span_mode)
        self.evt_span_setter = Mention.create_span_setter(conf.evt_span_mode)
        self.arg_span_getter = Mention.create_span_getter(conf.arg_span_mode)
        self.arg_span_setter = Mention.create_span_setter(conf.arg_span_mode)

    def _prep_sent(self, sent: Sent):
        conf: ZDecoderSRLConf = self.conf
        # --
        slen = len(sent)
        _loss_weight_non = getattr(sent, "_loss_weight_non", 1.)  # todo(+N): special name; loss_weight_non
        # note: for simplicity, assume no loss_weight_non for args
        # first for events
        evt_arr = np.full([slen], 0, dtype=np.int)  # [evt]
        arg_arr = np.full([slen, slen], 0, dtype=np.int)  # [evt, arg]
        arg2_arr = np.full([slen, slen], 0, dtype=np.int)  # [evt, arg]
        evt_items = np.full([slen], None, dtype=object)  # [evt]
        for f in sent.get_frames(conf.evt_ftag):  # note: assume no overlapping
            # predicate
            evt_widx, evt_wlen = self.evt_span_getter(f.mention)
            evt_label = f.label_idx
            assert evt_wlen==1 and evt_label>0, "For simplicity!!"  # note: use shead here!
            evt_items[evt_widx] = f
            evt_arr[evt_widx] = evt_label
            # arguments
            if conf.arg_only_rank1:
                cur_args = [a for a in f.args if a.info.get("rank", 1) == 1]
            else:
                cur_args = f.args
            # bio or not
            if conf.arg_use_bio:  # special
                arg_spans = [self.arg_span_getter(a.mention) + (a.label_idx,) for a in cur_args]
                tag_layers = self.vocab_arg.spans2tags_idx(arg_spans, slen)
                if len(tag_layers) > 1:
                    zwarn(f"Warning: 'Full args require multiple layers with {arg_spans}")
                arg_arr[evt_widx, :] = tag_layers[0][0]  # directly assign it!
            else:  # plain ones
                for a in cur_args:
                    arg_role = a.label_idx
                    arg_widx, arg_wlen = self.arg_span_getter(a.mention)
                    arg_arr[evt_widx, arg_widx:arg_widx+arg_wlen] = arg_role
            # arg2, only single ones!
            for a2 in cur_args:
                arg2_arr[evt_widx, a2.mention.shead_widx] = a2.label_idx
            # --
        return ZObject(sent=sent, slen=slen, loss_weight_non=_loss_weight_non,
                       evt_items=evt_items, evt_arr=evt_arr, arg_arr=arg_arr, arg2_arr=arg2_arr)

    # prepare inputs
    def prepare(self, insts: List[Sent], use_cache: bool):
        # get info
        zobjs = ZDecoderHelper.get_zobjs(insts, self._prep_sent, use_cache, f"_cache_srl")
        # batch things
        bsize, mlen = len(insts), max(z.slen for z in zobjs) if len(zobjs)>0 else 1  # at least put one as padding
        batched_shape = (bsize, mlen)
        arr_items = np.full(batched_shape, None, dtype=object)
        arr_evt_labels = np.full(batched_shape, 0, dtype=np.int)
        arr_arg_labels = np.full(batched_shape+(mlen,), 0, dtype=np.int)
        arr_arg2_labels = np.full(batched_shape+(mlen,), 0, dtype=np.int)
        for zidx, zobj in enumerate(zobjs):
            zlen = zobj.slen
            arr_items[zidx, :zlen] = zobj.evt_items
            arr_evt_labels[zidx, :zlen] = zobj.evt_arr
            arr_arg_labels[zidx, :zlen, :zlen] = zobj.arg_arr
            arr_arg2_labels[zidx, :zlen, :zlen] = zobj.arg2_arr
        expr_evt_labels = BK.input_idx(arr_evt_labels)  # [*, slen]
        expr_arg_labels = BK.input_idx(arr_arg_labels)  # [*, slen, slen]
        expr_arg2_labels = BK.input_idx(arr_arg2_labels)  # [*, slen, slen]
        expr_loss_weight_non = BK.input_real([z.loss_weight_non for z in zobjs])  # [*]
        return arr_items, expr_evt_labels, expr_arg_labels, expr_arg2_labels, expr_loss_weight_non

    # simple arg decode
    def simple_arg_decode(self, tag_idxes: List[int]):
        spans = []
        prev_start, prev_t = 0, 0
        # --
        def _close_prev(_start: int, _end: int, _t, _spans: List):
            if _t > 0:
                assert _end > _start
                _spans.append((_start, _end-_start, _t))
            return _end, 0
        # --
        for idx, tidx in enumerate(tag_idxes):
            if tidx == 0 or tidx != prev_t:
                prev_start, prev_t = _close_prev(prev_start, idx, prev_t, spans)
            prev_t = tidx
        _close_prev(prev_start, len(tag_idxes), prev_t, spans)
        return spans

    # decode arg
    def decode_arg(self, evt: Frame, arg_lidxes, arg_scores, vocab, real_vocab):
        conf: ZDecoderSRLConf = self.conf
        # --
        if conf.arg_use_bio:
            new_arg_results = vocab.tags2spans_idx(arg_lidxes)
        else:
            new_arg_results = self.simple_arg_decode(arg_lidxes)
        # --
        new_arg_labels = [vocab.idx2word(z) if z>0 else 'O' for z in arg_lidxes]
        evt.info["arg_lab"] = new_arg_labels
        for a_widx, a_wlen, a_lab in new_arg_results:
            a_lab = int(a_lab)
            assert a_lab > 0, "Error: should not extract 'O'!?"
            new_ef = evt.sent.make_frame(a_widx, a_wlen, self.conf.arg_ftag)  # make an ef as mention
            a_role = real_vocab.idx2word(a_lab)
            alink = evt.add_arg(new_ef, a_role, score=np.mean(arg_scores[a_widx:a_widx+a_wlen]).item())
            alink.set_label_idx(a_lab)  # set idx
            self.arg_span_setter(alink.mention, a_widx, a_wlen)
        # --

    # *[*, slen], *[*, slen, slen]
    def put_results(self, insts: List[Sent], all_arrs):
        conf: ZDecoderSRLConf = self.conf
        _evt_pred_use_posi = conf.evt_pred_use_posi
        vocab_evt = self.vocab_evt
        vocab_arg = self.vocab_arg
        if conf.arg_use_bio:
            real_vocab_arg = vocab_arg.base_vocab
        else:
            real_vocab_arg = vocab_arg
        # --
        # all_arrs = [BK.get_value(z) for z in [best_evt_labs, best_evt_scores, best_arg_labs, best_arg_scores]]
        for bidx, inst in enumerate(insts):
            inst.delete_frames(conf.arg_ftag)  # delete old args
            # --
            cur_len = len(inst)
            cur_evt_labs, cur_evt_scores, cur_arg_labs, cur_arg_scores = [z[bidx][:cur_len] for z in all_arrs]
            inst.info["evt_lab"] = [vocab_evt.idx2word(z) if z>0 else 'O' for z in cur_evt_labs]
            # --
            if _evt_pred_use_posi:  # special mode
                for evt in inst.get_frames(conf.evt_ftag):
                    # reuse posi but re-assign label!
                    one_widx = evt.mention.shead_widx
                    one_lab, one_score = cur_evt_labs[one_widx].item(), cur_evt_scores[one_widx].item()
                    evt.set_label(vocab_evt.idx2word(one_lab))
                    evt.set_label_idx(one_lab)
                    evt.score = one_score
                    # args
                    new_arg_scores = cur_arg_scores[one_widx][:cur_len]
                    new_arg_label_idxes = cur_arg_labs[one_widx][:cur_len]
                    self.decode_arg(evt, new_arg_label_idxes, new_arg_scores, vocab_arg, real_vocab_arg)
            else:  # pred everything!
                inst.delete_frames(conf.evt_ftag)
                for one_widx in range(cur_len):
                    one_lab, one_score = cur_evt_labs[one_widx].item(), cur_evt_scores[one_widx].item()
                    if one_lab == 0:
                        continue
                    # make new evt
                    new_evt = inst.make_frame(one_widx, 1, conf.evt_ftag, type=vocab_evt.idx2word(one_lab), score=one_score)
                    new_evt.set_label_idx(one_lab)
                    self.evt_span_setter(new_evt.mention, one_widx, 1)
                    # args
                    new_arg_scores = cur_arg_scores[one_widx][:cur_len]
                    new_arg_label_idxes = cur_arg_labs[one_widx][:cur_len]
                    self.decode_arg(new_evt, new_arg_label_idxes, new_arg_scores, vocab_arg, real_vocab_arg)
                # --
            # --

# --
# b msp2/tasks/zmtl/modules/dec_comps/srl:?
