#

# SRL: pred + arg

__all__ = [
    "MySRLConf", "MySRLNode", "MySRLHelper",
]

from typing import List
import numpy as np
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.nn.modules import LossHelper
from msp2.data.inst import Sent, Frame, Mention, DataPadder
from msp2.data.vocab import SimpleVocab, SeqVocab
from msp2.utils import Constants, zlog, ZObject, zwarn
from msp2.tasks.common.models.seqlab import BigramInferenceHelper
from msp2.tasks.common.models.iter import *

# =====

class MySRLConf(BasicConf):
    def __init__(self):
        super().__init__()
        self.isize = -1  # input expr size
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
        self.evt_conf = SingleIdecConf()  # predicate
        self.loss_evt = 1.0
        self.evt_label_smoothing = 0.
        self.evt_loss_weight_non = 1.0
        self.evt_loss_sample_neg = 1.0  # how much of neg to include
        # for argument
        self.arg_conf = PairwiseIdecConf()  # argument
        self.loss_arg = 1.0
        self.arg_label_smoothing = 0.
        self.arg_loss_evt_sample_neg = 0.  # need to include bad evts? (no args!): a small one is ok
        self.arg_loss_sample_neg = 1.0  # how much of neg to include
        # special for arguments (BIO tagging)
        self.arg_use_bio = True  # use BIO mode!
        self.arg_pred_use_seq_cons = True  # use bio constraints in prediction
        self.arg_beam_k = 20
        # special decoding!
        self.evt_pred_use_posi = False  # special mode: assume gold posi
        self.evt_pred_use_all = False  # special debug mode: assume all as targets!

@node_reg(MySRLConf)
class MySRLNode(BasicNode):
    def __init__(self, conf: MySRLConf, vocab_evt: SimpleVocab, vocab_arg: SimpleVocab, **kwargs):
        super().__init__(conf, **kwargs)
        conf: MySRLConf = self.conf
        self.vocab_evt = vocab_evt
        self.vocab_arg = vocab_arg
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
        self.helper = MySRLHelper(conf, self.vocab_evt, helper_vocab_arg)
        # --
        # predicate
        self.evt_node = SingleIdecNode(conf.evt_conf, ndim=conf.isize, nlab=(2 if conf.binary_evt else len(vocab_evt)))
        # argument
        self.arg_node = PairwiseIdecNode(conf.arg_conf, ndim=conf.isize, nlab=len(helper_vocab_arg))

    def get_idec_nodes(self):
        return [self.evt_node, self.arg_node]

    # ----
    # helper for loss
    def _prepare_loss_weights(self, mask_expr: BK.Expr, not_nil: BK.Expr, neg_rate: float, extra_weights=None):
        if neg_rate < 1.:
            ret_weights = ((BK.rand(mask_expr.shape) < neg_rate) | not_nil).float() * mask_expr
        else:
            ret_weights = mask_expr  # simply as it is
        if extra_weights is not None:
            ret_weights *= extra_weights
        return ret_weights

    # [*, slen, D], [*, slen]
    def loss(self, insts: List[Sent], input_expr: BK.Expr, mask_expr: BK.Expr):
        conf: MySRLConf = self.conf
        # --
        slen = BK.get_shape(mask_expr, -1)
        arr_items, expr_evt_labels, expr_arg_labels, expr_loss_weight_non = self.helper.prepare(insts, True)
        if conf.binary_evt:
            expr_evt_labels = (expr_evt_labels>0).long()  # either 0 or 1
        loss_items = []
        # =====
        # evt
        # -- prepare weights and masks
        evt_not_nil = (expr_evt_labels>0)  # [*, slen]
        evt_extra_weights = BK.where(evt_not_nil, mask_expr, expr_loss_weight_non.unsqueeze(-1)*conf.evt_loss_weight_non)
        evt_weights = self._prepare_loss_weights(mask_expr, evt_not_nil, conf.evt_loss_sample_neg, evt_extra_weights)
        # -- get losses
        _, all_evt_cfs, all_evt_scores = self.evt_node.get_all_values()  # [*, slen]
        all_evt_losses = []
        for one_evt_scores in all_evt_scores:
            one_losses = BK.loss_nll(one_evt_scores, expr_evt_labels, label_smoothing=conf.evt_label_smoothing)
            all_evt_losses.append(one_losses)
        evt_loss_results = self.evt_node.helper.loss(all_losses=all_evt_losses, all_cfs=all_evt_cfs)
        for loss_t, loss_alpha, loss_name in evt_loss_results:
            one_evt_item = LossHelper.compile_leaf_loss("evt"+loss_name, (loss_t*evt_weights).sum(), evt_weights.sum(),
                                                        loss_lambda=conf.loss_evt*loss_alpha, gold=evt_not_nil.float().sum())
            loss_items.append(one_evt_item)
        # =====
        # arg
        _arg_loss_evt_sample_neg = conf.arg_loss_evt_sample_neg
        if _arg_loss_evt_sample_neg > 0:
            arg_evt_masks = ((BK.rand(mask_expr.shape)<_arg_loss_evt_sample_neg) | evt_not_nil).float() * mask_expr
        else:
            arg_evt_masks = evt_not_nil.float()  # [*, slen]
        # expand/flat the dims
        arg_flat_mask = (arg_evt_masks > 0)  # [*, slen]
        flat_mask_expr = mask_expr.unsqueeze(-2).expand(-1, slen, slen)[arg_flat_mask]  # [*, 1->slen, slen] => [??, slen]
        flat_arg_labels = expr_arg_labels[arg_flat_mask]  # [??, slen]
        flat_arg_not_nil = (flat_arg_labels > 0)  # [??, slen]
        flat_arg_weights = self._prepare_loss_weights(flat_mask_expr, flat_arg_not_nil, conf.arg_loss_sample_neg)
        # -- get losses
        _, all_arg_cfs, all_arg_scores = self.arg_node.get_all_values()  # [*, slen, slen]
        all_arg_losses = []
        for one_arg_scores in all_arg_scores:
            one_flat_arg_scores = one_arg_scores[arg_flat_mask]  # [??, slen]
            one_losses = BK.loss_nll(one_flat_arg_scores, flat_arg_labels, label_smoothing=conf.evt_label_smoothing)
            all_arg_losses.append(one_losses)
        all_arg_cfs = [z[arg_flat_mask] for z in all_arg_cfs]  # [??, slen]
        arg_loss_results = self.arg_node.helper.loss(all_losses=all_arg_losses, all_cfs=all_arg_cfs)
        for loss_t, loss_alpha, loss_name in arg_loss_results:
            one_arg_item = LossHelper.compile_leaf_loss("arg"+loss_name, (loss_t*flat_arg_weights).sum(), flat_arg_weights.sum(),
                                                        loss_lambda=conf.loss_arg*loss_alpha, gold=flat_arg_not_nil.float().sum())
            loss_items.append(one_arg_item)
        # =====
        # return loss
        ret_loss = LossHelper.combine_multiple_losses(loss_items)
        return ret_loss

    # [*, slen, D], [*, slen]
    def predict(self, insts: List[Sent], input_expr: BK.Expr, mask_expr: BK.Expr):
        conf: MySRLConf = self.conf
        slen = BK.get_shape(mask_expr, -1)
        # --
        # =====
        # evt
        _, all_evt_cfs, all_evt_raw_scores = self.evt_node.get_all_values()  # [*, slen, Le]
        all_evt_scores = [z.log_softmax(-1) for z in all_evt_raw_scores]
        final_evt_scores = self.evt_node.helper.pred(all_logprobs=all_evt_scores, all_cfs=all_evt_cfs)  # [*, slen, Le]
        if conf.evt_pred_use_all or conf.evt_pred_use_posi:  # todo(+W): not an elegant way...
            final_evt_scores[:,:,0] += Constants.REAL_PRAC_MIN  # all pred sth!!
        pred_evt_scores, pred_evt_labels = final_evt_scores.max(-1)  # [*, slen]
        # =====
        # arg
        _, all_arg_cfs, all_arg_raw_score = self.arg_node.get_all_values()  # [*, slen, slen, La]
        all_arg_scores = [z.log_softmax(-1) for z in all_arg_raw_score]
        final_arg_scores = self.arg_node.helper.pred(all_logprobs=all_arg_scores, all_cfs=all_arg_cfs)  # [*, slen, slen, La]
        # slightly more efficient by masking valid evts??
        full_pred_shape = BK.get_shape(final_arg_scores)[:-1]  # [*, slen, slen]
        pred_arg_scores, pred_arg_labels = BK.zeros(full_pred_shape), BK.zeros(full_pred_shape).long()
        arg_flat_mask = (pred_evt_labels > 0)  # [*, slen]
        flat_arg_scores = final_arg_scores[arg_flat_mask]  # [??, slen, La]
        if not BK.is_zero_shape(flat_arg_scores):  # at least one predicate!
            if self.pred_cons_mat is not None:
                flat_mask_expr = mask_expr.unsqueeze(-2).expand(-1, slen, slen)[arg_flat_mask]  # [*, 1->slen, slen] => [??, slen]
                flat_pred_arg_labels, flat_pred_arg_scores = BigramInferenceHelper.inference_search(
                    flat_arg_scores, self.pred_cons_mat, flat_mask_expr, conf.arg_beam_k)  # [??, slen]
            else:
                flat_pred_arg_scores, flat_pred_arg_labels = flat_arg_scores.max(-1)  # [??, slen]
            pred_arg_scores[arg_flat_mask] = flat_pred_arg_scores
            pred_arg_labels[arg_flat_mask] = flat_pred_arg_labels
        # =====
        # assign
        self.helper.put_results(insts, pred_evt_labels, pred_evt_scores, pred_arg_labels, pred_arg_scores)
        # --

# helper for dealing with instances
class MySRLHelper:
    def __init__(self, conf: MySRLConf, vocab_evt: SimpleVocab, vocab_arg: SimpleVocab):
        self.conf = conf
        self.vocab_evt = vocab_evt
        self.vocab_arg = vocab_arg
        # --
        self.evt_span_getter = Mention.create_span_getter(conf.evt_span_mode)
        self.evt_span_setter = Mention.create_span_setter(conf.evt_span_mode)
        self.arg_span_getter = Mention.create_span_getter(conf.arg_span_mode)
        self.arg_span_setter = Mention.create_span_setter(conf.arg_span_mode)

    def _prep_sent(self, sent: Sent):
        conf: MySRLConf = self.conf
        slen = len(sent)
        _loss_weight_non = getattr(sent, "_loss_weight_non", 1.)  # todo(+N): special name; loss_weight_non
        # note: for simplicity, assume no loss_weight_non for args
        # first for events
        evt_arr = np.full([slen], 0, dtype=np.int)  # [evt]
        arg_arr = np.full([slen, slen], 0, dtype=np.int)  # [evt, arg]
        evt_items = np.full([slen], None, dtype=object)  # [evt]
        for f in sent.get_frames(conf.evt_ftag):  # note: assume no overlapping
            # predicate
            evt_widx, evt_wlen = self.evt_span_getter(f.mention)
            evt_label = f.label_idx
            assert evt_wlen==1 and evt_label>0, "For simplicity!!"
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
        return ZObject(sent=sent, slen=slen, loss_weight_non=_loss_weight_non,
                       evt_items=evt_items, evt_arr=evt_arr, arg_arr=arg_arr)

    # prepare inputs
    def prepare(self, insts: List[Sent], use_cache: bool):
        # get info
        if use_cache:
            zobjs = []
            attr_name = f"_cache_srl"  # should be unique
            for s in insts:
                one = getattr(s, attr_name, None)
                if one is None:
                    one = self._prep_sent(s)
                    setattr(s, attr_name, one)  # set cache
                zobjs.append(one)
        else:
            zobjs = [self._prep_sent(s) for s in insts]
        # batch things
        bsize, mlen = len(insts), max(z.slen for z in zobjs) if len(zobjs)>0 else 1  # at least put one as padding
        batched_shape = (bsize, mlen)
        arr_items = np.full(batched_shape, None, dtype=object)
        arr_evt_labels = np.full(batched_shape, 0, dtype=np.int)
        arr_arg_labels = np.full(batched_shape+(mlen,), 0, dtype=np.int)
        for zidx, zobj in enumerate(zobjs):
            zlen = zobj.slen
            arr_items[zidx, :zlen] = zobj.evt_items
            arr_evt_labels[zidx, :zlen] = zobj.evt_arr
            arr_arg_labels[zidx, :zlen, :zlen] = zobj.arg_arr
        expr_evt_labels = BK.input_idx(arr_evt_labels)  # [*, slen]
        expr_arg_labels = BK.input_idx(arr_arg_labels)  # [*, slen]
        expr_loss_weight_non = BK.input_real([z.loss_weight_non for z in zobjs])  # [*]
        return arr_items, expr_evt_labels, expr_arg_labels, expr_loss_weight_non

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
        conf: MySRLConf = self.conf
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
    def put_results(self, insts: List[Sent], best_evt_labs, best_evt_scores, best_arg_labs, best_arg_scores):
        conf: MySRLConf = self.conf
        _evt_pred_use_posi = conf.evt_pred_use_posi
        vocab_evt = self.vocab_evt
        vocab_arg = self.vocab_arg
        if conf.arg_use_bio:
            real_vocab_arg = vocab_arg.base_vocab
        else:
            real_vocab_arg = vocab_arg
        # --
        all_arrs = [BK.get_value(z) for z in [best_evt_labs, best_evt_scores, best_arg_labs, best_arg_scores]]
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
# b msp2/tasks/zsfp/mtl/srl:168
