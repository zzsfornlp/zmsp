#

# the srl decoder's inference helpers (at inference time)

__all__ = [
    "SrlInferenceHelperConf", "SrlInferenceHelper",
]

from typing import List, Dict
import numpy as np
from msp2.nn import BK
from msp2.nn.layers import *
from ...common import ZMediator
from .postprocess import *

# --
class SrlInferenceHelperConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        # ef
        self.pred_ef_nil_add = 0.
        # --
        # evt
        self.pred_evt_nil_add = 0.
        self.pred_given_evt = False  # predict with given event (position)
        self.pred_evt_label = True  # try to predict event label, otherwise leave it there or "UNK"
        # --
        # arg
        self.pred_arg_nil_add = 0.
        self.arg_pp = PostProcessorConf()  # post-processor for arg
        # --

@node_reg(SrlInferenceHelperConf)
class SrlInferenceHelper(BasicNode):
    def __init__(self, conf: SrlInferenceHelperConf, dec: 'ZDecoderSrl', **kwargs):
        super().__init__(conf, **kwargs)
        conf: SrlInferenceHelperConf = self.conf
        # --
        self.setattr_borrow('dec', dec)
        self.arg_pp = PostProcessor(conf.arg_pp)
        # --

    # note: this procedure follows: evt(predicate) -> arg
    def predict(self, med: ZMediator):
        conf: SrlInferenceHelperConf = self.conf
        dec = self.dec
        dec_conf = dec.conf
        # --
        # prepare
        ibatch = med.ibatch
        hid_t = med.get_enc_cache("hid").get_cached_value()  # last enc layer, [*, dlen, D]
        _ds_idxes = ibatch.seq_info.dec_sent_idxes  # [*, dlen]
        # --
        # evt
        evt_score_cache = med.get_cache((dec.name, 'evt'))
        evt_scores_t = dec.lab_evt.score_labels(evt_score_cache.vals, nil_add_score=conf.pred_evt_nil_add)  # [*, dlen, L]
        if conf.pred_given_evt:  # assume given evt position
            res_evt_idxes_t, res_evts = self.decode_evt_given(dec, med.ibatch, evt_scores_t)  # ([??], [??]), [??]
        else:
            res_evt_idxes_t, res_evts = self.decode_evt(dec, med.ibatch, evt_scores_t)  # ([??], [??]), [??]
        if len(res_evts) == 0:
            return  # note: no need for further since no evts predicted!
        # evt boundary
        if dec.boundary_evt is not None:
            _boundary_mask_t = (_ds_idxes[res_evt_idxes_t[0]] == _ds_idxes[res_evt_idxes_t].unsqueeze(-1)).float()  # [??, dlen]
            flat_indicators = dec.boundary_evt.prepare_indicators([res_evt_idxes_t[1]], BK.get_shape(_boundary_mask_t))
            flat_hid_t = hid_t[res_evt_idxes_t[0]]  # [??, dlen, D]
            _, _left_idxes, _right_idxes = dec.boundary_evt.decode(flat_hid_t, _boundary_mask_t, flat_indicators)  # [??]
            self.assign_boundaries(res_evts, _left_idxes, _right_idxes)
        # --
        # arg
        arg_score_cache = med.get_cache((dec.name, 'arg'))
        arg_scores_t = dec.lab_arg.score_labels(
            arg_score_cache.vals, preidx_t=res_evt_idxes_t, nil_add_score=conf.pred_arg_nil_add)  # [??, dlen, L]
        if dec_conf.arg_use_bio:  # if use BIO, then checking the seq will be fine
            self.decode_arg_bio(dec, res_evts, arg_scores_t)
        else:  # otherwise, still need two steps
            res_arg_idxes_t, res_args = self.decode_arg(dec, res_evts, arg_scores_t)  # [???]
            # arg(ef) boundary
            if len(res_args)>0 and dec.boundary_arg is not None:
                _ab_fidxes_t, _ab_awidxes_t = res_arg_idxes_t
                _ab_bidxes_t, _ab_ewidxes_t = [z[_ab_fidxes_t] for z in res_evt_idxes_t]  # [???]
                _ab_mask_t = (_ds_idxes[_ab_bidxes_t] == _ds_idxes[_ab_bidxes_t,_ab_awidxes_t].unsqueeze(-1)).float()
                _inds = dec.boundary_arg.prepare_indicators([_ab_ewidxes_t, _ab_awidxes_t], BK.get_shape(_ab_mask_t))  # [???, dlen]
                flat_hid_t = hid_t[_ab_bidxes_t]  # [???, dlen, D]
                _, _left_idxes, _right_idxes = dec.boundary_arg.decode(flat_hid_t, _ab_mask_t, _inds)  # [???]
                self.assign_boundaries(res_args, _left_idxes, _right_idxes)
        # --
        # final arg post-process
        for evt in res_evts:
            self.arg_pp.process(evt)  # modified inplace!
        # --
        return

    # --

    # return flattened info on predicted ones
    def decode_evt(self, dec, ibatch, evt_scores_t: BK.Expr):
        _pred_max_layer_evt = dec.conf.max_layer_evt
        _voc_evt = dec.voc_evt
        _pred_evt_label = self.conf.pred_evt_label
        # --
        evt_logprobs_t = evt_scores_t.log_softmax(-1)  # [*, dlen, L]
        pred_evt_scores, pred_evt_labels = evt_logprobs_t.topk(_pred_max_layer_evt)  # [*, dlen, K]
        arr_evt_scores, arr_evt_labels = BK.get_value(pred_evt_scores), BK.get_value(pred_evt_labels)  # [*, dlen, K]
        # put results
        res_bidxes, res_widxes, res_evts = [], [], []  # flattened results
        for bidx, item in enumerate(ibatch.items):  # for each item in the batch
            _dec_offsets = item.seq_info.dec_offsets
            for sidx, sent in enumerate(item.sents):
                # note: here we only predict for center if there is!
                if item.center_sidx is not None and sidx != item.center_sidx:
                    continue  # skip non-center sent in this mode!
                _start = _dec_offsets[sidx]
                _len = len(sent)
                _arr_scores, _arr_labels = arr_evt_scores[bidx][_start:_start+_len], arr_evt_labels[bidx][_start:_start+_len]
                for widx in range(_len):
                    for _score, _lab in zip(_arr_scores[widx], _arr_labels[widx]):  # [K]
                        if _lab == 0:  # note: idx=0 means NIL
                            break
                        # add new one!!
                        res_bidxes.append(bidx)
                        res_widxes.append(_start+widx)  # note: remember to add offset!
                        _new_evt = sent.make_event(
                            widx, 1, type=(_voc_evt.idx2word(_lab) if _pred_evt_label else "UNK"), score=float(_score))
                        _new_evt.set_label_idx(int(_lab))
                        _new_evt._tmp_sstart = _start  # todo(+N): ugly tmp value ...
                        _new_evt._tmp_sidx = sidx
                        _new_evt._tmp_item = item
                        res_evts.append(_new_evt)
        # return
        res_bidxes_t, res_widxes_t = BK.input_idx(res_bidxes), BK.input_idx(res_widxes)  # [??]
        return (res_bidxes_t, res_widxes_t), res_evts

    # assume given predicates (at least position), return flattened info on predicted ones
    def decode_evt_given(self, dec, ibatch, evt_scores_t: BK.Expr):
        _voc_evt = dec.voc_evt
        _assume_osof = dec.conf.assume_osof  # one seq one frame
        _pred_evt_label = self.conf.pred_evt_label
        # --
        if _pred_evt_label:
            evt_logprobs_t = evt_scores_t.log_softmax(-1)  # [*, dlen, L]
            pred_evt_scores, pred_evt_labels = evt_logprobs_t.max(-1)  # [*, dlen], note: maximum!
            arr_evt_scores, arr_evt_labels = BK.get_value(pred_evt_scores), BK.get_value(pred_evt_labels)  # [*, dlen]
        else:
            arr_evt_scores = arr_evt_labels = None
        # --
        # read given results
        res_bidxes, res_widxes, res_evts = [], [], []  # flattened results
        for bidx, item in enumerate(ibatch.items):  # for each item in the batch
            _trg_evts = [item.inst] if _assume_osof else \
                sum([sent.events for sidx,sent in enumerate(item.sents) if (item.center_sidx is None or sidx == item.center_sidx)],[])
            # --
            _dec_offsets = item.seq_info.dec_offsets
            for _evt in _trg_evts:
                sidx = item.sents.index(_evt.sent)
                _start = _dec_offsets[sidx]
                _full_hidx = _start+_evt.mention.shead_widx
                # add new one
                res_bidxes.append(bidx)
                res_widxes.append(_full_hidx)
                _evt._tmp_sstart = _start  # todo(+N): ugly tmp value ...
                _evt._tmp_sidx = sidx
                _evt._tmp_item = item
                res_evts.append(_evt)
                # assign label?
                if _pred_evt_label:
                    _lab = int(arr_evt_labels[bidx, _full_hidx])  # label index
                    _evt.set_label_idx(_lab)
                    _evt.set_label(_voc_evt.idx2word(_lab))
                    _evt.set_score(float(arr_evt_scores[bidx, _full_hidx]))
            # --
        # --
        # return
        res_bidxes_t, res_widxes_t = BK.input_idx(res_bidxes), BK.input_idx(res_widxes)  # [??]
        return (res_bidxes_t, res_widxes_t), res_evts

    # decode arg (shead)
    def decode_arg(self, dec, res_evts: List, arg_scores_t: BK.Expr):
        _pred_max_layer_arg = dec.conf.max_layer_arg
        _arg_allowed_sent_gap = dec.conf.arg_allowed_sent_gap
        _voc_arg = dec.voc_arg
        # --
        arg_logprobs_t = arg_scores_t.log_softmax(-1)  # [??, dlen, L]
        pred_arg_scores, pred_arg_labels = arg_logprobs_t.topk(_pred_max_layer_arg)  # [??, dlen, K]
        arr_arg_scores, arr_arg_labels = BK.get_value(pred_arg_scores), BK.get_value(pred_arg_labels)  # [??, dlen, K]
        # put results
        res_fidxes, res_widxes, res_args = [], [], []  # flattened results
        for fidx, evt in enumerate(res_evts):  # for each evt
            item = evt._tmp_item  # cached
            _evt_sidx = evt._tmp_sidx
            _dec_offsets = item.seq_info.dec_offsets
            for sidx, sent in enumerate(item.sents):
                if abs(sidx - _evt_sidx) > _arg_allowed_sent_gap:
                    continue  # larger than allowed sentence gap
                _start = _dec_offsets[sidx]
                _len = len(sent)
                _arr_scores, _arr_labels = arr_arg_scores[fidx][_start:_start+_len], arr_arg_labels[fidx][_start:_start+_len]
                for widx in range(_len):
                    for _score, _lab in zip(_arr_scores[widx], _arr_labels[widx]):  # [K]
                        if _lab == 0:  # note: idx=0 means NIL
                            break
                        # add new one!!
                        res_fidxes.append(fidx)
                        res_widxes.append(_start+widx)
                        _new_ef = sent.make_entity_filler(widx, 1)
                        _new_arg = evt.add_arg(_new_ef, role=_voc_arg.idx2word(_lab), score=float(_score))
                        _new_arg._tmp_sstart = _start  # todo(+N): ugly tmp value ...
                        res_args.append(_new_arg)
        # return
        res_fidxes_t, res_widxes_t = BK.input_idx(res_fidxes), BK.input_idx(res_widxes)  # [??]
        return (res_fidxes_t, res_widxes_t), res_args

    # decode arg bio
    def decode_arg_bio(self, dec, res_evts: List, arg_scores_t: BK.Expr):
        _arg_allowed_sent_gap = dec.conf.arg_allowed_sent_gap
        _vocab_bio_arg = dec.vocab_bio_arg
        _vocab_arg = _vocab_bio_arg.base_vocab
        # --
        arg_logprobs_t = arg_scores_t.log_softmax(-1)  # [??, dlen, L]
        # todo(+N): allow multiple bio seqs?
        pred_arg_scores, pred_arg_labels = arg_logprobs_t.max(-1)  # note: here only top1, [??, dlen]
        arr_arg_scores, arr_arg_labels = BK.get_value(pred_arg_scores), BK.get_value(pred_arg_labels)  # [??, dlen]
        # put results
        for fidx, evt in enumerate(res_evts):  # for each evt
            item = evt._tmp_item  # cached
            _evt_sidx = evt._tmp_sidx
            _dec_offsets = item.seq_info.dec_offsets
            for sidx, sent in enumerate(item.sents):
                if abs(sidx - _evt_sidx) > _arg_allowed_sent_gap:
                    continue  # larger than allowed sentence gap
                _start = _dec_offsets[sidx]
                _len = len(sent)
                _arr_scores, _arr_labels = arr_arg_scores[fidx][_start:_start+_len], arr_arg_labels[fidx][_start:_start+_len]
                # decode bio
                _arg_results = _vocab_bio_arg.tags2spans_idx(_arr_labels)
                for a_widx, a_wlen, a_lab in _arg_results:
                    a_lab = int(a_lab)
                    assert a_lab > 0, "Error: should not extract 'O'!?"
                    _new_ef = sent.make_entity_filler(a_widx, a_wlen)
                    a_role = _vocab_arg.idx2word(a_lab)
                    _new_arg = evt.add_arg(_new_ef, a_role, score=np.mean(_arr_scores[a_widx:a_widx+a_wlen]).item())
        # --
        return  # no need to return anything here

    # helper for assign boundaries
    def assign_boundaries(self, items: List, left_idxes: BK.Expr, right_idxes: BK.Expr):
        _arr_left, _arr_right = BK.get_value(left_idxes), BK.get_value(right_idxes)
        for ii, item in enumerate(items):
            _mention = item.mention
            _start = item._tmp_sstart  # need to minus this!!
            _left_widx, _right_widx = _arr_left[ii].item()-_start, _arr_right[ii].item()-_start
            _mention.set_span(*(_mention.get_span()), shead=True)  # first move to shead!
            _mention.set_span(_left_widx, _right_widx-_left_widx+1)
        # --

# --
# b msp2/tasks/zmtl2/zmod/dec/dec_srl/inference:??
