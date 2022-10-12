#

# the srl decoder's inference helpers (at inference time) (v2)

__all__ = [
    "SrlInferenceHelperConf", "SrlInferenceHelper",
]

from typing import List, Dict
from collections import defaultdict
import numpy as np
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.utils import default_pickle_serializer, Constants
from msp2.data.inst import set_ee_heads
from msp2.data.vocab import ZFrameCollectionHelper as fchelper
from ...common import ZMediator
from .postprocess import *

# --
class SrlInferenceHelperConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self.frames_file = ""  # load from file
        self.frames_name = ""  # find it in prebuilt resources
        # evt
        self.pred_evt_nil_add: float = None
        self.pred_given_evt = False  # predict with given event (position)
        self.pred_evt_label = True  # try to predict event label, otherwise leave it there or "UNK"
        self.pred_evt_boundary = True  # try to predict boundary (if there are Node)
        self.use_cons_evt = False
        self.cons_evt_tok = "None"  # (lambda tok: return key) (lu -> frame)
        self.cons_evt_frame = "None"  # (lambda frame: return key) (lu -> frame); note: special one for given frame!
        self.cons_evt_hit_delta0 = 0.  # if hit, add what delta to idx0
        # special for frame filtering
        self.pred_evt_filter = ""
        self.pred_evt_check_layer = -1  # check repeat for how many layers of evt type (for example, if 1 then no repeat L1)
        # --
        # ef
        self.pred_ef_nil_add: float = None
        self.pred_given_ef = False  # predict with given ef (position)
        self.pred_ef_label = True  # try to predict ef label, otherwise leave it there or "UNK"
        self.pred_ef_boundary = True  # try to predict boundary (if there are Node)
        self.pred_ef_first = False  # first decode all efs, then link them as args!
        # --
        # arg
        self.pred_arg_nil_add: float = None
        self.arg_pp = PostProcessorConf()  # post-processor for arg
        self.use_cons_arg = False  # use arg-cons (frame -> arg)
        # --
        # special decode boundary batch size
        self.boundary_bsize = 1024
        # --

    @classmethod
    def _get_type_hints(cls):
        return {'pred_evt_nil_add': float, 'pred_ef_nil_add': float, 'pred_arg_nil_add': float, 'frames_file': 'zglob1'}

    # --
    def get_cons_evt_tok(self):
        _pre_build = {
            'lemma': (lambda t: None if t.lemma is None else t.lemma.lower()),
            'lemma0': (lambda t: t.lemma),
        }
        ret = _pre_build.get(self.cons_evt_tok, None)
        if ret is None:
            ret = eval(self.cons_evt_tok)
        return ret
    # --
    def get_cons_evt_frame(self):
        _pre_build = {'lu': (lambda f: f.info['luName'])}
        ret = _pre_build.get(self.cons_evt_frame, None)
        if ret is None:
            ret = eval(self.cons_evt_frame)
        return ret
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
        self.lu_cons, self.role_cons = None, None
        if conf.frames_name:  # currently only frame->role
            from msp2.data.resources import get_frames_label_budgets
            flb = get_frames_label_budgets(conf.frames_name)
            _voc_ef, _voc_evt, _voc_arg = dec.ztask.vpack
            _role_cons = fchelper.build_constraint_arrs(flb, _voc_arg, _voc_evt)
            self.role_cons = BK.input_real(_role_cons)
        if conf.frames_file:
            _voc_ef, _voc_evt, _voc_arg = dec.ztask.vpack
            _fc = default_pickle_serializer.from_file(conf.frames_file)
            _lu_cons = fchelper.build_constraint_arrs(fchelper.build_lu_map(_fc), _voc_evt, warning=False)  # lexicon->frame
            _role_cons = fchelper.build_constraint_arrs(fchelper.build_role_map(_fc), _voc_arg, _voc_evt)  # frame->role
            self.lu_cons, self.role_cons = _lu_cons, BK.input_real(_role_cons)
        # --
        self.cons_evt_tok_f = conf.get_cons_evt_tok()
        self.cons_evt_frame_f = conf.get_cons_evt_frame()
        if self.dec.conf.arg_use_bio:  # extend for bio!
            self.cons_arg_bio_sels = BK.input_idx(self.dec.vocab_bio_arg.get_bio2origin())
        else:
            self.cons_arg_bio_sels = None
        # --
        from msp2.data.resources.frames import KBP17_TYPES
        self.pred_evt_filter = {'kbp17': KBP17_TYPES}.get(conf.pred_evt_filter, None)
        # --

    # decode for one frame (evt/ef)
    def decode_frame(self, ibatch, scores_t: BK.Expr, pred_max_layer: int, voc, pred_label: bool, pred_tag: str,
                     pred_check_layer: int):
        # --
        # first get topk for each position
        logprobs_t = scores_t.log_softmax(-1)  # [*, dlen, L]
        pred_scores, pred_labels = logprobs_t.topk(pred_max_layer)  # [*, dlen, K]
        arr_scores, arr_labels = BK.get_value(pred_scores), BK.get_value(pred_labels)  # [*, dlen, K]
        # put results
        res_bidxes, res_widxes, res_frames = [], [], []  # flattened results
        res_farrs = np.full(arr_scores.shape, None, dtype=object)  # [*, dlen, K]
        for bidx, item in enumerate(ibatch.items):  # for each item in the batch
            _dec_offsets = item.seq_info.dec_offsets
            for sidx, sent in enumerate(item.sents):
                # todo(+N): currently we only predict for center if there is!
                if item.center_sidx is not None and sidx != item.center_sidx:
                    continue  # skip non-center sent in this mode!
                _start = _dec_offsets[sidx]
                _len = len(sent)
                _arr_scores, _arr_labels = arr_scores[bidx][_start:_start+_len], arr_labels[bidx][_start:_start+_len]
                for widx in range(_len):
                    _full_widx = widx + _start  # idx in the msent
                    _tmp_set = set()
                    for _k in range(pred_max_layer):
                        _score, _lab = float(_arr_scores[widx][_k]), int(_arr_labels[widx][_k])
                        if _lab == 0:  # note: lab=0 means NIL
                            break
                        _type_str = (voc.idx2word(_lab) if pred_label else "UNK")
                        _type_str_prefix = '.'.join(_type_str.split('.')[:pred_check_layer])
                        if pred_check_layer>=0 and _type_str_prefix in _tmp_set:
                            continue  # ignore since constraint
                        _tmp_set.add(_type_str_prefix)
                        # add new one!
                        res_bidxes.append(bidx)
                        res_widxes.append(_full_widx)
                        _new_frame = sent.make_frame(
                            widx, 1, tag=pred_tag, type=_type_str, score=float(_score))
                        _new_frame.set_label_idx(int(_lab))
                        _new_frame._tmp_sstart = _start  # todo(+N): ugly tmp value ...
                        _new_frame._tmp_sidx = sidx
                        _new_frame._tmp_item = item
                        res_frames.append(_new_frame)
                        res_farrs[bidx, _full_widx, _k] = _new_frame
        # return
        res_bidxes_t, res_widxes_t = BK.input_idx(res_bidxes), BK.input_idx(res_widxes)  # [??]
        return (res_bidxes_t, res_widxes_t), res_frames, res_farrs  # [??], [*, dlen, K]

    # assume given positions
    def decode_frame_given(self, ibatch, scores_t: BK.Expr, pred_max_layer: int, voc, pred_label: bool, pred_tag: str, assume_osof: bool):
        if pred_label:  # if overwrite label!
            logprobs_t = scores_t.log_softmax(-1)  # [*, dlen, L]
            pred_scores, pred_labels = logprobs_t.max(-1)  # [*, dlen], note: maximum!
            arr_scores, arr_labels = BK.get_value(pred_scores), BK.get_value(pred_labels)  # [*, dlen]
        else:
            arr_scores = arr_labels = None
        # --
        # read given results
        res_bidxes, res_widxes, res_frames = [], [], []  # flattened results
        tmp_farrs = defaultdict(list)  # later assign
        for bidx, item in enumerate(ibatch.items):  # for each item in the batch
            _trg_frames = [item.inst] if assume_osof else \
                sum([sent.get_frames(pred_tag) for sidx,sent in enumerate(item.sents)
                     if (item.center_sidx is None or sidx == item.center_sidx)],[])  # still only pick center ones!
            # --
            _dec_offsets = item.seq_info.dec_offsets
            for _frame in _trg_frames:  # note: simply sort by original order!
                sidx = item.sents.index(_frame.sent)
                _start = _dec_offsets[sidx]
                _full_hidx = _start+_frame.mention.shead_widx
                # add new one
                res_bidxes.append(bidx)
                res_widxes.append(_full_hidx)
                _frame._tmp_sstart = _start  # todo(+N): ugly tmp value ...
                _frame._tmp_sidx = sidx
                _frame._tmp_item = item
                res_frames.append(_frame)
                tmp_farrs[(bidx, _full_hidx)].append(_frame)
                # assign/rewrite label?
                if pred_label:
                    _lab = int(arr_labels[bidx, _full_hidx])  # label index
                    _frame.set_label_idx(_lab)
                    _frame.set_label(voc.idx2word(_lab))
                    _frame.set_score(float(arr_scores[bidx, _full_hidx]))
            # --
        # --
        res_farrs = np.full(BK.get_shape(scores_t)[:-1]+[pred_max_layer], None, dtype=object)  # [*, dlen, K]
        for _key, _values in tmp_farrs.items():
            bidx, widx = _key
            _values = _values[:pred_max_layer]  # truncate if more!
            res_farrs[bidx, widx, :len(_values)] = _values
        # return
        res_bidxes_t, res_widxes_t = BK.input_idx(res_bidxes), BK.input_idx(res_widxes)  # [??]
        return (res_bidxes_t, res_widxes_t), res_frames, res_farrs  # [??], [*, dlen, K]

    # decode arg (shead)
    def decode_arg(self, res_evts: List, arg_scores_t: BK.Expr, pred_max_layer: int, voc, arg_allowed_sent_gap: int, arr_efs):
        # first get topk
        arg_logprobs_t = arg_scores_t.log_softmax(-1)  # [??, dlen, L]
        pred_arg_scores, pred_arg_labels = arg_logprobs_t.topk(pred_max_layer)  # [??, dlen, K]
        arr_arg_scores, arr_arg_labels = BK.get_value(pred_arg_scores), BK.get_value(pred_arg_labels)  # [??, dlen, K]
        # put results
        res_fidxes, res_widxes, res_args = [], [], []  # flattened results
        for fidx, evt in enumerate(res_evts):  # for each evt
            item = evt._tmp_item  # cached
            _evt_sidx = evt._tmp_sidx
            _dec_offsets = item.seq_info.dec_offsets
            for sidx, sent in enumerate(item.sents):
                if abs(sidx - _evt_sidx) > arg_allowed_sent_gap:
                    continue  # larger than allowed sentence gap
                _start = _dec_offsets[sidx]
                _len = len(sent)
                _arr_scores, _arr_labels = arr_arg_scores[fidx][_start:_start+_len], arr_arg_labels[fidx][_start:_start+_len]
                for widx in range(_len):
                    _full_widx = widx + _start  # idx in the msent
                    _new_ef = None
                    if arr_efs is not None:  # note: arr_efs should also expand to frames!
                        _new_ef = arr_efs[fidx, _full_widx, 0]  # todo(+N): only get the first one!
                        if _new_ef is None:
                            continue  # no ef!
                    for _score, _lab in zip(_arr_scores[widx], _arr_labels[widx]):  # [K]
                        if _lab == 0:  # note: idx=0 means NIL
                            break
                        # add new one!!
                        res_fidxes.append(fidx)
                        res_widxes.append(_full_widx)
                        if _new_ef is None:
                            _new_ef = sent.make_entity_filler(widx, 1)  # share them if possible!
                        _new_arg = evt.add_arg(_new_ef, role=voc.idx2word(_lab), score=float(_score))
                        _new_arg._tmp_sstart = _start  # todo(+N): ugly tmp value ...
                        res_args.append(_new_arg)
        # return
        res_fidxes_t, res_widxes_t = BK.input_idx(res_fidxes), BK.input_idx(res_widxes)  # [??]
        return (res_fidxes_t, res_widxes_t), res_args

    # decode arg bio
    def decode_arg_bio(self, res_evts: List, arg_scores_t: BK.Expr, pred_max_layer: int, voc_bio, arg_allowed_sent_gap: int, arr_efs):
        assert pred_max_layer == 1, "Currently BIO only allow one!"
        assert arr_efs is None, "Currently BIO does not allow pick given efs!"
        _vocab_bio_arg = voc_bio
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
                if abs(sidx - _evt_sidx) > arg_allowed_sent_gap:
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

    # helper for assign boundaries: [???, dlen], [???, dlen, D], *[???]
    def assign_boundaries(self, items: List, boundary_node, flat_mask_t: BK.Expr, flat_hid_t: BK.Expr, indicators: List):
        flat_indicators = boundary_node.prepare_indicators(indicators, BK.get_shape(flat_mask_t))
        # --
        _bsize, _dlen = BK.get_shape(flat_mask_t)  # [???, dlen]
        _once_bsize = max(1, int(self.conf.boundary_bsize / max(1, _dlen)))
        # --
        if _once_bsize >= _bsize:
            _, _left_idxes, _right_idxes = boundary_node.decode(flat_hid_t, flat_mask_t, flat_indicators)  # [???]
        else:
            _all_left_idxes, _all_right_idxes = [], []
            for ii in range(0, _bsize, _once_bsize):
                _, _one_left_idxes, _one_right_idxes = boundary_node.decode(
                    flat_hid_t[ii:ii+_once_bsize], flat_mask_t[ii:ii+_once_bsize], [z[ii:ii+_once_bsize] for z in flat_indicators])
                _all_left_idxes.append(_one_left_idxes)
                _all_right_idxes.append(_one_right_idxes)
            _left_idxes, _right_idxes = BK.concat(_all_left_idxes, 0), BK.concat(_all_right_idxes, 0)
        _arr_left, _arr_right = BK.get_value(_left_idxes), BK.get_value(_right_idxes)
        for ii, item in enumerate(items):
            _mention = item.mention
            _start = item._tmp_sstart  # need to minus this!!
            _left_widx, _right_widx = _arr_left[ii].item()-_start, _arr_right[ii].item()-_start
            # todo(+N): sometimes we can have repeated ones, currently simply over-write!
            if _mention.get_span()[1] == 1:
                _mention.set_span(*(_mention.get_span()), shead=True)  # first move to shead!
            _mention.set_span(_left_widx, _right_widx-_left_widx+1)
        # --

    # constraints
    # [*, dlen, L]
    def cons_score_lu2frame(self, evt_scores: BK.Expr, ibatch, given_f=None):
        dlen, llen = BK.get_shape(evt_scores)[-2:]
        _tok_f, _frame_f = self.cons_evt_tok_f, self.cons_evt_frame_f
        # --
        res = []
        for bidx, item in enumerate(ibatch.items):
            _dec_offsets = item.seq_info.dec_offsets
            one_res = [np.zeros(llen) for _ in range(dlen)]  # dlen*[L]
            for sidx, sent in enumerate(item.sents):
                _start = _dec_offsets[sidx]
                # tok
                if _tok_f is not None:
                    for widx, tok in enumerate(sent.get_tokens()):
                        _key = _tok_f(tok)
                        _arr = self.lu_cons.get(_key)
                        if _arr is not None:
                            one_res[_start+widx] += _arr
                # frame
                if _frame_f is not None and given_f is not None:
                    _frames = given_f(sent)
                    for ff in _frames:
                        _key = _frame_f(ff)
                        _arr = self.lu_cons.get(_key)
                        if _arr is not None:
                            one_res[_start+ff.mention.shead_widx] += _arr
            # --
            res.extend(one_res)
        # --
        valid_mask = BK.input_real(res).view(evt_scores.shape)  # [*, dlen, L]
        valid_mask.clamp_(max=1)
        hit_t = (valid_mask.sum(-1, keepdims=True)>0).float()  # at least one hit!
        exclude_mask = (1 - valid_mask) * hit_t  # no effects if no-hit!
        exclude_mask[:, :, 0] = 0.  # note: still do not exclude NIL, use "pred_evt_nil_add" for special mode!
        evt_scores = evt_scores + exclude_mask * Constants.REAL_PRAC_MIN  # [*, dlen, L]
        # --
        _d0 = self.conf.cons_evt_hit_delta0
        if _d0 != 0.:
            d_scores = hit_t * _d0  # [*, dlen, 1]
            d_scores = BK.pad(d_scores, [0, evt_scores.shape[-1]-1])  # [*, dlen, L]
            evt_scores = evt_scores + d_scores
        # --
        return evt_scores

    # [??, dlen, L], [??]
    def cons_score_frame2role(self, arg_scores: BK.Expr, evts: List):
        evt_idxes = [e.label_idx for e in evts]
        valid_mask = BK.input_real(self.role_cons[evt_idxes])  # [??, L]
        if self.cons_arg_bio_sels is not None:
            valid_mask = valid_mask[:, self.cons_arg_bio_sels]  # [??, L']
        valid_mask[:, 0] = 1.  # note: must preserve NIL!
        arg_scores = arg_scores + (1-valid_mask).unsqueeze(-2)*Constants.REAL_PRAC_MIN  # [??, dlen, L]
        return arg_scores

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
        # first predict the evt/ef frames
        frame_results = []
        # --
        _evt_settings = ['evt', dec.lab_evt, conf.pred_evt_nil_add, conf.pred_given_evt, dec.boundary_evt, dec_conf.max_layer_evt, dec.voc_evt, conf.pred_evt_label, conf.pred_evt_boundary, dec_conf.assume_osof, conf.pred_evt_check_layer]
        _ef_settings = ['ef', dec.lab_ef, conf.pred_ef_nil_add, conf.pred_given_ef, dec.boundary_ef, dec_conf.max_layer_ef, dec.voc_ef, conf.pred_ef_label, conf.pred_ef_boundary, False, -1]
        if not conf.pred_ef_first:
            _ef_settings[0] = None  # no predicting efs first!!
        # --
        for pred_tag, node_lab, pred_nil_add, pred_given, node_boundary, pred_max_layer, voc, \
            pred_label, pred_boundary, assume_osof, pred_check_layer in [_evt_settings, _ef_settings]:
            if pred_tag is None:
                frame_results.append((None, None, None))
                continue
            # --
            if pred_given:
                for item in ibatch.items:
                    set_ee_heads(item.sents)  # we may need a head-widx later!
            # --
            score_cache = med.get_cache((dec.name, pred_tag))
            scores_t = node_lab.score_labels(score_cache.vals, nil_add_score=pred_nil_add)  # [*, dlen, L]
            if pred_tag == 'evt' and conf.use_cons_evt:  # modify scores by constraints
                scores_t = self.cons_score_lu2frame(scores_t, ibatch, given_f=((lambda s: s.events) if pred_given else None))
            # -> ([??], [??]), [??], [*, dlen, K]
            if pred_given:
                res = self.decode_frame_given(ibatch, scores_t, pred_max_layer, voc, pred_label, pred_tag, assume_osof)
            else:
                res = self.decode_frame(ibatch, scores_t, pred_max_layer, voc, pred_label, pred_tag, pred_check_layer)
            # boundary?
            res_idxes_t, res_frames, _ = res
            if pred_boundary and node_boundary is not None and len(res_frames)>0:
                _flat_mask_t = (_ds_idxes[res_idxes_t[0]] == _ds_idxes[res_idxes_t].unsqueeze(-1)).float()  # [??, dlen]
                _flat_hid_t = hid_t[res_idxes_t[0]]  # [??, dlen, D]
                self.assign_boundaries(res_frames, node_boundary, _flat_mask_t, _flat_hid_t, [res_idxes_t[1]])
            # --
            frame_results.append(res)
        # --
        # then predict args!
        res_evt_idxes_t, res_evts, _ = frame_results[0]  # evt results!
        if len(res_evts) == 0:
            return  # note: no need to do anything!
        _, _, arr_efs = frame_results[1]  # ef results!
        if arr_efs is not None:
            arr_efs = arr_efs[BK.get_value(res_evt_idxes_t[0])]  # change to fidx at first: [??, dlen, K]
        # --
        arg_score_cache = med.get_cache((dec.name, 'arg'))
        # --
        base_mask_t = dec.get_dec_mask(ibatch, dec_conf.msent_pred_center)  # [bs, dlen]
        arg_seq_mask = base_mask_t.unsqueeze(-2).expand(-1, BK.get_shape(base_mask_t, -1), -1)  # [bs, dlen, dlen]
        # --
        arg_scores_t = dec.lab_arg.score_labels(
            arg_score_cache.vals, seq_mask_t=arg_seq_mask, preidx_t=res_evt_idxes_t,
            nil_add_score=conf.pred_arg_nil_add)  # [??, dlen, L]
        if conf.use_cons_arg:  # modify scores by constraints
            arg_scores_t = self.cons_score_frame2role(arg_scores_t, res_evts)
        _pred_max_layer, _arg_allowed_sent_gap = dec_conf.max_layer_arg, dec_conf.arg_allowed_sent_gap
        if dec_conf.arg_use_bio:  # if use BIO, then checking the seq will be fine
            self.decode_arg_bio(res_evts, arg_scores_t, _pred_max_layer, dec.vocab_bio_arg, _arg_allowed_sent_gap, arr_efs)
        else:  # otherwise, still need two steps
            res_arg_idxes_t, res_args = self.decode_arg(res_evts, arg_scores_t, _pred_max_layer, dec.voc_arg, _arg_allowed_sent_gap, arr_efs)  # [???]
            # arg(ef) boundary
            if dec.boundary_arg is not None and len(res_args)>0 and arr_efs is None:
                _ab_fidxes_t, _ab_awidxes_t = res_arg_idxes_t  # [???]
                _ab_bidxes_t, _ab_ewidxes_t = [z[_ab_fidxes_t] for z in res_evt_idxes_t]  # [???]
                _ab_mask_t = (_ds_idxes[_ab_bidxes_t] == _ds_idxes[_ab_bidxes_t,_ab_awidxes_t].unsqueeze(-1)).float()
                flat_hid_t = hid_t[_ab_bidxes_t]  # [???, dlen, D]
                self.assign_boundaries(res_args, dec.boundary_arg, _ab_mask_t, flat_hid_t, [_ab_ewidxes_t, _ab_awidxes_t])
        # --
        # final arg post-process
        if self.pred_evt_filter:
            for evt in list(res_evts):
                if evt.type not in self.pred_evt_filter:
                    evt.sent.delete_frame(evt, 'evt')
        for evt in res_evts:
            self.arg_pp.process(evt)  # modified inplace!
        # --
        return

    # --

# --
# b msp2/tasks/zmtl2/zmod/dec/dec_srl/inference2:??
