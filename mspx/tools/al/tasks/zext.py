#

# for frame extraction

__all__ = [
    "ALZextConf", "ALZextHelper",
]

import os
from collections import Counter, defaultdict
from mspx.data.inst import Doc, yield_sents, yield_frames, Frame
from mspx.utils import zlog, zwarn, ZHelper, ZObject, mkdir_p
from .base import *

class ALZextConf(ALTaskConf):
    def __init__(self):
        super().__init__()
        # --
        from mspx.tasks.zext.mod import ZTaskExtConf
        self.tconf = ZTaskExtConf().direct_update(name='ext0')
        # specific options
        self.ann_full_frames = True  # annotate full frames upon tokens

@ALZextConf.conf_rd()
class ALZextHelper(ALTaskHelper):
    def __init__(self, conf: ALZextConf, **kwargs):
        super().__init__(conf, **kwargs)
        # --
        conf: ALZextConf = self.conf
        self.name = conf.tconf.name
        assert len(self.name) > 0, "Must have a name!"
        # note: from ExtSeqlabMod
        self.KEY_PA = f"{self.name}_ispart"  # whether partial
        self.KEY_STRG0 = f"{self.name}_strg" + "0"  # soft target0: unary!
        self.KEY_STRG1 = f"{self.name}_strg" + "1"  # soft target1: binary!
        self.KEY_CALI = f"{self.name}_cali"  # for calibration!
        self.KEY_HID = f"repr_hid"  # hidden layer
        self.LAB_QUERY = '_Q_'
        self.LAB_NIL = '_NIL_'
        self.PAST_PREFIX = "**"  # prefix for past ann
        self.selector = conf.selector.make_node()
        self._init_specs()
        # --

    # --
    # helper

    # yield items from doc
    def _yield_items(self, doc, cc, yield_tok=False, yield_sent=False, skip_ann=False, yield_frame=False):
        conf: ALZextConf = self.conf
        frame_cate = conf.tconf.frame_cate
        _lab_nils = [self.LAB_NIL, self.PAST_PREFIX + self.LAB_NIL]
        # --
        assert isinstance(doc, Doc)
        cc['doc'] += 1
        for sent in yield_sents(doc):
            _len = len(sent)
            cc['sent'] += 1
            cc['tok'] += _len
            # check sent flag
            if sent.info.get(self.KEY_PA, False):  # is partial
                flag_ann = [0] * len(sent)
            else:  # already finished
                flag_ann = [1] * len(sent)
            # check frames
            for frame in sent.get_frames(cates=frame_cate):
                if frame.info.get('is_pred', False):
                    continue  # note: ignore predicted frames!
                ii0, wlen = frame.mention.get_span()
                ii1 = ii0 + wlen
                flag_ann[ii0:ii1] = [1] * wlen
                cc['frame'] += 1
                cc[f'frame_NIL'] += int(frame.label in _lab_nils)  # whether frame is NIL
                if frame.label == self.LAB_QUERY:
                    cc['frame_Q'] += 1
                    flag_ann[ii0:ii1] = [2] * wlen  # special mark!
                cc['tok_frame'] += wlen
                if yield_frame:
                    f_cand = ZObject(type='frame', gid=id(sent), frame=frame, budget=wlen)
                    yield f_cand
            # check tokens
            cc['tok_ann'] += sum([int(z>0) for z in flag_ann])
            cc['tok_unn'] += _len - sum([int(z>0) for z in flag_ann])
            if yield_tok:
                for widx, ann in enumerate(flag_ann):
                    if ann == 2:
                        cc['tok_annQ'] += 1
                    is_ann = (ann > 0)
                    if (not is_ann) or (not skip_ann):  # get one candidate
                        tok_cand = ZObject(type='tok', gid=id(sent), sent=sent, widx=widx,
                                           budget=1, budgetR=(1-int(is_ann)))
                        yield tok_cand
            # sent
            _num_ann = sum(flag_ann)
            cc['sentU'] += int(_num_ann == 0)  # empty Unannotated
            cc['sentA'] += int(_num_ann == _len)  # full Annotated
            cc['sentP'] += int(_num_ann > 0 and _num_ann < _len)  # partial
            if yield_sent and (not (skip_ann and _num_ann == _len)):
                sent_cand = ZObject(type='sent', gid=id(doc), sent=sent, budget=_len, budgetR=_len-_num_ann)
                yield sent_cand
        # --

    @property
    def main_entry(self):
        return "mspx.tasks.zext.main"

    @property
    def frame_cate(self):
        return self.conf.tconf.frame_cate

    # --
    # 0. setup

    # process inst at setup (inplace!)
    def setup_inst(self, inst: Doc, mark_unn: bool, mark_past=False):
        conf: ALZextConf = self.conf
        cc = Counter()
        all_objs = self._yield_items(inst, cc, yield_frame=True, yield_sent=True)
        for obj in all_objs:
            if mark_past:
                if obj.type == 'sent':
                    pass  # no need to change sent's info
                elif obj.type == 'frame':
                    if obj.frame.label != self.LAB_QUERY:
                        obj.frame.set_label(self.PAST_PREFIX + obj.frame.label)
                else:
                    raise RuntimeError()
            if mark_unn:  # extra operations!
                if obj.type == 'sent':
                    obj.sent.info[self.KEY_PA] = True  # mark partial!
                elif obj.type == 'frame':
                    obj.frame.del_self()  # delete existing ones!
                else:
                    raise RuntimeError()
        return cc

    # --
    # 1. query

    # yielding and add arrs!
    def _yield_cands(self, doc, cc, partial: bool):
        for one_cand in self._yield_items(doc, cc, yield_tok=partial, yield_sent=(not partial), skip_ann=True):
            # [L, V], [1+L, D]
            sent = one_cand.sent
            arr_strg, arr_hid = sent.arrs.get(self.KEY_STRG0), sent.arrs.get(self.KEY_HID)
            one_cand.score_cand = None
            if arr_strg is not None:
                one_cand.arr_strg = arr_strg[one_cand.widx].copy() if partial else arr_strg
            if arr_hid is not None:  # CLS or 1+widx
                one_cand.arr_hid = arr_hid[(1+one_cand.widx) if partial else 0].copy()
            yield one_cand

    # obtain all cands
    def _obtain_all_cands(self, data_stream, partial: bool):
        cc = Counter()
        all_docs = []
        cands = []  # List of unann candidates
        for doc in data_stream:
            all_docs.append(doc)  # store it!
            for one_cand in self._yield_cands(doc, cc, partial):
                cands.append(one_cand)
        return all_docs, cands, cc

    # actual querying
    def _do_query(self, data_stream, dev_stream, ref_stream=None, refD_stream=None, no_strg=False, repr_helper=None):
        conf: ALZextConf = self.conf
        frame_cate = conf.tconf.frame_cate
        cc = Counter()
        _query_partial = conf.curr_is_partial(self.curr_iter)
        _score_randomly = (no_strg or (not conf.query_use_unc))
        # --
        if not conf.query_selv2:
            all_docs, cands, cc0 = self._obtain_all_cands(data_stream, _query_partial)
            cc += cc0
            query_cands = self.selector.select(cands, self.curr_budget, conf.budget_group, score_randomly=_score_randomly)
        else:
            all_docs, cands, cc0 = self._obtain_all_cands(data_stream, True)  # first get all tokens
            cc += cc0
            all_sents = yield_sents(all_docs)
            if conf.selv2_only_empty:
                all_s2_sents = [z for z in all_sents if not any(not z2.info.get('is_pred', False) for z2 in z.get_frames(cates=frame_cate))]
            else:
                all_s2_sents = all_sents
            cand_items = [cands]
            cand_sc = [conf.selv2_sc0]
            cand_ratios = [self._get_curr_val(conf.selv2_ratios0)]
            cand_threshs = [self.curr_selv2_ths[0]]
            comb_params = [1.]
            dev_items = None if dev_stream is None else [self._yield_dev_cands(dev_stream, self.KEY_CALI)]
            _final_cands = self.selector.select_v2(all_s2_sents, [self.curr_budget, conf.budget_group], None, cand_items, dev_items, cand_sc, cand_ratios, cand_threshs, _score_randomly, _query_partial, conf.selv2_ratio_sentwise, 'score', comb_params, ref_helper=self._get_ref_helper(ref_stream), repr_helper=repr_helper)
            query_cands = _final_cands[0] if _query_partial else _final_cands  # fit here!
        # --
        # then prepare the query insts
        for doc in all_docs:  # whether clear all?
            self.setup_inst(doc, mark_unn=(not conf.qann_with_past), mark_past=conf.qann_with_past)
        cc['q_candA'] += len(cands)
        cc['q_candQ'] += len(query_cands)
        hit_sents = {}  # (doc.id, sid) -> sent
        hit_toks = set()
        for one_cand in query_cands:  # note: here do not care which frame_cate!
            cc['q_budget'] += one_cand.budget
            hit_sents[one_cand.sent.dsids] = one_cand.sent
            if one_cand.type == 'tok':  # each tok individual!
                frame = one_cand.sent.make_frame(one_cand.widx, 1, self.LAB_QUERY, frame_cate)
                hit_toks.add((id(one_cand.sent), one_cand.widx))
            elif one_cand.type == 'span':  # mark the span
                frame = one_cand.sent.make_frame(one_cand.span[0], one_cand.span[1], self.LAB_QUERY, frame_cate)
                hit_toks.update([(id(one_cand.sent), z) for z in range(one_cand.span[0], one_cand.span[0]+one_cand.span[1])])
            else:  # simply query a full one
                assert one_cand.type == 'sent'
                frame = one_cand.sent.make_frame(0, len(one_cand.sent), self.LAB_QUERY, frame_cate)
                hit_toks.update([(id(one_cand.sent), z) for z in range(len(one_cand.sent))])
            frame.score = one_cand.score_cand  # set utility score to frame.score!
        hit_doc_ids = set(z[0] for z in hit_sents.keys())
        if conf.qann_with_hit:
            ret_docs = [d for d in all_docs if d.id in hit_doc_ids]  # if there are queries
        else:
            ret_docs = all_docs  # still put them all
        cc['q_doc'] += len(hit_doc_ids)
        cc['q_sent'] += len(hit_sents)
        cc['q_tok'] += len(hit_toks)
        assert cc['q_tok'] == cc['q_budget']  # no overlapping!
        return ret_docs, hit_sents, cc

    # --
    # 2./3. ann & comb

    # check sent frames
    def _get_tok2frames(self, sent, frame_cate: str):
        ret = [[] for _ in range(len(sent))]
        for frame in sent.yield_frames(cates=frame_cate):  # note: no need copy-list
            widx, wlen = frame.mention.get_span()
            for ii in range(widx, widx+wlen):
                ret[ii].append(frame)
        return ret

    # simul annotate frame; return cost
    def _simul_ann_frame(self, q_frame, ref_map, hit_ones, hit_sents, frame_marks=None):
        conf: ALZextConf = self.conf
        frame_cate = conf.tconf.frame_cate
        _frame_marks = [] if frame_marks is None else frame_marks
        # --
        # annotate:
        q_sent = q_frame.sent
        hit_sents.add(id(q_sent))
        r_sent = ref_map[q_sent.doc.id].sents[q_sent.sid]  # must be there
        r_t2f = self._get_tok2frames(r_sent, frame_cate)
        q_widx, q_wlen = q_frame.mention.get_span()
        q_cost = 0
        for ii in range(q_widx, q_widx + q_wlen):
            _key = (id(q_sent), ii)
            if _key in hit_ones:  # already hit!
                continue
            hit_ones.add(_key)
            # --
            _rfs = r_t2f[ii]
            if len(_rfs) == 0:  # make an NIL if no frames
                q_sent.make_frame(ii, 1, self.LAB_NIL, frame_cate)
                q_cost += 1
            else:
                if len(_rfs) > 1:
                    zwarn(f"Meet overlapping frames: {_rfs}")
                if conf.ann_full_frames:  # annotate full frames!
                    for _rf in _rfs:
                        if id(_rf) in hit_ones:
                            continue  # no repeating!
                        hit_ones.add(id(_rf))
                        r_widx, r_wlen = _rf.mention.get_span()
                        newf = q_sent.make_frame(r_widx, r_wlen, _rf.label, _rf.cate)
                        newf.info.update(_rf.info)  # update info
                        for _mm in _frame_marks:
                            newf.info[_mm] = True
                        q_cost += r_wlen
                else:  # "annotate" partial frames (no checking repeating!)
                    assert len(_rfs) == 1, "Only allow one frame in partial-frame mode!"
                    _rf = _rfs[0]
                    r_widx, r_wlen = _rf.mention.get_span()
                    _prefix = "B-" if r_widx == ii else "I-"  # todo(+2): simply use BIO!
                    newf = q_sent.make_frame(ii, 1, _prefix + _rf.label, _rf.cate)
                    for _mm in _frame_marks:
                        newf.info[_mm] = True
                    q_cost += 1
        # --
        return q_cost

    # simulated ann according to ref
    def do_simul_ann(self, query_insts, ref_map, last_model=None):
        conf: ALZextConf = self.conf
        frame_cate = conf.tconf.frame_cate
        cc = Counter()
        # --
        # to be annotated
        all_frames = [z for z in yield_frames(query_insts, cates=frame_cate) if z.label==self.LAB_QUERY]
        all_frames.sort(key=(lambda f: f.score), reverse=True)
        # iter frames according to utility score
        all_budget = self.curr_budget
        remaining_budget = all_budget
        hit_ones = set()  # already hit toks or frames
        hit_sents = set()
        for q_frame in all_frames:
            cc['a_Fall'] += 1
            q_frame.del_self()  # no matter what delete self!
            if remaining_budget <= 0:  # skip since no budgets
                cc['a_Fdel'] += 1
            else:
                cc['a_Fann'] += 1
                q_cost = self._simul_ann_frame(q_frame, ref_map, hit_ones, hit_sents)
                remaining_budget -= q_cost
        # --
        cc['q_sent_hit'] = len(hit_sents)
        cc['budgetC'] = all_budget - remaining_budget  # cost
        cc['budgetR'] = remaining_budget  # could be negative but could be neglected!
        return cc

    # score the matching of two frames
    def _score_match(self, f1, f2):
        start1, len1 = f1.mention.get_span()
        start2, len2 = f2.mention.get_span()
        overlap = min(start1 + len1, start2 + len2) - max(start1, start2)
        overlap = max(0, overlap)  # overlapped tokens
        posi_match = overlap / (len1 + len2 - overlap)  # using Jaccard Index
        # remove PAST_PREFIX for label checking!
        _pp = self.PAST_PREFIX
        lab1, lab2 = [(z[len(_pp):] if z.startswith(_pp) else z) for z in [f1.label, f2.label]]
        lab_match = float(f"{f1.cate}_{lab1}" == f"{f2.cate}_{lab2}")
        return (posi_match, lab_match)

    # find a frame match
    def _match_frame(self, q_frame, r_sent, frame_cate):
        r_t2f = self._get_tok2frames(r_sent, frame_cate)
        q_widx, q_wlen = q_frame.mention.get_span()
        best_ref, best_ref_score = None, (0., 0.)  # best-ref frame, match-(posi,label)
        for ii in range(q_widx, q_widx + q_wlen):  # at least span overlap!
            for _rf in r_t2f[ii]:
                _score = self._score_match(q_frame, _rf)
                if _score > best_ref_score:
                    best_ref, best_ref_score = _rf, _score
        return best_ref, best_ref_score  # None for no matching!

    # combine a new frame
    def _comb_frame(self, a_frame, t_sent, cc):
        _pp = self.PAST_PREFIX
        _widx, _wlen = a_frame.mention.get_span()
        _label = a_frame.label
        # --
        ret_frame = None
        has_new_ann = False
        if _label.startswith(_pp):  # check previous existing
            _aa_frame, _aa_score = self._match_frame(a_frame, t_sent, a_frame.cate)
            if _aa_frame is None:  # no match?
                zwarn(f"Ignoring since no past annotation for: {a_frame}")
                cc['combF_aaN'] += 1
            elif _aa_score != (1., 1.):
                zwarn(f"Correction against past annotation: {a_frame} vs {_aa_frame}")
                _aa_frame.mention.set_span(_widx, _wlen)
                _aa_frame.set_label(_label[len(_pp):])
                _aa_frame.clear_cached_vals()
                cc['combF_aaC'] += 1
                has_new_ann = True
            else:
                cc['combF_aa'] += 1  # already annotated
            ret_frame = _aa_frame  # simply the first one!
        elif _label == self.LAB_QUERY or a_frame.info.get('is_pred', False):  # ignore unannotated ones
            cc['combF_unn'] += 1
        else:  # add new one
            has_new_ann = True
            ret_frame = t_sent.make_frame(_widx, _wlen, _label, a_frame.cate)
            ret_frame.info.update(a_frame.info)
            not_nil = (ret_frame.label != self.LAB_NIL)
            if not_nil:  # not-NIL ones
                cc['combF_nnew'] += 1
                cc['combF_nnewx2'] += 2  # for ACE-REL
                cc['combF_nnew_len'] += _wlen
            cc['combF_new'] += 1
            for _tok in ret_frame.mention.get_tokens():
                cc['combF_new_len'] += 1
                if not_nil or _tok.upos in {'PROPN', 'ADJ'}:  # for NER-C03-en
                    cc['combF_pN0new_len'] += 1
                if not_nil or _tok.upos in {'PROPN', 'NOUN', 'ADJ'}:  # for NER?
                    cc['combF_pN1new_len'] += 1
                if not_nil or _tok.upos in {'PROPN', 'NUM', 'NOUN', 'ADJ'}:  # for NER-onto
                    cc['combF_pN2new_len'] += 1
                if not_nil or _tok.upos in {'PRON', 'PROPN', 'NOUN', 'ADJ', 'VERB'}:  # for evt/ent(ACE)
                    cc['combF_pEnew_len'] += 1
                if not_nil or _tok.upos in {'PRON', 'PROPN', 'NOUN', 'ADJ'}:  # for rel-ent(ACE)
                    cc['combF_pRnew_len'] += 1
        return ret_frame

    # combine new insts into trg
    def _do_comb(self, ann_inst, trg_map, cc):
        conf: ALZextConf = self.conf
        for one in self._yield_items(ann_inst, cc, yield_frame=True):
            a_frame = one.frame
            a_sent = a_frame.sent
            t_sent = trg_map[a_sent.doc.id].sents[a_sent.sid]  # must be there
            self._comb_frame(a_frame, t_sent, cc)
        # --

    # --
    # 5. training
    # ...

# --
# b mspx/tools/al/tasks/zext:174
