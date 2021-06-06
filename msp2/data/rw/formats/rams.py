#

# RAMS

from typing import Dict
import json
import math
import re
from .base import DataFormator
from msp2.data.inst import Doc, Sent, Mention

@DataFormator.reg_decorator("rams")
class RamsDataFormator(DataFormator):
    _OTHER_DOC_FIELDS = ['language_id', 'source_url', 'split']

    def to_obj(self, inst: Doc) -> str:
        info = inst.info
        d = {"doc_key": inst.id}
        d.update({k: info.get(k) for k in RamsDataFormator._OTHER_DOC_FIELDS})
        d.update({"sentences": [], "ent_spans": [], "evt_triggers": [], "gold_evt_links": []})
        # add sents
        _accu_tidx = 0
        sent_offsets = []
        sent_offset_maps = {}
        for sent in inst.sents:
            d["sentences"].append(sent.seq_word.vals)
            sent_offsets.append(_accu_tidx)
            sent_offset_maps[id(sent)] = _accu_tidx
            _accu_tidx += len(d["sentences"][-1])
        # -----
        def _get_span(_m: Mention):
            _soff = sent_offset_maps[id(_m.sent)]
            _wid, _wlen = _m.widx, _m.wlen
            return _soff+_wid, _soff+_wid+_wlen-1
        # --
        # add them
        for sent in inst.sents:
            for ent in sent.entity_fillers:
                ent_span = _get_span(ent.mention)
                d["ent_spans"].append([ent_span[0], ent_span[1], [[a.role, math.exp(a.score)] for a in ent.as_args]])
            for evt in sent.events:
                evt_span = _get_span(evt.mention)
                d["evt_triggers"].append([evt_span[0], evt_span[1], [[evt.type, math.exp(evt.score)]]])
                for arg in evt.args:
                    arg_span = _get_span(arg.arg.mention)
                    d["gold_evt_links"].append([[evt_span[0], evt_span[1]], [arg_span[0], arg_span[1]], arg.role])
        # --
        return json.dumps(d)

    def from_obj(self, s: str):
        d = json.loads(s)
        doc = Doc.create(id=d["doc_key"])
        doc.info.update({k: d.get(k) for k in RamsDataFormator._OTHER_DOC_FIELDS})
        # add sents
        _accu_tidx = 0
        tok_idx = []
        for toks in d["sentences"]:
            sent = Sent.create(toks)
            doc.add_sent(sent)
            for _i, _ in enumerate(toks):
                tok_idx.append((_i, sent))
            _accu_tidx += len(toks)
        # -----
        def _get_span(_e):
            _i_begin, _sent_begin = tok_idx[_e[0]]
            _i_endm1, _sent_endm1 = tok_idx[_e[1]]
            assert _sent_begin is _sent_endm1
            _wid, _wlen = _i_begin, _i_endm1-_i_begin+1
            return _wid, _wlen, _sent_begin
        def _norm_role(_r):
            return re.split(r'\d+', _r)[-1]
        # --
        # add entities
        ent_map = {}
        for e in d["ent_spans"]:
            _span_key = (e[0], e[1])
            wid, wlen, sent = _get_span(_span_key)
            ent = sent.make_entity_filler(wid, wlen, type="UNK")  # todo(note): ignore ent types!
            ent_map[_span_key] = ent
        # add events
        evt_map = {}
        assert len(d["evt_triggers"]) == 1, "Currently only one evt per Inst!"
        for e in d["evt_triggers"]:
            _span_key = (e[0], e[1])
            wid, wlen, sent = _get_span(_span_key)
            assert len(e[-1])==1, "Currently only one type per evt!"
            for type, score in e[-1]:
                evt = sent.make_event(wid, wlen, type=type, score=math.log(score))
                evt_map[_span_key] = evt
        # add args
        for link in d["gold_evt_links"]:
            _span_evt, _span_ent, role = link[:3]
            evt, ent = evt_map[tuple(_span_evt)], ent_map[tuple(_span_ent)]  # must be there
            role = _norm_role(role)
            evt.add_arg(ent, role)
        # --
        return doc
