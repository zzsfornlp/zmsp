#

# Old Zie format of Doc
# -- merge entities and fillers

from typing import Dict
import json
from .base import DataFormator
from msp2.utils import zwarn
from msp2.data.inst import Doc, Sent, Mention, Frame

@DataFormator.reg_decorator("zdoc")
class ZDocDataFormator(DataFormator):
    _OTHER_DOC_FIELDS = ['extra_info', 'dataset', 'source', 'lang']

    def to_obj(self, inst: Doc) -> str:
        info = inst.info
        d = {"doc_id": inst.id, "extra_info": info}
        d.update({k: info.get(k) for k in ZDocDataFormator._OTHER_DOC_FIELDS})
        d.update({"sents": [], "entity_mentions": [], "event_mentions": []})
        # -----
        def _get_posi(sid: int, _m: Mention):
            _widx, _wlen = _m.widx, _m.wlen
            return [(sid, z) for z in range(_widx, _widx+_wlen)]
        # --
        # mark no items?
        if all(sent.entity_fillers is None for sent in inst.sents):
            d["entity_mentions"] = None
        if all(sent.events is None for sent in inst.sents):
            d["event_mentions"] = None
        # stablize all ids
        for sent in inst.sents:
            for field in ["entity_fillers", "events"]:
                f_items = getattr(sent, field)
                if f_items is not None:
                    for f in f_items:
                        s_id, f_id = sent.id, f.id
                        if not f_id.startswith(s_id):  # avoid accumulation
                            f.set_id(f"{s_id}_{f_id}")
        # ---
        for sid, sent in enumerate(inst.sents):
            # TODO(+W): write other fields?
            _snt = {"text": sent.seq_word.vals}
            if sent.id is not None:
                _snt["id"] = sent.id
            d["sents"].append(_snt)
            # --
            if sent.entity_fillers is not None:
                for ent in sent.entity_fillers:
                    d_ef = {"id": ent.id, "posi": _get_posi(sid, ent.mention),
                            "type": ent.type, "score": ent.score}
                    d_ef.update({k: ent.info.get(k) for k in ["extra_info", "gid"] if k in ent.info})
                    d["entity_mentions"].append(d_ef)
            if sent.events is not None:
                for evt in sent.events:
                    d_evt = {"id": evt.id, "trigger": {"posi": _get_posi(sid, evt.mention)},
                             "type": evt.type, "score": evt.score, "extra_info": evt.info}
                    d_evt.update({k: evt.info.get(k) for k in ["extra_info", "gid", "realis", "realis_score"] if k in evt.info})
                    if evt.args is None:
                        d_evt["em_arg"] = None
                    else:
                        d_evt["em_arg"] = []
                        for arglink in evt.args:
                            d_arg = {"aid": arglink.arg.id, "role": arglink.role, "score": arglink.score}
                            d_arg.update({k: arglink.info.get(k) for k in ["is_aug", "extra_info"] if k in arglink.info})
                            d_evt["em_arg"].append(d_arg)
                    d["event_mentions"].append(d_evt)
        # --
        return json.dumps(d)

    def from_obj(self, s: str) -> Doc:
        d = json.loads(s)
        doc = Doc.create(id=d["doc_id"])
        doc.info.update({k: d.get(k) for k in ZDocDataFormator._OTHER_DOC_FIELDS})
        # add sents
        for one_sent in d["sents"]:
            sent = Sent.create(one_sent["text"], id=one_sent.get("id"))
            if "positions" in one_sent:
                sent.build_word_positions(one_sent["positions"])
            if "lemma" in one_sent:
                sent.build_lemmas(one_sent["lemma"])
            if "upos" in one_sent:
                sent.build_uposes(one_sent["upos"])
            if "governor" in one_sent and "dependency_relation" in one_sent:
                sent.build_dep_tree(one_sent["governor"], one_sent["dependency_relation"])
            doc.add_sent(sent)
        # --
        failed_items = {"ef": [], "evt": [], "arg": []}
        args_maps = {}  # id -> Frame
        # entities and fillers
        if d.get("entity_mentions") is None and d.get("fillers") is None:
            # no entities info
            for sent in doc.sents:
                sent.mark_no_entity_fillers()
        else:
            ef_items = d.get("entity_mentions", []) + d.get("fillers", [])
            for one_ef_item in ef_items:
                mention = self._parse_mention(one_ef_item, doc)
                if mention is None:
                    failed_items["ef"].append(one_ef_item)
                else:
                    ef = Frame.create(mention, type=one_ef_item["type"], score=one_ef_item.get("score", 0.), id=one_ef_item["id"])
                    ef.info.update({k: one_ef_item[k] for k in ["extra_info", "gid"] if k in one_ef_item})
                    # todo(note): no checking for possibly repeat efs
                    assert ef.id not in args_maps
                    args_maps[ef.id] = ef
                    mention.sent.add_entity_filler(ef)
        # events
        if d.get("event_mentions") is None:
            # no events info
            for sent in doc.sents:
                sent.mark_no_events()
        else:
            for one_evt_item in d["event_mentions"]:
                mention = self._parse_mention(one_evt_item["trigger"], doc)
                if mention is None:
                    failed_items["evt"].append(one_evt_item)
                else:
                    evt = Frame.create(mention, type=one_evt_item["type"], score=one_evt_item.get("score", 0.), id=one_evt_item["id"])
                    evt.info.update({k: one_evt_item[k] for k in ["extra_info", "gid", "realis", "realis_score"] if k in one_evt_item})
                    assert evt.id not in args_maps
                    args_maps[evt.id] = evt
                    mention.sent.add_event(evt)
        # args
        for one_evt_item in d.get("event_mentions", []):
            if one_evt_item["id"] not in args_maps:
                assert one_evt_item["trigger"]["posi"] is None
                continue
            evt = args_maps[one_evt_item["id"]]  # must be there
            em_args = one_evt_item.get("em_arg", None)
            if em_args is None:
                evt.mark_no_args()
            else:
                for one_arg in em_args:
                    aid, role = one_arg["aid"], one_arg["role"]
                    if aid not in args_maps:
                        failed_items["arg"].append(one_arg)
                    else:
                        arg_arg = args_maps[aid]
                        arglink = evt.add_arg(arg_arg, role, score=one_arg.get("score", 0.))
                        arglink.info.update({k: one_arg[k] for k in ["is_aug", "extra_info"] if k in one_arg})
        # --
        if any(len(v)>0 for k,v in failed_items.items()):
            zwarn(f"Failed when reading Doc({doc.id}): {[(k,len(v)) for k,v in failed_items.items()]}")
        return doc

    def _read_posi(self, posi):
        assert len(set([z[0] for z in posi])) == 1  # same sid
        sid, widx = posi[0]
        wlen = len(posi)
        assert all(z[1] == widx + i for i, z in enumerate(posi))  # continuous
        return sid, widx, wlen

    def _parse_mention(self, mention: Dict, doc: Doc) -> Mention:
        # get mention
        main_posi_info = mention.get("posi")
        if main_posi_info is None:
            return None  # no posi info!!
        sid, widx, wlen = self._read_posi(main_posi_info)
        ret = Mention.create(doc.sents[sid], widx, wlen)
        # possible head span?
        head_posi_info = mention.get("head", {}).get("posi")
        if head_posi_info is not None:
            head_sid, head_widx, head_wlen = self._read_posi(head_posi_info)
            if head_sid != sid or not (head_widx>=widx and head_widx+head_wlen<=widx+wlen):
                zwarn(f"Error in head: {head_posi_info} vs. {main_posi_info}")
            else:  # make sure things are correct! otherwise simply discard!!
                ret.set_span(head_widx, head_wlen, hspan=True)
        return ret

#
# b msp2/data/rw/formats/zdoc.py:137
