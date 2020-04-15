#

# evaluation for the extractions

from typing import List
from msp.utils import Conf
from msp.zext.evaler import LabelF1Evaler

from .data import DocInstance

#
class MyIEEvalerConf(Conf):
    def __init__(self):
        self.ef_mode = "head"
        self.evt_mode = "head"
        self.evt_match_realis = False  # further use event realis
        self.arg_mode = "head"
        self.arg_match_evt_mention = True  # require that arg should also match evt mention posi (head)
        self.arg_match_evt_type = True  # require that the evt to which arg is mapped to should have the correct type
        # -----
        self.res_list = ["argument", "event"]

class MyIEEvaler:
    def __init__(self, conf: MyIEEvalerConf):
        self.conf = conf

    def _get_key(self, doc_id, mention, sents, mode, extra_type=None):
        if mention is None:
            x = (doc_id, None)  # for some of the missed gold ones
        else:
            hard_span = mention.hard_span
            if mode == "span":
                x = (doc_id, hard_span.sid, hard_span.wid, hard_span.length)
            elif mode == "head":
                x = (doc_id, hard_span.sid, hard_span.head_wid)
            elif mode == "sent":
                x = (doc_id, hard_span.sid, )
            elif mode == "char":
                x = (doc_id, mention.get_origin_or_hspan_char_posi(sents))
            else:
                raise NotImplementedError()
        if extra_type is None:
            return x
        else:
            return x + (extra_type, )

    # todo(+N): specific [-2, 2]
    SDIST2EIDX = {s: max(-2, min(s, 2))+2 for s in range(-20000, 20000)}

    def eval(self, gold_docs: List[DocInstance], pred_docs: List[DocInstance], quite=True, breakdown=False, use_pred=True):
        ef_evaler = LabelF1Evaler("entity_filler")
        evt_evaler = LabelF1Evaler("event")
        arg_evaler = LabelF1Evaler("argument")
        sdist2idx = MyIEEvaler.SDIST2EIDX
        # =====
        # special evals
        layered_evt_evalers = [LabelF1Evaler(f"event_L{i+1}") for i in range(3)]
        sdist_arg_evalers = [LabelF1Evaler(f"argument_S{i}") for i in range(-2,3)]
        # =====
        # todo(note): this eval is very similar to the previous one
        # arg2_evaler = LabelF1Evaler("argument2")
        ef_mode, evt_mode, arg_mode = self.conf.ef_mode, self.conf.evt_mode, self.conf.arg_mode
        arg_match_evt_mention, arg_match_evt_type = self.conf.arg_match_evt_mention, self.conf.arg_match_evt_type
        # add golds
        for one_gold in gold_docs:
            doc_id = one_gold.doc_id
            sents = one_gold.sents
            if one_gold.entity_fillers is not None:
                for one_ef in one_gold.entity_fillers:
                    ef_evaler.add_gold(self._get_key(doc_id, one_ef.mention, sents, ef_mode), one_ef.type)
            if one_gold.events is not None:
                for one_evt in one_gold.events:
                    evt_mention_key = self._get_key(doc_id, one_evt.mention, sents, evt_mode,
                                                    extra_type=(one_evt.realis if self.conf.evt_match_realis else None))
                    evt_evaler.add_gold(evt_mention_key, one_evt.type)
                    for layer, layer_eval in enumerate(layered_evt_evalers):
                        layer_eval.add_gold(evt_mention_key, ".".join(one_evt.type.split(".")[:layer+1]))
                    # arg should match the frame first
                    evt_keyer = []
                    if self.conf.arg_match_evt_mention:
                        evt_keyer.append(evt_mention_key)
                    if self.conf.arg_match_evt_type:
                        evt_keyer.append(one_evt.type)
                    evt_keyer = tuple(evt_keyer)
                    if one_evt.links is not None:
                        for one_arg in one_evt.links:
                            # arg eval
                            arg_evaler_key = self._get_key(doc_id, one_arg.ef.mention, sents, arg_mode, extra_type=evt_keyer)
                            arg_evaler.add_gold(arg_evaler_key, one_arg.role)
                            # sdist
                            if one_evt.mention is not None and one_arg.ef.mention is not None:
                                evt_sid = one_evt.mention.hard_span.sid
                                ef_sid = one_arg.ef.mention.hard_span.sid
                                sdist_arg_evalers[sdist2idx[ef_sid-evt_sid]].add_gold(arg_evaler_key, one_arg.role)
                            # # special arg eval (arg2): first no not-found events/args, then cross-sent-PROPN use head-str
                            # if one_evt.mention is not None and one_arg.ef.mention is not None:
                            #     evt_sid = one_evt.mention.hard_span.sid
                            #     ef_sid, ef_hwid = one_arg.ef.mention.hard_span.sid, one_arg.ef.mention.hard_span.head_wid
                            #     if ef_sid != evt_sid and sents[ef_sid].uposes.vals[ef_hwid] == "PROPN":
                            #         # here only need to hit the event and the surface string
                            #         arg2_evaler.add_gold((sents[ef_sid].words.vals[ef_hwid].lower(), evt_keyer), one_arg.role)
                            #     else:
                            #         arg2_evaler.add_gold(arg_evaler_key, one_arg.role)
        # add preds
        for one_pred in pred_docs:
            doc_id = one_pred.doc_id
            sents = one_pred.sents
            # if one_pred.entity_fillers is not None:
            if 1:
                for one_ef in (one_pred.pred_entity_fillers if use_pred else one_pred.entity_fillers):
                    ef_evaler.add_pred(self._get_key(doc_id, one_ef.mention, sents, ef_mode), one_ef.type)
            # if one_pred.events is not None:
            if 1:
                for one_evt in (one_pred.pred_events if use_pred else one_pred.events):
                    evt_mention_key = self._get_key(doc_id, one_evt.mention, sents, evt_mode,
                                                    extra_type=(one_evt.realis if self.conf.evt_match_realis else None))
                    evt_evaler.add_pred(evt_mention_key, one_evt.type)
                    for layer, layer_eval in enumerate(layered_evt_evalers):
                        layer_eval.add_pred(evt_mention_key, ".".join(one_evt.type.split(".")[:layer+1]))
                    # arg should match the frame first
                    evt_keyer = []
                    if self.conf.arg_match_evt_mention:
                        evt_keyer.append(evt_mention_key)
                    if self.conf.arg_match_evt_type:
                        evt_keyer.append(one_evt.type)
                    evt_keyer = tuple(evt_keyer)
                    for one_arg in one_evt.links:
                        # arg eval
                        arg_evaler_key = self._get_key(doc_id, one_arg.ef.mention, sents, arg_mode, extra_type=evt_keyer)
                        arg_evaler.add_pred(arg_evaler_key, one_arg.role)
                        # sdist
                        if one_evt.mention is not None and one_arg.ef.mention is not None:
                            evt_sid = one_evt.mention.hard_span.sid
                            ef_sid = one_arg.ef.mention.hard_span.sid
                            sdist_arg_evalers[sdist2idx[ef_sid-evt_sid]].add_pred(arg_evaler_key, one_arg.role)
                        # # special arg eval (arg2): first no not-found events/args, then cross-sent-PROPN use head-str
                        # if one_evt.mention is not None and one_arg.ef.mention is not None:
                        #     evt_sid = one_evt.mention.hard_span.sid
                        #     ef_sid, ef_hwid = one_arg.ef.mention.hard_span.sid, one_arg.ef.mention.hard_span.head_wid
                        #     if ef_sid != evt_sid and sents[ef_sid].uposes.vals[ef_hwid] == "PROPN":
                        #         # here only need to hit the event and the surface string
                        #         arg2_evaler.add_pred((sents[ef_sid].words.vals[ef_hwid].lower(), evt_keyer), one_arg.role)
                        #     else:
                        #         arg2_evaler.add_pred(arg_evaler_key, one_arg.role)
                        # special arg eval (arg2)
        # -----
        detailed_results = []
        for one_evaler in layered_evt_evalers + sdist_arg_evalers:
            all_f_u, all_f_l, label_fs = one_evaler.eval(quite, breakdown)
            detailed_results.append(f"{one_evaler.name}: {all_f_u}||{all_f_l}")
        ret = {
            "entity_filler": ef_evaler.eval(quite, breakdown),
            "event": evt_evaler.eval(quite, breakdown),
            "argument": arg_evaler.eval(quite, breakdown),
            # "argument2": arg2_evaler.eval(quite, breakdown),
            "zdetails": " ~~~ ".join(detailed_results),
        }
        return ret

# b tasks/zie/common/eval.py:59
