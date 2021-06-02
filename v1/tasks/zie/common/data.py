#

# data for information extraction

import json
from typing import Iterable, List, Set, Dict, Union
import numpy as np
import pickle
import math

from msp.utils import zfatal, zopen, zwarn, Random, Helper, MathHelper, zcheck, zlog, PickleRW, JsonRW
from msp.data import Instance, FileOrFdStreamer, VocabHelper, AdapterStreamer, FAdapterStreamer
from msp.zext.seq_data import InstanceHelper, SeqFactor, InputCharFactor

from .data_helper import get_label_normer, ExternalEntityVocab

#
class DocInstance(Instance):
    def __init__(self, sents: List['Sentence'], orig_props: Dict):
        super().__init__()
        self.sents: List[Sentence] = sents
        self.orig_props = orig_props
        self.doc_id: str = orig_props["doc_id"]
        self.dataset: str = orig_props["dataset"]
        # overall gold ones (also containing the failed to find position ones)
        self.entity_fillers: List[EntityFiller] = None
        self.events: List[Event] = None
        # extra info
        self.extra_info = orig_props.get("extra_info", None)
        if self.extra_info is None:
            self.extra_info = {}

    # mainly for convenience
    def get_entity_fillers(self, use_pred):
        return self.pred_entity_fillers if use_pred else self.entity_fillers

    def get_events(self, use_pred):
        return self.pred_events if use_pred else self.events

    # ===== grouping each sents ones for predictions
    @property
    def pred_entity_fillers(self):
        return Helper.join_list(x.pred_entity_fillers for x in self.sents)

    @property
    def pred_events(self):
        return Helper.join_list(x.pred_events for x in self.sents)

# todo(note): token idx in program will be added an offset 1 to the ones in json-inputs
#  to remember: in-class widx = in-dict widx + 1
class Sentence(Instance):
    # todo(note): always include special ROOT symbol to make heads more convenient
    ROOT_SYMBOL = VocabHelper.convert_special_pattern("r")

    def __init__(self, sid, words, lemmas, uposes, ud_heads, ud_labels, positions, entity_labels):
        super().__init__()
        self.sid: int = sid
        self.doc = None
        self.length = 1+len(words)  # add one extra ROOT node
        # =====
        # features (inputs)
        _tmp_root_list = [Sentence.ROOT_SYMBOL]
        self.words = SeqFactor(_tmp_root_list + words)
        self.lemmas = SeqFactor(_tmp_root_list + lemmas)
        self.chars = InputCharFactor([""] + words)  # empty pad chars
        self.uposes = SeqFactor(_tmp_root_list + uposes)
        self.ud_heads = SeqFactor([0] + ud_heads)
        self.ud_labels = SeqFactor(_tmp_root_list + ud_labels)
        self.positions = SeqFactor(None if positions is None else ([(0, 0)] + positions))  # (offset, length)
        if entity_labels is None:
            self.entity_labels = SeqFactor(None)
        else:
            # todo(note): idx the entity-BIO-labels with the fixed vocab
            entity_labels_vals = _tmp_root_list+entity_labels
            entity_labels_idxes = ExternalEntityVocab.idx_seq(entity_labels_vals)
            self.entity_labels = SeqFactor(entity_labels_vals)
            self.entity_labels.set_idxes(entity_labels_idxes)
        # extra inputs, for example, those from mbert
        self.extra_features = {"aux_repr": None}
        # =====
        # ie related
        self.entity_fillers: List = None  # entity or fillers (to be the arg of frames)
        self.events: List = None  # frames (including related args)
        # predictions
        self.pred_entity_fillers = []
        self.pred_events = []
        # -----
        # prepared for training or others; cached for efficiency
        self.preps = {}

    def __repr__(self):
        return f"Sent(sid={self.sid}, length={self.length-1}+1)"

    # return (i, word, lemma, pos, ud_head, ud_label)
    def get_tokens(self):
        return [(i, self.words.vals[i], self.lemmas.vals[i], self.uposes.vals[i], self.ud_heads.vals[i], self.ud_labels.vals[i])
                for i in range(self.length)]

    # mainly for convenience
    def get_entity_fillers(self, use_pred):
        return self.pred_entity_fillers if use_pred else self.entity_fillers

    def get_events(self, use_pred):
        return self.pred_events if use_pred else self.events

    # get distance-to-root with the heads
    def get_ddist(self):
        heads = self.ud_heads.vals
        ddist = [-1] * len(heads)
        ddist[0] = 0
        for m in range(len(heads)):
            path = []
            cur_idx = m
            while ddist[cur_idx] < 0:
                path.append(cur_idx)
                cur_idx = heads[cur_idx]
            up_dist = ddist[cur_idx]
            for i, idx in enumerate(reversed(path)):
                ddist[idx] = up_dist+i+1
        return ddist

# argument link between Event and EntityFiller
class ArgLink:
    def __init__(self, evt, ef, role: str, role_idx=None, score=0., is_aug=False, extra_info=None):
        self.evt: Event = evt
        self.ef: EntityFiller = ef
        self.role: str = role
        self.role_idx = role_idx
        self.score = score  # confidence score todo(note): scores are currently logprobs
        self.is_aug = is_aug  # whether this arg is augmented
        self.extra_info = {} if extra_info is None else extra_info

    def __repr__(self):
        return f"{self.evt}->{self.ef}({self.role})"

# entity or filler: to be the argument of some frames
class EntityFiller:
    def __init__(self, ef_id, mention, type: str, mtype, is_entity, type_idx=None, score=0., extra_info=None, gid=None):
        self.id: str = ef_id
        self.mention: Mention = mention
        # multiple levels of types
        self.type: str = type
        self.type_idx = type_idx
        # special mtype for entity
        self.is_entity = is_entity
        self.mtype = mtype
        # links
        self.links: List[ArgLink] = []
        # confidence score
        self.score = score
        # extra info
        self.extra_info = {} if extra_info is None else extra_info
        self.gid = gid

    def __repr__(self):
        return f"{self.mention}({self.type})"

# -----
EVENT_REALIS_LIST = ["actual", "generic", "other"]
EVENT_REALIS_MAP = {k:i for i,k in enumerate(EVENT_REALIS_LIST)}
# -----

# the frame itself
class Event:
    def __init__(self, evt_id, mention, type: str, type_idx=None, score=0., extra_info=None, gid=None,
                 realis=None, realis_score=None):
        self.id: str = evt_id
        self.mention: Mention = mention  # trigger
        # multiple levels of type
        self.type: str = type
        self.type_idx = type_idx
        self.realis: str = None
        self.realis_idx: int = None
        self.realis_score: float = None
        if realis is not None:
            self.set_realis(realis_str=realis, realis_score=realis_score)
        # links
        self.links: List[ArgLink] = []
        # confident score
        self.score = score
        # extra info
        self.extra_info = {} if extra_info is None else extra_info
        self.tmp_info = {}  # storing tmp values
        self.gid = gid

    def set_realis(self, realis_idx=None, realis_str=None, realis_score=None):
        if realis_idx is not None:
            self.realis_idx = realis_idx
            self.realis = EVENT_REALIS_LIST[realis_idx]  # must be one of the three
        elif realis_str is not None:
            realis_str = str.lower(realis_str)
            self.realis = realis_str
            self.realis_idx = EVENT_REALIS_MAP[realis_str]
        else:
            raise RuntimeError("To set realis, must provide idx or str!")
        self.realis_score = realis_score

    def add_arg(self, ef: EntityFiller, role: str, **kwargs):
        link = ArgLink(self, ef, role, **kwargs)
        self.links.append(link)
        ef.links.append(link)

    def __repr__(self):
        return f"{self.mention}({self.type})"

# hard spans (sid, wid, length, head-wid)
class HardSpan:
    def __init__(self, sid, head_wid, wid, length):
        self.sid: int = sid  # sentence if
        # there may be situations when there are only head word
        self.head_wid: int = head_wid  # head word id
        # full span info
        self.wid: int = wid  # start word offset
        self.length: int = length
        #
        # for partial evaluation: first assume the head word is the span
        if self.wid is None and self.length is None and self.head_wid is not None:
            self.wid = self.head_wid
            self.length = 1

    def position(self, headed=True):
        return (self.sid, self.head_wid, 1) if headed else (self.sid, self.wid, self.length)

    def __repr__(self):
        return f"({self.sid},{self.head_wid},{self.wid},{self.length})"

class SoftSpan:
    def __init__(self):
        self.sid: int = None
        self.weights = None  # [0-1] prob-like weights to indicate the span

# mentions (justifications)
class Mention:
    def __init__(self, hard_span: HardSpan, soft_span: SoftSpan = None, origin_char_posi=None):
        self.hard_span: HardSpan = hard_span
        self.soft_span: SoftSpan = soft_span
        # original position (char-start, char-length) (only read from gold annotation)
        self.origin_char_posi = origin_char_posi

    # the original one
    def get_origin_char_posi(self):
        assert self.origin_char_posi is not None
        return self.origin_char_posi

    # the one from token-level hard-span
    def get_hspan_char_posi(self, sents: List[Sentence], headed):
        sid, wid, wlen = self.hard_span.position(headed)
        cur_positions = sents[sid].positions.vals
        char_start, char_end = cur_positions[wid][0], cur_positions[wid+wlen-1][0] + cur_positions[wid+wlen-1][1]
        return char_start, char_end-char_start  # (offset, length)

    # return origin if have one (mainly for eval and printing)
    def get_origin_or_hspan_char_posi(self, sents: List[Sentence]):
        if self.origin_char_posi is not None:
            return self.get_origin_char_posi()
        else:
            return self.get_hspan_char_posi(sents, False)  # headed=False

    def __repr__(self):
        return str(self.hard_span)

# =====
# reader

# specific json format (prep/{xml2json,json2tok,tok2anno}.py)
class MyDocReader(FileOrFdStreamer):
    NOUN_HEAD_SCORES = {"NOUN": 0, "PROPN": -1, "NUM": -2, "VERB": -3, "PRON": -4, "ADJ": -7, "ADV": -10, "SYM": -11, "X": -12}
    VERB_HEAD_SCORES = {"VERB": 1, "NOUN": 0, "PROPN": -1, "NUM": -2, "PRON": -4, "ADJ": -7, "ADV": -10, "SYM": -11, "X": -12}

    def __init__(self, file_or_fd, use_la0=False, noef_link0=False, alter_carg_by_coref=False, max_evt_layers=100):
        super().__init__(file_or_fd)
        self.use_la0 = use_la0
        self.noef_link0 = noef_link0
        self.alter_carg_by_coref = alter_carg_by_coref
        self.max_evt_layers = max_evt_layers

    # parse basic sentence
    def parse_sent(self, sid, one_sent: Dict):
        words = one_sent["text"]
        slen = len(words)
        lemmas = [("_" if z is None else z) for z in one_sent.get("lemma", ['_']*slen)]  # can be None for puncts
        uposes = one_sent.get("upos", ['_']*slen)
        ud_heads = one_sent.get("governor", [0]*slen)
        ud_labels = one_sent.get("dependency_relation", ['_']*slen)
        if self.use_la0:  # use only first layer
            ud_labels = [z.split(":")[0] for z in ud_labels]
        positions = one_sent.get("positions", None)  # not every dataset prepared this field
        entity_labels = one_sent.get("entity_labels", None)  # not eveyr one get this, BIO tags
        return Sentence(sid, words, lemmas, uposes, ud_heads, ud_labels, positions, entity_labels)

    # heuristically find head
    @staticmethod
    def find_head(posi, sentences: List[Sentence], is_event: bool):
        sid, wid, wlen = posi
        idx_start, idx_end = wid, wid+wlen
        assert wlen>0
        if wlen==1:  # only one word
            return wid
        cur_ddists = sentences[sid].get_ddist()
        cur_heads = sentences[sid].ud_heads.vals
        cur_poses = sentences[sid].uposes.vals
        # todo(note): rule 1: simply find the highest node (nearest to root and not punct)
        # first pass by ddist
        min_ddist = min(cur_ddists[z] for z in range(idx_start, idx_end))
        cand_idxes1 = [z for z in range(idx_start, idx_end) if cur_ddists[z]<=min_ddist]
        assert len(cand_idxes1) > 0
        if len(cand_idxes1) == 1:
            return cand_idxes1[0]
        # next pass by POS
        POS_SCORES_MAP = MyDocReader.VERB_HEAD_SCORES if is_event else MyDocReader.NOUN_HEAD_SCORES
        pos_scores = [POS_SCORES_MAP.get(cur_poses[z], -100) for z in cand_idxes1]
        max_pos_score = max(pos_scores)
        cand_idxes2 = [v for i,v in enumerate(cand_idxes1) if pos_scores[i]>=max_pos_score]
        assert len(cand_idxes2) > 0
        if len(cand_idxes2) == 1:
            return cand_idxes2[0]
        # todo(note): rule 2: if same head and same pos, use the rightmost one
        # todo(+N): fine only for English?
        cand_idxes = cand_idxes2
        cand_heads, cand_poses = [cur_heads[z] for z in cand_idxes], [cur_poses[z] for z in cand_idxes]
        if all(z==cand_heads[0] for z in cand_heads) and all(z==cand_poses[0] for z in cand_poses):
            return cand_idxes[-1]
        if all(z=="PROPN" for z in cand_poses):
            return cand_idxes[-1]
        if all(z=="NUM" for z in cand_poses):
            return cand_idxes[-1]
        # todo(note): extra one: AUX+PART like "did not"
        if cand_poses == ["AUX", "PART"]:
            return cand_idxes[0]
        # todo(note): rule final: simply the rightmost
        if 1:
            cur_words = sentences[sid].words.vals
            ranged_words = cur_words[idx_start:idx_end]
            ranged_ddists = cur_ddists[idx_start:idx_end]
            ranged_heads = cur_heads[idx_start:idx_end]
            ranged_poses = cur_poses[idx_start:idx_end]
            zwarn(f"Cannot heuristically set head (is_event={is_event}), use the last one: "
                  f"{ranged_words} {ranged_ddists} {ranged_heads} {ranged_poses}")
        return cand_idxes[-1]

    # parse for one mention
    def parse_mention(self, mention: Dict, sentences: List[Sentence], is_event: bool):
        # =====
        def _get_posi(posi):
            if posi is None:
                return None
            else:
                assert len(set([z[0] for z in posi])) == 1  # same sid
                sid, wid = posi[0]
                assert all(z[1] == wid+i for i, z in enumerate(posi))
                # todo(note): here += 1 for ROOT offset
                return sid, wid+1, len(posi)  # sid, wid, length
        # =====
        # first general span: return None if cannot find position (pre-processing problems)
        posi = _get_posi(mention["posi"])
        if posi is None:
            return None
        sid, wid, length = posi
        # then get head word
        head_posi0 = mention.get("head", None)
        if head_posi0 is not None:
            head_posi0 = head_posi0["posi"]
        head_posi = _get_posi(head_posi0)
        if head_posi is None:
            # use the original whole span
            head_posi = posi
        # guess head by heuristic rules
        head_sid, head_wid, head_length = head_posi
        if not(head_sid==sid and head_wid>=wid and head_wid+head_length<=wid+length):
            zwarn(f"Head span is not inside full span, use full posi instead: {mention}")
            head_posi = posi
        head_wid = self.find_head(head_posi, sentences, is_event)
        # we only have original position from the origin gold file
        if "offset" in mention and "length" in mention:
            origin_char_posi = (mention["offset"], mention["length"])
        else:
            origin_char_posi = None
        ret = Mention(HardSpan(sid, head_wid, wid, length), origin_char_posi=origin_char_posi)
        if is_event and ret.hard_span.length>3:
            zwarn(f"Strange long event span: {sid}/{wid}/{length}/head={head_wid}: "
                  f"{sentences[sid].words.vals[wid:wid+length]}", level=2)
            zwarn("", level=2)
        return ret

    # parse the json dict
    def parse_doc(self, doc_dict: Dict):
        doc_id, dataset, source = doc_dict["doc_id"], doc_dict["dataset"], doc_dict["source"]
        label_normer = get_label_normer(dataset)
        # build all the sentences (basic inputs)
        sentences = [self.parse_sent(sid, one_sent) for sid, one_sent in enumerate(doc_dict["sents"])]
        # build entities, events and arguments
        args_maps = {}  # id -> EntityFiller
        # doc = DocInstance(sentences, {k:v for k,v in doc_dict.items() if isinstance(v, str)})  # record simple fields as props
        doc = DocInstance(sentences, doc_dict)
        for s in sentences:
            s.doc = doc  # link back
        # ----- entity and fillers
        sig2ef = {}  # todo(note): used to merge ef, but not for evt
        for cur_name in ["entity_mentions", "fillers"]:
            is_entity = (cur_name == "entity_mentions")
            cur_mentions = doc_dict.get(cur_name, None)
            if cur_mentions is not None:
                # first prepare the lists
                for s in sentences:
                    if s.entity_fillers is None:
                        s.entity_fillers = []
                if doc.entity_fillers is None:
                    doc.entity_fillers = []
                # then parse them
                for cur_mention in cur_mentions:
                    new_mention = self.parse_mention(cur_mention, sentences, is_event=False)
                    # assert cur_mention["type"] is not None
                    cur_type = label_normer.norm_ef_label(cur_mention["type"])
                    mtype = cur_mention.get("mtype", "")
                    new_id = cur_mention["id"]
                    kwargs = {k:cur_mention.get(k) for k in ["score", "extra_info", "gid"] if k in cur_mention}
                    new_ef = EntityFiller(new_id, new_mention, cur_type, mtype, is_entity, **kwargs)
                    # todo(note): check for repeat
                    if new_mention is not None:
                        cur_sig = (new_mention.hard_span.position(False), cur_type)  # same span and type
                        repeat_ef = sig2ef.get(cur_sig)
                        if repeat_ef is not None:
                            assert new_id not in args_maps
                            args_maps[new_id] = repeat_ef
                            # if is_entity:
                            #     zwarn(f"Repeated entity: {repeat_ef} <- {new_ef}")
                            continue  # not adding new one, only put id for later arg finding
                        else:
                            sig2ef[cur_sig] = new_ef
                    # if not repeat
                    assert new_id not in args_maps
                    args_maps[new_id] = new_ef
                    doc.entity_fillers.append(new_ef)
                    if new_mention is not None:
                        sentences[new_mention.hard_span.sid].entity_fillers.append(new_ef)
                    else:
                        # zwarn(f"Cannot find posi for a entity/filler mention: {cur_mention}")
                        pass
        # --- read entity corefs for solving some cross-sent distance args
        entity_chains = {}
        entity_corefs = doc_dict.get("entities")
        if entity_corefs is not None:
            entity_chains = {z['id']: z['mentions'] for z in entity_corefs}
        # --- events and arguments
        event_mentions = doc_dict.get("event_mentions", None)
        if event_mentions is not None:
            # first prepare the lists
            for s in sentences:
                s.events = []
            doc.events = []
            # then parse them
            for cur_mention in event_mentions:
                new_mention = self.parse_mention(cur_mention["trigger"], sentences, is_event=True)
                cur_type = label_normer.norm_evt_label(cur_mention["type"])
                # -----
                # cutoff event type layers
                cur_type = ".".join(cur_type.split(".")[:self.max_evt_layers])
                # -----
                kwargs = {k: cur_mention.get(k) for k in
                          ["score", "extra_info", "gid", "realis", "realis_score"] if k in cur_mention}
                new_evt = Event(cur_mention["id"], new_mention, cur_type, **kwargs)
                em_args = cur_mention.get("em_arg", None)
                if em_args is None:
                    new_evt.links = None  # annotation not available
                else:
                    cur_evt_sid = None if new_mention is None else new_mention.hard_span.sid
                    for cur_arg in cur_mention["em_arg"]:
                        aid, role = cur_arg["aid"], label_normer.norm_role_label(cur_arg["role"])
                        cur_ef = args_maps.get(aid)
                        if cur_ef is None:
                            zwarn(f"Cannot find event argument: {cur_arg}", level=3)
                            continue
                        # =====
                        # todo(note): change for same-sent args
                        if self.alter_carg_by_coref and cur_evt_sid is not None:
                            cur_ef_sid = None if cur_ef.mention is None else cur_ef.mention.hard_span.sid
                            if cur_ef_sid is not None and cur_ef_sid != cur_evt_sid:
                                coref_chain_ef_mentions = entity_chains.get(cur_ef.gid)
                                if coref_chain_ef_mentions is not None:
                                    for alter_aid in coref_chain_ef_mentions:
                                        alter_ef = args_maps[alter_aid]
                                        alter_ef_sid = None if alter_ef.mention is None else alter_ef.mention.hard_span.sid
                                        if alter_ef_sid == cur_evt_sid:
                                            cur_ef = alter_ef  # find the alternative
                                            break
                        # =====
                        # todo(WARN): there can be no-position arguments
                        kwargs = {k: cur_arg.get(k) for k in ["is_aug", "score", "extra_info"] if k in cur_arg}
                        new_evt.add_arg(cur_ef, role, **kwargs)
                        if cur_ef.mention is None:
                            zwarn(f"Cannot find posi for an event argument: {cur_arg}", level=2)
                # add it
                doc.events.append(new_evt)
                if new_mention is not None:
                    sentences[new_mention.hard_span.sid].events.append(new_evt)
                else:
                    zwarn(f"Cannot find posi for an event mention: {cur_mention}", level=2)
        # =====
        # special mode
        if self.noef_link0:
            tmp_ff = lambda x: len(x.links)>0
            if doc.entity_fillers is not None:
                doc.entity_fillers = list(filter(tmp_ff, doc.entity_fillers))
                for one_sent in doc.sents:
                    one_sent.entity_fillers = list(filter(tmp_ff, one_sent.entity_fillers))
        return doc

    def _next(self):
        line = self.fd.readline()
        if len(line)==0:
            return None
        doc_dict = json.loads(line)
        one = self.parse_doc(doc_dict)
        one.init_idx = self.count()
        return one

# ===== Data Reader
def get_data_reader(file_or_fd, input_format, use_la0, noef_link0, aux_repr_file=None, max_evt_layers=100):
    if input_format == "json":
        r = MyDocReader(file_or_fd, use_la0, noef_link0, alter_carg_by_coref=True, max_evt_layers=max_evt_layers)
    else:
        zfatal("Unknown input_format %s" % input_format)
        r = None
    if aux_repr_file is not None and len(aux_repr_file)>0:
        r = AuxDataReader(r, aux_repr_file, "aux_repr")
    return r

# pre-computed auxiliary data
class AuxDataReader(AdapterStreamer):
    def __init__(self, base_streamer, aux_repr_file, aux_name):
        super().__init__(base_streamer)
        self.file = aux_repr_file
        self.fd = None
        self.aux_name = aux_name

    def __del__(self):
        if self.fd is not None:
            self.fd.close()

    def _restart(self):
        self.base_streamer_.restart()
        if isinstance(self.file, str):
            if self.fd is not None:
                self.fd.close()
            self.fd = zopen(self.file, mode='rb', encoding=None)
        else:
            zcheck(self.restart_times_ == 0, "Cannot restart a FdStreamer")

    def _next(self):
        one = self.base_streamer_.next()
        if self.base_streamer_.is_eos(one):
            return None
        res = PickleRW.load_list(self.fd, len(one.sents))
        assert len(res) == len(one.sents), "Unmatched length"
        for one_res, one_sent in zip(res, one.sents):
            assert len(one_res) == one_sent.length, "Unmatched length for the aux_repr arr"
            one_sent.extra_features[self.aux_name] = one_res
        return one

#
class BerterDataAuger(AdapterStreamer):
    def __init__(self, base_streamer, berter, aux_name):
        super().__init__(base_streamer)
        self.berter = berter
        self.aux_name = aux_name

    def _next(self):
        one = self.base_streamer_.next()
        if self.base_streamer_.is_eos(one):
            return None
        # get all tokens
        cur_doc_tokens = [s.words.vals[1:] for s in one.sents]  # exclude root at this time
        cur_subwords = [self.berter.subword_tokenize(ones, True) for ones in cur_doc_tokens]
        res = self.berter.extract_features(cur_subwords)
        #
        assert len(res) == len(one.sents), "Unmatched length"
        for one_res, one_sent in zip(res, one.sents):
            assert len(one_res) == one_sent.length, "Unmatched length for the aux_repr arr"
            one_sent.extra_features[self.aux_name] = one_res
        return one

# =====
# Data Writer

def get_data_writer(file_or_fd, output_format):
    if output_format == "json":
        return MyDocWriter(file_or_fd)
    else:
        zfatal("Unknown output_format %s" % output_format)

# format means writing in which format, use_pred means use predicted ones
class MyDocWriter:
    def __init__(self, file_or_fd, format='json', use_pred=True):
        if isinstance(file_or_fd, str):
            self.fd = zopen(file_or_fd, "w")
        else:
            self.fd = file_or_fd
        self.format = format
        self.use_pred = use_pred
        #
        self._write_f = {"json": self.write_json, "txt": self.write_txt,
                         "tbf": self.write_tbf, "ann": self.write_ann}[format]

    def finish(self):
        self.fd.close()
        self.fd = None

    def __del__(self):
        if self.fd is not None:
            self.fd.close()

    # transform to json
    def transform_json(self, inst: DocInstance, **kwargs):
        # =====
        def _set_posi(d: Dict, mention: Mention):
            if mention is None:
                d["posi"] = None
            else:
                hspan = mention.hard_span
                sid, wid, wlen = hspan.sid, hspan.wid, hspan.length
                # todo(note): exclude ROOT for output idx
                if wid is not None:
                    d["posi"] = [(sid, z-1) for z in range(wid, wid+wlen)]
                if hspan.head_wid is not None:
                    d["head"] = {"posi": [(sid, hspan.head_wid-1)]}
        # =====
        # todo(note): specific as the reverse of reading; output predictions
        ret = {"sents": inst.orig_props["sents"], "extra_info": inst.extra_info,
               "entity_mentions": [], "fillers": [], "event_mentions": []}
        ret.update({k:v for k,v in inst.orig_props.items() if isinstance(v, str)})
        #
        cur_entity_fillers = inst.get_entity_fillers(self.use_pred)
        cur_events = inst.get_events(self.use_pred)
        #
        if cur_entity_fillers is not None:
            for one_ef in cur_entity_fillers:
                one_key = "entity_mentions" if one_ef.is_entity else "fillers"
                d_ef = {"type": one_ef.type, "id": one_ef.id, "score": one_ef.score, "extra_info": one_ef.extra_info}
                _set_posi(d_ef, one_ef.mention)
                ret[one_key].append(d_ef)
        if cur_events is not None:
            for one_event in cur_events:
                d_evt = {"type": one_event.type, "id": one_event.id, "trigger": {}, "em_arg": [],
                         "score": one_event.score, "realis": one_event.realis, "realis_score": one_event.realis_score,
                         "extra_info": one_event.extra_info}
                _set_posi(d_evt["trigger"], one_event.mention)
                # args
                if one_event.links is not None:
                    for one_arg in one_event.links:
                        d_evt["em_arg"].append({"aid": one_arg.ef.id, "role": one_arg.role, "is_aug": one_arg.is_aug,
                                                "score": one_arg.score, "extra_info": one_arg.extra_info})
                ret["event_mentions"].append(d_evt)
        return ret

    # write json
    def write_json(self, inst: DocInstance, **kwargs):
        ret = self.transform_json(inst, **kwargs)
        JsonRW.save_list([ret], self.fd)  # save one inst for one line

    # write in txt format for easy seeing
    def write_txt(self, inst: DocInstance, write_ef=True, write_arg=True, use_span=False, **kwargs):
        doc = inst
        ss = []
        ss.append(f"#BeginOfDocument {doc.doc_id} {doc.dataset}")
        # doc summaries
        sents = inst.sents
        cur_entity_fillers = inst.get_entity_fillers(self.use_pred)
        cur_events = inst.get_events(self.use_pred)
        ss.append(f"Doc-Summary: #EntityFiller={len(cur_entity_fillers) if cur_entity_fillers is not None else 'None'}, "
                  f"#Events={len(cur_events) if cur_events is not None else 'None'}, #Sents={len(sents)}")
        # ----
        def _get_mention_info(_hspan):
            hwid = _hspan.head_wid
            if use_span:
                wid, wlen = _hspan.wid, _hspan.length
                r0 = list(range(wid, wid+wlen))
                return r0, [f"_w{'H' if i==hwid else z}" for z,i in enumerate(r0)]
            else:
                return [hwid], [""]
        def _put_mention_info(_hspan, _prefix, _tofill):
            _wids, _suffixes = _get_mention_info(_hspan)
            for _i, _s in zip(_wids, _suffixes):
                _tofill[_i].append(_prefix+_s)
        # ----
        # sentence
        for sid, sent in enumerate(sents):
            tokens = sent.words.vals
            length = len(tokens)
            list_events = [[] for _ in range(length)]
            list_efs = [[] for _ in range(length)]
            list_args = [[] for _ in range(length)]
            # --
            s_events = sent.get_events(self.use_pred)
            if s_events is not None:
                for one_event in s_events:
                    if one_event.mention is None:
                        continue
                    _put_mention_info(one_event.mention.hard_span, f"{one_event.id}-{one_event.type}", list_events)
                    if write_arg and one_event.links is not None:
                        for one_link in one_event.links:
                            if one_link.ef.mention is None:
                                continue
                            _hdspan = one_link.ef.mention.hard_span
                            if _hdspan.sid == sid:
                                _put_mention_info(_hdspan, f"{one_link.evt.id}-{one_link.evt.type}-{one_link.role}", list_args)
            # --
            s_entity_fillers = sent.get_entity_fillers(self.use_pred)
            if write_ef and s_entity_fillers is not None:
                for one_ef in s_entity_fillers:
                    if one_ef.mention is not None:
                        _put_mention_info(one_ef.mention.hard_span, f"{one_ef.id}-{one_ef.type}", list_efs)
            # actually writing
            ss.append(f"#BefinOfSentence SID={sid} LEN={length}")
            ss.append("#"+" ".join(tokens))  # also write the sent in one line
            for one_tok, one_events, one_efs, one_args in zip(tokens, list_events, list_efs, list_args):
                ss.append(f"{one_tok}\t{one_events}\t{one_efs}\t{one_args}")
            ss.append(f"#EndOfSentence")
        ss.append(f"#EndOfDocument")
        self.fd.write("\n".join(ss)+"\n")

    # write to tbf format as in KBP
    def write_tbf(self, inst, **kwargs):
        # print(f"write {doc.doc_id}")  # debug
        doc = inst
        sents = inst.sents
        ss = []
        ss.append(f"#BeginOfDocument {doc.doc_id}")
        cur_events = inst.get_events(self.use_pred)
        # sort by trigger position
        cur_events.sort(key=lambda x: (-1,-1) if x.mention is None else (x.mention.hard_span.sid, x.mention.hard_span.wid))
        #
        for cur_i, cur_evt in enumerate(cur_events):
            if cur_evt.mention is not None:
                cur_span = cur_evt.mention.hard_span
                sid, wid, wlen = cur_span.sid, cur_span.wid, cur_span.length
                cur_positions = sents[sid].positions.vals
                char_start, char_end = cur_positions[wid][0], cur_positions[wid+wlen-1][0]+cur_positions[wid+wlen-1][1]
                cur_text = " ".join(cur_positions[wid:wid+wlen])
            else:
                char_start, char_end = -1, -1
                cur_text = "UNK"
            cur_type = "_".join(cur_evt.type.split("."))
            ss.append("\t".join(["z", doc.doc_id, f"E{cur_i}", f"{char_start},{char_end}", cur_text, cur_type, "UNK"]))
        ss.append(f"#EndOfDocument")
        self.fd.write("\n".join(ss) + "\n")

    # write to ann format for brat-visualization
    def write_ann(self, insts: Union[List, DocInstance], write_ef=True, write_arg=True, ignore_span0=False, **kwargs):
        if isinstance(insts, DocInstance):
            insts = [insts]
            single_input = True
        else:
            single_input = False
        #
        source = None
        if write_arg:
            assert write_ef
        #
        all_mentions = {}  # id -> brat-id
        all_events = {}  # id -> brat-id
        all_lines = []
        for one_idx, one_inst in enumerate(insts):
            one_prefix = "" if single_input else f"S{one_idx}."
            if source is not None:
                assert one_inst.orig_props["source"] == source
            else:
                source = one_inst.orig_props["source"]
            # doc summaries
            sents = one_inst.sents
            cur_entity_fillers = one_inst.get_entity_fillers(self.use_pred)
            cur_events = one_inst.get_events(self.use_pred)
            # sort by trigger position
            cur_entity_fillers.sort(key=lambda x: (-1,-1) if x.mention is None else (x.mention.hard_span.sid, x.mention.hard_span.wid))
            cur_events.sort(key=lambda x: (-1,-1) if x.mention is None else (x.mention.hard_span.sid, x.mention.hard_span.wid))
            # =====
            # put all the mentions
            # entity and filler
            if write_ef:
                for one_ef in cur_entity_fillers:
                    if ignore_span0 and one_ef.mention is None:
                        continue
                    one_ef_id = one_ef.id
                    one_ef_type = one_prefix + one_ef.type  # here use prefix to tell which set
                    one_ef_char_start, one_ef_char_length = one_ef.mention.get_hspan_char_posi(sents, False) \
                        if one_ef.mention is not None else (0, 0)
                    one_ef_char_end = one_ef_char_start + one_ef_char_length
                    assert (one_idx, one_ef_id) not in all_mentions
                    cur_bid = len(all_mentions)
                    all_mentions[(one_idx, one_ef_id)] = cur_bid
                    all_lines.append("\t".join([f"T{cur_bid}", f"{one_ef_type} {one_ef_char_start} {one_ef_char_end}",
                                                source[one_ef_char_start:one_ef_char_end]]))
            # event
            for one_evt in cur_events:
                if ignore_span0 and one_evt.mention is None:
                    continue
                one_evt_id = one_evt.id
                one_evt_type = one_prefix + one_evt.type
                one_evt_char_start, one_evt_char_length = one_evt.mention.get_hspan_char_posi(sents, False) \
                    if one_evt.mention is not None else (0, 0)
                one_evt_char_end = one_evt_char_start + one_evt_char_length
                assert (one_idx, one_evt_id) not in all_mentions
                cur_bid = len(all_mentions)
                all_mentions[(one_idx, one_evt_id)] = cur_bid
                # mention line
                all_lines.append("\t".join([f"T{cur_bid}", f"{one_evt_type} {one_evt_char_start} {one_evt_char_end}",
                                            source[one_evt_char_start:one_evt_char_end]]))
                # event and arg line
                assert (one_idx, one_evt_id) not in all_events
                cur_eid = len(all_events)
                all_events[(one_idx, one_evt_id)] = cur_eid
                evt_line = f"E{cur_eid}\t{one_evt_type}:T{cur_bid}"
                if write_arg:
                    for one_link in one_evt.links:
                        one_ef, one_role = one_link.ef, one_link.role
                        aug_prefix = "aug_" if one_link.is_aug else ""
                        arg_ef_id = all_mentions.get((one_idx, one_ef.id))
                        if arg_ef_id is None:
                            assert ignore_span0
                            continue
                        evt_line += f" {aug_prefix+one_role}:T{arg_ef_id}"
                all_lines.append(evt_line)
        self.fd.write("\n".join(all_lines) + "\n")

    def write(self, insts, **kwargs):
        if isinstance(insts, Iterable):
            for one in insts:
                self._write_f(one, **kwargs)
        else:
            self._write_f(insts, **kwargs)


# updated json format
# {"doc_id": ~, "dataset": ~, "source": str, "lang": str[2],
# "sents": List[Dict],
# todo(note): for the groups, currently still use the original separate types!
# "entities": [{id, type, mentions}], "relations": [{id, type, mentions}], "hoppers": [{id, mentions}],
# todo(note): type and subtype and subsubtype are merged into type with sep="."
# extra fields for these: posi, score=0., extra_info={}, is_aug=False
# "fillers": [{id, offset, length, type}],
# "entity_mentions": [id, gid, offset, length, type, mtype, (head)],
# "relation_mentions": [id, gid, type, rel_arg1{aid, role}, rel_arg2{aid, role}],
# "event_mentions": [id, gid, type, trigger{offset, length}, em_arg[{aid, role}]]}

# example
"""
# for others
PYTHONPATH=../src/ python3
from tasks.zie.common.data import MyDocReader, MyDocWriter
docs = list(MyDocReader("../data5/outputs_split/en.ere.test.json", False, False))
MyDocWriter("_gold.txt", "txt", False).write(docs, write_ef=False, write_arg=False)
docs = list(MyDocReader("zout.json.dev1", False, False))
MyDocWriter("_out.txt", "txt", False).write(docs, write_ef=False, write_arg=False)
#
docs = list(MyDocReader("zout.json.dev0", False, False))
MyDocWriter("outputP.txt", "tbf", False).write(docs)
docs2 = list(MyDocReader("../data3/outputs_split/ace.dev.json", False, False))
MyDocWriter("outputG.txt", "tbf", False).write(docs2)
# todo(note): scorer_v1.8.py:764 comb[:-2] to ignore realis
"""
