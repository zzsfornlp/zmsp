#

# assign anns from another file by matching 'seq_words'
# note: mainly for conll12 (en & zh & ar)

import sys
import re
import string
from typing import List
from collections import defaultdict
from msp2.utils import Conf, zlog, init_everything, zwarn, Constants, AlgoHelper
from msp2.data.inst import yield_sents
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

# --
# convert for zh
# # todo(note): extras: "URL" "X"
# UPOS2CTB = {
#     "ADJ": ["JJ", "VA", ],
#     "ADV": ["AD", ],
#     "INTJ": ["IJ", ],
#     "NOUN": ["M", "NN", "NT", ],
#     "PROPN": ["NR", ],
#     "VERB": ["VC", "VV", "VE", ],
#     "ADP": ["LC", "P", ],
#     "AUX": ["BA", "LB", "MSP", "SB", ],
#     "CCONJ": ["CC", "CS", ],
#     "DET": ["DT", "INF"],  # note: INF is a strange one, but seems to be here
#     "NUM": ["CD", "OD", ],
#     "PART": ["AS", "DEC", "DEG", "DER", "DEV", "SP"],
#     "PRON": ["PN", ],
#     "SCONJ": [],
#     "PUNCT": ["PU", ],
#     "SYM": ["FW", "URL"],
#     "X": ["ETC", "ON", "X"],
# }
# CTB2UPOS = {}
# for _t_ud, _ts_ctb in UPOS2CTB.items():
#     for z in _ts_ctb:
#         CTB2UPOS[z] = _t_ud
# use this more carefully-specified one!
from msp2.data.resources.ud_zh import UPOS2CTB, CTB2UPOS

UD2_LABEL_LIST = ["punct", "case", "nsubj", "det", "root", "nmod", "advmod", "obj", "obl", "amod", "compound", "aux", "conj", "mark", "cc", "cop", "advcl", "acl", "xcomp", "nummod", "ccomp", "appos", "flat", "parataxis", "discourse", "expl", "fixed", "list", "iobj", "csubj", "goeswith", "vocative", "reparandum", "orphan", "dep", "dislocated", "clf"]
UD1to2 = {w:w for w in UD2_LABEL_LIST}
UD1to2.update({"dobj": "obj", "neg": "advmod", "mark:clf": "clf",
               "nsubjpass": "nsubj:pass", "csubjpass": "csubj:pass", "auxpass": "aux:pass",
               "name": "compound", "erased": "dep", "etc": "dep"})  # note: the last three are not super clear ...
# nmod/obl
UPOS_NMOD = {"NOUN", "PROPN", "PRON", "NUM", "PUNCT", "DET", "SYM", "X", "INTJ"}
UPOS_OBL = {"ADJ", "ADV", "VERB", "AUX", "PART", "ADP"}

def convert_zh(pos_vals, head_vals, deplab_vals):
    new_pos_vals = [CTB2UPOS[z] for z in pos_vals]
    # note: simply split here!
    new_deplab_vals = []
    for ii, zz in enumerate(deplab_vals):
        # map
        vv = "clf" if zz=="mark:clf" else UD1to2[zz.split(":")[0]]
        # nmod/obl
        if vv == "nmod":
            head_upos = new_pos_vals[head_vals[ii]-1]
            if head_upos in UPOS_OBL:
                vv = "obl"
            else:
                assert head_upos in UPOS_NMOD
        new_deplab_vals.append(vv)
    return new_pos_vals, head_vals, new_deplab_vals
# --

# ==
class MainConf(Conf):
    def __init__(self):
        self.input = ReaderGetterConf()
        self.aux = ReaderGetterConf()
        self.output = WriterGetterConf()
        # --
        self.word_map = "PTB"
        self.output_sent_and_discard_nonhit = False
        self.convert_f = ""  # further do conversion, like "convert_zh"
        self.change_words = False  # whether change to words from trg sent?
        # --
        # indexer conf
        self.delete_char_scheme = ""  # ar
        self.change_char_scheme = ""  # zh/en
        self.fuzzy_no_repeat_query = False  # no repeat for fuzzy match!
        self.fuzzy_word_cnum = 0  # allow how many chars in one word to be diff?
        self.fuzzy_seq_wnum = Constants.INT_PRAC_MAX  # allow how many words in seq to be diff?
        self.fuzzy_seq_wrate = 0.  # allow how much ratio of words in seq to be diff?
        self.no_exact_match = False  # only do fuzzy match (just debugging)
        # --

# --
class MyIndexer:
    def __init__(self, conf: MainConf):
        self.conf = conf
        # --
        self.delete_char_set = {
            "ar": {chr(z) for z in [0x64c, 0x64d, 0x64e, 0x64f, 0x650, 0x651, 0x652]},
        }.get(conf.delete_char_scheme, set())
        self.change_char_map = {
            "en": {'\\': "<"},
            "zh": {'＜': "<", '＞': ">", '［': "<", '］': ">", '{': "<", '}': ">", '｛': "<", '｝': ">", '〈': "<", '〉': ">"},
        }.get(conf.change_char_scheme, {})
        # --
        self.exact_map = {}
        self.length_maps = defaultdict(list)  # seq_len -> List[sent]
        self.special_set = set("0123456789()")  # note: be careful about these!
        # --
        self.fuzzy_hit_ids = set()

    def __len__(self):
        return len(self.exact_map)

    def _get_words(self, sent):
        words = [''.join([c for c in w if c not in self.delete_char_set]) for w in sent.seq_word.vals]
        words = [''.join([self.change_char_map.get(c, c) for c in w]) for w in words]
        return words

    def put(self, sent):
        key = tuple(self._get_words(sent))
        if key not in self.exact_map:
            self.exact_map[key] = sent
            self.length_maps[len(key)].append(sent)
        # --

    def query(self, sent):
        conf = self.conf
        # --
        key = tuple(self._get_words(sent))
        if (not conf.no_exact_match) and key in self.exact_map:
            return self.exact_map[key]
        else:  # allow fuzzy match with only some char differences!
            good_ones = []  # (sent, (err_word, err_char))
            for cand in self.length_maps.get(len(key), []):
                # --
                if id(cand) in self.fuzzy_hit_ids:
                    continue
                # --
                cand_key = self._get_words(cand)
                assert len(cand_key) == len(key)
                _budget_words = min(conf.fuzzy_seq_wnum, int(conf.fuzzy_seq_wrate * len(key)))
                is_good = True
                cand_err_word, cand_err_char = 0, 0
                for w1, w2 in zip(cand_key, key):
                    # simple filter
                    if len(w1) - len(w2) > conf.fuzzy_word_cnum:
                        is_good = False
                        break
                    # digits filter
                    if (all(c in self.special_set for c in w1) or all(c in self.special_set for c in w2)) and (w1 != w2):
                        is_good = False
                        break
                    # --
                    # special distance: only allow same_len diff, prefix, suffix
                    _err = self._get_edit_distance(w1, w2)
                    # --
                    if _err > 0:
                        cand_err_word += 1
                        cand_err_char += _err
                        if _err <= conf.fuzzy_word_cnum:
                            _budget_words -= 1
                            if _budget_words < 0:  # differ too much overall
                                is_good = False
                                break
                        else:  # differ too much in one word
                            is_good = False
                            break
                is_good = (is_good and (_budget_words>=0))
                if is_good:
                    good_ones.append((cand, (cand_err_word, cand_err_char)))
            # return the least err one!
            if len(good_ones) >= 2:
                zwarn(f"Get multiple options for {key}")
            # return None if len(good_ones)<=0 else min(good_ones, key=lambda x: x[-1])[0]
            # note: only return if there is "the only one"!!
            ret = None if len(good_ones)!=1 else good_ones[0][0]
            if conf.fuzzy_no_repeat_query and ret is not None:
                self.fuzzy_hit_ids.add(id(ret))
            return ret

    # edit distance
    def _get_edit_distance(self, word1: str, word2: str):
        return AlgoHelper.edit_distance(word1, word2)

# --
def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    word_map_f = {
        "PTB": lambda x: re.sub("-LRB-", "(", re.sub("-RRB-", ")", x)),
    }.get(conf.word_map, lambda x: x)
    convert_f = None if conf.convert_f == "" else globals()[conf.convert_f]
    # --
    # first read aux ones
    aux_insts = list(conf.aux.get_reader())
    aux_index = MyIndexer(conf)
    num_aux_sent = 0
    for sent in yield_sents(aux_insts):
        num_aux_sent += 1
        aux_index.put(sent)
    zlog(f"Read from {conf.aux.input_path}: insts={len(aux_insts)}, sents={num_aux_sent}, len(index)={len(aux_index)}")
    # then read input
    input_insts = list(conf.input.get_reader())
    output_sents = []
    num_input_sent = 0
    num_rebuild_sent = 0
    num_reset_sent = 0
    num_hit_sent = 0
    for sent in yield_sents(input_insts):
        num_input_sent += 1
        # --
        new_word_vals = [word_map_f(w) for w in sent.seq_word.vals]
        if new_word_vals != sent.seq_word.vals:
            num_rebuild_sent += 1
            sent.build_words(new_word_vals)
        # --
        trg_sent = aux_index.query(sent)
        # -- debug
        # if trg_sent is None:
        #     breakpoint()
        #     trg_sent = aux_index.query(sent)
        # --
        if trg_sent is not None:
            num_hit_sent += 1
            # --
            # note: currently we replace upos & tree_dep
            upos_vals, head_vals, deplab_vals = \
                trg_sent.seq_upos.vals, trg_sent.tree_dep.seq_head.vals, trg_sent.tree_dep.seq_label.vals
            if convert_f is not None:
                upos_vals, head_vals, deplab_vals = convert_f(upos_vals, head_vals, deplab_vals)
            # --
            if conf.change_words and sent.seq_word.vals != trg_sent.seq_word.vals:
                num_reset_sent += 1
                sent.seq_word.set_vals(trg_sent.seq_word.vals)  # reset it!
            # --
            sent.build_uposes(upos_vals)
            sent.build_dep_tree(head_vals, deplab_vals)
            # --
            output_sents.append(sent)
        else:
            zlog(f"Miss sent: {sent.seq_word}")
            if not conf.output_sent_and_discard_nonhit:
                output_sents.append(sent)
    zlog(f"Read from {conf.input.input_path}: insts={len(input_insts)}, sents={num_input_sent}, (out-sent-{len(output_sents)})"
         f"rebuild={num_rebuild_sent}({num_rebuild_sent/num_input_sent:.4f}), "
         f"reset={num_reset_sent}({num_reset_sent/num_input_sent:.4f}) hit={num_hit_sent}({num_hit_sent/num_input_sent:.4f})")
    # write
    with conf.output.get_writer() as writer:
        if conf.output_sent_and_discard_nonhit:
            writer.write_insts(output_sents)
        else:  # write the original insts
            writer.write_insts(input_insts)
    # --

# --
# python3 assign_anns.py input.input_path:?? aux.input_path:?? output.output_path:??
# -- aux.input_format:conllu aux.use_multiline:1 "aux.mtl_ignore_f:'ignore_#'"
# note: this seems ok for ar-conll12, cover 97%+, but will there be false positive??
# -- delete_char_scheme:ar fuzzy_word_cnum:3 fuzzy_seq_wrate:0.5 change_words:1
if __name__ == '__main__':
    main(*sys.argv[1:])
