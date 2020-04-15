#

# external annotator
# standfordnlp / StanfordCoreNLP
"""
- For CoreNLP, need the models:
## use be_quite=False to debug!!
# wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
# unzip stanford-corenlp-full-2018-10-05.zip
# (mainly for Chinese, for which UD-GSD seems to be not good enough)
# wget http://nlp.stanford.edu/software/stanford-chinese-corenlp-2018-10-05-models.jar
- For standfordnlp, need the models
import stanfordnlp
stanfordnlp.download('en')
stanfordnlp.download('ru')
stanfordnlp.download('uk')
stanfordnlp.download('es')
stanfordnlp.download('zh')
"""

from typing import Set, List, Dict, Iterable, Union
import sys
import os
import json
import argparse
from collections import Counter
import re

import stanfordnlp
from stanfordnlp.server import CoreNLPClient

# =====
# helpers
def zlog(s):
    print(str(s), file=sys.stderr, flush=True)

def zwarn(s):
    zlog("!!"+str(s))

# =====
LANG_NAME_MAP = {'es': 'spanish', 'zh': 'chinese', 'en': 'english'}

# map CTB pos tags to UD (partially reference "Developing Universal Dependencies for Mandarin Chinese")
CORENLP_POS_TAGS = [
    "VA", "VC", "VE", "VV", "NR", "NT", "NN", "LC", "PN", "DT", "CD", "OD", "M", "AD", "P", "CC", "CS",
    "DEC", "DEG", "DER", "DEV", "SP", "AS", "ETC", "MSP", "IJ", "ON", "PU", "JJ", "FW", "LB", "SB", "BA",
]
# todo(note): extras: "URL" "X"
UD2CORENLP = {
    "ADJ": ["JJ", "VA", ],
    "ADV": ["AD", ],
    "INTJ": ["IJ", ],
    "NOUN": ["M", "NN", "NT", ],
    "PROPN": ["NR", ],
    "VERB": ["VC", "VV", "VE", ],
    "ADP": ["LC", "P", ],
    "AUX": ["BA", "LB", "MSP", "SB", ],
    "CCONJ": ["CC", "CS", ],
    "DET": ["DT", ],
    "NUM": ["CD", "OD", ],
    "PART": ["AS", "DEC", "DEG", "DER", "DEV", "SP"],
    "PRON": ["PN", ],
    "SCONJ": [],
    "PUNCT": ["PU", ],
    "SYM": ["FW", "URL"],
    # "X": ["ETC", "ON", "X"],
    "X": ["ETC", "ON"],
}
CORENLP2UD = {}
for _t_ud, _ts_ctb in UD2CORENLP.items():
    for z in _ts_ctb:
        assert z not in CORENLP2UD
        CORENLP2UD[z] = _t_ud
    assert _t_ud not in CORENLP2UD
    CORENLP2UD[_t_ud] = _t_ud

# =====
class StanfordPipeline:
    def __init__(self, lang: str, annotators: Union[str,List[str]], use_corenlp=False, models_dir="",
                 pretokenized=False, merge_one_sent=False, sep_syms=""):
        self.lang = lang
        self.corenlp_language_name = LANG_NAME_MAP[lang]
        self.use_corenlp = use_corenlp
        self.models_dir = models_dir
        if isinstance(annotators, str):
            self.annotators = annotators.split(",")
        else:
            self.annotators = annotators
        self.annotators_set = set(self.annotators)
        self.pretokenized = pretokenized
        self.merge_one_sent = merge_one_sent
        self.need_pos, self.need_lemma, self.need_dep = [(z in self.annotators_set) for z in ['pos','lemma','depparse']]
        self.sep_syms_set = set(list(sep_syms))  # separate these symbols (add spaces) apart especially
        if len(self.sep_syms_set)>0:
            assert not self.pretokenized, "No processing for sep_syms if pretokenized!!"
        # -----
        # currently only support certain operations
        if use_corenlp:
            assert all((z in {'tokenize','ssplit','pos','lemma','depparse'}) for z in self.annotators)
            assert "CORENLP_HOME" in os.environ, "CORENLP_HOME not found, please set this for CORENLP!!"
            zlog(f"Start tokenizer with corenlp, home={os.environ['CORENLP_HOME']} and self = {self.__dict__}")
            # self.corenlp = None  # binded at running time
            self.nlp = None
        else:  # no mwt since that may split words (which we ignore for now)
            # assert all((z in {'tokenize','mwt','pos','lemma','depparse'}) for z in self.annotators)
            assert all((z in {'tokenize','pos','lemma','depparse'}) for z in self.annotators)
            zlog(f"Start tokenizer with stanfordnlp, self = {self.__dict__}")
            # self.corenlp = None
            annotators_str = ",".join(self.annotators)
            if len(models_dir) > 0:
                self.nlp = stanfordnlp.Pipeline(processors=annotators_str, lang=self.lang,
                                                tokenize_pretokenized=pretokenized, models_dir=models_dir)
            else:  # use default dir
                self.nlp = stanfordnlp.Pipeline(processors=annotators_str, lang=self.lang,
                                                tokenize_pretokenized=pretokenized)

    # annotate on iter of raw texts
    def annotate(self, input_iter: Union[Iterable[str], Iterable[List[str]]]):
        stat = Counter()
        annotators, pretokenized, merge_one_sent = self.annotators, self.pretokenized, self.merge_one_sent
        if self.use_corenlp:
            with CoreNLPClient(annotators=annotators, timeout=60000, memory='16G',
                               properties=self.corenlp_language_name, endpoint="http://localhost:9001") as client:
                # using CoreNLP, currently we annotate each inst at one time
                for one_inst in input_iter:
                    if pretokenized:
                        assert isinstance(one_inst, List)
                        one_input_str = " ".join(one_inst)
                    else:
                        one_input_str = self.sep_syms_for_str(one_inst)
                    # annotate
                    ann = client.annotate(one_input_str, annotators=annotators)
                    one_res = self.unpack_ann(ann, one_inst, pretokenized, merge_one_sent, stat)
                    yield one_res
        else:
            # to make it easy, here we also annotate each inst at one time
            for one_inst in input_iter:
                if pretokenized:
                    assert isinstance(one_inst, List)
                    one_input = [one_inst]
                else:
                    one_input = self.sep_syms_for_str(one_inst)
                # annotate
                ann = self.nlp(one_input)
                one_res = self.unpack_ann(ann, one_inst, pretokenized, merge_one_sent, stat)
                yield one_res
        # -----
        zlog(f"This annotation, stat={stat}")

    # -----
    # helpers

    def sep_syms_for_str(self, input_str):
        sep_syms_set = self.sep_syms_set
        all_chars = []
        for s in input_str:
            if s in sep_syms_set:
                all_chars.extend([" ", s, " "])  # especially separate apart
            else:
                all_chars.append(s)
        return "".join(all_chars)

    def unpack_ann(self, ann, inst, pretokenized: bool, merge_one_sent: bool, stat):
        ret_sents = self.unpack_ann_corenlp(ann) if self.use_corenlp else self.unpack_ann_nlp(ann)
        # checking
        if pretokenized:
            if len(ret_sents) != 1 or len(ret_sents[0]["text"]) != len(inst):
                zwarn(f"WARN: Pre-tokenizing not matched: {inst} vs {ret_sents}")
        # merge one sent
        if pretokenized or merge_one_sent:
            final_ret_sents = self.merge_sents(ret_sents)
        else:
            final_ret_sents = ret_sents
        # stat
        stat["inst"] += 1
        stat["sent_orig"] += len(ret_sents)
        stat["sent_final"] += len(final_ret_sents)
        return final_ret_sents

    def merge_sents(self, sents: List):
        if len(sents) <= 1:
            return sents
        new_sent = {"text": []}
        if self.need_pos:
            new_sent["upos"] = []
        if self.need_lemma:
            new_sent["lemma"] = []
        if self.need_dep:
            new_sent["governor"] = []
            new_sent["dependency_relation"] = []
        # -----
        cur_idx = 0
        for cur_sent in sents:
            cur_tokens = cur_sent["text"]
            cur_slen = len(cur_tokens)
            # -----
            new_sent["text"].extend(cur_tokens)
            if self.need_pos:
                new_sent["upos"].extend(cur_sent["upos"])
            if self.need_lemma:
                new_sent["lemma"].extend(cur_sent["lemma"])
            if self.need_dep:
                cur_deps, cur_rels = cur_sent["governor"], cur_sent["dependency_relation"]
                # todo(+N): allow multiple roots!!
                new_sent["governor"].extend([0 if z==0 else z+cur_idx for z in cur_deps])  # add offset!!
                new_sent["dependency_relation"].extend(cur_rels)
            # -----
            cur_idx += cur_slen
        return [new_sent]

    def unpack_ann_corenlp(self, ann):
        sents = []
        for s in ann.sentence:
            one_sent = {}
            # tokens
            one_sent["text"] = [w.originalText for w in s.token]
            # pos
            if self.need_pos:
                ud_poses = []
                for w in s.token:
                    orig_pos = w.pos
                    ud_pos = CORENLP2UD.get(orig_pos, None)
                    if ud_pos is None:
                        zwarn(f"UNK pos of {orig_pos}")
                        ud_pos = "X"
                    ud_poses.append(ud_pos)
                one_sent["upos"] = ud_poses
            # lemma
            if self.need_lemma:
                one_sent["lemma"] = [w.lemma for w in s.token]
            # dep
            if self.need_dep:
                slen = len(s.token)
                ud_heads = [None] * slen
                ud_deps = [None] * slen
                ud_count = 0
                for one_edge in s.basicDependencies.edge:
                    ud_count += 1
                    h, m, la = one_edge.source, one_edge.target, one_edge.dep
                    midx = m - 1
                    assert ud_heads[midx] is None and ud_deps[midx] is None
                    ud_heads[midx] = h
                    ud_deps[midx] = la
                for r in s.basicDependencies.root:
                    ud_count += 1
                    ridx = r - 1
                    assert ud_heads[ridx] is None and ud_deps[ridx] is None
                    ud_heads[ridx] = 0
                    ud_deps[ridx] = "root"
                assert ud_count == slen
                one_sent["governor"] = ud_heads
                one_sent["dependency_relation"] = ud_deps
            # -----
            sents.append(one_sent)
        return sents

    def unpack_ann_nlp(self, ann):
        sents = []
        for s in ann.sentences:
            one_sent = {}
            # tokens
            slen = len(s.tokens)
            one_sent["text"] = [w.text for w in s.tokens]
            words = s.words
            # pos
            if self.need_pos:
                one_pos = [w.upos for w in words]
                assert len(one_pos) == slen
                one_sent["upos"] = one_pos
            # lemma
            if self.need_lemma:
                one_lemma = [w.lemma for w in words]
                assert len(one_lemma) == slen
                one_sent["lemma"] = one_lemma
            # dep
            if self.need_dep:
                one_deps, one_rels = [w.governor for w in words], [w.dependency_relation for w in words]
                assert len(one_deps)==slen and len(one_rels)==slen
                one_sent["governor"] = one_deps
                one_sent["dependency_relation"] = one_rels
            # -----
            sents.append(one_sent)
        return sents

# =====
# running example
# -- one instance per line

def main():
    parser = argparse.ArgumentParser("Annotate data with stanford nlp.")
    parser.add_argument("-i", "--input", type=str, default='')
    parser.add_argument("-o", "--output", type=str, default='')
    parser.add_argument("-l", "--lang", type=str, required=True)
    parser.add_argument("-m", "--models_dir", type=str, default='')
    parser.add_argument("-a", "--annotators", type=str, required=True)
    parser.add_argument("-s", "--sep_syms", type=str, default="[]")  # first separate certain symbols apart
    parser.add_argument("--input_format", type=str, default="plain")
    parser.add_argument("--output_format", type=str, default="json")
    parser.add_argument("--use_corenlp", type=int, default=0)  # use corenlp or the other one
    parser.add_argument("--pretokenized", type=int, default=0)  # already separated by spaces
    parser.add_argument("--merge_one_sent", type=int, default=1)  # only one sent per inst
    args = parser.parse_args()
    # -----
    p = StanfordPipeline(lang=args.lang, annotators=args.annotators, use_corenlp=bool(args.use_corenlp),
                         models_dir=args.models_dir, pretokenized=bool(args.pretokenized),
                         merge_one_sent=bool(args.merge_one_sent), sep_syms=args.sep_syms)
    fin = sys.stdin if (args.input == "-" or args.input == "") else open(args.input)
    fout = sys.stdout if (args.output == "-" or args.output == "") else open(args.output, "w")
    # Input
    all_lines = [line.rstrip() for line in fin]
    all_nonempty_lines = [z for z in all_lines if len(z)>0]  # do not process empty lines but keep them for output
    if args.input_format == "plain":
        input_insts = all_nonempty_lines
        if bool(args.pretokenized):  # if pre-token, use " " to separate
            input_insts = [line.split(" ") for line in input_insts]
    elif args.input_format == "json":
        input_insts = [json.loads(line)["text"] for line in all_nonempty_lines]
        assert bool(args.pretokenized), "If not pretokenized, use the plain input"
    else:
        raise NotImplementedError()
    # Process
    all_annos = list(p.annotate(input_insts))
    # Output
    nonempty_idx = 0
    for one_input_line in all_lines:
        if len(one_input_line) == 0:  # simply output empty
            fout.write("\n")
        else:
            one_res = all_annos[nonempty_idx]
            nonempty_idx += 1
            if args.output_format == "plain":
                all_tokens = []
                for one_sent in one_res:
                    all_tokens.extend(one_sent["text"])
                fout.write(" ".join(all_tokens) + "\n")
            elif args.output_format == "json":
                for one_sent in one_res:
                    fout.write(json.dumps(one_sent)+"\n")  # should we write to one line?
            else:
                raise NotImplementedError()
    assert nonempty_idx == len(all_annos)
    fin.close()
    fout.close()

if __name__ == '__main__':
    main()

# =====
# typical use case
# use corenlp for chinese
# python3 stanford.py -l zh -a "tokenize,ssplit,pos,lemma,depparse" --output_format json --use_corenlp 1 --pretokenized 0 --merge_one_sent 1  -i ? -o ?
# use stanfordnlp for english/spanish
# python3 stanford.py -a "tokenize,pos,lemma,depparse" --output_format json --use_corenlp 0 --pretokenized 0 --merge_one_sent 1 -i ? -o ? -l ?
