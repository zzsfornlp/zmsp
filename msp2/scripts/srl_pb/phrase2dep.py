#

# use stanford parser tool to convert phrase-tree to dep-tree

import os
from typing import List
from msp2.utils import zlog, zwarn, zopen, system
from msp2.data.inst import Sent, yield_sents
from msp2.data.rw import get_reader, ReaderGetterConf
from msp2.tools.annotate import AnnotatorConf, Annotator

# =====
"""
# prepare stanford nlp 4.1.0
wget http://nlp.stanford.edu/software/stanford-corenlp-4.1.0.zip
unzip stanford-corenlp-4.1.0.zip
wget http://nlp.stanford.edu/software/stanford-corenlp-4.1.0-models-english.jar
# get the upos mapper rules
wget https://raw.githubusercontent.com/stanfordnlp/CoreNLP/master/data/edu/stanford/nlp/upos/ENUniversalPOS.tsurgeon
mkdir -p edu/stanford/nlp/models/upos/
mv ENUniversalPOS.tsurgeon edu/stanford/nlp/models/upos/
jar cf stanford-parser-missing-file.jar edu/stanford/nlp/models/upos/ENUniversalPOS.tsurgeon
mv stanford-parser-missing-file.jar stanford-corenlp-4.1.0
"""

# =====
# some converters

# todo(note): extras: "URL" "X"
UPOS2CTB = {
    "ADJ": ["JJ", "VA", ],
    "ADV": ["AD", ],
    "INTJ": ["IJ", ],
    "NOUN": ["M", "NN", "NT", ],
    "PROPN": ["NR", ],
    "VERB": ["VC", "VV", "VE", ],
    "ADP": ["LC", "P", ],
    "AUX": ["BA", "LB", "MSP", "SB", ],
    "CCONJ": ["CC", "CS", ],
    "DET": ["DT", "INF"],  # note: INF is a strange one, but seems to be here
    "NUM": ["CD", "OD", ],
    "PART": ["AS", "DEC", "DEG", "DER", "DEV", "SP"],
    "PRON": ["PN", ],
    "SCONJ": [],
    "PUNCT": ["PU", ],
    "SYM": ["FW", "URL"],
    "X": ["ETC", "ON", "X"],
}
CTB2UPOS = {}
for _t_ud, _ts_ctb in UPOS2CTB.items():
    for z in _ts_ctb:
        CTB2UPOS[z] = _t_ud

UD2_LABEL_LIST = ["punct", "case", "nsubj", "det", "root", "nmod", "advmod", "obj", "obl", "amod", "compound", "aux", "conj", "mark", "cc", "cop", "advcl", "acl", "xcomp", "nummod", "ccomp", "appos", "flat", "parataxis", "discourse", "expl", "fixed", "list", "iobj", "csubj", "goeswith", "vocative", "reparandum", "orphan", "dep", "dislocated", "clf"]
UD1to2 = {w:w for w in UD2_LABEL_LIST}
UD1to2.update({"dobj": "obj", "neg": "advmod", "nsubjpass": "nsubj", "csubjpass": "csubj", "auxpass": "aux",
               "name": "compound", "erased": "dep", "etc": "dep"})  # note: the last three are not super clear ...

# =====
# tools

# sent -> tree
def pieces2tree(words: List[str], xposes: List[str], parses: List[str]):
    rets = []
    assert len(words)==len(xposes) and len(words)==len(parses)
    for w, xp, pp in zip(words, xposes, parses):
        try:
            p0, p1 = pp.split("*")  # must be two pieces
        except:  # note: this can be caused by empty [word]!
            zwarn(f"Bad parse-bit: {pp}, assume that is '*'")
            p0, p1 = '', ''
            if xp in ["*", "-"]:
                xp = "XX"  # also fix pos
        new_w = []
        for c in w:
            # note: for simplicity, change "{" to "[" to avoid -LCB-, ...
            new_w.append(
                {'(': "-LRB-", ')': "-RRB-", '＜': "<", '＞': ">", '［': "<", '］': ">",
                 '{': "<", '}': ">", '｛': "<", '｝': ">", '〈': "<", '〉': ">"}.get(c, c))
        if xp=='(': xp = "-LRB-"
        elif xp==')': xp = "-RRB-"
        rets.append(f"{p0} ({xp} {''.join(new_w)}) {p1}")
    tree_ret = " ".join(rets)
    tree_fix = check_and_fix_tree(tree_ret)
    return tree_fix

# todo(+W): currently very simple check and fix!
def check_and_fix_tree(s: str):
    cur_depth = 0
    hit_zero = 0
    for c in s:
        if c == "(":
            cur_depth += 1
        elif c == ")":
            cur_depth -= 1
        else:
            continue  # ignore others!
        assert cur_depth >= 0
        if cur_depth == 0:
            hit_zero += 1
    assert cur_depth == 0
    if hit_zero != 1:
        zwarn(f"Strange tree pieces={hit_zero}: {s}")
        return f"(S {s} )"  # simple fix
    else:
        return s

def sent2tree(sent: Sent):
    return pieces2tree(sent.seq_word.vals, sent.info["xpos"], sent.info["parse"])
# =====

class AnnotatorP2DConf(AnnotatorConf):
    def __init__(self):
        super().__init__()
        # specify corenlp
        self.p2d_home = ""  # use "this/stanford-corenlp-4.1.0.jar:this/stanford-parser-missing-file.jar"
        self.p2d_tmp_dir = "./"  # tmp dir (by default current dir)
        self.p2d_log = ""  # output of semafor running
        self.p2d_lang = "en"  # en(ud), en-sd
        self.p2d_use_xpos = False  # instead of read upos, use xpos instead
        self.p2d_change_words = True  # whether change words if mismatch?
        # converters?
        self.p2d_upos_converter = "lambda x: x"
        self.p2d_udp_converter = "lambda x: x"
        # --

@Annotator.reg_decorator("p2d", conf=AnnotatorP2DConf)
class AnnotatorP2D(Annotator):
    def __init__(self, conf: AnnotatorP2DConf):
        super().__init__(conf)
        conf: AnnotatorP2DConf = self.conf
        # --
        self.corenlp_home = conf.p2d_home
        if self.corenlp_home == "":
            # try to read env
            self.corenlp_home = os.environ.get("CORENLP_HOME", "")
        assert self.corenlp_home != "", "Please provide 'corenlp_home': either by conf or ${CORENLP_HOME}"
        # --
        self.cmd = f"java -Xmx8g -cp {self.corenlp_home}/stanford-corenlp-4.1.0.jar:{self.corenlp_home}/stanford-parser-missing-file.jar edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -conllx -keepPunct -language {conf.p2d_lang}"
        # --
        self.p2d_upos_converter = eval(conf.p2d_upos_converter)
        self.p2d_udp_converter = eval(conf.p2d_udp_converter)
        # --

    # note: should be run in large batch!
    def annotate(self, insts: List):
        conf: AnnotatorP2DConf = self.conf
        # --
        # get all sentences and run in batch
        all_sents = list(yield_sents(insts))
        tmp_input = os.path.join(conf.p2d_tmp_dir, "_input.penn")
        with zopen(tmp_input, 'w') as fd:
            for sent in all_sents:
                fd.write(sent2tree(sent)+"\n")
        # run
        tmp_output = os.path.join(conf.p2d_tmp_dir, "_output.conllu")
        log_cmd = f'2>{conf.p2d_log}' if conf.p2d_log else ''
        system(f"{self.cmd} -treeFile {tmp_input} >{tmp_output} {log_cmd}")
        # read output and add back
        conll_reader_conf = ReaderGetterConf()
        conll_reader_conf.input_conf.use_multiline = True
        conll_reader_conf.input_conf.mtl_ignore_f = "'ignore_#'"
        conll_reader_conf.input_format = "conllu"
        conll_reader_conf.input_path = tmp_output
        conll_reader = get_reader(conll_reader_conf)
        new_sents = list(conll_reader)
        # --
        assert len(all_sents) == len(new_sents)
        for s0, s1 in zip(all_sents, new_sents):
            assert len(s0) == len(s1)
            mismatched_tokens = [(v1,v2) for v1,v2 in zip(s0.seq_word.vals, s1.seq_word.vals) if v1!=v2]
            if len(mismatched_tokens) > 0:
                zwarn(f"Mismatch token NUM={len(mismatched_tokens)}: {mismatched_tokens}")
                if conf.p2d_change_words:
                    s0.build_words(s1.seq_word.vals)  # use the other one!!
                # breakpoint()
            # note: build again!
            s0.build_dep_tree(s1.tree_dep.seq_head.vals, [self.p2d_udp_converter(z) for z in s1.tree_dep.seq_label.vals])
            if conf.p2d_use_xpos:
                trg_pos_list = s1.info.get["xpos"]
            else:
                trg_pos_list = s1.seq_upos.vals
            s0.build_uposes([self.p2d_upos_converter(z) for z in trg_pos_list])
        # --

# example
#PYTHONPATH=../src/ python3 -m pdb -m msp2.cli.annotate msp2.scripts.srl_pb.phrase2dep/p2d ann_batch_size:1000000 p2d_home:stanford-corenlp-4.1.0 input_path:_tmp.json output_path:_tmp2.json
