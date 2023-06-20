#

# use stanford parser tool to convert phrase-tree to dep-tree

import os
from collections import Counter
from typing import List
from mspx.utils import zlog, zwarn, zopen, system, Conf, init_everything
from mspx.data.inst import Sent, yield_sents
from mspx.data.rw import WriterGetterConf, ReaderGetterConf

# =====
def corenlp_getter(version='4.5.1'):
    CORENLP_GETTER = f"""
    # prepare stanford nlp {version}
    wget http://nlp.stanford.edu/software/stanford-corenlp-{version}.zip
    unzip stanford-corenlp-{version}.zip
    #wget http://nlp.stanford.edu/software/stanford-corenlp-{version}-models-english.jar
    # get the upos mapper rules
    wget https://raw.githubusercontent.com/stanfordnlp/CoreNLP/master/data/edu/stanford/nlp/upos/ENUniversalPOS.tsurgeon
    mkdir -p edu/stanford/nlp/models/upos/
    mv ENUniversalPOS.tsurgeon edu/stanford/nlp/models/upos/
    jar cf stanford-parser-missing-file.jar edu/stanford/nlp/models/upos/ENUniversalPOS.tsurgeon
    mv stanford-parser-missing-file.jar stanford-corenlp-{version}
    """
    return CORENLP_GETTER

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
    # return pieces2tree(sent.seq_word.vals, sent.info["xpos"], sent.info["parse"])
    return pieces2tree(sent.seq_word.vals, sent.info["xpos"], sent.info["xparse"])
# =====

class MainConf(Conf):
    def __init__(self):
        # specify corenlp
        self.p2d_home = ""  # use "this/stanford-corenlp-4.5.1.jar:this/stanford-parser-missing-file.jar"
        self.p2d_version = "4.5.1"  # corenlp version
        self.p2d_tmp_dir = "./"  # tmp dir (by default current dir)
        self.p2d_log = ""  # output of semafor running
        self.p2d_lang = "en"  # en(ud), en-sd
        self.p2d_use_xpos = False  # instead of read upos, use xpos instead
        self.p2d_change_words = True  # whether change words if mismatch?
        # converters?
        self.p2d_upos_converter = "lambda x: x"
        self.p2d_udp_converter = "lambda x: x"
        # --
        self.R = ReaderGetterConf()
        self.W = WriterGetterConf()
        # --

class AnnotatorP2D:
    def __init__(self, conf: MainConf):
        self.conf = conf
        _corenlp_home = conf.p2d_home
        if _corenlp_home == "":
            _corenlp_home = os.environ.get("CORENLP_HOME", "")  # try to read env
        if _corenlp_home == "":
            _corenlp_home = f"stanford-corenlp-{conf.p2d_version}"
            if not os.path.exists(_corenlp_home):
                zwarn("Obtain corenlp!")
                _cmd = corenlp_getter(conf.p2d_version)
                system(_cmd)
        self.corenlp_home = _corenlp_home
        # --
        self.cmd = f"java -Xmx8g -cp {self.corenlp_home}/stanford-corenlp-{conf.p2d_version}.jar:{self.corenlp_home}/stanford-parser-missing-file.jar edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -conllx -keepPunct -language {conf.p2d_lang}"
        # --
        self.p2d_upos_converter = eval(conf.p2d_upos_converter)
        self.p2d_udp_converter = eval(conf.p2d_udp_converter)
        # --

    # note: should be run in large batch!
    def annotate(self, insts: List, cc):
        conf: MainConf = self.conf
        # --
        # get all sentences and run in batch
        all_sents = list(yield_sents(insts))
        tmp_input = os.path.join(conf.p2d_tmp_dir, "_input.penn")
        with zopen(tmp_input, 'w') as fd:
            for sent in all_sents:
                fd.write(sent2tree(sent)+"\n")
        cc['inst'] += len(insts)
        cc['sent'] += len(all_sents)
        # run
        tmp_output = os.path.join(conf.p2d_tmp_dir, "_output.conllu")
        log_cmd = f'2>{conf.p2d_log}' if conf.p2d_log else ''
        system(f"{self.cmd} -treeFile {tmp_input} >{tmp_output} {log_cmd}", pp=True)
        # read output and add back
        new_sents = list(yield_sents(ReaderGetterConf().get_reader(input_path=tmp_output, input_format='conllu')))
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

def main(args):
    conf: MainConf = init_everything(MainConf(), args)
    cc = Counter()
    # --
    ann = AnnotatorP2D(conf)
    insts = list(conf.R.get_reader())  # simply one batch!
    ann.annotate(insts, cc)
    if conf.W.has_path():
        with conf.W.get_writer() as writer:
            writer.write_insts(insts)
    zlog(f"Finish with {cc}")
    # --

# python3 -m mspx.scripts.data.onto.phrase2dep input_path:??
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
