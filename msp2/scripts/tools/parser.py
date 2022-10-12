#

# a simple parser (using stanza)

import sys
from typing import List
from msp2.utils import Conf, zlog, init_everything
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.data.inst import MyPrettyPrinter, yield_sents
from msp2.tools.annotate.ann_stanza import AnnotatorStanzaConf, AnnotatorStanza
from msp2.nn import BK

class MainConf(Conf):
    def __init__(self):
        self.R = ReaderGetterConf.direct_conf(input_format='raw_sent')  # one sentence one line!
        # self.W = WriterGetterConf()
        # --
        self.stanza = AnnotatorStanzaConf.direct_conf(
            stanza_lang='en', stanza_processors='tokenize,pos,lemma,depparse'.split(','), stanza_use_gpu=False)
        self.do_ann = True
        self.do_print = True
        self.debug = False
        # --

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    reader = conf.R.get_reader()
    annotator = AnnotatorStanza(conf.stanza) if conf.do_ann else None
    all_insts = []
    zlog(">> ", end='')
    for inst in yield_sents(reader):
        if conf.do_ann:
            annotator.annotate([inst])
        if conf.do_print:
            zlog(MyPrettyPrinter.str_fnode(inst, inst.tree_dep.fnode))
        if conf.debug:
            breakpoint()
        all_insts.append(inst)
        zlog(">> ", end='')
    # --

# --
# PYTHONPATH=../src/ python3 -m msp2.scripts.tools.parser
# PYTHONPATH=../src/ python3 -m msp2.scripts.tools.parser input_format:zjson input_path:?? do_ann:0
if __name__ == '__main__':
    main(*sys.argv[1:])
