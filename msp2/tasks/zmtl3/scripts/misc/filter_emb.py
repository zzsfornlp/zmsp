#

# filter embedding file according to input files

import sys
from copy import deepcopy
from collections import Counter, defaultdict, OrderedDict
from msp2.data.inst import yield_sents, set_ee_heads
from msp2.utils import Conf, zlog, zglob, init_everything, ZObject, wrap_color, default_pickle_serializer
from msp2.data.vocab import WordVectors
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

# --

class MainConf(Conf):
    def __init__(self):
        self.emb = ""
        self.inputs = []
        self.output = ""

# --
def main(*args):
    conf = MainConf()
    conf: MainConf = init_everything(conf, args)
    # --
    # read emb
    wv = WordVectors.load(conf.emb)
    sf = []
    for s in ['<s>', '</s>', '<unk>']:
        r = wv.find_key(s)
        if r is not None:
            sf.append(r)
    zlog(f"Find special keys: {sf}")
    # read files
    input_files = sum([zglob(f) for f in conf.inputs], [])
    for f in input_files:
        zlog(f"Read from {f}")
        for inst in ReaderGetterConf().get_reader(input_path=f):
            for sent in yield_sents(inst):
                seqs = sent.seq_word.vals + ([] if sent.seq_lemma is None else sent.seq_lemma.vals)
                seqs = [z for z in seqs if z is not None]
                seqs = seqs + [z.lower() for z in seqs]  # allow lowercase
                for w in seqs:
                    wv.find_key(w)
    # save file
    if conf.output:
        wv.save_hits(conf.output)
    # --

# --
if __name__ == '__main__':
    main(*sys.argv[1:])

"""
# get embs
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
unzip crawl-300d-2M-subword.zip
# for debug
cat crawl-300d-2M-subword.vec | head -1000 >debug.vec
# filter by all event files
python3 -m msp2.tasks.zmtl3.scripts.misc.filter_emb emb:crawl-300d-2M-subword.vec "inputs:../../events/data/data21f/en.*2.train.json,../../events/data/data21f/en.*2.dev.json,../../events/data/data21f/en.*2.test.json" output:cc2ms.filter_evt.vec |& tee _log.filter_evt
# --
# run with it
python3 -m msp2.tasks.zmtl3.main.test "conf_sbase:data:ace;test_case:case1" device:0 test1.group_files: evt0.bconf:wvec evt0.b_model:../embs/cc2ms.filter_evt.vec evt0.bert_lidx:-1 evt0.neg_delta.init:0.05
# -> 30+, similar to bert-layer0! (see _debug_nndec211208 for more details)
"""
