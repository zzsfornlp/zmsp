#

# get and print (BIO) label transitions

from collections import Counter
import numpy as np
from msp2.data.inst import yield_sents, set_ee_heads
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.data.vocab import SimpleVocab, SeqVocab
from msp2.utils import ZObject, OtherHelper, default_pickle_serializer, zlog

#
def main(vocab_file: str, input_path: str, output_file='lt.pkl'):
    # first get vocab
    vocabs = default_pickle_serializer.from_file(vocab_file)
    arg_voc = vocabs[0]['arg']
    zlog(f"Read {arg_voc} from {vocab_file}")
    # make it to BIO-vocab
    bio_voc = SeqVocab(arg_voc)
    zlog(f"Build bio-voc of {bio_voc}")
    # read insts
    insts = list(ReaderGetterConf().get_reader(input_path=input_path))  # read from stdin
    all_sents = list(yield_sents(insts))
    # --
    mat = np.ones([len(bio_voc), len(bio_voc)], dtype=np.float32)  # add-1 smoothing!
    cc = Counter()
    for sent in all_sents:
        for evt in sent.events:
            labels = ['O'] * len(sent)
            for arg in evt.args:
                widx, wlen = arg.mention.get_span()
                labels[widx:wlen] = ["B-"+arg.role] + ["I-"+arg.role] * (wlen-1)
            for a,b in zip(labels, labels[1:]):
                cc[f"{a}->{b}"] += 1
                mat[bio_voc[a], bio_voc[b]] += 1
        # --
    # --
    v = SimpleVocab()
    for name, count in cc.items():
        v.feed_one(name, count)
    v.build_sort()
    print(v.get_info_table()[:50].to_string())
    # OtherHelper.printd(cc)
    # --
    # normalize & log according to row and save
    mat = mat / mat.sum(-1, keepdims=True)
    mat = np.log(mat)
    default_pickle_serializer.to_file(mat, output_file)
    # --

# PYTHONPATH=../src/ python3 get_label_transition.py [vocab] [input]
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
