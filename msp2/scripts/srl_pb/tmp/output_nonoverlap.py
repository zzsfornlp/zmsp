#

# filter sentences with non-overlapping args

from msp2.data.inst import yield_sents
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
import numpy as np

#
def main(*aligned_files):
    RANDOM_DROP_EVT_RATE = 0.15
    # input
    aligned_insts = []
    for f in aligned_files:
        one_reader = ReaderGetterConf().get_reader(input_path=f)
        one_insts = list(one_reader)
        aligned_insts.append([z for z in yield_sents(one_insts)])
    # filter
    good_idxes = []
    for idx in range(len(aligned_insts[0])):
        sent_good = True
        for sent in yield_sents([z[idx] for z in aligned_insts]):
            if RANDOM_DROP_EVT_RATE>0:
                for evt in list(sent.events):
                    if np.random.random_sample()<RANDOM_DROP_EVT_RATE:
                        sent.delete_frame(evt, "evt")
            for evt in sent.events:
                hits = set()
                for arg in evt.args:
                    widx, wlen = arg.arg.mention.get_span()
                    for ii in range(widx, widx+wlen):
                        if ii in hits:
                            sent_good = False
                        hits.add(ii)
        if sent_good:
            good_idxes.append(idx)
    # output
    output_prefix = "_tmp.json"
    for outi, insts in enumerate(aligned_insts):
        filtered_insts = [insts[ii] for ii in good_idxes]
        writer = WriterGetterConf().get_writer(output_path=f"{output_prefix}{outi}")
        writer.write_insts(filtered_insts)
        writer.close()
    # --

# PYTHONPATH=../src/ python3 output_nonoverlap.py ...
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
