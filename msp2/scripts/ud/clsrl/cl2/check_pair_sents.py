#

# tool for checking pair's sentences

from collections import Counter
from msp2.utils import Conf, zlog, init_everything, zwarn, Constants, AlgoHelper
from msp2.data.inst import yield_sents, yield_sent_pairs
from msp2.data.rw import ReaderGetterConf, WriterGetterConf


def main(f1: str, f2: str, strip_ar='0'):
    strip_ar = int(strip_ar)
    data1 = ReaderGetterConf().get_reader(input_path=f1)
    data2 = ReaderGetterConf().get_reader(input_path=f2)
    cc = Counter()
    for s1, s2 in yield_sent_pairs(list(data1), list(data2)):
        words1, words2 = s1.seq_word.vals, s2.seq_word.vals
        if strip_ar:
            import pyarabic.araby as araby
            # strip_harakat or strip_tashkeel
            words1 = [araby.strip_tashkeel(w) for w in words1]
            words2 = [araby.strip_tashkeel(w) for w in words2]
        assert len(words1) == len(words2)
        cc['sent'] += 1
        cc['tok'] += len(words1)
        all_toks = []
        mismatched = 0
        for w1, w2 in zip(words1, words2):
            if w1 != w2:
                cc['tok_mismatch'] += 1
                all_toks.append(f"{w1}({w2})")
                mismatched = 1
            else:
                all_toks.append(w1)
        cc['sent_mismatch'] += mismatched
        if mismatched:
            print(" ".join(all_toks))
    print(cc)
    # --

# --
# python3 check_pair_sents.py ...
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
