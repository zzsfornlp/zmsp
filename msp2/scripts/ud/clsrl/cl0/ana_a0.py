#

# specifically analyze A0
# -- since gcn's fr's A0 is much worse

from typing import List
import numpy as np
from msp2.data.inst import Sent, ArgLink, Frame, Token, Mention, \
    yield_sent_pairs, yield_sents, MyPrettyPrinter, set_ee_heads
from msp2.data.rw import ReaderGetterConf
from msp2.utils import Conf, init_everything
from msp2.proc.eval import ItemMatcher

# --
def main(files: List):
    all_sents = [list(yield_sents(ReaderGetterConf().get_reader(input_path=f))) for f in files]
    num_file = len(files)
    # --
    filter_f = (lambda a: a.label == "A0")
    matches = {}
    for idx1 in range(num_file):
        for idx2 in range(idx1+1, num_file):
            sents1, sents2 = all_sents[idx1], all_sents[idx2]
            assert len(sents1) == len(sents2)
            _cur_match = [[], [], []]  # matched, unmatched1, unmatched2
            for s1, s2 in zip(sents1, sents2):
                args1 = list(a for f in s1.events for a in f.args if filter_f(a))
                args2 = list(a for f in s2.events for a in f.args if filter_f(a))
                # match them
                matching_arr = [[
                    float(a1.mention.get_span()==a2.mention.get_span() and a1.main.mention.get_span()==a2.main.mention.get_span())
                    for a2 in args2] for a1 in args1]
                matched_pairs, unmatched1, unmatched2 = ItemMatcher.match_simple(
                    np.asarray(matching_arr).reshape((len(args1), len(args2))))
                _cur_match[0].extend([(args1[i], args2[j]) for i,j in matched_pairs])
                _cur_match[1].extend([args1[i] for i in unmatched1])
                _cur_match[2].extend([args2[j] for j in unmatched2])
            # --
            _num_match = len(_cur_match[0])
            results = [f"{_num_match}/{len(z)+_num_match}={_num_match/(len(z)+_num_match):.4f}" for z in _cur_match[1:]]
            print(f"{(idx1, idx2)}: {results}")
            matches[(idx1, idx2)] = _cur_match
    # --
    breakpoint()
    # --

# PYTHONPATH=../src/ python3 ana_a0.py ...
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
