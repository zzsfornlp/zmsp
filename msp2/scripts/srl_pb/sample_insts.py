#

# sample instances
# -- also remove repeat (slightly similar to "fn_filter_exemplars")

import os
from typing import List
from collections import defaultdict
from msp2.utils import zlog, zwarn, zopen, system, Random, OtherHelper
from msp2.data.inst import Sent, yield_sents
from msp2.data.rw import get_reader, ReaderGetterConf, get_writer, WriterGetterConf

def get_stat(sents):
    ret = defaultdict(int)
    ret["sent"] += len(sents)
    ret["events"] += sum(len(s.events) for s in sents)
    ret["args"] += sum(len(e.args) for s in sents for e in s.events)
    return ret

def main(input_file: str, output_file: str, checking_file: str, keep_rate: float):
    keep_rate = float(keep_rate)
    _gen = Random.get_np_generator(12345)
    rstream = Random.stream(_gen.random_sample)
    # --
    # read input
    stat = {}
    input_sents = list(yield_sents(ReaderGetterConf().get_reader(input_path=input_file)))
    stat["input"] = get_stat(input_sents)
    if checking_file:
        checking_sents = list(yield_sents(ReaderGetterConf().get_reader(input_path=checking_file)))
        stat["check"] = get_stat(checking_sents)
        # collect keys
        hit_keys = set()
        for one_check_sent in checking_sents:
            tok_key = ''.join(one_check_sent.seq_word.vals).lower()
            tok_key = ''.join(tok_key.split())  # split and join again
            hit_keys.add(tok_key)
        # filter
        filtered_sents = []
        for one_input_sent in input_sents:
            tok_key = ''.join(one_input_sent.seq_word.vals).lower()
            tok_key = ''.join(tok_key.split())  # split and join again
            if tok_key not in hit_keys:
                filtered_sents.append(one_input_sent)
    else:
        filtered_sents = input_sents
    stat["filter"] = get_stat(filtered_sents)
    # sample
    if keep_rate < 1.:
        sample_sents = [s for r,s in zip(rstream, filtered_sents) if r<keep_rate]
    elif keep_rate > 10:
        sample_sents = [z for z in filtered_sents]
        for _ in range(10):
            _gen.shuffle(sample_sents)
        sample_sents = sample_sents[:int(keep_rate)]
    else:
        sample_sents = filtered_sents
    stat["sample"] = get_stat(sample_sents)
    # write
    if os.path.exists(output_file):
        assert False, f"File exists: {output_file}, delete it first!"
    if output_file:
        with WriterGetterConf().get_writer(output_path=output_file) as writer:
            writer.write_insts(sample_sents)
    # stat
    zlog(f"Read {input_file}, check {checking_file}, output {output_file}, stat:")
    OtherHelper.printd(stat)

# PYTHONPATH=../../../zsp2021/src/ python3 sample_insts.py ../conll12/train.conll.ud.json train2.conll.ud.json train.conll.ud.json 1.0
# PYTHONPATH=../../../zsp2021/src/ python3 sample_insts.py dev.conll.ud.json dev2.conll.ud.json '' 0.2
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
