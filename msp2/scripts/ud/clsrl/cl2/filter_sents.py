#

# filter out certain sents
# similar to "fn_filter_exemplars.py"

import sys
import os
import json
from collections import Counter
from msp2.utils import zlog, OtherHelper
from msp2.data.inst import yield_sents, yield_sent_pairs
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

# --
def guess_format(f: str):
    if f.endswith('conllu'):
        return 'conllu'
    else:
        return 'zjson'

def guess_reader(f: str):
    conf = ReaderGetterConf.direct_conf(input_path=f, input_format=guess_format(f))
    conf.validate()
    return conf.get_reader()
# --

def main(main_input: str, main_output: str, *exclude_files: str):
    hit_keys = set()
    cc = Counter()
    # first read all excluding ones
    for f in exclude_files:
        d = list(guess_reader(f))
        for s in d:
            cc['e_sent'] += 1
            _key = ' '.join(s.seq_word.vals)
            hit_keys.add(_key)
    cc['e_key'] = len(hit_keys)
    # filter main file
    survived = []
    for s in list(guess_reader(main_input)):
        cc['m_sent'] += 1
        _key = ' '.join(s.seq_word.vals)
        if _key in hit_keys:
            cc['m_exclude'] += 1
        else:
            survived.append(s)
    cc['m_survived'] = len(survived)
    # write
    assert not os.path.exists(main_output), "Avoid accident overwrite!"
    with WriterGetterConf().get_writer(output_path=main_output, output_format=guess_format(main_output)) as writer:
        writer.write_insts(survived)
    # --
    OtherHelper.printd(cc)

# python3 filter_sents.py IN OUT ...
if __name__ == '__main__':
    main(*sys.argv[1:])
