#

# especially prepare sst from original dataset

import sys
import os
import string
from mspx.data.inst import Sent
from mspx.data.rw import WriterGetterConf
from mspx.utils import zopen, zlog

def read_file(f: str, sep: str, skip_first: bool):
    rets = []
    with zopen(f, encoding='utf8') as fd:
        if skip_first:
            fd.readline()  # skip first line
        for line in fd:
            line = line.strip()
            if line:
                fields = line.split(sep)
                rets.append(fields)
    return rets

def main(orig_dir: str, output_prefix: str):
    _printable_chars = set(string.printable)
    # --
    # read all sentences
    _all_sents = read_file(os.path.join(orig_dir, 'datasetSentences.txt'), '\t', True)
    all_sents = {z[0]: z[1] for z in _all_sents}
    assert len(_all_sents) == len(all_sents)
    zlog(f"Read {len(_all_sents)} from datasetSentences.")
    # read dict
    _all_dict = read_file(os.path.join(orig_dir, 'dictionary.txt'), '|', False)
    all_dict = {(''.join([c for c in z[0] if c in _printable_chars])): z[1] for z in _all_dict}
    zlog(f"Read dict {len(_all_dict)} vs {len(all_dict)}")
    # read label
    _all_labels = read_file(os.path.join(orig_dir, 'sentiment_labels.txt'), '|', True)
    all_labels = {z[0]: z[1] for z in _all_labels}
    # read split
    _all_splits = read_file(os.path.join(orig_dir, 'datasetSplit.txt'), ',', True)
    all_splits = {z[0]: {'1':'train','2':'test','3':'dev'}[z[1]] for z in _all_splits}
    # --
    # put them all together
    data = {'train': [], 'dev': [], 'test': []}
    CC1 = {'-LRB-': '(', '-RRB-': ')'}
    CC = {'``': '"', "''": '"', '-LRB-': '(', '-RRB-': ')'}
    for s_id, s_str in all_sents.items():
        # --
        s_str0 = ''.join([c for c in s_str if c in _printable_chars])
        if s_str0 not in all_dict:
            for a, b in CC1.items():
                s_str0 = s_str0.replace(a, b)
        # --
        _score = float(all_labels[all_dict[s_str0]])
        if _score > 0.4 and _score <= 0.6:
            continue  # skip neutral!
        _label = int(_score >= 0.5)  # note: binary!
        _wset = all_splits[s_id]
        # --
        ss = Sent([CC.get(z,z).lower() for z in s_str.split()])  # note: lowercase!
        ss.info.update({'id': s_id, 'label': _label})
        data[_wset].append(ss.make_singleton_doc())
    # --
    # write
    for k, vs in data.items():
        _output = f"{output_prefix}.{k}.json"
        with WriterGetterConf().get_writer(output_path=_output) as writer:
            zlog(f"Write to {_output}: {len(vs)}")
            writer.write_insts(vs)
    # --

if __name__ == '__main__':
    main(*sys.argv[1:])

# --
# cmds
"""
wget http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
unzip stanfordSentimentTreebank.zip -d sst_orig
python3 -m mspx.scripts.data.glue.prep_sst sst_orig/stanfordSentimentTreebank/ orig_sst2
->
# Read 11855 from datasetSentences.
# Write to orig_sst2.train.json: 6920
# Write to orig_sst2.dev.json: 872
# Write to orig_sst2.test.json: 1820
"""
