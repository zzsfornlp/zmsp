#

# prepare ace05 for ent&rel

import os
import sys
from collections import OrderedDict, Counter
from msp2.utils import zopen, zlog, default_json_serializer, zwarn
from msp2.data.inst import yield_sents, yield_frames, Sent, Doc
from msp2.data.rw import ReaderGetterConf, WriterGetterConf


def convert_dygiepp(data: dict, stat):
    # first create doc
    doc = Doc.create(id=data['doc_key'])
    stat["doc"] += 1
    # first build doc index
    doc_token_idx = []  # token-idx -> (sid, wid)
    for sid, ss in enumerate(data['sentences']):
        doc_token_idx.extend([(sid, ii) for ii in range(len(ss))])
    # then add sents
    for sid, ss in enumerate(data['sentences']):
        sent = Sent.create(ss)
        doc.add_sent(sent)
        stat["sent"] += 1
        # --
        # add ef
        ef_map = {}  # (start,end) -> Frame
        for idx0, idx1, tt in data['ner'][sid]:
            _sid, _widx = doc_token_idx[idx0]
            assert _sid == sid
            # assert (idx0,idx1) not in ef_map
            if (idx0, idx1) in ef_map:
                zwarn(f"Repeated ef in {data['ner'][sid]}")
                continue
            ef = sent.make_event(_widx, idx1-idx0+1, type=tt)
            ef_map[(idx0,idx1)] = ef
            stat['ef'] += 1
        # --
        # add rel
        for idx00, idx01, idx10, idx11, rr in data['relations'][sid]:
            stat['rel'] += 1
            a0, a1 = ef_map[(idx00, idx01)], ef_map[(idx10, idx11)]
            a0.add_arg(a1, rr)
        # --
    # --
    return doc

# --
def main(input_file: str, output_file: str):
    cc = Counter()
    docs = []
    for dd in default_json_serializer.load_list(input_file):
        one_doc = convert_dygiepp(dd, cc)
        docs.append(one_doc)
    zlog(f"Read from {input_file} to {output_file}: {cc}")
    if output_file:
        with WriterGetterConf().get_writer(output_path=output_file) as writer:
            writer.write_insts(docs)
    # --

# python3 -m msp2.tasks.zmtl3.mat.prep.prep_ace ...
if __name__ == '__main__':
    main(*sys.argv[1:])

# --
# pre-process
"""
# use those by "DyGIE"
git clone https://github.com/luanyi/DyGIE
cd DyGIE/preprocessing/
ACE2004_DIR=../../raw_ace04/
ACE2005_DIR=../../raw_ace05/
# download stanford-NLP
cd common
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2015-04-20.zip
wget http://nlp.stanford.edu/software/stanford-postagger-2015-04-20.zip
unzip stanford-corenlp-full-2015-04-20.zip
unzip stanford-postagger-2015-04-20.zip
cd ..
# ace04
# --
# The current run.zsh has issues for preprocessing in ace2004. To fix it,
#     change java -cp ".:../stanford-corenlp-full-2015-04-20/* to java -cp ".:../common/stanford-corenlp-full-2015-04-20/*
#     delete &>! log & in adjust offsets
# --
# note: to clean it: rm -rf corpus result text fixed
cp -r ${ACE2004_DIR}/*/English ace2004/english
cd ace2004
# zsh run.zsh
bash run04.sh
mkdir -p ../../data/ace04/json/train
mkdir -p ../../data/ace04/json/test
# change "ace2json.py":L157 train->test
python ace2json.py
cd ..
# ace05
cp -r ${ACE2005_DIR}/*/English ace2005/
cd ace2005
# zsh run.zsh
bash run05.sh
mkdir -p ../../data/ace05/json/
# change "ace2json.py":L93 print fn -> print(fn)
python ace2json.py
cd ..
# ==
# back to main dir
cd ../..
for wset in train dev test; do
python3 -m msp2.tasks.zmtl3.mat.prep.prep_ace DyGIE/data/ace05/json/${wset}.json ace05.${wset}.json
done
# Read from DyGIE/data/ace05/json/train.json to ace05.train.json: Counter({'ef': 26470, 'sent': 10051, 'rel': 4788, 'doc': 351})
# Read from DyGIE/data/ace05/json/dev.json to ace05.dev.json: Counter({'ef': 6338, 'sent': 2424, 'rel': 1131, 'doc': 80})
# Read from DyGIE/data/ace05/json/test.json to ace05.test.json: Counter({'ef': 5476, 'sent': 2050, 'rel': 1151, 'doc': 80})
for wset in train test; do
for ii in {0..4}; do
python3 -m msp2.tasks.zmtl3.mat.prep.prep_ace DyGIE/data/ace04/json/${wset}/${ii}.json ace04.${ii}.${wset}.json
done
done
#Read from DyGIE/data/ace04/json/train/0.json to ace04.0.train.json: Counter({'ef': 18062, 'sent': 6898, 'rel': 3292, 'doc': 279})
#Read from DyGIE/data/ace04/json/test/0.json to ace04.0.test.json: Counter({'ef': 4669, 'sent': 1785, 'rel': 795, 'doc': 69})
# python3 -mpdb -m msp2.cli.analyze frame gold:
"""
