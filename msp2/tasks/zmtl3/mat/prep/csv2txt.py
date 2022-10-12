#

import os
import sys
from collections import Counter, OrderedDict
import pandas as pd

def main(output_prefix: str, *input_files):
    all_txts = OrderedDict()
    for f in input_files:
        f0 = os.path.basename(f)
        if '.' in f0:
            f0 = '.'.join(f0.split('.')[:-1])
        df = pd.read_csv(f)
        print(f"Read {f}: L={len(df)}")
        for _, d in df.iterrows():
            num = d['FileName'].split('/')[-1]
            key = f"{f0}_{num}"
            assert key not in all_txts
            all_txts[key] = d['Abstract']
    print(f"Read all = {len(all_txts)}, write to {output_prefix}*")
    for key, txt in all_txts.items():
        outf = output_prefix + key + '.txt'
        with open(outf, 'w') as fd:
            fd.write(txt)
    # --

# python3 csv2txt.py ...
# python3 -m msp2.tasks.zmtl3.mat.prep.csv2txt ...
if __name__ == '__main__':
    main(*sys.argv[1:])

# test0527
"""
# prepare data
mkdir -p data
python3 -m msp2.tasks.zmtl3.mat.prep.csv2txt data/ *.csv
python3 -m msp2.tasks.zmtl3.mat.prep.brat2json data mat.t0527.json
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:en stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:mat.t0527.json output_path:mat.t0527.ud2.json
# --
# Read AM_MechanicalProperties.csv: L=94
# Read AM_MeltpoolNet.csv: L=74
# Read all = 168, write to data/*
# Read from data to mat.t0527.json: Counter({'sent': 1220, 'doc': 168})
# --
# predict data
function run_pred () {
IN_FILE=$1
OUT_PREFIX=$2
DIR_ENT=$3
DIR_REL=$4
python3 -m msp2.tasks.zmtl3.main.test ${DIR_ENT}/_conf model_load_name:${DIR_ENT}/zmodel.best.m vocab_load_dir:${DIR_ENT}/ nn.device:0 log_stderr:1 testM.group_files:${IN_FILE} testM.output_file:${OUT_PREFIX}.s1.json testM.group_tasks:matM
python3 -m msp2.tasks.zmtl3.main.test ${DIR_REL}/_conf model_load_name:${DIR_REL}/zmodel.best.m vocab_load_dir:${DIR_REL}/ nn.device:0 log_stderr:1 testM.group_files:${OUT_PREFIX}.s1.json testM.output_file:${OUT_PREFIX}.s2.json testM.group_tasks:matR
python3 -m msp2.tasks.zmtl3.mat.prep.json2brat ${OUT_PREFIX}.s2.json ${OUT_PREFIX}/
zip -r ${OUT_PREFIX}.zip ${OUT_PREFIX}/
}
# --
run_pred mat.t0527.ud2.json output_data0524 ../run_mgo_0524v0M_0/ ../run_mgo_0524v0R_0/ |& tee _log
Read from output_data0524.s2.json to output_data0524/: Counter({'evt': 7159, 'rel': 4654, 'sent': 1220, 'inst': 168})
"""
