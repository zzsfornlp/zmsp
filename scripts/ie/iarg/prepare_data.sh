#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname ${BASH_SOURCE[0]}  )" && pwd )"

# step 0: get RAMS data
# -- as a result, we have the original data in RAMS_1.0/data/*.jsonlines
wget http://www.cs.jhu.edu/~paxia/rams/RAMS_1.0.tar.gz
tar -zxvf RAMS_1.0.tar.gz
DATA_DIR="./RAMS_1.0/data/"

# step 1: transform to our own format (simply for convenience) and parse using stanfordnlp
# Note: need (a relative old version) stanfordnlp=0.2.0 and corresponding model
# (in Python): import stanfordnlp; stanfordnlp.download('en')
for dset in train dev test; do
python3 ${SCRIPT_DIR}/rams2tok.py ${DATA_DIR}/${dset}.jsonlines ${DATA_DIR}/en.rams.${dset}.tok.json RAMS en
CUDA_VISIBLE_DEVICES=0 python3 ${SCRIPT_DIR}/tok2parse.py ${DATA_DIR}/en.rams.${dset}.tok.json ${DATA_DIR}/en.rams.${dset}.parse.json en
ln -s en.rams.${dset}.parse.json ${DATA_DIR}/en.rams.${dset}.json
done

# step 2: do training and testing
# please refer to "./go20.sh"
