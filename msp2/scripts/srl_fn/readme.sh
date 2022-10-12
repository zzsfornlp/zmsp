#

# prepare fn data

# =====
# get data
# m1: fill form and download
#https://framenet.icsi.berkeley.edu/fndrupal/framenet_request_data
# m2: nltk
# import nltk
# nltk.download('framenet_v15') or nltk.download('framenet_v17')
# --
# => ${DATA_DIR}/fn/fndata-{1.5,1.7}

# -----
# step0: paths
DATA_DIR="$HOME/working/data/"
SRC_DIR="../../src/"

# -----
# step 1: read data into doc format ("fn_reader.py")
PYTHONPATH=${SRC_DIR} python3 -m msp2.scripts.srl_fn.fn_reader ${DATA_DIR}/fn/fndata-1.5 fn15 |& tee _log.fn15
PYTHONPATH=${SRC_DIR} python3 -m msp2.scripts.srl_fn.fn_reader ${DATA_DIR}/fn/fndata-1.7 fn17 |& tee _log.fn17

# -----
# step 2: some more processings
for ff in fn15 fn17; do
# filter examplars
PYTHONPATH=${SRC_DIR} python3 -m msp2.scripts.srl_fn.fn_filter_exemplars ${ff}/fulltext.json ${ff}/exemplars.json ${ff}/exemplars.filtered.json
# split data
PYTHONPATH=${SRC_DIR} python3 -m msp2.scripts.srl_fn.fn_split_ft ${ff}/fulltext.json ${ff}/fulltext
done

# -----
# optional: print and check
# exmaple: print arg_repeat frames
PYTHONPATH=${SRC_DIR} python3 -m msp2.scripts.srl_fn.fn_print_frame input_path:fn15/fulltext.json "filter_code:arg_repeat"

# -----
# optional: stat
mkdir -p logs
for ff in fn15 fn17; do
export _CACHE_FRAMES="$ff/frames.json"
PYTHONPATH=${SRC_DIR} python3 -m msp2.scripts.srl_fn.fn_stat frames $ff/frames.json 1 |& tee logs/_${ff}_frames.log
PYTHONPATH=${SRC_DIR} python3 -m msp2.scripts.srl_fn.fn_stat sents $ff/exemplars.json 1 |& tee logs/_${ff}_exemplars.log
PYTHONPATH=${SRC_DIR} python3 -m msp2.scripts.srl_fn.fn_stat sents $ff/exemplars.filtered.json 1 |& tee logs/_${ff}_exemplars_filtered.log
for ff2 in train dev test test1; do
PYTHONPATH=${SRC_DIR} python3 -m msp2.scripts.srl_fn.fn_stat docs $ff/fulltext.$ff2.json 1 |& tee logs/_${ff}_${ff2}.log
done
unset _CACHE_FRAMES
done

# -----
# step?: pos, lemma and depparse (assume pre-tokenized)
mkdir -p parsed
for ff in fn15 fn17; do
for ff2 in train dev test test1; do
PYTHONPATH=${SRC_DIR} python3 -m msp2.cli.annotate 'stanza' input_path:${ff}/fulltext.${ff2}.json output_path:parsed/${ff}_fulltext.${ff2}.json stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized
done
PYTHONPATH=${SRC_DIR} python3 -m msp2.cli.annotate 'stanza' input_path:${ff}/exemplars.filtered.json output_path:parsed/${ff}_exemplars.filtered.json stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:32
done
