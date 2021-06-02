#!/usr/bin/env bash

# train on one source language and apply to others

# =====
# basic
train_opt="init_from_pretrain:1 emb_conf.dim_char:0 emb_conf.word_freeze:1 use_label0:1 pretrain_init_nohit:0."
train_opt+=" lrate.min_val:0.0001 lrate.k:0.75 patience:8 max_epochs:500"
# =====
# decoder
train_opt+=" partype:graph dec_algorithm:unproj output_normalizing:local loss_function:prob margin.init_val:0."
# =====
if [[ ${USE_ATT} == 1 ]]; then
# encoder
# -----
train_opt+=" emb_proj_dim:512 enc_hidden:512 enc_rnn_layer:0"
# self-att
train_opt+=" enc_att_layer:6 enc_att_add_wrapper:addnorm enc_att_rel_clip:10 enc_att_rel_neg:0 enc_att_dropout:0.1"
# att-training
train_opt+=" max_epochs:300 train_skip_length:80 tconf.batch_size:80 split_batch:4"
train_opt+=" lrate.init_val:0.001 lrate_warmup:-1 drop_embed:0.1 dropmd_embed:0. drop_hidden:0.1"
fi
# -----

if [[ ${DELEX} == 1 ]]; then
train_opt+=" dim_word:0 init_from_pretrain:0 pretrain_file:"
fi

function run_one () {
#
SRC_DIR="../src/"
DATA_DIR="../data/ud13/"
CUR_RUN=0
#
# train
CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.train train:${DATA_DIR}/${CUR_LANG}_train.conllu dev:${DATA_DIR}/${CUR_LANG}_dev.conllu test:${DATA_DIR}/${CUR_LANG}_test.conllu pretrain_file:${DATA_DIR}/wiki.multi.${CUR_LANG}.vec device:0 ${train_opt} "niconf.random_seed:9347${CUR_RUN}" "niconf.random_cuda_seed:9349${CUR_RUN}" conf_output:./_confm |& tee log.train
# test
for cl in de en fa ca es fr it ro bg cs hr ru zh et tr;
do
if [[ ${DELEX} == 1 ]]; then
test_opt=""
else
test_opt=" test_extra_pretrain_files:${DATA_DIR}/wiki.multi.${cl}.vec"
fi
    CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.test ./_confm conf_output: device:0 test:${DATA_DIR}/${cl}_test.conllu model_load_name:./zmodel.best ${test_opt} output_file:output.${cl}_test.conllu;
done |& tee log.test
}

#
run_one

# RGPU=0 CUR_LANG=en DELEX=0 USE_ATT=0 bash _go.sh |& tee log
