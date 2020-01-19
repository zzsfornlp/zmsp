#!/usr/bin/env bash

# train from g1 init

# =====
function run_one () {
# check envs
echo "The environment variables: RGPU=${RGPU} CUR_LANG=${CUR_LANG} CUR_RUN=${CUR_RUN}";

# =====
# basis (same as basic g1 for architecture)
#base_opt="conf_output:_conf init_from_pretrain:1 drop_embed:0.33 dropmd_embed:0.33 singleton_thr:2 fix_drop:1"
base_opt="conf_output:_conf init_from_pretrain:1 drop_embed:0.2 dropmd_embed:0.2 singleton_thr:2 fix_drop:1"
base_opt+=" arc_space:512 lab_space:128 transform_act:elu biaffine_div:1. biaffine_init_ortho:1"

# =====
# lrate and margin
if [[ -n "${G1MODEL}" ]]; then
# pre-load and pre-valid
base_opt+=" g1_pretrain_path:${G1MODEL} g1_pretrain_init:1 validate_first:1 lambda_g1_arc_testing:0. lambda_g1_lab_testing:0."
base_opt+=" lrate_warmup:0 lrate.val:0.0001 tconf.batch_size:32 split_batch:1 lrate.m:0.75 min_save_epochs:10 max_epochs:200 patience:5 anneal_times:8"
base_opt+=" margin.val:0.5 margin.max_val:0.5 margin.mode:linear margin.b:0.5 margin.k:0.01 reg_scores_lambda:0.01"
base_opt+=" enc_lrf.val:1. enc_lrf.mode:none enc_lrf.start_bias:0 enc_lrf.b:0. enc_lrf.k:1. enc_optim.weight_decay:0."
base_opt+=" mid_lrf.val:1. mid_lrf.mode:none mid_lrf.start_bias:0 mid_lrf.b:0. mid_lrf.k:1. mid_optim.weight_decay:0."
base_opt+=" dec_lrf.val:0. dec_lrf.mode:none dec_lrf.start_bias:0 dec_lrf.b:0. dec_lrf.k:1. dec_optim.weight_decay:0."
arc_k=5
else
# train from start
base_opt+=" lrate_warmup:0 lrate.val:0.001 tconf.batch_size:32 split_batch:1 lrate.m:0.75 min_save_epochs:150 max_epochs:300 patience:8 anneal_times:10"
base_opt+=" margin.val:0.5 margin.max_val:0.5 margin.mode:linear margin.b:0. margin.k:0.01 reg_scores_lambda:0.01"
base_opt+=" enc_lrf.val:1. enc_lrf.mode:none enc_lrf.start_bias:0 enc_lrf.b:0. enc_lrf.k:1. enc_optim.weight_decay:0."
base_opt+=" mid_lrf.val:1. mid_lrf.mode:none mid_lrf.start_bias:0 mid_lrf.b:0. mid_lrf.k:1. mid_optim.weight_decay:0."
base_opt+=" dec_lrf.val:0. dec_lrf.mode:none dec_lrf.start_bias:0 dec_lrf.b:0. dec_lrf.k:1. dec_optim.weight_decay:0."
base_opt+=" aabs.val:5. aabs.max_val:5. aabs.mode:linear aabs.start_bias:100 aabs.k:0.01"
arc_k=1
fi

# system
base_opt+=" partype:ef zero_extra_output_params:0 iconf.ef_mode:free tconf.ef_mode:free system_labeled:1"
base_opt+=" chs_num:10 chs_att.out_act:linear chs_att.att_dropout:0.1 chs_att.head_count:2"
# loss
base_opt+=" tconf.ending_mode:maxv cost0_weight.val:1. loss_div_weights:0 loss_div_fullbatch:1"
# features for ef
base_opt+=" fdrop_chs:0. fdrop_par:0. use_label_feat:1 use_chs:1 use_par:1 ef_ignore_chs:func"
# search
#arc_k=1  # set previously
lab_k=1
base_opt+=" iconf.plain_k_arc:${arc_k} iconf.plain_k_label:${lab_k} iconf.plain_beam_size:${arc_k}"
base_opt+=" tconf.plain_k_arc:${arc_k} tconf.plain_k_label:${lab_k} tconf.plain_beam_size:${arc_k}"

# different POS modes
if [[ "${MPOS}" == "nope" ]]; then
# no pos input
base_opt+=" dim_extras:[] extra_names:[]"
elif [[ "${MPOS}" == "joint" ]]; then
# no pos input
base_opt+=" dim_extras:[] extra_names:[]"
# jpos (split one layer for jpos)
base_opt+=" enc_conf.enc_rnn_layer:2 jpos_multitask:1 jpos_enc.enc_rnn_layer:1"
# jpos lambda
base_opt+=" jpos_lambda:0.1"
fi

# run
SRC_DIR="../src/"
DATA_DIR="../data/UD_RUN/ud23/"
#
CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.train train:${DATA_DIR}/${CUR_LANG}_train.conllu dev:${DATA_DIR}/${CUR_LANG}_dev.conllu test:${DATA_DIR}/${CUR_LANG}_test.conllu pretrain_file:${DATA_DIR}/wiki.${CUR_LANG}.filtered.vec device:0 ${base_opt} "niconf.random_seed:9347${CUR_RUN}" "niconf.random_cuda_seed:9349${CUR_RUN}"
}

# actual run
#G1MODEL=../zen_ef/zmodel.best CUR_LANG=en CUR_RUN=1 run_one  # set init0=false
#G1MODEL=../zen_g1/zmodel.best CUR_LANG=en CUR_RUN=1 run_one
#G1MODEL='' CUR_LANG=en CUR_RUN=1 run_one
G1MODEL='' CUR_LANG=en CUR_RUN=1 MPOS=nope run_one

# RGPU=0 bash _go.sh |& tee log
