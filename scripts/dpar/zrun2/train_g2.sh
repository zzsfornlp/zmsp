#!/usr/bin/env bash

# train g2

# TODO(WARN): not fully tuned hyper-parameters

# =====
function run_one () {
# check envs
echo "The environment variables: RGPU=${RGPU} CUR_LANG=${CUR_LANG} CUR_RUN=${CUR_RUN}";

# =====
# basis
base_opt="conf_output:_conf init_from_pretrain:1 drop_embed:0.33 dropmd_embed:0.33 singleton_thr:2 fix_drop:1"
base_opt+=" arc_space:512 lab_space:128 transform_act:elu biaffine_div:1. biaffine_init_ortho:1"
# =====
# which system
base_opt+=" partype:g2 zero_extra_output_params:0 train_skip_length:90 infer_single_length:90 system_labeled:1"
base_opt+=" pruning_use_topk:1 pruning_topk:4 pruning_gap:20. pruning_use_marginal:1 pruning_mthresh:0.02 pruning_mthresh_rel:1"
base_opt+=" filter_pruned:1 filter_margin:1 gm_type:o3gsib mb_dec_lb:128 mb_dec_sb:80000"  # be careful for oom
# =====
# aux input scores
base_opt+=" g1_use_aux_scores:1 aux_score_train:${AUX_DIR}/train.scores.pkl aux_score_dev:${AUX_DIR}/dev.scores.pkl aux_score_test:${AUX_DIR}/test.scores.pkl"
# =====
if [[ -n "${G1MODEL}" ]]; then
# pre-load and pre-valid
base_opt+=" g1_pretrain_path:${G1MODEL} g1_pretrain_init:1 validate_first:1 lambda_g1_arc_testing:1. lambda_g1_lab_testing:1."
base_opt+=" lrate_warmup:0 lrate.val:0.0001 tconf.batch_size:32 split_batch:1 lrate.m:0.75 min_save_epochs:10 max_epochs:200 patience:5 anneal_times:8"
base_opt+=" margin.val:0.5 margin.max_val:0.5 margin.mode:linear margin.b:0.5 margin.k:0.05 reg_scores_lambda:0.01"
base_opt+=" enc_lrf.val:1. enc_lrf.mode:none enc_lrf.start_bias:0 enc_lrf.b:0. enc_lrf.k:1. enc_optim.weight_decay:0."
base_opt+=" mid_lrf.val:1. mid_lrf.mode:none mid_lrf.start_bias:0 mid_lrf.b:0. mid_lrf.k:1. mid_optim.weight_decay:0."
base_opt+=" dec_lrf.val:0. dec_lrf.mode:none dec_lrf.start_bias:0 dec_lrf.b:0. dec_lrf.k:1. dec_optim.weight_decay:0."
else
# train from start
base_opt+=" lrate_warmup:0 lrate.val:0.001 tconf.batch_size:32 split_batch:1 lrate.m:0.75 min_save_epochs:150 max_epochs:300 patience:8 anneal_times:10"
base_opt+=" margin.val:0.5 margin.max_val:0.5 margin.mode:linear margin.b:0. margin.k:0.01 reg_scores_lambda:0."
base_opt+=" enc_lrf.val:1. enc_lrf.mode:none enc_lrf.start_bias:0 enc_lrf.b:0. enc_lrf.k:1. enc_optim.weight_decay:0."
base_opt+=" mid_lrf.val:1. mid_lrf.mode:none mid_lrf.start_bias:0 mid_lrf.b:0. mid_lrf.k:1. mid_optim.weight_decay:0."
base_opt+=" dec_lrf.val:0. dec_lrf.mode:none dec_lrf.start_bias:0 dec_lrf.b:0. dec_lrf.k:1. dec_optim.weight_decay:0."
fi

# no POS for ptb
if [[ ${NOPOS} == 1 ]]; then
base_opt+=" dim_extras:[] extra_names:[]"
fi

# run
SRC_DIR="../src/"
DATA_DIR="../data/UD_RUN/ud23/"
#
CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.train train:${DATA_DIR}/${CUR_LANG}_train.conllu dev:${DATA_DIR}/${CUR_LANG}_dev.conllu test:${DATA_DIR}/${CUR_LANG}_test.conllu pretrain_file:${DATA_DIR}/wiki.${CUR_LANG}.filtered.vec device:0 ${base_opt} "niconf.random_seed:9347${CUR_RUN}" "niconf.random_cuda_seed:9349${CUR_RUN}"
}

# actual run
G1MODEL='../scores/en/_model.all.best' AUX_DIR='../scores/en' CUR_LANG=en CUR_RUN=1 run_one

# RGPU=0 bash _go.sh |& tee log
