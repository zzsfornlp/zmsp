#!/usr/bin/env bash

# train raw

# =====
function run_one () {
# check envs
echo "The environment variables: RGPU=${RGPU} CUR_LANG=${CUR_LANG} CUR_RUN=${CUR_RUN}";

# =====
# basis
#base_opt="conf_output:_conf init_from_pretrain:1 drop_embed:0.33 dropmd_embed:0.33 singleton_thr:2 fix_drop:1"
base_opt="conf_output:_conf init_from_pretrain:1 drop_embed:0.2 dropmd_embed:0.2 singleton_thr:2 fix_drop:1"
base_opt+=" lrate.val:0.001 tconf.batch_size:32 lrate.m:0.75 min_save_epochs:150 max_epochs:300 patience:8 anneal_times:10"
base_opt+=" loss_function:hinge margin.val:0.5 margin.max_val:0.5 margin.mode:linear margin.b:0. margin.k:0.01 reg_scores_lambda:0."
base_opt+=" arc_space:512 lab_space:128 transform_act:elu biaffine_div:1. biaffine_init_ortho:1"
base_opt+=" enc_lrf.val:1. enc_lrf.mode:none enc_lrf.start_bias:0 enc_lrf.b:0. enc_lrf.k:1. enc_optim.weight_decay:0."
base_opt+=" dec_lrf.val:0. dec_lrf.mode:none dec_lrf.start_bias:0 dec_lrf.b:0. dec_lrf.k:1. dec_optim.weight_decay:0."
# =====
# which system
# todo(note): simply chaning this to "graph" is also ok
base_opt+=" partype:g1"

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

# =====
function test_one () {
# run
SRC_DIR="../src/"
DATA_DIR="../data/UD_RUN/ud23/"
#
# test with g1
echo "Decode with graph:"; CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.test _conf test:${DATA_DIR}/${CUR_LANG}_test.conllu device:0
## test with ef
#for k1 in {1..10}; do for k2 in {1..5}; do
#echo "Decode with ef: arc=${k1} label=${k2}"; CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.test _conf test:${DATA_DIR}/${CUR_LANG}_test.conllu device:0 partype:ef use_label_feat:1 use_chs:0 use_par:0 iconf.plain_k_arc:${k1} iconf.plain_k_label:${k2} iconf.plain_beam_size:${k1};
#done; done
# todo(note): -> arc=3/label=3 works best, but arc=2/label=1 works also fine!
# test with ef (new version)
for k1 in {1..10}; do for k2 in {1..5}; do
echo "Decode with ef: arc=${k1} label=${k2}"; CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.test _conf test:${DATA_DIR}/${CUR_LANG}_test.conllu device:0 partype:ef iconf.plain_k_arc:${k1} iconf.plain_k_label:${k2} iconf.plain_beam_size:${k1} model_load_name: g1_pretrain_path:zmodel.best g1_pretrain_init:1 g1_pretrain_init0:1;
done; done
# test g1 with pruning
for t1 in 0.01 0.001 0.00075 0.0005 0.00025 0.0002 0.0001; do for rel1 in 1 0;
do echo "Decode with prune: thresh=${t1}, rel=${rel1}"; CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.test _conf test:${DATA_DIR}/${CUR_LANG}_test.conllu device:0 use_pruning:1 pruning_mthresh_rel:${rel1} pruning_mthresh:${t1};
done; done;
}

# mainly about the scales of the scores
function try_prescores () {
# CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../src/ python3 ../src/tasks/cmd.py zdpar.main.test ./_conf test:../data/UD_RUN/ud23/en_dev.conllu
SRC_DIR="../src/"
DATA_DIR="../data/UD_RUN/ud23/"
CUR_LANG=en
RGPU=3
RUN_DIR="../scores/en/"
# test prune with pre-scores
CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.test ${RUN_DIR}/_conf.all dict_dir:${RUN_DIR}/ test:${DATA_DIR}/${CUR_LANG}_dev.conllu aux_score_test:${RUN_DIR}/dev.scores.pkl debug_use_aux_scores:1 partype:g1 model_load_name: device:0 use_pruning:1 pruning_use_topk:0 pruning_topk:4 pruning_gap:20. pruning_use_marginal:1 pruning_mthresh:0.02 pruning_mthresh_rel:1
# test original
CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.test ${RUN_DIR}/_conf.all dict_dir:${RUN_DIR}/ test:${DATA_DIR}/${CUR_LANG}_dev.conllu model_load_name:${RUN_DIR}/_model.all.best device:0 partype:g1
# test with ef
CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.test ${RUN_DIR}/_conf.all dict_dir:${RUN_DIR}/ test:${DATA_DIR}/${CUR_LANG}_dev.conllu model_load_name: device:0 partype:ef g1_pretrain_path:${RUN_DIR}/_model.all.best g1_pretrain_init:1 zero_extra_output_params:1 partype:ef iconf.plain_k_label:1 iconf.plain_k_arc:2 iconf.plain_beam_size:2
# test with s2
CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.test ${RUN_DIR}/_conf.all dict_dir:${RUN_DIR}/ test:${DATA_DIR}/${CUR_LANG}_dev.conllu model_load_name: device:0 partype:ef g1_pretrain_path:${RUN_DIR}/_model.all.best g1_pretrain_init:1 zero_extra_output_params:1 partype:s2 sl_conf.use_par:0 sl_conf.use_chs:0 dprune.pruning_mthresh:0. sprune.pruning_mthresh:0.
}

# actual run
#PARTYPE=g1 CUR_LANG=en CUR_RUN=1 run_one
PARTYPE=g1 CUR_LANG=en CUR_RUN=1 MPOS=nope run_one

# RGPU=0 bash _go.sh |& tee log
