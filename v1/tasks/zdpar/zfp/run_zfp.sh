#!/usr/bin/env bash

# running for zfp

# --
# go: create new dir?
if [[ ${USE_RDIR} != "" ]]; then
echo "Going into a new dir for the running: ${USE_RDIR}"
mkdir -p ${USE_RDIR}
cd ${USE_RDIR}
fi
# --

# =====
# find the dir
SRC_DIR="../src/"
if [[ ! -d ${SRC_DIR} ]]; then
SRC_DIR="../../src/"
if [[ ! -d ${SRC_DIR} ]]; then
SRC_DIR="../../../src/"
fi
fi
DATA_DIR="${SRC_DIR}/../data/"  # set data dir

# =====
function decode_one () {
if [[ -z ${WSET} ]]; then WSET="test"; fi
for cl in en ar eu zh 'fi' he hi it ja ko ru sv tr;
do
# running for one
    CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.test ${MDIR}/_conf conf_output: test:${DATA_DIR}/${cl}_${WSET}.conllu output_file:crout_${cl}${OUTINFFIX}.out model_load_name:${MDIR}/zmodel.best dict_dir:${MDIR} log_file:
done
}
# RGPU=? MDIR=../? OUTINFFIX= WSET= decode_one |& tee log.decode

# =====
function run_one () {
# check envs
echo "The environment variables: RGPU=${RGPU} CUR_LANG=${CUR_LANG} CUR_RUN=${CUR_RUN}";

# =====
# basis
base_opt="conf_output:_conf init_from_pretrain:0"
base_opt+=" drop_hidden:0.2 drop_embed:0.2 fix_drop:0 dropmd_embed:0."
base_opt+=" lrate.val:0.00005 lrate.min_val:0.000001 tconf.batch_size:16 lrate.m:0.5 min_save_epochs:10 max_epochs:100 patience:5 anneal_times:5"

# which system
base_opt+=" partype:fp use_label0:1"

if [[ ${NOFTBERT} == 1 ]]; then
# simply use features
base_opt+=" bert2_model:bert-base-multilingual-cased bert2_output_layers:[6,8,10] bert2_trainable_layers:0 lrate.val:0.0002"
else
# fine-tune bert
base_opt+=" bert2_model:bert-base-multilingual-cased bert2_output_layers:[-1] bert2_trainable_layers:13 lrate.val:0.00005"
fi

# decoder
base_opt+=" arc_space:512 lab_space:128"
base_opt+=" use_biaffine:1 biaffine_div:0. use_ff1:1 use_ff2:0 ff2_hid_layer:0"

# encoder
# by default no encoder
if [[ ${USERNN} == 1 ]]; then
base_opt+=" enc_conf.enc_rnn_layer:3 enc_conf.enc_att_layer:0 lrate.val:0.001"
fi
if [[ ${USEATT} == 1 ]]; then
# specific for self-att encoder
base_opt+=" middle_dim:512 enc_conf.enc_hidden:512"
base_opt+=" enc_conf.enc_rnn_layer:0 enc_conf.enc_att_layer:6 enc_conf.enc_att_add_wrapper:addnorm enc_conf.enc_att_conf.clip_dist:10 enc_conf.enc_att_conf.use_neg_dist:0 enc_conf.enc_att_conf.att_dropout:0.1 enc_conf.enc_att_conf.use_ranges:1 enc_conf.enc_att_fixed_ranges:5,10,20,30,50,100 enc_conf.enc_att_final_act:linear"
base_opt+=" max_epochs:100 train_skip_length:80 tconf.batch_size:80 split_batch:2 biaffine_div:0."
base_opt+=" lrate.val:0.0002 lrate_warmup:-2 drop_embed:0.1 dropmd_embed:0. drop_hidden:0.1"
fi

# run
data_opt="train:${DATA_DIR}/${CUR_LANG}_train.conllu dev:${DATA_DIR}/${CUR_LANG}_dev.conllu test:${DATA_DIR}/${CUR_LANG}_test.conllu"
LOG_PREFIX="_log"
# train
if [[ -n ${DEBUG} ]]; then
  CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 -m pdb ${SRC_DIR}/tasks/cmd.py zdpar.main.train device:0 ${data_opt} ${base_opt} "niconf.random_seed:9347${CUR_RUN}" "niconf.random_cuda_seed:9349${CUR_RUN}" ${EXTRA_ARGS}
else
  CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.train device:0 ${data_opt} ${base_opt} "niconf.random_seed:9347${CUR_RUN}" "niconf.random_cuda_seed:9349${CUR_RUN}" ${EXTRA_ARGS} |& tee ${LOG_PREFIX}.train
fi
# decode targets
RGPU=${RGPU} MDIR=./ OUTINFFIX= WSET= decode_one |& tee ${LOG_PREFIX}.decode
}

# actual run
if [[ -z "${CUR_LANG}" ]]; then
  CUR_LANG="en"
fi
# --
run_one
# --
if [[ ${USE_RDIR} != "" ]]; then
cd $OLDPWD
fi

# =====
# fine-tune bert
# RGPU=1 DEBUG= CUR_LANG=en CUR_RUN=1 USERNN=0 USEATT=0 NOFTBERT=0 USE_RDIR= bash _go.sh
# bert-features + 6-layer transformer
# RGPU=1 DEBUG= CUR_LANG=en CUR_RUN=1 USERNN=0 USEATT=1 NOFTBERT=1 USE_RDIR= bash _go.sh
