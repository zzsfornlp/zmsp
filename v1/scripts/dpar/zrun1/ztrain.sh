#!/usr/bin/env bash

# train for various languages for loss/methods/3-runs

# =====
# base options
base_opt="conf_output:_conf partype:graph init_from_pretrain:1 lrate.init_val:0.001 drop_embed:0.33 dropmd_embed:0.33 singleton_thr:2 fix_drop:1 max_epochs:300"
# possible methods
single_m="dec_algorithm:unproj output_normalizing:single loss_single_sample:2."
local_m="dec_algorithm:unproj output_normalizing:local"
global_unproj_m="dec_algorithm:unproj output_normalizing:global"
global_proj_m="dec_algorithm:proj output_normalizing:global"
# possible losses
hinge_loss="loss_function:hinge margin.init_val:2.0"
prob_loss="loss_function:prob margin.init_val:0."

# =====
function run_one () {
# check envs
echo "The environment variables: RGPU=${RGPU} CUR_LANG=${CUR_LANG} CUR_METHOD=${CUR_METHOD} CUR_LOSS=${CUR_LOSS} CUR_RUN=${CUR_RUN}";
if [[ -z "${RGPU}${CUR_LANG}${CUR_METHOD}${CUR_LOSS}${CUR_RUN}" ]];
then
echo "Please provide the missing environment variables!!";
exit 1;
fi
# which method
if [[ ${CUR_METHOD} == "single" ]]; then method=${single_m};
elif [[ ${CUR_METHOD} == "local" ]]; then method=${local_m};
elif [[ ${CUR_METHOD} == "unproj" ]]; then method=${global_unproj_m};
elif [[ ${CUR_METHOD} == "proj" ]]; then method=${global_proj_m};
else echo "Unknown method!"; exit 1;
fi
# which loss
if [[ ${CUR_LOSS} == "hinge" ]]; then loss=${hinge_loss};
elif [[ ${CUR_LOSS} == "prob" ]]; then loss=${prob_loss};
else echo "Unknown loss!"; exit 1;
fi
# run
RUN_DIR="./zfr_${CUR_LANG}/z_${CUR_METHOD}_${CUR_LOSS}_${CUR_RUN}"
mkdir -p ${RUN_DIR}
MAIN_DIR=`pwd`
#
cd ${RUN_DIR}
SRC_DIR="../../src/"
if [[ ${CUR_LANG} == "ptb" || ${CUR_LANG} == "ctb" ]]; then DATA_DIR="../../data/pc/";
else DATA_DIR="../../data/ud23/";
fi;
echo RUNNING in ${RUN_DIR};
CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.train train:${DATA_DIR}/${CUR_LANG}_train.conllu dev:${DATA_DIR}/${CUR_LANG}_dev.conllu test:${DATA_DIR}/${CUR_LANG}_test.conllu pretrain_file:${DATA_DIR}/wiki.${CUR_LANG}.vec device:0 ${base_opt} ${method} ${loss} "niconf.random_seed:9347${CUR_RUN}" "niconf.random_cuda_seed:9349${CUR_RUN}"
# go back
cd ${MAIN_DIR};
}

# =====
# group run
#for rr in 1 2 3; do
#for rr in 1; do
#for mm in single local unproj proj; do
RGPU=${RGPU} CUR_LANG=${CUR_LANG} CUR_LOSS=${CUR_LOSS} CUR_METHOD=${CUR_METHOD} CUR_RUN=1 run_one;
#done
#done

# RGPU=0 CUR_LANG=ptb CUR_LOSS=hinge bash ztrain.sh >/dev/null 2>&1 &
# specifically for large ones: ptb ctb ru cs
# for cur_lang in ptb ctb bg ca cs de en es fr it nl no ro ru; do RGPU=0 CUR_LANG=${cur_lang} CUR_LOSS=hinge bash ztrain.sh >/dev/null 2>&1; done
