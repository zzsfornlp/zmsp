#!/usr/bin/env bash

# modify original "zrun1/ztrain.sh" to make it runnable at 2020's msp

# =====
# train for various languages for loss/methods/3-runs

# =====
# base options
base_opt="conf_output:_conf partype:graph init_from_pretrain:1 lrate.val:0.001 drop_embed:0.33 dropmd_embed:0.33 singleton_thr:2 fix_drop:1 max_epochs:300"
# possible methods
single_m="dec_algorithm:unproj output_normalizing:single loss_single_sample:2."
local_m="dec_algorithm:unproj output_normalizing:local"
global_unproj_m="dec_algorithm:unproj output_normalizing:global"
global_proj_m="dec_algorithm:proj output_normalizing:global"
# possible losses
hinge_loss="loss_function:hinge margin.val:2.0"
prob_loss="loss_function:prob margin.val:0."

# =====
# find the dir
SRC_DIR="../src/"
if [[ ! -d ${SRC_DIR} ]]; then
SRC_DIR="../../src/"
if [[ ! -d ${SRC_DIR} ]]; then
SRC_DIR="../../../src/"
fi
fi
DATA_DIR="${SRC_DIR}/../data/UD_RUN/ud24/"

# =====
function run_one () {
# check envs
echo "The environment variables: RGPU=${RGPU} CUR_LANG=${CUR_LANG} CUR_METHOD=${CUR_METHOD} CUR_LOSS=${CUR_LOSS} CUR_POS=${CUR_POS} CUR_RUN=${CUR_RUN}";
if [[ -z "${RGPU}${CUR_LANG}${CUR_METHOD}${CUR_LOSS}${CUR_POS}${CUR_RUN}" ]];
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
# which pos
if [[ ${CUR_POS} == "1" ]]; then base_opt+=" dim_extras:50 extra_names:pos";
else base_opt+=" dim_extras: extra_names:";  # no pos
fi
# run
CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.train train:${DATA_DIR}/${CUR_LANG}_train.conllu dev:${DATA_DIR}/${CUR_LANG}_dev.ppos.conllu test:${DATA_DIR}/${CUR_LANG}_test.ppos.conllu pretrain_file:${DATA_DIR}/wiki.${CUR_LANG}.filtered.vec device:0 ${base_opt} ${method} ${loss} "niconf.random_seed:9347${CUR_RUN}" "niconf.random_cuda_seed:9349${CUR_RUN}"
}

# =====
## group run
##for rr in 1 2 3; do
#for rr in 1; do
#for mm in single local unproj proj; do
#RGPU=${RGPU} CUR_LANG=${CUR_LANG} CUR_LOSS=${CUR_LOSS} CUR_METHOD=${mm} CUR_RUN=${rr} run_one;
#done
#done

# example run (with or without pos)
RGPU=$RGPU CUR_LANG=en CUR_LOSS=prob CUR_METHOD=local CUR_POS=1 CUR_RUN=1 run_one |& tee _log_p1
RGPU=$RGPU CUR_LANG=en CUR_LOSS=prob CUR_METHOD=local CUR_POS=0 CUR_RUN=1 run_one |& tee _log_p0

# RGPU=3 bash _go.sh
