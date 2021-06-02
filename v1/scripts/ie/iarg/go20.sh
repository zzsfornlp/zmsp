#

# train m3a on RAMS

LANGUAGE="en"
DATA_SET="en.rams"
TEST_NAME="test"
# find data
SRC_DIR="../src/"
DATA_DIR="../RAMS_1.0/data//"
if [[ ! -d ${SRC_DIR} ]]; then
SRC_DIR="../../src/"
DATA_DIR="../../RAMS_1.0/data//"
if [[ ! -d ${SRC_DIR} ]]; then
SRC_DIR="../../../src/"
DATA_DIR="../../../RAMS_1.0/data/"
fi
fi
# =====
function run_one () {
# check envs
echo "The environment variables: RGPU=${RGPU} CUR_RUN=${CUR_RUN}";

# =====
# basis

# basic
base_opt="model_type:m3a conf_output:_conf init_from_pretrain:0 word_freeze:1"

# enc
base_opt+=" bert2_model:bert-base-cased bert_use_special_typeids:1 bert2_output_layers:[-1] bert2_trainable_layers:13"
base_opt+=" m2e_use_basic_plus:0 m2e_use_basic_dep:0 ms_extend_step:2 ms_extend_budget:250"  # avoid oom?
base_opt+=" m3_enc_conf.enc_rnn_layer:0 m3_enc_conf.enc_hidden:512"  # extra LSTM layer?

# training and dropout
base_opt+=" tconf.train_msent_based:1 tconf.batch_size:18 split_batch:3 iconf.batch_size:64"
base_opt+=" lrate.val:0.00005 lrate.m:0.5 min_save_epochs:0 max_epochs:20 patience:2 anneal_times:5"
base_opt+=" drop_embed:0.1 dropmd_embed:0.1 drop_hidden:0.2 gdrop_rnn:0.33 idrop_rnn:0.1 fix_drop:0"
base_opt+=" bert2_training_mask_rate:0."

# bucket
base_opt+=" benc_bucket_range:20 enc_bucket_range:20"

# =====

# overall
base_opt+=" train_skip_noevt_rate:1.0 res_list:argument"  # only keep center sentences

# cand
base_opt+=" lambda_cand.val:0.5"
base_opt+=" c_cand.train_neg_rate:0.02 c_cand.nil_penalty:100."
base_opt+=" c_cand.pred_sent_ratio:0.4 c_cand.pred_sent_ratio_sep:1"

# arg
base_opt+=" eval_conf.arg_mode:span"  # eval on span
base_opt+=" lambda_arg.val:1."  # loss lambda
#
base_opt+=" ps_conf.nil_score0:1 arc_space:0 lab_space:0"  # scorer architecture
base_opt+=" c_arg.share_adp:1 c_arg.share_scorer:1"  # param sharing
base_opt+=" use_borrowed_evt_hlnode:0 use_borrowed_evt_adp:0"  # extra param sharing (for evt-emb and evt-adp)
base_opt+=" c_arg.use_cons_frame:0 which_frame_cons:rams"  # use constraints (mainly for predicted evt types)
base_opt+=" max_sdist:2 max_pairwise_role:2 max_pairwise_ef:1"  # other dec-constraints
base_opt+=" center_train_neg_rate:0.5 outside_train_neg_rate:0.5"  # arg neg rate
base_opt+=" c_arg.norm_mode:acand c_arg.use_evt_label:0 c_arg.use_ef_label:0 c_arg.use_sdist:0"  # arg mode

# arg expander
base_opt+=" lambda_span.val:0.5 max_range:7 use_lstm_scorer:0 use_binary_scorer:0"

# special reg & cut-lrate deterministically
#base_opt+=" biaffine_div:0. biaffine_freeze:1"
base_opt+=" lrate.which_idx:eidx lrate.start_bias:5 lrate.scale:4 lrate.m:0.5 lrate_warmup:0 lrate.min_val:0.000001 min_save_epochs:10"
base_opt+=" lrate.val:0.00005 tconf.batch_size:12 split_batch:3 lrate.m:0.5 lrate_warmup:-2 bert2_training_mask_rate:0.15"

# final group
#base_opt+=" use_lstm_scorer:1 use_binary_scorer:1"
base_opt+=" use_lstm_scorer:0 use_binary_scorer:0 lrate.val:0.000075 lrate.m:0.5 lrate_warmup:0 arc_space:0 lab_space:256 min_save_epochs:7"

# =====
# note: these are for "gold_span" linking experiments
#base_opt+=" c_arg.use_sdist:0 pred_span:0 lambda_span.val:0."
#base_opt+=" iconf.lookup_ef:1 mix_pred_ef_rate:0 bert_use_center_typeids:0 bert_use_special_typeids:0 bert2_output_layers:[9,10,11,12] bert2_trainable_layers:0"
#base_opt+=" lrate.val:0.0002 drop_hidden:0.33 tconf.batch_size:64 split_batch:1"
# =====

# training
if [[ -n ${DEBUG} ]]; then
DEBUG_OPTION="-m pdb"
else
DEBUG_OPTION=""
fi
CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${DEBUG_OPTION} ${SRC_DIR}/tasks/cmd.py zie.main.train train:${DATA_DIR}/${DATA_SET}.train.json dev:${DATA_DIR}/${DATA_SET}.dev.json test:${DATA_DIR}/${DATA_SET}.${TEST_NAME}.json pretrain_file: device:-1 ${base_opt} "niconf.random_seed:9347${CUR_RUN}" "niconf.random_cuda_seed:9349${CUR_RUN}" ${EXTRA_ARGS}
}

run_one

# RGPU=2 EXTRA_ARGS="device:0" DEBUG=1 bash _go.sh

# to test
#CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 -m pdb ${SRC_DIR}/tasks/cmd.py zie.main.test ${RUN_DIR}/_conf device:-1 dict_dir:${RUN_DIR}/ model_load_name:${RUN_DIR}/zmodel.best
