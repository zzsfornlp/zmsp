#

# some runnings for stat2

# _run1028.sh
SRC_DIR="../src/"
#DATA_DIR="../data/UD_RUN/ud24/"
DATA_DIR="../data/ud24/"
CUR_LANG=en
RGPU=
DEVICE=-1
for b_mask_mode in all first one pass; do
for b_mask_repl in MASK UNK PAD; do
for dist_f in cos mse; do
pic_name="_${CUR_LANG}_dev.${b_mask_mode}.${b_mask_repl}.${dist_f}.pic"
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.stat2.decode3 device:${DEVICE} input_file:${DATA_DIR}/${CUR_LANG}_dev.ppos.conllu output_pic:${pic_name} b_mask_mode:${b_mask_mode} "b_mask_repl:[${b_mask_repl}]" dist_f:${dist_f}
done; done; done |& tee _log1028

# _run1029.sh
SRC_DIR="../src/"
#DATA_DIR="../data/UD_RUN/ud24/"
DATA_DIR="../data/ud24/"
CUR_LANG=en
RGPU=
DEVICE=-1
for fpic in *.pic; do
for which_fold in {1..9} {10..12}; do
for use_fdec in 0 1; do
  echo "ZRUN ${fpic} ${which_fold} ${use_fdec}"
  CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.stat2.decode3 device:${DEVICE} input_file:$fpic already_pre_computed:1 which_fold:${which_fold} use_fdec:${use_fdec}
done
done
done |& tee _log1029
