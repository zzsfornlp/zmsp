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

# _run1029.sh -> fold 7/8/9 slightly better
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

# _run1113.sh
# bash _run1113.sh |& tee _log1113
SRC_DIR="../src/"
#DATA_DIR="../data/UD_RUN/ud24/"
DATA_DIR="../data/ud24/"
CUR_LANG=en
RGPU=
DEVICE=-1
for hf_use_pos_rule in 0 1; do
for fpic in *.pic; do
for which_fold in 12 9 8 7; do
  echo "ZRUN POS_RULE=${hf_use_pos_rule} ${fpic} FOLD=${which_fold}"
  CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.stat2.decode3 device:${DEVICE} input_file:$fpic already_pre_computed:1 mdec_method:cky hf_use_ir:1 ps_pp:1 hf_use_pos_rule:${hf_use_pos_rule} which_fold:${which_fold}
done
done
done

# _run1114a.sh -> row=1 col=0/-1
# bash _run1114a.sh |& tee _log1114a
SRC_DIR="../src/"
#DATA_DIR="../data/UD_RUN/ud24/"
DATA_DIR="../data/ud24/"
CUR_LANG=en
RGPU=
DEVICE=-1
for hf_use_pos_rule in 0 1; do
fpic="_en_dev.all.MASK.mse.pic"
which_fold=8
for ps_method in "add_avg" "max_avg"; do
for hf_win_row in 0 1; do
for hf_win_col in 1 0 -1; do
  echo "ZRUN hf_use_pos_rule:${hf_use_pos_rule} ps_method:${ps_method} hf_win_row:${hf_win_row} hf_win_col:${hf_win_col}"
  CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.stat2.decode3 device:${DEVICE} input_file:$fpic already_pre_computed:1 mdec_method:cky which_fold:${which_fold} hf_use_ir:1 ps_pp:1 hf_use_pos_rule:${hf_use_pos_rule} ps_method:${ps_method} hf_win_row:${hf_win_row} hf_win_col:${hf_win_col}
done
done
done
done

# _run1114b.sh -> not much differences on the good ones ~31
# bash _run1114b.sh |& tee _log1114b
SRC_DIR="../src/"
#DATA_DIR="../data/UD_RUN/ud24/"
DATA_DIR="../data/ud24/"
CUR_LANG=en
RGPU=
DEVICE=-1
#
hf_use_pos_rule=0
fpic="_en_dev.all.MASK.mse.pic"
which_fold=8
for ps_method in "add_avg" "max_avg"; do
for hf_win_row in 0 1; do
for hf_win_col in 1 0 -1; do
for hf_wout_row in 0 1 -1; do
for hf_wout_col in 0 1 -1; do
for hf_self_headed in 0 1; do
for hf_other_headed in 0 1; do
  echo "ZRUN ps_method:${ps_method} hf_win_row:${hf_win_row} hf_win_col:${hf_win_col} hf_wout_row:${hf_wout_row} hf_wout_col:${hf_wout_col} hf_self_headed:${hf_self_headed} hf_other_headed:${hf_other_headed}"
  CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.stat2.decode3 device:${DEVICE} input_file:$fpic already_pre_computed:1 mdec_method:cky which_fold:${which_fold} hf_use_ir:0 ps_pp:1 ps_method:${ps_method} hf_win_row:${hf_win_row} hf_win_col:${hf_win_col} hf_wout_row:${hf_wout_row} hf_wout_col:${hf_wout_col} hf_self_headed:${hf_self_headed} hf_other_headed:${hf_other_headed}
done; done; done; done; done; done; done
