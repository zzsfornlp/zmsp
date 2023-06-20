#!/usr/bin/env bash

# train with a cross-validation style

if [[ -z ${ZPIECE} ]]; then
  ZPIECE=5  # by default 5 piece
fi
if [[ -z ${ZPIECE} ]]; then
  ZDIR='.'
fi
mkdir -p ${ZDIR}
echo "train_cv with ${ZTRAIN} ${ZDEV} ${ZTEST} ${ZEXTRA} piece=${ZPIECE}"

# first splitting training set
python3 -m mspx.scripts.tools.sample_shuffle shuffle_times:1 split_piece:${ZPIECE} input_path:${ZTRAIN} output_path:${ZDIR}/_train.json

# then train them all
for (( ii=0; ii<"${ZPIECE}"; ii+=1 )); do
ONE_TRAIN=$(python -c "print(','.join([f'../_train.{i}.json' for i in range(${ZPIECE}) if i!=${ii}]))")
echo "TRAIN ${ii} with ${ONE_TRAIN}"
python3 -m mspx.tasks.zext.main zdir:${ZDIR}/p${ii} log_stderr:0 log_file:_log device:0 conf_sbase:bert_name:xlm-roberta-base conf_output:_conf train0.group_files:${ONE_TRAIN} dev0.group_files:${ZDEV} test0.group_files:${ZTEST} fs:build,train,test ${ZEXTRA}
python3 -m mspx.tasks.zext.main _conf zdir:${ZDIR}/p${ii} log_stderr:0 log_file:_log2 device:0 fs:test test0.group_files:../_train.${ii}.json test0.output_file:_zout.${ii}.json pred_do_strg:1 pred_no_strg1:1
done

# example run
# CUDA_VISIBLE_DEVICES=0 ZDIR=_tmp ZTRAIN=__data/ner/data/en.train.json ZDEV=__data/ner/data/en.dev.json ZTEST=__data/ner/data/en.test.json ZEXTRA="record_best_start_cidx:1 max_uidx:2000" bash train_cv.sh
# for cl in en de es nl; do ZDIR=run_cvner_${cl} ZTRAIN=__data/ner/data/${cl}.train.json ZDEV=__data/ner/data/${cl}.dev.json ZTEST=__data/ner/data/${cl}.test.json ZPIECE=3 bash train_cv.sh; done
