#

# related runnings for genia

# --
# get data and tools
wget http://bionlp-st.dbcls.jp/GE/2011/downloads/BioNLP-ST_2011_genia_train_data_rev1.tar.gz
wget http://bionlp-st.dbcls.jp/GE/2011/downloads/BioNLP-ST_2011_genia_devel_data_rev1.tar.gz
wget http://bionlp-st.dbcls.jp/GE/2011/downloads/BioNLP-ST_2011_genia_test_data.tar.gz
wget http://bionlp-st.dbcls.jp/GE/2011/downloads/BioNLP-ST_2011_genia_tools_rev1.tar.gz
for ff in *.tar.gz; do
  tar -zxvf $ff
done

# --
# prepare beesl
git clone https://github.com/cosbi-research/beesl
cd beesl
conda create --name beesl-env python=3.7  # create an python 3.7 env called beesl-env
conda activate beesl-env                  # activate the environment
python -m pip install -r requirements.txt # install the packages from requirements.txt
bash download_data.sh
curl -O https://www.cosbi.eu/fx/2354/model.tar.gz
#tar -zxvf model.tar.gz -C models/beesl-model/
pip install gdown
gdown https://drive.google.com/uc?id=1R84voFKHfWV9xjzeLzWBbmY1uOMYpnyD
# Extract the model, convert it to pytorch, and clean the directory
tar xC models -f biobert_v1.1_pubmed.tar.gz
pytorch_transformers bert models/biobert_v1.1_pubmed/model.ckpt-1000000 models/biobert_v1.1_pubmed/bert_config.json models/biobert_v1.1_pubmed/pytorch_model.bin
rm models/biobert_v1.1_pubmed/model.ckpt*
# pre-process:
# note: fix a bug on: bioscripts/utils/corpus_er.py:157 (avoid repeated adding by "and matched_mention not in spans_to_merge")
# note: also rm previous ones since they create files with 'append' mode!!
rm -rf data/GE11/
python bioscripts/preprocess.py --masking type
# predict
pip install overrides==3.1.0  # downgrade to solve allennlp issue
mkdir -p outputs
python predict.py model.tar.gz data/GE11/masked/dev.mt.1 outputs/dev.mt.1 --device 0
python predict.py model.tar.gz data/GE11/masked/test.mt.1 outputs/test.mt.1 --device 0
# unmask & convert
python bioscripts/preprocess.py --masking no
for wset in dev test; do
  python bio-mergeBack.py outputs/${wset}.mt.1 data/GE11/not-masked/${wset}.mt.1 2 >outputs/${wset}.conll
  python bioscripts/postprocess.py --filepath outputs/${wset}.conll
  mv outputs/output/ outputs/out_${wset}/
  perl bioscripts/eval/a2-normalize.pl -g data/corpora/GE11/${wset}/ -o outputs/norm_${wset}/ outputs/out_${wset}/*.a2
  perl bioscripts/eval/a2-evaluate.pl -t1 -sp -g data/corpora/GE11/${wset}/ outputs/norm_${wset}/*.a2
done
# return
cd ..
# --
# train one!
#python train.py --name run --dataset_config ./config/mt.1.mh.0.50.json --parameters_config ./config/params.json --device 3
# ...

# --
# read beesl
for rr in '' 'Protein'; do
  for wset in train dev test; do
    python3 -m msp2.tasks.zmtl3.scripts.genia.read_beesl beesl/data/GE11/not-masked/${wset}.mt.1 en.bio11_gold_${rr}.${wset}.json "$rr"
  done
  for wset in dev test; do
    python3 -m msp2.tasks.zmtl3.scripts.genia.read_beesl beesl/outputs/${wset}.conll en.bio11_pred_${rr}.${wset}.json "$rr"
  done
done
#Read 908 from beesl/data/GE11/not-masked/train.mt.1 to en.bio11.train.json: Counter({'line': 239393, 'ef': 11625, 'arg': 10186, 'evt': 8728, 'sent': 8656, 'evt_trig_1': 8466, 'arg_N=1': 2916, 'doc': 908, 'evt_trig_2': 131, 'arg_N=2': 37})
#Read 259 from beesl/data/GE11/not-masked/dev.mt.1 to en.bio11.dev.json: Counter({'line': 77222, 'ef': 4688, 'arg': 3396, 'sent': 2888, 'evt': 2648, 'evt_trig_1': 2556, 'arg_N=1': 868, 'doc': 259, 'evt_trig_2': 46, 'arg_N=2': 12})
#Read 347 from beesl/data/GE11/not-masked/test.mt.1 to en.bio11.test.json: Counter({'line': 93456, 'ef': 5300, 'sent': 3365, 'doc': 347})
#Read 259 from beesl/outputs/dev.conll to en.bio11.dev.json: Counter({'line': 77222, 'ef': 4688, 'arg': 3028, 'sent': 2888, 'evt': 2691, 'evt_trig_1': 2567, 'arg_N=1': 730, 'arg_strange': 272, 'doc': 259, 'evt_trig_2': 62, 'arg_N=2': 18})
#Read 347 from beesl/outputs/test.conll to en.bio11.test.json: Counter({'line': 93456, 'ef': 5300, 'arg': 3784, 'evt': 3423, 'sent': 3365, 'evt_trig_1': 3321, 'arg_N=1': 944, 'arg_strange': 453, 'doc': 347, 'evt_trig_2': 51, 'arg_N=2': 11, 'ef_strange': 2, 'arg_N=0': 1})
# and reparse
for ff in en.bio11*.{train,dev,test}.json; do
 python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:en stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:${ff} "output_path:${ff%.json}.ud2.json"
done
#python3 -m pdb -m msp2.tasks.zmtl3.scripts.misc.ana_arg gold:en.bio11.dev.ud2.json preds:en.bio11P.dev.ud2.json

# --
# do predictions
# prun data_key dir_name model_name extra_args
function prun () {
  PATH0="$(pwd)"
  MPATH="$(readlink -f $3)"
  BPATH="$(readlink -f beesl)"
  DKEY="bio11_$1"
  mkdir -p $2
  cd $2
  PATH1="$(pwd)"
  {
  COMMEN_ARGS="device:0 arg0.mix_evt_ind:0.5 test0.input_dir:.. test0.group_files:en.${DKEY}.dev.ud2.json,en.${DKEY}.test.ud2.json arg0.pred_bc:10 arg0.extend_span:0 test0.gold_file::s/bio11_pred/bio11_gold arg0.clear_arg_ef:0"
  # run predictions
  if [[ -f _zout0.dev.json && -f _zout0.test.json ]]; then
    echo "Use previous values!"
  else
    python3 -m msp2.tasks.zmtl3.main.test 'conf_sbase:data:bio11;task:arg' arg0.arg_mode:tpl "model_load_name:${MPATH}###DMarg0.qmod.emb_frame,DMarg0.qmod.emb_role" $COMMEN_ARGS $4
    mv "_zout.ee_en.${DKEY}.dev.ud2.json_test.json" _zout0.dev.json
    mv "_zout.ee_en.${DKEY}.test.ud2.json_test.json" _zout0.test.json
  fi
  # postprocess
  python3 -m msp2.tasks.zmtl3.scripts.genia.postprocess _zout0.dev.json _zout.dev.json
  python3 -m msp2.tasks.zmtl3.scripts.genia.postprocess _zout0.test.json _zout.test.json
  # write
  python3 -m msp2.tasks.zmtl3.scripts.genia.write_beesl _zout.dev.json _zout.dev.conll
  python3 -m msp2.tasks.zmtl3.scripts.genia.write_beesl _zout.test.json _zout.test.conll
  # eval dev
  rm -f ./*.fixed
  conda activate beesl-env
  cd ${BPATH}
  for wset in test dev; do
    # --
#    python -c "from udify.dataset_readers.ge11_eval import evaluate_asrm; evaluate_asrm('data/GE11/masked/${wset}.mt.1', '${PATH1}/_zout.${wset}.conll')"
    # --
    python bio-mergeBack.py ${PATH1}/_zout.${wset}.conll data/GE11/not-masked/${wset}.mt.1 2 >${PATH1}/_zout.${wset}.conll.fixed
    python bioscripts/postprocess.py --filepath ${PATH1}/_zout.${wset}.conll.fixed
    mv ${PATH1}/output/ ${PATH1}/${wset}_output
    perl bioscripts/eval/a2-normalize.pl -g data/corpora/GE11/${wset}/ -o ${PATH1}/${wset}_output_norm/ ${PATH1}/${wset}_output/*.a2
    perl bioscripts/eval/a2-evaluate.pl -t1 -sp -g data/corpora/GE11/${wset}/ ${PATH1}/${wset}_output_norm/*.a2 >${PATH1}/${wset}_results.txt
    cat ${PATH1}/${wset}_results.txt
    # --
    cd ${PATH1}/${wset}_output
    tar -czf ${wset}_output.tar.gz *.a2
    mv ${wset}_output.tar.gz ..
    cd ${OLDPWD}
    # --
  done
  cd ${PATH1}
  conda deactivate
  } |& tee _log_test
  # --
  cd $PATH0
}
# --
# preliminary runnings
#prun pred_ _tmp "../run_zzf_augsrl_ALL/run_a6/zmodel.srl_a6.m" "arg0.arg_mode:tpl"
#prun pred_Protein _tmp "../run_zzf_augsrl_ALL/run_a6/zmodel.srl_a6.m" "arg0.arg_mode:tpl"
# ok, use Protein since (R=48.94/P=60.86/F=54.25) > (R=44.65/P=61.44/F=51.72)
#prun pred_Protein _tmp "../run_zzf_qa_ALL/zmodel.qa.m" "arg0.arg_mode:mrc arg0.mrc_use_rques:1"
#prun pred_Protein _tmp "../run_zzf_qa_ALL/zmodel.qa.m" "arg0.arg_mode:mrc arg0.mrc_use_rques:0"
# ok, use rques since (36.36/54.85/43.73) > (1.97/41.56/3.77)(too few preds)
#python3 -mpdb -m msp2.tasks.zmtl3.scripts.misc.ana_arg gold:en.bio11_gold_.dev.ud2.json preds:run_qa/_zout.dev.json,run_srl/_zout.dev.json
# --
# for safety: delete args
#python3 -m msp2.tasks.zmtl3.scripts.genia.rm_args en.bio11_pred_Protein.dev.ud2.json en.bio11_pred_Protein_RARG.dev.ud2.json
#python3 -m msp2.tasks.zmtl3.scripts.genia.rm_args en.bio11_pred_Protein.test.ud2.json en.bio11_pred_Protein_RARG.test.ud2.json
#python3 -m msp2.tasks.zmtl3.scripts.genia.rm_args en.bio11_gold_Protein.train.ud2.json en.bio11_gold_Protein_RALL.train.ud2.json 1
#prun pred_Protein_RARG run_srl "../run_zzf_augsrl_ALL/run_a6/zmodel.srl_a6.m" "arg0.arg_mode:tpl"
#prun pred_Protein_RARG run_qa "../run_zzf_qa_ALL/zmodel.qa.m" "arg0.arg_mode:mrc arg0.mrc_use_rques:1"
# note: ok, hit it, no problems here, simply take origin ones for easier eval
#prun pred_Protein run_srl "../run_zzf_augsrl_ALL/run_a6/zmodel.srl_a6.m" "arg0.arg_mode:tpl"
#prun pred_Protein run_qa "../run_zzf_qa_ALL/zmodel.qa.m" "arg0.arg_mode:mrc arg0.mrc_use_rques:1"
# --

# --
# auto-parse with PB
if false; then
# parse PB
FDIR=../../try0428evt1/
PB_MDIR=../../try0428evt1/models_en/pb/
python3 -m msp2.tasks.zmtl2.main.test ${PB_MDIR}/_conf model_load_name:${PB_MDIR}/zmodel.best.m vocab_load_dir:${PB_MDIR}/ log_stderr:1 testM.group_tasks:pb1 "testM.group_files:en.bio11_gold_Protein_RALL.train.ud2.json" "testM.output_file:en.bio11_gold_Protein_RALL.train.ud2_pb.json" "pb1.frames_file:${FDIR}/frames.pb3.pkl" pb1.use_cons_evt:1 pb1.cons_evt_frame:None pb1.cons_evt_tok:lemma0 |& tee _log_parse_pb
# further parse PB args
python3 -m msp2.tasks.zmtl3.main.stream device:0 'conf_sbase:data:pbfn;task:arg' arg0.extend_span:0 arg0.np_getter:fn 'arg0.filter_noncore:+spec,-*' arg0.build_oload:pbfn arg0.arg_mode:tpl 'model_load_name:../run_zzf_srl_ALL/run_tpl/zmodel.srl_tpl.m###DMarg0.qmod.emb_frame,DMarg0.qmod.emb_role' arg0.mix_evt_ind:0.5 stream_input:en.bio11_gold_Protein_RALL.train.ud2_pb.json stream_output:en.bio11_gold_Protein_RALL.train.ud2_pbplus.json
# python3 -mpdb -m msp2.tasks.zmtl3.scripts.misc.ana_arg gold:en.bio11_gold_Protein.train.ud2.json preds:en.bio11_gold_Protein_RALL.train.ud2_pb.json,en.bio11_gold_Protein_RALL.train.ud2_pbplus.json
# -> posi-frame: 5246.0/34974.0=0.1500; 5246.0/8728.0=0.6011; 0.2401
# use that in "qadistill.sh"
$CCMD_Q1 stream_input:en.bio11_gold_Protein_RALL.train.ud2_pb.json stream_output:../qadistill/en.bio11_gold_Protein_RALL.train.ud2_pb.q1.json
$CCMD_Q1 stream_input:en.bio11_gold_Protein_RALL.train.ud2_pbplus.json stream_output:../qadistill/en.bio11_gold_Protein_RALL.train.ud2_pbplus.q1.json
fi

# --
# ok, final run
for spec in qa:run_zzf_qa_ALL asrl:run_zzf_augsrl_ALL/run_a6 gsrl:run_zzf_augsrl_ALL/run_genia; do
  ii=0
  IFS=':' read -r ss0 ss1 <<< $spec
  if [[ $ss0 == 'qa' ]]; then
    aa="arg0.arg_mode:mrc arg0.mrc_use_rques:1"
  else
    aa="arg0.arg_mode:tpl"
  fi
  for mm in ../$ss1/zmodel.*.m ../$ss1/*/zmodel.best.m; do
    echo ZRUN $mm
    prun pred_Protein run_${ss0}_${ii} $mm "$aa"
    ii=$((ii+1))
  done
done
#run_asrl_0/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1607)     2617 ( 1606)    49.55    61.37    54.83 (*)
#run_asrl_1/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1584)     2566 ( 1583)    48.84    61.69    54.52
#run_asrl_2/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1566)     2528 ( 1565)    48.29    61.91    54.26
#run_asrl_3/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1567)     2551 ( 1566)    48.32    61.39    54.08
#run_asrl_4/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1567)     2529 ( 1566)    48.32    61.92    54.28
#run_asrl_5/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1548)     2506 ( 1547)    47.73    61.73    53.84
#run_gsrl_0/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1643)     2694 ( 1642)    50.66    60.95    55.33
#run_gsrl_1/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1607)     2589 ( 1606)    49.55    62.03    55.09
#run_gsrl_2/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1643)     2653 ( 1642)    50.66    61.89    55.72 (*)
#run_gsrl_3/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1629)     2628 ( 1628)    50.23    61.95    55.48
#run_gsrl_4/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1613)     2599 ( 1612)    49.74    62.02    55.21
#run_gsrl_5/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1591)     2586 ( 1590)    49.06    61.48    54.57
#run_qa_0/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1379)     3116 ( 1377)    42.52    44.19    43.34
#run_qa_1/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1454)     2828 ( 1452)    44.84    51.34    47.87 (*)
#run_qa_2/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1415)     3174 ( 1413)    43.63    44.52    44.07
#run_qa_3/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1429)     2827 ( 1427)    44.06    50.48    47.05
#run_qa_4/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1489)     3032 ( 1487)    45.91    49.04    47.43
#run_qa_5/dev_results.txt:    ==[ALL-TOTAL]==       3243 ( 1458)     3110 ( 1456)    44.96    46.82    45.87
# -> submit
