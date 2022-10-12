# --
# prepare event datasets
# todo(note): deprecated by those in "prep"

# --
# specify the path!!
export PYTHONPATH=../../zsp2021/src/:${PYTHONPATH}
# --

# read ace/ere
DATA_DIR="../../../working/zop20/data5/outputs_split/"
for dd in en.ace.{train,dev,test} {en,es,zh}.ere.{train,dev,test54,test55}; do
  echo "#=====\nPrepare $dd"
  python3 -m msp2.cli.change_format R.input_path:${DATA_DIR}/$dd.json R.input_format:zdoc W.output_path:$dd.json
done |& tee _log_ae

# read rams
wget https://nlp.jhu.edu/rams/RAMS_1.0b.tar.gz
tar -zxvf RAMS_1.0b.tar.gz
for dd in train dev test; do
  echo "#=====\nPrepare $dd"
  python3 -m msp2.cli.change_format R.input_path:RAMS_1.0/data/$dd.jsonlines R.input_format:rams W.output_path:en.rams.$dd.json
done |& tee _log_rams

# annotate with stanza
for ff in en.*.json; do
  echo "#===== Parse(stanza) $ff"
  CUDA_VISIBLE_DEVICES=1 python3 -m msp2.cli.annotate 'stanza' stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:32 input_path:${ff} output_path:${ff%.json}.ud.json
done |& tee _log_stanza
