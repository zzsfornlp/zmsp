#

# prepare the event datasets

# --
# prepare for the data-preparation
if [[ -z ${LDC_DIR} ]]; then
  LDC_DIR="../data/raw/"
fi
LDC_DIR=$(readlink -f ${LDC_DIR})
echo "Before running, specify ENV as: LDC_DIR=${LDC_DIR} PYTHONPATH=${PYTHONPATH}"
# --

# --
# ACE05 using dygiepp (ace1)

echo "Please follow 'https://github.com/dwadden/dygiepp#ace05-event' to obtain the pre-processed data, which should be available at 'dygiepp/data/ace-event/processed-data/default-settings/json/'"
# --
#git clone https://github.com/dwadden/dygiepp
#cd dygiepp
##conda deactivate
##conda create --name ace-event-preprocess python=3.7
#conda activate ace-event-preprocess
##pip install -r scripts/data/ace-event/requirements.txt
##python -m spacy download en
#bash ./scripts/data/ace-event/collect_ace_event.sh ${LDC_DIR}/LDC2006T06/ace_2005_td_v7/
#python ./scripts/data/ace-event/parse_ace_event.py default-settings
#conda deactivate
#cd ..
# --
for wset in train dev 'test'; do
  python3 -m msp2.scripts.event.prep.s3_convert dygiepp/data/ace-event/processed-data/default-settings/json/${wset}.json en.ace1.${wset}.json convert_dygiepp;
done |& tee _log_en_ace1

# --
# ACE05/ERE using oneie (ace2,ere2)

echo "Please first download oneie-v0.4.8 from 'http://blender.cs.illinois.edu/software/oneie/' and decompress it!"
# ace
mkdir -p _en_ace
python3 oneie_v0.4.8/preprocessing/process_ace.py -i ${LDC_DIR}/LDC2006T06/ace_2005_td_v7/data/ -o _en_ace -s oneie_v0.4.8/resource/splits/ACE05-E -l english
# ere
mkdir -p _en_ere{A,B,C} _en_ere
python3 oneie_v0.4.8/preprocessing/process_ere.py -i ${LDC_DIR}/LDC2015E29/data -o _en_ereA -d normal -l english
python3 oneie_v0.4.8/preprocessing/process_ere.py -i ${LDC_DIR}/LDC2015E68/data -o _en_ereB -d r2v2 -l english
python3 oneie_v0.4.8/preprocessing/process_ere.py -i ${LDC_DIR}/LDC2015E78/data -o _en_ereC -d parallel -l english
cat _en_ere?/*.oneie* >_en_ere/english.oneie.json
python3 -m msp2.tasks.zmtl3.scripts.data.split_data _en_ere/english.oneie.json _en_ere oneie_v0.4.8/resource/splits/ERE-EN
# convert
for dset in ace ere; do
for wset in train dev test; do
  python3 -m msp2.tasks.zmtl3.scripts.data.oneie2mine _en_${dset}/${wset}.oneie.json en.${dset}2.${wset}.json
done
done |& tee _log_en_oneie
# --
# run with oneie
# python train.py -c ./config/ace.json
# for wset in dev test; do python decode_oneie.py -m ./_model_ace/20211115_133948/best.role.mdl -i _en_ace/${wset}.oneie.json -o ./_model_ace/20211115_133948/best_out.${wset}.json --gpu; done
# or since they store best-dev results:
# for wset in dev test; do python3 -m msp2.tasks.zmtl3.scripts.data.oneie2mine ./_model_ace/20211115_133948/{result.${wset}.json,zout.${wset}.json}; python decode_oneie.py -m ./_model_ace/20211115_133948/best.role.mdl -i _en_ace/${wset}.oneie.json --eval_path ./_model_ace/20211115_133948/result.${wset}.json; python3 -m msp2.scripts.event.eeval gold.input_path:../events/data/data21f/en.ace2.${wset}.json pred.input_path:./_model_ace/20211115_133948/zout.${wset}.json; done

# --
# RAMS_v1.0

wget https://nlp.jhu.edu/rams/RAMS_1.0b.tar.gz
tar -zxvf RAMS_1.0b.tar.gz
for wset in train dev 'test'; do
  python3 -m msp2.cli.change_format R.input_path:./RAMS_1.0/data/${wset}.jsonlines R.input_format:rams W.output_path:en.rams.${wset}.json
done |& tee _log_rams

# =====
# parse them all
#for ff in en.*.json; do
# python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:en stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:${ff} "output_path:${ff%.json}.ud2.json"
#done |& tee _log_parse

# stat them all
#PYTHONPATH=?? python3 -m pdb -m msp2.cli.analyze frame frame_getter:evt gold:??
#for ff in en.*.ud2.json; do
#  python3 -m msp2.scripts.event.prep.sz_stat input_path:$ff  # stat
#done |& tee _log_stat

# =====
# --
# prepare more oneie data
#for dd in ace ere; do
#  mkdir -p _en_${dd}_no5
#  for wset in train dev test; do
#    python3 -m msp2.tasks.zmtl3.scripts.data.filter_data _en_${dd}/${wset}.oneie.json _en_${dd}_no5/${wset}.oneie.json "${dd}.-s5,+*"
#  done |& tee _en_${dd}_no5/_log
#done
# --
# train
# go.sh
#for mm in ace ace_no5 ere ere_no5; do
# note: need to add a line to utils.py: if relation not in relation_type_vocab: continue
#for mm in ere ere_no5; do
#  MMDIR="_model_$mm"
#  mkdir -p $MMDIR
#  python train.py -c ./config/$mm.json
#  mv $MMDIR/*/* $MMDIR/  # move it above!
#  for wset in dev test; do
#    python3 -m msp2.tasks.zmtl3.scripts.data.oneie2mine $MMDIR/{result.${wset}.json,zout.${wset}.json};
#    python decode_oneie.py -m $MMDIR/best.role.mdl -i "_en_$mm/${wset}.oneie.json" --eval_path $MMDIR/result.${wset}.json;
#  done |& tee $MMDIR/_log
#done
# --

# --
# to eval these?
#for mm in ace; do
#  MMDIR="_model_$mm"
#for wset in dev test; do
#  GFILE=../events/data/data21f/en.ace2.${wset}.json
#  PFILE=$MMDIR/zout.${wset}.json
#  python3 -m msp2.cli.analyze frame do_loop:0 gold:$GFILE preds:$PFILE match_arg_with_frame:1
#  python3 -m msp2.cli.analyze frame do_loop:0 gold:$GFILE preds:$PFILE match_arg_with_frame:0
#  python3 -m msp2.scripts.event.eeval gold.input_path:$GFILE pred.input_path:$PFILE
#done |& tee $MMDIR/_logE
#done
# --
