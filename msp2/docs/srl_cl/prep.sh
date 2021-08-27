#

# one script to prepare data (srl and ud) for srl-cl experiments
# note: require PYTHONPATH to include msp2

# --
set -e
ROOT_DIR=`pwd`
mkdir -p data/ud; cd data/ud  # root data dir
#PYTHONPATH=?? PATB_INT=?? EWT_FOLDER=?? CONLL12D=?? CONLL09=?? PTB3=?? CTB6=?? bash prep.sh
# --

# step 0: prepare
echo "Please make sure you have set the env variables of (use ABSolute path!!):"
echo "'PYTHONPATH' should point to the root dir of the msp code repo."
echo "'PATB_INT' should point to integrated dir of PATB files (all integrated files including the three parts) as indicated by 'https://github.com/UniversalDependencies/UD_Arabic-NYUAD/tree/r2.7#data'"
echo "'EWT_FOLDER' points to the dir containing EWT pb files."
echo "'CONLL12D' should point to the dir of 'pb/conll12d' prepared by 'srl_span/prep.sh'"
echo "'CONLL09' should point to a folder where there are original CoNLL09 data, we are expecting \$CONLL09/CoNLL2009-ST-*"
echo "'PTB3' should point to the folder of PTB3, we are expecting \$PTB3/parsed/mrg/wsj/*/*.mrg"
echo "'CTB6' should point to the folder of CTB6, we are expecting \$CTB6/data/utf8/bracketed/*.fid"
echo "Current settings are: $PYTHONPATH, $PATB_INT, $EWT_FOLDER, $CONLL12D, $CONLL09, $PTB3, $CTB6"
read -p "Press any key to continue if they are all set well:" _TMP

# step 1: download UD
echo "Step 1: download and prepare UD"
# --
# UD v1.4
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1827/ud-treebanks-v1.4.tgz
tar -zxvf ud-treebanks-v1.4.tgz
# --
# UD v2.7
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3424/ud-treebanks-v2.7.tgz
tar -zxvf ud-treebanks-v2.7.tgz
# --
# specific for Arabic-NYUAD
cd ud-treebanks-v2.7/UD_Arabic-NYUAD/
for ff in *.conllu; do
  java -jar merge.jar $ff $PATB_INT
  mv $ff $ff.blank
  mv ${ff%.conllu}.merged.conllu $ff
done
cd ${OLDPWD}
# --

# step 2: prepare ewt, upb, fipb
# --
# upb1.0
# (21.01.24): 60e2fb824e304c90cbee692aa3adadcf54f5c73f
echo "Prepare UPB"
git clone https://github.com/System-T/UniversalPropositions
mv UniversalPropositions/UP_English-EWT UniversalPropositions/_UP_English-EWT
for ff in UniversalPropositions/UP*/*.conllu; do
  python3 -m msp2.cli.change_format R.input_path:$ff R.input_format:conllup W.output_path:${ff%.conllu}.json
done |& tee _log_up_change
# simply combine the two spanish ones
mkdir -p UniversalPropositions/UP_Spanish2/
for wset in train dev test; do
  cat UniversalPropositions/UP_Spanish/es-up-${wset}.json UniversalPropositions/UP_Spanish-AnCora/es_ancora-up-${wset}.json > UniversalPropositions/UP_Spanish2/es2-up-${wset}.json;
done
# --
mkdir -p pb2; cd pb2;
# --
# ewt
echo "Prepare EWT"
cp ${EWT_FOLDER}/ewt.*.conll .
for wset in train dev test; do
  python3 -m msp2.scripts.ud.assign_p2d.assign_anns input.input_path:ewt.${wset}.conll aux.input_path:../ud-treebanks-v1.4/UD_English/en-ud-${wset}.conllu output.output_path:en_span.ewt.${wset}.json input.input_format:conllpb aux.input_format:conllu output_sent_and_discard_nonhit:1
  python3 -m msp2.scripts.ud.prep.span2dep en_span.ewt.${wset}.json en.ewt.${wset}.json
done |& tee _log.ewt  # hit for train/dev/test -> 0.9999/1.0000/0.9990
# --
# fi-pb (simply read it into zjson)
# (21.01.26): 77a694a765a93d4f944bb9302ea5d1f2132d9cdd
for wset in train dev test; do
  # note: it seems that other fields are exactly the same as udv14
  wget https://raw.githubusercontent.com/TurkuNLP/Finnish_PropBank/data/fipb-ud-${wset}.conllu
  python3 -m msp2.scripts.ud.prep.read_fipb fipb-ud-${wset}.conllu fipb-ud-${wset}.json
done |& tee _log.fipb
# --
cd ..;
# --

# step 2.5: prepare stanford-nlp for further preprocessing
echo "Step 2.5: prepare stanford NLP"
wget http://nlp.stanford.edu/software/stanford-corenlp-4.2.0.zip
unzip stanford-corenlp-4.2.0.zip
wget http://nlp.stanford.edu/software/stanford-corenlp-4.2.0-models-english.jar
wget https://raw.githubusercontent.com/stanfordnlp/CoreNLP/master/data/edu/stanford/nlp/upos/ENUniversalPOS.tsurgeon
mkdir -p edu/stanford/nlp/models/upos/
mv ENUniversalPOS.tsurgeon edu/stanford/nlp/models/upos/
jar cf stanford-parser-missing-file.jar edu/stanford/nlp/models/upos/ENUniversalPOS.tsurgeon
mv stanford-corenlp-4.2.0-models-english.jar stanford-parser-missing-file.jar stanford-corenlp-4.2.0

# step 3: prepare ontonotes
echo "Step 3: prepare ontonotes"
mkdir -p pb12; cd pb12;
# --
CORENLP_HOME="../stanford-corenlp-4.2.0/"
C12_DIR=${CONLL12D}/conll-2012/
ONTO5_DIR=${CONLL12D}/../conll12/ontonotes-release-5.0/
CONV_CMD0="java -Xmx8g -cp ${CORENLP_HOME}/stanford-corenlp-4.2.0.jar:${CORENLP_HOME}/stanford-parser-missing-file.jar edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -conllx -keepPunct -language"
# --
# gather
rm -f _*.files
for wset in train development test; do
find ${C12_DIR}/v4/data/${wset}/data/english/ -name "*.v4_gold_conll" | sed -e 's/\(.*\)\(english\)\(.*\)/\2\3/g' >>_en.files
find ${C12_DIR}/v4/data/${wset}/data/chinese/ -name "*.v4_gold_conll" | sed -e 's/\(.*\)\(chinese\)\(.*\)/\2\3/g' >>_zh.files
done
# convert
for cl in en zh; do
while IFS= read -r line; do
# [nope] echo "# ${line}"
  cat "${ONTO5_DIR}/data/files/data/${line%.v4_gold_conll}.parse" | python3 -m msp2.scripts.ud.assign_p2d.change_trees ${cl}
done <_${cl}.files >_${cl}.penn
${CONV_CMD0} ${cl} -treeFile _${cl}.penn >_${cl}.conllu
done
# directly get ar
cat ../ud-treebanks-v2.7/UD_Arabic-NYUAD/*.conllu >_ar.conllu
# --
# get data
cp ${CONLL12D}/*.conll.json .
for wset in train dev test; do
  cp ${CONLL12D}/../conll12b/${wset}.conll.json en.${wset}.conll.json
done
# first parse them with stanza
for cl in en ar zh; do
for wset in train dev test; do
 CUDA_VISIBLE_DEVICES= python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:${cl} stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:16 input_path:${cl}.${wset}.conll.json output_path:${cl}.${wset}.conll.ud2.json
done; done |& tee _log.ud2
# assign ud
for wset in train dev test; do
  python3 -m msp2.scripts.ud.assign_p2d.assign_anns input.input_path:en.${wset}.conll.ud2.json aux.input_path:_en.conllu output.output_path:en.${wset}.conll.ud.json aux.input_format:conllu change_char_scheme:en
  python3 -m msp2.scripts.ud.assign_p2d.assign_anns input.input_path:zh.${wset}.conll.ud2.json aux.input_path:_zh.conllu output.output_path:zh.${wset}.conll.ud.json aux.input_format:conllu change_char_scheme:zh convert_f:convert_zh
  python3 -m msp2.scripts.ud.assign_p2d.assign_anns input.input_path:ar.${wset}.conll.ud2.json aux.input_path:_ar.conllu output.output_path:ar.${wset}.conll.ud.json aux.input_format:conllu delete_char_scheme:ar fuzzy_word_cnum:3 fuzzy_seq_wrate:0.5 change_words:1
done |& tee _log.assign
# --
# further another set for zh with udapy
python3 -m msp2.scripts.ud.assign_p2d.upos_ctb2ud _zh.conllu _zh2.conllu
udapy -s ud.Convert1to2 <_zh2.conllu >_zh2p.conllu 2>_zh2p.log
for wset in train dev test; do
  python3 -m msp2.scripts.ud.assign_p2d.assign_anns input.input_path:zh.${wset}.conll.ud2.json aux.input_path:_zh2p.conllu output.output_path:zh2.${wset}.conll.ud.json aux.input_format:conllu change_char_scheme:zh convert_f:
done |& tee _log.assign.zh2
# --
cd ..;

# step 4: prepare conll09
echo "Step 3: prepare ontonotes"
mkdir -p conll09; cd conll09;
# --
declare -A CL_MAP=(
  [English]="en" [Chinese]="zh"
  [Catalan]="ca" [Czech]="cs" [German]="de" [Spanish]="es"
)
# convert
for cl in English Chinese Catalan Czech German Spanish; do
  cl2="${CL_MAP[${cl}]}"
  echo $cl2
  python3 -m msp2.cli.change_format R.input_path:${CONLL09}/CoNLL2009-ST-${cl}/CoNLL2009-ST-${cl}-train.txt R.input_format:conll09 W.output_path:${cl2}.train.json
  python3 -m msp2.cli.change_format R.input_path:${CONLL09}/CoNLL2009-ST-${cl}/CoNLL2009-ST-${cl}-development.txt R.input_format:conll09 W.output_path:${cl2}.dev.json
  python3 -m msp2.cli.change_format R.input_path:${CONLL09}/CoNLL2009-ST-${cl}/CoNLL2009-ST-evaluation-${cl}.txt R.input_format:conll09 W.output_path:${cl2}.test.json
done |& tee _log.data
# obtain ud
CORENLP_HOME="../stanford-corenlp-4.2.0/"
CONV_CMD0="java -Xmx8g -cp ${CORENLP_HOME}/stanford-corenlp-4.2.0.jar:${CORENLP_HOME}/stanford-parser-missing-file.jar edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -conllx -keepPunct -language"
# en from ptb3
cat ${PTB3}/parsed/mrg/wsj/*/*.mrg >_en.penn
${CONV_CMD0} en -treeFile _en.penn >_en.conllu
# zh from ctb6
cat ${CTB6}/data/utf8/bracketed/*.fid | grep -v -E "^\s*<.*>" >_zh.penn
${CONV_CMD0} zh -treeFile _zh.penn >_zhv1.conllu
python3 -m msp2.scripts.ud.assign_p2d.upos_ctb2ud _zhv1.conllu _zhv1p.conllu
udapy -s ud.Convert1to2 <_zhv1p.conllu >_zh.conllu 2>_zhv1pto2.log
# ca from ancora
cat ../ud-treebanks-v2.7/UD_Catalan-AnCora/*.conllu >_ca.conllu
# es from ancora
cat ../ud-treebanks-v2.7/UD_Spanish-AnCora/*.conllu >_es.conllu
# cs from pdt
cat ../ud-treebanks-v2.7/UD_Czech-PDT/*.conllu >_cs.conllu
# assign ud and conv
for cl in en zh ca cs es; do
for wset in train dev test; do
  echo "#ann $cl.$wset"
  python3 -m msp2.scripts.ud.assign_p2d.assign_anns_v2 input.input_path:${cl}.${wset}.json aux.input_path:_${cl}.conllu output.output_path:${cl}.${wset}.udA.json aux.input_format:conllu
  echo "conv $cl.$wset"
  python3 -m msp2.scripts.ud.conll09.convert_depsrl src_input.input_path:${cl}.${wset}.json trg_input.input_path:${cl}.${wset}.udA.json output.output_path:${cl}.${wset}.ud.json method:path
done; done |& tee _log.udA
# --
cd ..

# step 5: final preparation for different experiments (shuffling and put into specific dirs)
# --
# cl0
mkdir -p cl0; cd cl0;
# first change ewt's data to short arg labels
for wset in train dev test; do
  python3 -m msp2.scripts.ud.prep.change_arg_label input_path:../pb2/en.ewt.${wset}.json output_path:en.ewt.${wset}.json
done
# then get others' ud and shuffle!
declare -A UDTABLE=(
  [en]="UD_English" [fi]="UD_Finnish" [fr]="UD_French" [de]="UD_German"
  [it]="UD_Italian" [pt_bosque]="UD_Portuguese-Bosque" [es_ancora]="UD_Spanish-AnCora" [es]="UD_Spanish"
)
# get them
for CL in "${!UDTABLE[@]}"; do
  FULL_CL=${UDTABLE[${CL}]}
  echo ${CL} ${FULL_CL};
  for wset in train dev test; do
    python3 -m msp2.cli.change_format R.input_path:../ud-treebanks-v1.4/${FULL_CL}/${CL}-ud-${wset}.conllu R.input_format:conllu W.output_path:_tmp.json  # change to json
    python3 -m msp2.scripts.tools.sample_shuffle input:_tmp.json output:${CL}.ud.${wset}.json shuffle:1
  done
done |& tee _log
# combine the spanish ones
for wset in train dev test; do
  cat es_ancora.ud.${wset}.json es.ud.${wset}.json >_tmp.json
  python3 -m msp2.scripts.tools.sample_shuffle input:_tmp.json output:es2.ud.${wset}.json shuffle:1
done
# --
cd ..
# --
# cl1
mkdir -p cl1; cd cl1;
# simply cp cl0's en set!
cp ../pb2/en.ewt.* .
# and shuffle fi ones
for wset in train dev test; do
  python3 -m msp2.scripts.tools.sample_shuffle input:../pb2/fipb-ud-${wset}.json output:fipb.${wset}.json shuffle:1
done
cd ..
# --
# cl2
mkdir -p cl2; cd cl2;
# get them
for wset in train dev test; do
  python3 -m msp2.scripts.tools.sample_shuffle input:../pb12/en.${wset}.conll.ud.json output:en.${wset}.ud.json shuffle_times:1
  python3 -m msp2.scripts.tools.sample_shuffle input:../pb12/ar.${wset}.conll.ud.json output:ar.${wset}.ud.json shuffle_times:1
  # use zh2 to overwrite ud ones!
  python3 -m msp2.scripts.tools.sample_shuffle input:../pb12/zh2.${wset}.conll.ud.json output:zh.${wset}.ud.json shuffle_times:1
done
cd ..
# --
# cl3
mkdir -p cl3; cd cl3;
# sample them
for cl in en zh ca cs es; do
for wset in train dev test; do
  python3 -m msp2.scripts.tools.sample_shuffle input:../conll09/${cl}.${wset}.ud.json output:./${cl}.${wset}.ud.json shuffle_times:1
done
done
cd ..
# --
