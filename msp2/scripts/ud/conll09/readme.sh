#

# preparing conll09 data
# from LDC2012T03 & LDC2012T04
# -> (merge 2009_conll_p* with "mv 2009_conll_p*/data/CoNLL2009-ST-* 2009_conll")

# --
# note: first export PYTHONPATH
#export PYTHONPATH=../../src/
# --

# --
declare -A CL_MAP=(
  [English]="en" [Chinese]="zh"
  [Catalan]="ca" [Czech]="cs" [German]="de" [Spanish]="es"
)
# --

# --
# convert format
for cl in English Chinese Catalan Czech German Spanish; do
  cl2="${CL_MAP[${cl}]}"
  echo $cl2
  python3 -m msp2.cli.change_format R.input_path:2009_conll/CoNLL2009-ST-${cl}/CoNLL2009-ST-${cl}-train.txt R.input_format:conll09 W.output_path:${cl2}.train.json
  python3 -m msp2.cli.change_format R.input_path:2009_conll/CoNLL2009-ST-${cl}/CoNLL2009-ST-${cl}-development.txt R.input_format:conll09 W.output_path:${cl2}.dev.json
  python3 -m msp2.cli.change_format R.input_path:2009_conll/CoNLL2009-ST-${cl}/CoNLL2009-ST-evaluation-${cl}.txt R.input_format:conll09 W.output_path:${cl2}.test.json
  python3 -m msp2.cli.change_format R.input_path:2009_conll/CoNLL2009-ST-${cl}/CoNLL2009-ST-evaluation-${cl}-ood.txt R.input_format:conll09 W.output_path:${cl2}.ood.json
done |& tee _log.data

# --
# stat
for ff in *.json; do
  echo "#Stat $ff"
  python3 stat_udsrl.py zjson $ff
done |& tee _log.stat

# --
# parse with stanza as ud2
#[stanza.download(z) for z in ['en','zh','es','ar']]
#[stanza.download(z) for z in ['ca', 'cs', 'de']]
for cl in en zh ca cs de es; do
#for wset in train dev test ood; do
for wset in train dev test; do
 CUDA_VISIBLE_DEVICES= python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:${cl} stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:16 input_path:${cl}.${wset}.json output_path:${cl}.${wset}.ud2.json
done; done |& tee _log.ud2

# --
# obtain ud
# prepare ptb & ctb & others (see "assign_p2d/assign.sh" for preps)
CORENLP_HOME=~/stanza_corenlp/
CONV_CMD0="java -Xmx8g -cp ${CORENLP_HOME}/stanford-corenlp-4.2.0.jar:${CORENLP_HOME}/stanford-parser-missing-file.jar edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -conllx -keepPunct -language"
# en from ptb3
cat treebank_3/parsed/mrg/wsj/*/*.mrg >_en.penn
${CONV_CMD0} en -treeFile _en.penn >_en.conllu
# zh from ctb6
cat ctb_v6/data/utf8/bracketed/*.fid | grep -v -E "^\s*<.*>" >_zh.penn
${CONV_CMD0} zh -treeFile _zh.penn >_zhv1.conllu
python3 upos_ctb2ud.py _zhv1.conllu _zhv1p.conllu
~/.local/bin/udapy -s ud.Convert1to2 <_zhv1p.conllu >_zh.conllu 2>_zhv1pto2.log
# ca from ancora
cat ../ud-treebanks-v2.7/UD_Catalan-AnCora/*.conllu >_ca.conllu
# es from ancora
cat ../ud-treebanks-v2.7/UD_Spanish-AnCora/*.conllu >_es.conllu
# cs from pdt
cat ../ud-treebanks-v2.7/UD_Czech-PDT/*.conllu >_cs.conllu
# --
# assign ud
for cl in en zh ca cs es; do
#for wset in train dev test ood; do
for wset in train dev test; do
  echo "#ann $cl.$wset"
  python3 assign_anns_v2.py input.input_path:${cl}.${wset}.json aux.input_path:_${cl}.conllu output.output_path:${cl}.${wset}.udA.json aux.input_format:conllu
  echo "#eval $cl.$wset"
  python3 -m msp2.cli.analyze ud gold:${cl}.${wset}.udA.json preds:${cl}.${wset}.json,${cl}.${wset}.ud2.json </dev/null
done; done |& tee _log.udA

# --
# finally, convert arguments!
mkdir convert; cd convert;
# --
zgo () {
python3 convert_depsrl.py src_input.input_path:${SRC} trg_input.input_path:${TRG} output.output_path:${OUT1} method:${METHOD}
python3 convert_depsrl.py src_input.input_path:${OUT1} trg_input.input_path:${SRC} output.output_path:${OUT2} method:${METHOD}
python3 -m msp2.cli.analyze frame gold:${SRC} preds:${OUT1},${OUT2} econf:pb no_join_c:1 auto_save_name: </dev/null
# zz = filter fl "(lambda x: [z.mention.widx for z in x])(d.gold.args) != (lambda x: [z.mention.widx for z in x])(d.pred.args)"
}
# --
for cl in en zh ca cs es; do
#for wset in train dev test ood; do
for wset in train dev test; do
  echo "#conv $cl.$wset"
  SRC=../$cl.$wset.json TRG=../$cl.$wset.udA.json OUT1=$cl.$wset.ud.json OUT2=$cl.$wset.back.json METHOD=path zgo
done; done |& tee _log.conv
# --
# extra comparing with span
#for cl in en zh ca cs es; do
##for wset in train dev test ood; do
#for wset in dev; do
#  echo "#conv $cl.$wset"
#  SRC=../$cl.$wset.json TRG=../$cl.$wset.udA.json OUT1=$cl.$wset.ud.json OUT2=$cl.$wset.back.json METHOD=span zgo
#done; done |& tee _log.conv2
# --
