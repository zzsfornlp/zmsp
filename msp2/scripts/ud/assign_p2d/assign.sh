#

# gather trees from treebanks and then lookup

# --
# use the newer version of CoreNLP 4.2.0 (downloaded by stanza)
CORENLP_HOME=~/stanza_corenlp/
# get the upos mapper rules
wget https://raw.githubusercontent.com/stanfordnlp/CoreNLP/master/data/edu/stanford/nlp/upos/ENUniversalPOS.tsurgeon
mkdir -p edu/stanford/nlp/models/upos/
mv ENUniversalPOS.tsurgeon edu/stanford/nlp/models/upos/
jar cf stanford-parser-missing-file.jar edu/stanford/nlp/models/upos/ENUniversalPOS.tsurgeon
mv stanford-parser-missing-file.jar ${CORENLP_HOME}

# --
# get trees from ontonotes
C12_DIR=../../pb/conll12d/conll-2012/
ONTO5_DIR=../../pb/conll12/ontonotes-release-5.0/
CONV_CMD0="java -Xmx8g -cp ${CORENLP_HOME}/stanford-corenlp-4.2.0.jar:${CORENLP_HOME}/stanford-parser-missing-file.jar edu.stanford.nlp.trees.EnglishGrammaticalStructure -basic -conllx -keepPunct -language"
# --
# gather
rm _*.files
for wset in train development test; do
find ${C12_DIR}/v4/data/${wset}/data/english/ -name "*.v4_gold_conll" | sed -e 's/\(.*\)\(english\)\(.*\)/\2\3/g' >>_en.files
find ${C12_DIR}/v4/data/${wset}/data/chinese/ -name "*.v4_gold_conll" | sed -e 's/\(.*\)\(chinese\)\(.*\)/\2\3/g' >>_zh.files
done
# convert
for cl in en zh; do
while IFS= read -r line; do
# [nope] echo "# ${line}"
  cat "${ONTO5_DIR}/data/files/data/${line%.v4_gold_conll}.parse" | python3 change_trees.py ${cl}
done <_${cl}.files >_${cl}.penn
${CONV_CMD0} ${cl} -treeFile _${cl}.penn >_${cl}.conllu
done
#${CONV_CMD0} en -treeFile _en.penn >_en.conllu
#${CONV_CMD0} zh -treeFile _zh.penn >_zh.conllu
# directly get ar
cat ../ud-treebanks-v2.7/UD_Arabic-NYUAD/*.conllu >_ar.conllu

# --
export PYTHONPATH=../../../zsp2021/src/
# --

# --
# parse them with stanza as ud2
for cl in en ar zh; do
for wset in train dev test; do
 CUDA_VISIBLE_DEVICES= python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:${cl} stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:16 input_path:${cl}.${wset}.conll.json output_path:${cl}.${wset}.conll.ud2.json
done; done |& tee _log.ud2

# --
# assign them on
for wset in train dev test; do
  python3 assign_anns.py input.input_path:en.${wset}.conll.ud2.json aux.input_path:_en.conllu output.output_path:en.${wset}.conll.ud.json aux.input_format:conllu change_char_scheme:en
  python3 assign_anns.py input.input_path:zh.${wset}.conll.ud2.json aux.input_path:_zh.conllu output.output_path:zh.${wset}.conll.ud.json aux.input_format:conllu change_char_scheme:zh convert_f:convert_zh
  python3 assign_anns.py input.input_path:ar.${wset}.conll.ud2.json aux.input_path:_ar.conllu output.output_path:ar.${wset}.conll.ud.json aux.input_format:conllu delete_char_scheme:ar fuzzy_word_cnum:3 fuzzy_seq_wrate:0.5 change_words:1
done |& tee _log.assign
# eval: note: still not super consistent with stanza-ud2: upos/uas/las very low: en(0.9437/0.8451/0.8060), zh(0.8284/0.5260/0.3349), ar(0.7846/0.6208/0.4559)
# -- check head agreement, head/frame: en(0.9580/0.8704), zh(0.9033/0.7381), ar(0.8695/0.6280)

# --
# further more ud2 for zh
python3 upos_ctb2ud.py _zh.conllu _zh2.conllu
~/.local/bin/udapy -s ud.Convert1to2 <_zh2.conllu >_zh2p.conllu 2>_zh2p.log
for wset in train dev test; do
  python3 assign_anns.py input.input_path:zh.${wset}.conll.ud2.json aux.input_path:_zh2p.conllu output.output_path:zh2.${wset}.conll.ud.json aux.input_format:conllu change_char_scheme:zh convert_f:
done |& tee _log.assign.zh2
# eval:
for wset in train dev test; do
  python3 eval_arg_head.py gold.input_path:zh2.${wset}.conll.ud.json pred.input_path:zh.${wset}.conll.ud.json
  python3 eval_arg_head.py gold.input_path:zh2.${wset}.conll.ud.json pred.input_path:zh.${wset}.conll.ud2.json
  python3 eval_arg_head.py gold.input_path:zh2.${wset}.conll.ud.json pred.input_path:zh.${wset}.conll.ud3.json
done |& tee _log.eval.zh2
# -> still similar, but better pos, on dev (head agreement is almost the same):
# zh2 vs ud;;ud2;;ud3
#pos: 106924.0/110034.0=0.9717;; 92080.0/110034.0=0.8368;; 94670.0/110034.0=0.8604
#uas: 109495.0/110034.0=0.9951;; 58242.0/110034.0=0.5293;; 61674.0/110034.0=0.5605
#las: 106630.0/110034.0=0.9691;; 37802.0/110034.0=0.3435;; 42419.0/110034.0=0.3855
