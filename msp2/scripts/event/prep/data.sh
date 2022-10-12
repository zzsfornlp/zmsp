#

# --
# what data do we have?

# =====
# first, the harder to prepare ones: ace/ere

# ACE2005 (LDC2006T06/ace_2005_td_v7)
# en/zh/ar: 599/633/(403-1=402)

# ERE(TAC-KBP17)
# en+kbp15(693+360 -> 866/20/167):
# -- LDC2015E29, LDC2015E68, LDC2016E31, LDC2016E73(LDC2017E02), LDC2017E54/LDC2017E55, kbp15
# zh(693 -> 506/20/167):
# -- LDC2015E105, LDC2015E112, LDC2015E78, LDC2016E73(LDC2017E02), LDC2017E54/LDC2017E55
# es(403 -> 217/20/166): LDC2015E107, LDC2016E34, LDC2016E73(LDC2017E02), LDC2017E54/LDC2017E55

# step 1: extract
mkdir -p ./extract
for cc in en.ace en.ere en.kbp15 zh.ace zh.ere es.ere ar.ace; do
  python3 s1_extract.py ./raw/ ${cc} ./extract/${cc}.json;
done |& tee ./extract/_log

# step 2: tokenize and settle positions
for toker in corenlp stanza; do
#for toker in corenlp; do
#for toker in stanza; do
mkdir -p ./"tokenize_${toker}";
for cc in en.ace en.ere en.kbp15 zh.ace zh.ere es.ere ar.ace; do
  python3 s2_tokenize.py -i ./extract/${cc}.json -o ./"tokenize_${toker}"/${cc}.json --tokenizer ${toker}
done >./"tokenize_${toker}"/_log.stdout 2>./"tokenize_${toker}"/_log.stderr
done
ln -s tokenize_corenlp tokenize  # choose to use corenlp, which generally seems to get less mismatch

# step 3: convert to my format
export PYTHONPATH="../../../zsp2021/src/"
mkdir -p convert
for cc in en.ace en.ere en.kbp15 zh.ace zh.ere es.ere ar.ace; do
  python3 s3_convert.py ./tokenize/${cc}.json ./convert/${cc}.json;
done |& tee ./convert/_log0

# =====
# then some easier to prepare (pre-tokenized) ones: rams/maven/ace2-dygiepp

# ramsv1.0
# -- directly use internal converter
wget https://nlp.jhu.edu/rams/RAMS_1.0b.tar.gz -O ./raw/RAMS_1.0b.tar.gz
tar -zxvf ./raw/RAMS_1.0b.tar.gz -C ./raw/
for dd in train dev 'test'; do
  python3 -m msp2.cli.change_format R.input_path:./raw/RAMS_1.0/data/${dd}.jsonlines R.input_format:rams W.output_path:./convert/en.rams.${dd}.json
done |& tee ./convert/_log_rams

# ace2-dygiepp
for dd in train dev 'test'; do
  python3 s3_convert.py ./raw/ace2/${dd}.json ./convert/en.ace2.${dd}.json convert_dygiepp;
done |& tee ./convert/_log_ace2

# maven
for dd in train dev 'test'; do
  python3 s3_convert.py ./raw/maven/${dd}.jsonl ./convert/en.maven.${dd}.json convert_maven;
done |& tee ./convert/_log_maven

# =====
# further use stanza to parse (lemma/upos/parse)

mkdir -p parse
for cl in en zh es ar; do
  for ff in ./convert/${cl}.*.json; do
    CUDA_VISIBLE_DEVICES=1 python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:1 stanza_lang:${cl} stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:2 input_path:${ff} "output_path:./parse/`basename ${ff}`"
  done |& tee ./parse/_log.${cl}
done

# --
# also split on ace/ere
mkdir -p split
cp parse/*.json split/
for cc in en.ace en.ere zh.ace zh.ere es.ere ar.ace; do
  python3 s4_split.py ./split/${cc}.json ./split/${cc};
done |& tee ./split/_log_split

# --
# stat
for ff in ./split/*.json; do
  echo "#== ${ff}"
  python3 sz_stat.py input_path:${ff}
done |& tee ./split/_log_stat

# --
# note: further use those in "tune0207ud.py" to parse to split2
# and further shuffle the training files to splitS
for ff in ../split2/*.train.json ../split2/en.kbp15.json; do
  python3 -m  msp2.scripts.tools.sample_shuffle "input:$ff" "output:`basename $ff`" shuffle_times:1
done
