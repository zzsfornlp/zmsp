#

# one script to prepare srl-pb (conll05 and conll12)
# note: require PYTHONPATH to include msp2

# --
set -e
ROOT_DIR=`pwd`
mkdir -p data/pb; cd data/pb  # root data dir
# PYTHONPATH=?? PTB3=?? ONTO5=?? bash prep.sh
# --

# step 0: prepare
echo "[1/3] Please make sure you have set the env variables of (use ABSolute path!!): PYTHONPATH, PTB3, ONTO5"
echo "Current settings are: $PYTHONPATH, $PTB3, $ONTO5"
read -p "Press any key to continue if they are all set well:" _TMP
echo "[2/3] Please prepare a conda environment of py27 which is python 2.7"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py27
conda deactivate
read -p "Press any key to continue if this is prepared:" _TMP
echo "[3/3] Please download certain files manually, which seems not able to be obtained automatically with wget, and put them at here: ${ROOT_DIR}"
echo http://ontonotes.cemantix.org/download/conll-formatted-ontonotes-5.0-scripts.tar.gz
echo http://conll.cemantix.org/2012/download/conll-2012-train.v4.tar.gz
echo http://conll.cemantix.org/2012/download/conll-2012-development.v4.tar.gz
echo http://conll.cemantix.org/2012/download/test/conll-2012-test-official.v9.tar.gz
echo http://conll.cemantix.org/2012/download/test/conll-2012-test-key.tar.gz
echo http://conll.cemantix.org/2012/download/conll-2012-scripts.v3.tar.gz
read -p "Press any key to continue if these files are well downloaded:" _TMP

# step 1: conll05
echo "Step 1: prepare for conll05"
mkdir -p conll05; cd conll05;
# step 1.1: get data
wget https://www.cs.upc.edu/~srlconll/conll05st-release.tar.gz
tar -xzvf conll05st-release.tar.gz
wget https://www.cs.upc.edu/~srlconll/conll05st-tests.tar.gz
tar -xzvf conll05st-tests.tar.gz
wget https://www.cs.upc.edu/~srlconll/srlconll-1.1.tgz
tar -xzvf srlconll-1.1.tgz
# step 1.2: use "prep05.py" to extract them (need to provide PTB3)!
ln -s "`readlink -f ${PTB3}`" TREEBANK_3
python3 -m msp2.scripts.srl_pb.prep05
# step 1.3: rename
for wset in train devel test.wsj test.brown; do ln -s conll05st-release/$wset.conll .; done
mv devel.conll dev.conll
# step 1.4: json
for f in *.conll; do
  python3 -m msp2.cli.change_format R.input_path:$f R.input_format:conll05 R.use_multiline:1 W.output_path:${f}.json
done
# --
cd ..

# step 2: conll12
echo "Step 2: prepare for conll12"
mkdir -p conll12; cd conll12;
# step 2.1: get data
wget https://github.com/ontonotes/conll-formatted-ontonotes-5.0/archive/v12.tar.gz
tar -zxvf v12.tar.gz
mv conll-formatted-ontonotes-5.0-12/* .
rmdir conll-formatted-ontonotes-5.0-12
# step 2.2: get script and convert to conll
#wget http://ontonotes.cemantix.org/download/conll-formatted-ontonotes-5.0-scripts.tar.gz
tar -zxvf ${ROOT_DIR}/conll-formatted-ontonotes-5.0-scripts.tar.gz
# need to use python2
ln -s "`readlink -f ${ONTO5}`" ontonotes-release-5.0
echo "Notice that we need to use python2 for this, please create a conda environment of 'py27'"
conda activate py27
bash conll-formatted-ontonotes-5.0/scripts/skeleton2conll.sh -D ontonotes-release-5.0/data/files/data/ conll-formatted-ontonotes-5.0/
conda deactivate
# step 2.3: simply concat them together
for wset in train development test conll-2012-test; do
  find conll-formatted-ontonotes-5.0/data/${wset} -name "*.gold_conll" -exec cat {} \; >${wset}.conll
done
ln -s development.conll dev.conll
# --
cd ..

# step 3: conll12b
echo "Step 3: prepare for conll12b (conll12 subset of onto5)"
mkdir -p conll12b; cd conll12b;
# step 3.1: get data
#wget http://conll.cemantix.org/2012/download/conll-2012-train.v4.tar.gz
#wget http://conll.cemantix.org/2012/download/conll-2012-development.v4.tar.gz
#wget http://conll.cemantix.org/2012/download/test/conll-2012-test-official.v9.tar.gz
for ff in ${ROOT_DIR}/{conll-2012-train.v4.tar.gz,conll-2012-development.v4.tar.gz,conll-2012-test-official.v9.tar.gz}; do tar -zxvf $ff; done
# step 3.2: get doc ids
find conll-2012/v4/data/train/data/english/ -name "*.v4_auto_skel" | sed "s/conll-2012\/v4\/data\/train\///;s/\.v4_auto_skel//" | sort >ids_train.txt
find conll-2012/v4/data/development/data/english/ -name "*.v4_auto_skel" | sed "s/conll-2012\/v4\/data\/development\///;s/\.v4_auto_skel//"  | sort >ids_dev.txt
find conll-2012/v9/data/test/data/english/ -name "*.v9_auto_skel" | sed "s/conll-2012\/v9\/data\/test\///;s/\.v9_auto_skel//" | sort >ids_test.txt
# step 3.3: concat all files
data_dir=../conll12/conll-formatted-ontonotes-5.0/data/
ln -s ${data_dir}/train data_train
ln -s ${data_dir}/development data_dev
ln -s ${data_dir}/conll-2012-test data_test
for wset in train dev test; do
while IFS= read -r line; do
  cat data_${wset}/${line}.gold_conll;
done <ids_${wset}.txt >${wset}.conll
done
# step 3.4: json
for f in *.conll; do
  python3 -m msp2.cli.change_format R.input_path:$f R.input_format:conll12 R.use_multiline:1 W.output_path:${f}.json "R.mtl_ignore_f:'ignore_#'"
done
# --
cd ..

# step 4: conll12d
echo "Step 4: prepare for conll12d (conll12 subset of ar/zh)"
mkdir -p conll12d; cd conll12d;
# 4.1: get data
#wget http://conll.cemantix.org/2012/download/conll-2012-train.v4.tar.gz
#wget http://conll.cemantix.org/2012/download/conll-2012-development.v4.tar.gz
#wget http://conll.cemantix.org/2012/download/test/conll-2012-test-key.tar.gz
#wget http://conll.cemantix.org/2012/download/conll-2012-scripts.v3.tar.gz
for ff in ${ROOT_DIR}/{conll-2012-train.v4.tar.gz,conll-2012-development.v4.tar.gz,conll-2012-test-key.tar.gz,conll-2012-scripts.v3.tar.gz}; do tar -zxvf $ff; done
# 4.2: convert
conda activate py27
bash conll-2012/v3/scripts/skeleton2conll.sh -D ../conll12/ontonotes-release-5.0/data/files/data/ conll-2012/
conda deactivate
# concat them together
for cl in arabic chinese; do
for wset in train development test; do
  find conll-2012/v4/data/${wset}/data/${cl}/ -name "*.v4_gold_conll" -exec cat {} \; >${cl}.${wset}.conll
done
done
# 4.3: some slight fixings
python3 -m msp2.scripts.srl_pb.tmp.fix_conll_data input:chinese.train.conll output:zh.train.conll do_fix_lemma_zh:1
python3 -m msp2.scripts.srl_pb.tmp.fix_conll_data input:chinese.development.conll output:zh.dev.conll do_fix_lemma_zh:1
python3 -m msp2.scripts.srl_pb.tmp.fix_conll_data input:chinese.test.conll output:zh.test.conll do_fix_lemma_zh:1
python3 -m msp2.scripts.srl_pb.tmp.fix_conll_data input:arabic.train.conll output:ar.train.conll do_fix_word_ar:1
python3 -m msp2.scripts.srl_pb.tmp.fix_conll_data input:arabic.development.conll output:ar.dev.conll do_fix_word_ar:1
python3 -m msp2.scripts.srl_pb.tmp.fix_conll_data input:arabic.test.conll output:ar.test.conll do_fix_word_ar:1
# 4.4: json
for f in {zh,ar}.{train,dev,test}.conll; do
  python3 -m msp2.cli.change_format R.input_path:$f R.input_format:conll12 R.use_multiline:1 W.output_path:${f}.json "R.mtl_ignore_f:'ignore_#'"
done
# --
cd ..

# --
# step 5: convert to UD
echo "Step 5: convert to UD"
# 5.1: first get corenlp
wget http://nlp.stanford.edu/software/stanford-corenlp-4.1.0.zip
unzip stanford-corenlp-4.1.0.zip
wget http://nlp.stanford.edu/software/stanford-corenlp-4.1.0-models-english.jar
wget https://raw.githubusercontent.com/stanfordnlp/CoreNLP/master/data/edu/stanford/nlp/upos/ENUniversalPOS.tsurgeon
mkdir -p edu/stanford/nlp/models/upos/
mv ENUniversalPOS.tsurgeon edu/stanford/nlp/models/upos/
jar cf stanford-parser-missing-file.jar edu/stanford/nlp/models/upos/ENUniversalPOS.tsurgeon
mv stanford-corenlp-4.1.0-models-english.jar stanford-parser-missing-file.jar stanford-corenlp-4.1.0
# 5.2: convert
for f in conll{05,12b}/*.conll.json; do
  python3 -m msp2.cli.annotate msp2.scripts.srl_pb.phrase2dep/p2d ann_batch_size:1000000 p2d_home:stanford-corenlp-4.1.0 input_path:$f output_path:${f%.json}.ud.json
done
# --

# --
# step 6: conll12c
echo "Step 6: prepare for conll12c (conll12 subset of en with domain splits)"
mkdir -p conll12c; cd conll12c;
# 6.1: link files
for wset in train dev test; do
  ln -s ../conll12b/${wset}.conll.ud.json ./en.${wset}.conll.ud.json
done
# 6.2: split
for wset in train dev test; do
  python3 -m msp2.scripts.srl_pb.tmp.split_domain input:./en.${wset}.conll.ud.json
done
# --
cd ..
