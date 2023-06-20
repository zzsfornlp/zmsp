#

# prepare ontonotes data

# --
set -e
ROOT_DIR=`pwd`
# --

# step 0: prepare
echo "Please make sure you have set the env variables of (use ABSolute path!!): PYTHONPATH"
echo "'PYTHONPATH' should point to the root dir of the msp code repo."
echo "'PATH_ONTO5' should point to OntoNotes5.0 of 'ontonotes-release-5.0'."
echo "Please prepare a conda environment of py27 which is python 2.7"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate py27
conda deactivate
echo "Current settings are: $PYTHONPATH, $PATH_ONTO5"
read -p "Press any key to continue if this is prepared:" _TMP

# --
PATH_ONTO5=$(readlink -f $PATH_ONTO5)
# --

# step 1: obtain data & prepare
# get data
wget https://github.com/ontonotes/conll-formatted-ontonotes-5.0/archive/v12.tar.gz
tar -zxvf v12.tar.gz
mv conll-formatted-ontonotes-5.0-12/* .
rmdir conll-formatted-ontonotes-5.0-12
# get script and convert to conll
wget http://ontonotes.cemantix.org/download/conll-formatted-ontonotes-5.0-scripts.tar.gz
tar -zxvf conll-formatted-ontonotes-5.0-scripts.tar.gz
conda activate py27
bash conll-formatted-ontonotes-5.0/scripts/skeleton2conll.sh -D ${PATH_ONTO5}/data/files/data/ conll-formatted-ontonotes-5.0/
conda deactivate
# simply concat them together
for conf in train:train development:dev test:test conll-2012-test:test12; do
  IFS=: read -r wset0 wset <<< $conf
  find conll-formatted-ontonotes-5.0/data/${wset0} -name "*.gold_conll" -exec cat {} \; >${wset}.conll
done

# step 2: convert from conll to json
# -> convert
for wset in train dev test test12; do
  python3 -m mspx.scripts.data.onto.prep_onto input_path:${wset}.conll output_path:onto.${wset}.json add_syntax:1 add_ner:1
done
#Finish reading-onto: {'coref_m': 155560, 'doc': 115812, 'file': 1, 'ner': 128738, 'sent': 115812, 'srl_arg': 852053, 'srl_frame': 253070, 'word': 2200865}
#Finish reading-onto: {'coref_m': 19156, 'doc': 15680, 'file': 1, 'ner': 20354, 'sent': 15680, 'srl_arg': 118659, 'srl_frame': 35297, 'word': 304701}
#Finish reading-onto: {'coref_m': 19764, 'doc': 12217, 'file': 1, 'ner': 12586, 'sent': 12217, 'srl_arg': 88431, 'srl_frame': 26715, 'word': 230118}
#Finish reading-onto: {'coref_m': 19764, 'doc': 9479, 'file': 1, 'ner': 11257, 'sent': 9479, 'srl_arg': 80242, 'srl_frame': 24462, 'word': 169579}
# -> convert UD tree
for wset in train dev test test12; do
  python3 -m mspx.scripts.data.onto.phrase2dep input_path:onto.${wset}.json output_path:onto.all.${wset}.ud.json
done
# -> split domain
for wset in train dev test test12; do
  python3 -m mspx.scripts.tools.split_domain input_path:onto.all.${wset}.ud.json 'domain_key_f:x.sents[0].info["doc_id"].split("/")[0]' output_path:onto.ZZKEYZZ.${wset}.ud.json
done
# --
