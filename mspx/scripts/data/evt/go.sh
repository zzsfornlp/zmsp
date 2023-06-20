#

# prepare evt related data

# --
# prepare ACE05-E+ with oneie
echo "Please first download oneie-v0.4.8 from 'http://blender.cs.illinois.edu/software/oneie/' and decompress it!"
tar -zxvf oneie_v0.4.8.tar.gz
echo "Please provide input folder LDC_DIR for LDC2006T06"
if [[ -z ${LDC_DIR} ]]; then
  exit 1
fi
mkdir -p _en_ace
python3 oneie_v0.4.8/preprocessing/process_ace.py -i ${LDC_DIR}/ace_2005_td_v7/data/ -o _en_ace -s oneie_v0.4.8/resource/splits/ACE05-E -l english
# --
mkdir -p data
for wset in train dev test; do
  python3 -m mspx.scripts.data.evt.conv_oneie2mine _en_ace/${wset}.oneie.json data/en.ace05.${wset}.json
done |& tee data/_log.ace05

# --
# prepare ACE05-ar/zh with xgear
# -- pre-process; note: no need to prepare english ...
git clone https://github.com/PlusLabNLP/X-Gear
cd X-Gear/preprocessing
conda env create -f ../environment.yml
conda activate xgear
mkdir -p ../processed_data/ace05_zh_mT5
mv ~/stanza_resources/ ~/tmp_stanza/
if [[ -d ~/_stanza1.2/ ]]; then
  mv ~/_stanza1.2/ ~/stanza_resources/
else
  python -c "import stanza; stanza.download('en'); stanza.download('ar'); stanza.download('zh')"
fi
python src/process_ace.py -i ${LDC_DIR}/ace_2005_td_v7/data/ -o ../processed_data/ace05_zh_mT5 -s src/splits/ACE05-ZH -b google/mt5-large -w 1 -l chinese
mv ~/stanza_resources/ ~/_stanza1.2/
mv ~/tmp_stanza/ ~/stanza_resources/
conda deactivate
cd $OLDPWD
# -- convert
mkdir -p data
for cl in zh ar; do
for wset in train dev test; do
  python3 -m mspx.scripts.data.evt.conv_oneie2mine X-Gear/processed_data/ace05_${cl}_mT5/${wset}.json data/${cl}.ace05.${wset}.json
done; done |& tee data/_log.ace05arzh

# --
# convert to brat for checking
if false; then
mkdir -p brat
for ff in data/*.json; do
  python3 -m mspx.scripts.data.brat.json2brat $ff "brat/_$(basename $ff)/"
done
fi

# --
# prepare for sent level
mkdir -p dataS
for cl in en zh ar; do
for wset in train dev test; do
  python3 -m mspx.scripts.data.evt.conv_oneie2mine *_${cl}_*/${wset}*.json dataS/${cl}.ace05.${wset}.json '0'
done; done |& tee dataS/_log.ace05arzh

# --
# filter sents
for cl in en zh ar; do
python3 -m mspx.scripts.data.evt.filter_data input_path:dataS/${cl}.ace05.train.json min_len:3 min_alpha_ratio:0.5 output_path:dataS/${cl}.ace05f.train.json
done
# 'sum((any(str.isalpha(c) for c in t)) for t in d[0].seq_word.vals) / len(d[0]) < 0.5 or len(d[0]) <= 2'
#Process dataS/en.ace05.train.json -> dataS/en.ace05f.train.json: Counter({'frame': 51973, 'frameK': 48695, 'doc': 19204, 'sent': 19204, 'sentV': 14193, 'docV': 14193, 'sentD': 5011, 'docD': 5011, 'frameD': 3278})
#Process dataS/zh.ace05.train.json -> dataS/zh.ace05f.train.json: Counter({'frame': 32607, 'frameK': 32514, 'doc': 6305, 'sent': 6305, 'sentV': 5742, 'docV': 5742, 'sentD': 563, 'docD': 563, 'frameD': 93})
#Process dataS/ar.ace05.train.json -> dataS/ar.ace05f.train.json: Counter({'frame': 17886, 'frameK': 17796, 'doc': 3218, 'sent': 3218, 'sentV': 2111, 'docV': 2111, 'sentD': 1107, 'docD': 1107, 'frameD': 90})

# --
# parse with stanza
for cl in en zh ar; do
python3 -m mspx.cli.annotate anns:stanza stanza_lang:$cl input_path:dataS/${cl}.ace05f.train.json output_path:dataS/${cl}.ace05f.train.ud2.json
done
# --
