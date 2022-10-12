#

# download and extract wiki data

# tool
#git clone https://github.com/attardi/wikiextractor

# CUR_LANG?
if [[ -z ${CUR_LANG} ]]; then
  CUR_LANG="en"
fi

# download the data (be explicit about time)
#wget -nc "https://dumps.wikimedia.org/${CUR_LANG}wiki/latest/${CUR_LANG}wiki-latest-pages-articles.xml.bz2"
wget -nc https://dumps.wikimedia.org/${CUR_LANG}wiki/20210220/${CUR_LANG}wiki-20210220-pages-articles.xml.bz2

# extract
mkdir -p extracted
PYTHONPATH=./wikiextractor/ python3 -m wikiextractor.WikiExtractor ${CUR_LANG}wiki-20210220-pages-articles.xml.bz2 --json -b 200M -o extracted --no-template --processes 8 |& tee extracted/log

# convert
mkdir -p raw
python3 run_para.py -i ./extracted/AA/wiki_* -c "python3 wiki2raw.py [IN] [OUT]" -o "lambda x: 'raw/'+basename(x)"

# tokenizer (corenlp seems fast enough!)
mkdir -p tokenized
python3 run_para.py -n 1 -i ./raw/wiki_* -c "OMP_NUM_THREADS=1 python3 raw2tok.py input:[IN] output:[OUT] lang:en corenlp_threads:8" -o "lambda x: 'tokenized/'+basename(x)+'.json'"

# stat & shuffle
mkdir -p shuffled
#python3 stat_shuffle.py input_prefix:tokenized/wiki output_prefix:shuffled/wiki |& tee shuffled/_log
python3 stat_shuffle.py do_unescape:1 do_shorten_http:1 input_prefix:tokenized/wiki output_prefix:shuffled/wiki |& tee shuffled/_log
