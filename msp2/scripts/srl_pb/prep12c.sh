#

# prepare ar/zh from conll12 (en from "prep12b.sh"), also split domains!
# note: ar/zh moved to prep12d!!

# =====
# first get ar/zh from conll12
# https://conll.cemantix.org/2012/data.html

# get data
wget http://conll.cemantix.org/2012/download/conll-2012-train.v4.tar.gz
wget http://conll.cemantix.org/2012/download/conll-2012-development.v4.tar.gz
wget http://conll.cemantix.org/2012/download/test/conll-2012-test-official.v9.tar.gz
wget http://conll.cemantix.org/2012/download/test/conll-2012-test-supplementary.v9.tar.gz
wget http://conll.cemantix.org/2012/download/test/conll-2012-test-key.tar.gz
# get scripts (have to use the ones from conll12)
wget http://conll.cemantix.org/2012/download/conll-2012-scripts.v3.tar.gz
# decompress
for ff in *.gz; do tar -zxvf $ff; done
# convert to conll (note: need to use python2)
conda activate py27
bash conll-2012/v3/scripts/skeleton2conll.sh -D ../conll12/ontonotes-release-5.0/data/files/data/ conll-2012/
conda deactivate
# concat them together
for cl in arabic chinese english; do
for wset in train development; do  # note: currently no test!
  find conll-2012/v4/data/${wset}/data/${cl}/ -name "*.v4_gold_conll" -exec cat {} \; >${cl}.${wset}.conll
done
done

# =====
# note: setup the correct path to msp!!
#export PYTHONPATH=../src/

for wset in train dev test; do
ln -s ../conll12b/${wset}.conll.json ./en.${wset}.conll.json
for ud in ud ud2; do
  ln -s ../conll12b/${wset}.conll.${ud}.json ./en.${wset}.conll.${ud}.json
done; done

# =====
# split domains according to "doc_id"
for f in *.conll.json *.ud*.json; do
  python3 split_domain.py input:$f
done

# =====
# stat
#for f in *.conll.json *.ud*.json; do python3 frame_stat.py input_path:$f; done
#python3 frame_stat_res.py res.json ZZZ ar.train.*.json ar.*.train.*.json zh.train.*.json zh.*.train.*.json en.train.*.ud.json en.*.train.*.ud.json
