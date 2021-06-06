#

# again cl-pb12, but with some modifications (no do_discard_nonflat)
# -- updated version upon "prep12d.sh"
# note: actually only ar changes! no change for en/zh

# --
# first obtain data and skeleton2conll with "prep12c"
# ...
ln -s ../conll12c/conll-2012/ .

# concat them together
for cl in arabic chinese english; do
for wset in train development test; do  # note: currently no test!
  find conll-2012/v4/data/${wset}/data/${cl}/ -name "*.v4_gold_conll" -exec cat {} \; >${cl}.${wset}.conll
done
done

# link english ones
for wset in train dev dev2 test; do
ln -s ../conll12b/${wset}.conll.json en.${wset}.conll.json
ln -s ../conll12b/${wset}.conll.ud.json en.${wset}.conll.ud.json
ln -s ../conll12b/${wset}.conll.ud2.json en.${wset}.conll.ud2.json
done

# =====
# note: setup the correct path to msp!!
#export PYTHONPATH=../src/

# =====
# sightly fix for ar/zh and change_format (note: with updated get_f_args, should not discard anything!)
python3 fix_conll_data.py input:chinese.train.conll output:zh.train.conll do_fix_lemma_zh:1
python3 fix_conll_data.py input:chinese.development.conll output:zh.dev.conll do_fix_lemma_zh:1
python3 fix_conll_data.py input:chinese.test.conll output:zh.test.conll do_fix_lemma_zh:1
python3 fix_conll_data.py input:arabic.train.conll output:ar.train.conll do_fix_word_ar:1
python3 fix_conll_data.py input:arabic.development.conll output:ar.dev.conll do_fix_word_ar:1
python3 fix_conll_data.py input:arabic.test.conll output:ar.test.conll do_fix_word_ar:1
for f in {zh,ar}.{train,dev,test}.conll; do
  python3 -m msp2.cli.change_format R.input_path:$f R.input_format:conll12 R.use_multiline:1 W.output_path:${f}.json "R.mtl_ignore_f:'ignore_#'"
done

# --
# note: see "ud/assign_p2d" for later processings
