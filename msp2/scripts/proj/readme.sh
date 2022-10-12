#

# tools for doing annotation projection (translation)

# --
#export PYTHONPATH=??

# --
# step 0:
# repos to prepare
git clone https://github.com/clab/fast_align
# note: also need to compile fast_align
git clone https://github.com/moses-smt/mosesdecoder

# --
# step 1: prepare parallel data
# see align/readme.sh

# --
# step 2: prepare src and translations and (tokenize)
cat ../../cl1/en.ewt.{train,dev,test}.json >en.ewt.json
python3 -m msp2.cli.change_format R.input_path:en.ewt.json W.output_path:en.ewt.txt W.output_format:plain_sent
# then translate to "_trans.en-${cl}.raw.txt" (untokenized!)
# -> note: here use external tools for the translation!
# then tokenize
SCRIPTS=../mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
for cl in de fr it es pt 'fi'; do
  cat _trans.en-${cl}.raw.txt | perl $NORM_PUNC -l $cl | perl $TOKENIZER -threads 8 -a -no-escape -l $cl | sed 's/@-@/-/g' >_trans.en-${cl}.tok.txt
done

# --
# step 3: align
export OMP_NUM_THREADS=8
for cl in de fr it es pt 'fi'; do
  echo "START ${cl} [$(date)]"
  OUT_DIR=_align_en_${cl} bash align.sh ../europarlv7/europarl-v7.${cl}-en.en-${cl}.clean.fastalign en.ewt.txt _trans.en-${cl}.tok.txt
  echo "END ${cl} [$(date)]"
done |& tee _log.align

# --
# step 4: do projection
for cl in de fr it es pt 'fi'; do
  python3 proj_anns.py input_path:en.ewt.cl0.json output_path:${cl}.ewtT.cl0.json src_txt:en.ewt.txt trg_txt:_trans.en-${cl}.tok.txt align_txt:_align_en_${cl}/_output.align fa_prefix:_align_en_${cl}/fwd_
done |& tee _log.cl0_proj
for cl in 'fi'; do
  python3 proj_anns.py input_path:en.ewt.cl1.json output_path:${cl}.ewtT.cl1.json src_txt:en.ewt.txt trg_txt:_trans.en-${cl}.tok.txt align_txt:_align_en_${cl}/_output.align fa_prefix:_align_en_${cl}/fwd_
done |& tee _log.cl1_proj
