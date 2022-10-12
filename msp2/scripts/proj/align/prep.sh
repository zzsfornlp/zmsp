#

# prepare (tokenize and clean) parallel corpus

# --
# usage: bash preprocess.sh src trg file_prefix
SRC=$1
TRG=$2
FILE_PREFIX=$3
OUTPUT_PREFIX="${FILE_PREFIX}.${SRC}-${TRG}"
echo "Prepare corpus from $FILE_PREFIX.{$SRC,$TRG} to $OUTPUT_PREFIX.clean.{$SRC,$TRG}"

# step 0: get moses and setup paths
#git clone https://github.com/moses-smt/mosesdecoder.git
if [[ -z ${MOSES_HOME} ]]; then
  MOSES_HOME="../mosesdecoder/"
fi

# --
SCRIPTS=$MOSES_HOME/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

# step 1: tokenize and clean
for CL in $SRC $TRG; do
cat $FILE_PREFIX.$CL | perl $NORM_PUNC -l $CL | perl $TOKENIZER -threads 8 -a -no-escape -l $CL | sed 's/@-@/-/g' >$OUTPUT_PREFIX.tok.$CL
done
perl $CLEAN $OUTPUT_PREFIX.tok $SRC $TRG $OUTPUT_PREFIX.clean 1 100
# todo(+1): do we need truecase? currently nope.
wc $OUTPUT_PREFIX.*

# step2: combine into fastalign format
paste $OUTPUT_PREFIX.clean.$SRC $OUTPUT_PREFIX.clean.$TRG  | sed 's/ *\t */ ||| /g' >$OUTPUT_PREFIX.clean.fastalign
