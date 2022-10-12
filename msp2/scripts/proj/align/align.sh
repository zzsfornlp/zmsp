#

# align files

# usage: OUT_DIR=?? bash align.sh [LARGE.fastalign] SRC TRG

# read args and prepare
base_file=$(readlink -f $1)
src_file=$(readlink -f $2)
trg_file=$(readlink -f $3)
if [[ -z ${FA_BIN} ]]; then
  FA_BIN="../fast_align/build/"
fi
FA_BIN_ABSPATH=$(readlink -f ${FA_BIN})
if [[ -z ${OUT_DIR} ]]; then
  OUT_DIR="_tmp"
fi
mkdir -p ${OUT_DIR}
cd ${OUT_DIR}

# align
cp ${base_file} _align.train
paste ${src_file} ${trg_file}  | sed 's/ *\t */ ||| /g' >>_align.train

${FA_BIN_ABSPATH}/fast_align -i _align.train -d -o -v -p fwd_params >_forward.align 2>fwd_err
${FA_BIN_ABSPATH}/fast_align -i _align.train -d -o -v -r -p rev_params >_reverse.align 2>rev_err
${FA_BIN_ABSPATH}/atools -i _forward.align -j _reverse.align -c grow-diag-final-and >_both.align

tail -n "$(wc -l $src_file | cut -d ' ' -f 1)" _both.align >_output.align
