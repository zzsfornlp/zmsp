#

# prepare ar/zh from conll12
# -- updated version upon "prep12c.sh"

# --
# first obtain data and skeleton2conll with "prep12c"
# ...
ln -s ../conll12c/conll-2012/ .

# concat them together
for cl in arabic chinese english; do
for wset in train development test; do  # note: currently no test!
  find conll-2012/v4/data/${wset}/data/${cl}/ -name "*.v4_gold_conll" -exec cat {} \; >${cl}.${wset}.conll
done
# nope, use v4-test from key files!!
#find conll-2012/v9/data/test/data/${cl}/ -name "*.v9_gold_parse_conll" -exec cat {} \; >${cl}.test.conll
done

# =====
# note: setup the correct path to msp!!
#export PYTHONPATH=../src/

# =====
# sightly fix for ar/zh and change_format
python3 fix_conll_data.py input:chinese.train.conll output:zh.train.conll do_fix_lemma_zh:1 do_discard_nonflat:1
python3 fix_conll_data.py input:chinese.development.conll output:zh.dev.conll do_fix_lemma_zh:1 do_discard_nonflat:1
python3 fix_conll_data.py input:chinese.test.conll output:zh.test.conll do_fix_lemma_zh:1 do_discard_nonflat:1
python3 fix_conll_data.py input:arabic.train.conll output:ar.train.conll do_fix_word_ar:1 do_discard_nonflat:1
python3 fix_conll_data.py input:arabic.development.conll output:ar.dev.conll do_fix_word_ar:1 do_discard_nonflat:1
python3 fix_conll_data.py input:arabic.test.conll output:ar.test.conll do_fix_word_ar:1 do_discard_nonflat:1
for f in {zh,ar}.{train,dev,test}.conll; do
  python3 -m msp2.cli.change_format R.input_path:$f R.input_format:conll12 R.use_multiline:1 W.output_path:${f}.json "R.mtl_ignore_f:'ignore_#'"
done

# parse them with stanza as ud2
for cl in ar zh; do
for wset in train dev test; do
CUDA_VISIBLE_DEVICES=1 python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:${cl} stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:2 input_path:${cl}.${wset}.conll.json output_path:${cl}.${wset}.conll.ud2.json
done; done

# get ud
# --
## first get from nyuad for ar
#for wset in train dev test; do
#python3 assign_anns.py input.input_path:ar.${wset}.conll.ud2.json aux.input_path:../patb/ar_nyuad-ud-all.merged.conllu output.output_path:ar.${wset}.conll.ud.json aux.input_format:conllu aux.use_multiline:1 "aux.mtl_ignore_f:'ignore_#'" delete_char_scheme:ar fuzzy_word_cnum:3 fuzzy_seq_wrate:0.5
#done |& tee _log_ar_convert
# see "ud/assign_p2d"
# --
# then use corenlp converter for zh; note: the labels may be strange and not in udv2 ...
for wset in train dev test; do
python3 -m msp2.cli.annotate msp2.scripts.srl_pb.phrase2dep/p2d ann_batch_size:1000000 p2d_home:../convert/stanford-corenlp-4.1.0 input_path:zh.${wset}.conll.ud2.json output_path:zh.${wset}.conll.ud.json p2d_lang:zh p2d_change_words:0 "p2d_upos_converter: lambda x: CTB2UPOS[x]" "p2d_udp_converter:lambda x: UD1to2[x.split(':')[0]]"
done |& tee _log_zh_convert

# compare?
for cl in ar zh; do
for wset in train dev test; do
python3 eval_arg_head.py gold.input_path:${cl}.${wset}.conll.ud.json pred.input_path:${cl}.${wset}.conll.ud2.json
done; done |& tee _log_eval_head
# note: very low UAS/LAS, maybe not good to use them for training, but head-arg matches are relatively high (>85%)!
# -> for ar, padt -> nyuad drops much, 95/87/83 -> 81/63/45; for zh, gsd-simp: 95/83/79
# note: discrepancies maybe because of 1) different seg criteria, 2) zh-converter not good enough

# =====
# split domains according to "doc_id"
for f in *.conll.json *.ud*.json; do
  python3 split_domain.py input:$f
done
