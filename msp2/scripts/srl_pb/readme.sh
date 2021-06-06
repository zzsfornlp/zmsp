#

# prepare pb data

# =====
# CoNLL04:
#https://www.cs.upc.edu/~srlconll/st04/st04.html
#https://www.aclweb.org/anthology/W04-2412.pdf

# CoNLL05: see "prep05.py"
#https://www.cs.upc.edu/~srlconll/spec.html
#https://www.aclweb.org/anthology/W05-0620.pdf
#https://www.cs.upc.edu/~srlconll/conll05st-release.tar.gz
#https://www.cs.upc.edu/~srlconll/srlconll-1.1.tgz

# CoNLL12: see "prep12.sh"
#http://conll.cemantix.org/2012/
#http://cemantix.org/data/ontonotes.html

# PB-Release: see "prepP.sh"
#https://github.com/propbank/propbank-release

# =====
# convert everyone into json formats

# --
# pb_json.sh
export PYTHONPATH="`readlink -f ../../zsp2021/src/`:${PYTHONPATH}"
for f in conll05/*.conll; do
  python3 -m msp2.cli.change_format R.input_path:$f R.input_format:conll05 R.use_multiline:1 W.output_path:${f}.json
done
for f in conll12/*.conll conll12b/*.conll; do
  python3 -m msp2.cli.change_format R.input_path:$f R.input_format:conll12 R.use_multiline:1 W.output_path:${f}.json "R.mtl_ignore_f:'ignore_#'"
done
for f in pb/*.conll; do
  python3 -m msp2.cli.change_format R.input_path:$f R.input_format:conllpb R.use_multiline:1 W.output_path:${f}.json
done
# --
bash pb_json.sh |& tee _log_pb_json
cat _log_pb_json | grep -E "Ready to annotate|Annotate Finish"
# note: see "prep12c.sh" for more on pb12c!!

# =====

# =====
# convert phrase-tree to ud-dep (at pb/convert/)
# note: this slightly changes some tokens: like "1\/3" -> "1/3", and '-LRB-', ...
for f in ../*/*.conll.json; do
PYTHONPATH=../../../zsp2021/src/ python3 -m msp2.cli.annotate msp2.scripts.srl_pb.phrase2dep/p2d ann_batch_size:1000000 p2d_home:stanford-corenlp-4.1.0 input_path:$f output_path:${f%.json}.ud.json
done |& tee _log_convert
# get auto-parses with stanza
# note: can further use "eval_arg_head.py" to eval
for f in ../conll05/*.conll.ud.json ../conll12b/*.conll.ud.json; do
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=../../../zsp2021/src/ python3 -m msp2.cli.annotate 'stanza' stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:32 input_path:${f} output_path:${f%.ud.json}.ud2.json
done |& tee _log_stanza

# =====
# finally for conll12b (at conll12b)
# PYTHONPATH=../../../zsp2021/src/ python3 sample_insts.py ../conll12/train.conll.ud.json train2.conll.ud.json train.conll.ud.json 1.0
# PYTHONPATH=../../../zsp2021/src/ python3 sample_insts.py dev.conll.ud.json dev2.conll.ud.json '' 0.2
# -> logs
#Read dev.conll.ud.json, check , output dev2.conll.ud.json, stat:
#filter: defaultdict(<class 'int'>, {'sent': 9603, 'events': 23910})
#input: defaultdict(<class 'int'>, {'sent': 9603, 'events': 23910})
#sample: defaultdict(<class 'int'>, {'sent': 1945, 'events': 4762})
#Read ../conll12/train.conll.ud.json, check train.conll.ud.json, output train2.conll.ud.json, stat:
#check: defaultdict(<class 'int'>, {'sent': 75187, 'events': 188922})
#filter: defaultdict(<class 'int'>, {'sent': 39497, 'events': 63840})
#input: defaultdict(<class 'int'>, {'sent': 115812, 'events': 253070})
#sample: defaultdict(<class 'int'>, {'sent': 39497, 'events': 63840})
# --
