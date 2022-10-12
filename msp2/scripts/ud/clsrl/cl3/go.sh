#

# prepare for clsrl3

# note: since fix random seed by default, sampling will be the same
for cl in en zh ca cs es; do
for wset in train dev test; do
  # ud + converted args
  python3 sample_shuffle.py input:../conll09/convert/${cl}.${wset}.ud.json output:./${cl}.${wset}.ud.json shuffle_times:1
  # ud + original args
  python3 sample_shuffle.py input:../conll09/${cl}.${wset}.udA.json output:./${cl}.${wset}.udA.json shuffle_times:1
  # original syntax & args
  python3 sample_shuffle.py input:../conll09/${cl}.${wset}.json output:./${cl}.${wset}.orig.json shuffle_times:1
done
done
## further prepare ewt for ud
for ff in en_ewt es_gsd es_ancora ca_ancora fr_gsd it_isdt pt_gsd pt_bosque; do
python3 -m msp2.cli.change_format "R.input_path:`readlink -f ../ud-treebanks-v2.7/UD_*/${ff}-ud-train.conllu`" R.input_format:conllu W.output_path:${ff}.train.json
done
# --
