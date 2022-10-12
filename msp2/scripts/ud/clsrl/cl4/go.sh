#

# --
# first obtain turkish and latvian probbanks
for wset in train dev test; do
wget https://raw.githubusercontent.com/LUMII-AILab/FullStack/master/PropBank/data/lv-up-${wset}.conllu -O lv/lv-up-${wset}.conllu
done

# --
# convert from conll to zjson
for cl in tr lv; do
  for wset in train dev test; do
    for ff in ${cl}/${cl}*${wset}*.conllu; do
      if [[ ${cl} == "tr" ]]; then
        python3 -m msp2.cli.change_format R.input_path:$ff R.input_format:conlltrpb W.output_path:${cl}.orig.${wset}.json
      else
        python3 -m msp2.cli.change_format R.input_path:$ff R.input_format:conllup W.output_path:${cl}.orig0.${wset}.json
      fi
    done
  done
done
# --
# further merge sentences for lv
for wset in train dev test; do
  python3 merge_sents.py input_path:lv.orig0.${wset}.json output_path:lv.orig.${wset}.json
done
#Stat: Counter({'orig_arg': 27705, 'new_arg': 27705, 'orig_sent': 15503, 'orig_evt': 15503, 'new_evt': 15503, 'new_sent': 7778})
#Stat: Counter({'orig_arg': 4477, 'new_arg': 4477, 'orig_sent': 2518, 'orig_evt': 2518, 'new_evt': 2518, 'new_sent': 1259})
#Stat: Counter({'orig_arg': 3560, 'new_arg': 3560, 'orig_sent': 2034, 'orig_evt': 2034, 'new_evt': 2034, 'new_sent': 1012})
# --
## move tr-ud14 and lv-ud27 here!
#python3 -m msp2.cli.change_format R.input_path:../ud-treebanks-v1.4/UD_Turkish/tr-ud-train.conllu R.input_format:conllu W.output_path:tr.ud.train.json
#python3 -m msp2.cli.change_format R.input_path:../ud-treebanks-v2.7/UD_Latvian-LVTB/lv_lvtb-ud-train.conllu R.input_format:conllu W.output_path:lv.ud.train.json
# --
# shuffle
for cl in tr lv; do
  for wset in train dev test; do
    python3 sample_shuffle.py input:${cl}.orig.${wset}.json output:${cl}.${wset}.json shuffle_times:1
  done
done
# --
# finally put corresponding EWT here!
cp ../../pb/pb/ewt.*.conll .
for wset in train dev test; do
  python3 assign_anns.py input.input_path:ewt.${wset}.conll aux.input_path:../ud-treebanks-v1.4/UD_English/en-ud-${wset}.conllu output.output_path:_tmp1.json input.input_format:conllpb aux.input_format:conllu output_sent_and_discard_nonhit:1
  python3 span2dep.py _tmp1.json _tmp2.json
  python3 change_arg_label.py input_path:_tmp2.json output_path:en.ud1.${wset}.json
done |& tee _log.ewt.ud1  # hit for train/dev/test -> 0.9999/1.0000/0.9990
for wset in train dev test; do
  python3 assign_anns.py input.input_path:ewt.${wset}.conll aux.input_path:../ud-treebanks-v2.7/UD_English-EWT/en_ewt-ud-${wset}.conllu output.output_path:_tmp1.json input.input_format:conllpb aux.input_format:conllu output_sent_and_discard_nonhit:1
  python3 span2dep.py _tmp1.json _tmp2.json
  python3 change_arg_label.py input_path:_tmp2.json output_path:en.ud2.${wset}.json
done |& tee _log.ewt.ud2  # hit for train/dev/test -> 0.9921/0.9944/0.9981
# --
# stat
for ff in en.ud1.*.json en.ud2.*.json tr.{train,dev,test}.json lv.{train,dev,test}.json; do
  python3 stat_udsrl.py zjson $ff
done |& tee _log.stat.udsrl
