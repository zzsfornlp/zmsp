#

# prepare NLU data

# --
# read atis data
git clone https://github.com/howl-anderson/ATIS_dataset
python3 -m mspx.scripts.data.nlu.read_nlu input_path:ATIS_dataset/data/standard_format/rasa/train.json output_path:atis.train0.json
python3 -m mspx.scripts.data.nlu.read_nlu input_path:ATIS_dataset/data/standard_format/rasa/test.json output_path:atis.test.json
# split 500 for dev
python3 -m mspx.scripts.tools.sample_shuffle shuffle_times:1 split_sep:500,10000 split_names:dev,train input_path:atis.train0.json output_path:atis.ZZKEYZZ.json
# for ff in atis.*; do echo -n "$ff: "; python3 -m mspx.scripts.tools.count_stat input_path:$ff |& grep -o "Finish.*"; done
#dev: Finish: Counter({'c_word': 5613, 'c_Fef': 1633, 'c_doc': 500, 'c_sent': 500, 'c_Aef': 0})
#test: Finish: Counter({'c_word': 9198, 'c_Fef': 2837, 'c_doc': 893, 'c_sent': 893, 'c_Aef': 0})
#train0: Finish: Counter({'c_word': 56591, 'c_Fef': 16560, 'c_doc': 4978, 'c_sent': 4978, 'c_Aef': 0})
#train: Finish: Counter({'c_word': 50978, 'c_Fef': 14927, 'c_doc': 4478, 'c_sent': 4478, 'c_Aef': 0})

# --
# read snips
git clone https://github.com/MiuLab/SlotGated-SLU  # for splitting
git clone https://github.com/sonos/nlu-benchmark
# note: need to delete two invalid utf8 characters in "train_PlayMusic_full.json"
python3 -m mspx.scripts.data.nlu.read_nlu data:snips input_path:nlu-benchmark/2017-06-custom-intent-engines/*/train_*_full.json filter_data:SlotGated-SLU/data/snips/test/seq.in output_path:snips.test.json
python3 -m mspx.scripts.data.nlu.read_nlu data:snips input_path:nlu-benchmark/2017-06-custom-intent-engines/*/train_*_full.json filter_data:SlotGated-SLU/data/snips/train/seq.in output_path:snips.train.json
python3 -m mspx.scripts.data.nlu.read_nlu data:snips input_path:nlu-benchmark/2017-06-custom-intent-engines/*/validate_*.json filter_data:SlotGated-SLU/data/snips/valid/seq.in output_path:snips.dev.json

# --
# parse
for ff in {atis,snips}.{train,dev,test}.json; do
python3 -m mspx.cli.annotate anns:stanza stanza_lang:en input_path:$ff output_path:${ff%.json}.ud2.json stanza_dpar_level:1
done

# --
# add UD for atis
cat ../../ud/data/en1_atis.*.json >en1_atis.all.ud.json
cat atis.{train,dev,test}.ud2.json >en1_atis.all.ud2.json  # backoff
for wset in train dev test; do
  python3 -m mspx.scripts.data.nlu.assign_ud input_path:atis.${wset}.json output_path:atis.${wset}.ud.json ud_paths:en1_atis.all.ud.json,en1_atis.all.ud2.json
done |& tee _log_ud
# train: 4447+31=4478, dev: 495+5=500, test: 877+16=893
