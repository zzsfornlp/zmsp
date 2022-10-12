#

# --
# note:
# for pb/fn, simply use previous ones:
# asmara:{pb,fn} -> windheok:pbfn/ (pb with onto/ewt+ud, fn with ud2/ud3)

# for nb, see here:

# --
# get data
wget https://nlp.cs.nyu.edu/meyers/nombank/nombank.1.0.tgz
tar -zxvf nombank.1.0.tgz
#ln -s ../../pb/conll05/TREEBANK_3/ .

# --
# prepare
python3 -m msp2.tasks.zmtl3.scripts.nombank.prep_nb nombank.1.0 TREEBANK_3 nb.all.json |& tee _log_read
python3 -m msp2.cli.annotate msp2.scripts.srl_pb.phrase2dep/p2d ann_batch_size:1000000 p2d_home:_corenlp p2d_version:4.2.0 input_path:nb.all.json output_path:nb.all.ud.json |& tee _log_p2d
python3 -m msp2.tasks.zmtl3.scripts.nombank.split_nb nb.all.ud.json _nb
for wset in train dev test; do
  mv _nb.${wset}.json nb.${wset}.ud.json
  python3 -m msp2.scripts.event.prep.sz_stat input_path:nb.${wset}.ud.json
done |& tee _log_stat
cp nb.*.ud.json ../pbfn/

# --
# read nb frames
# --
grep 'roleset.*source' nombank.1.0/frames/*.xml | wc
# 3494   21548  395070
ls nombank.1.0/frames/*.xml | wc
# 4705    4705  149322
python3 -m msp2.scripts.ud.frames.read_frames onto:pb dir:./nombank.1.0/frames/ save_txt:frames.nb.txt save_pkl:frames.nb.pkl
# Read from ./nombank.1.0/frames/: 4705 files and 5577 frames!
# note: there are some "sources" not necessarily the same as the noun-frame, but let's just go with it and see ...
python3 -m msp2.tasks.zmtl3.scripts.nombank.get_nb_map frames.nb.pkl map.nb.json
#frame: 5577 || frame_hit: 3486 || frame_miss: 2091
