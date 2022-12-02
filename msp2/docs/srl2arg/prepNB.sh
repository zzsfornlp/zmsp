#

# prepare NomBank data

# step 0: prepare
echo "Please make sure you have set the env variables of (use ABSolute path!!):"
echo "'PYTHONPATH' should point to the root dir of the msp code repo."
echo "'PATH_CORENLP' should point to a CORENLP path for tree conversion, we are using: 'stanford-corenlp-4.2.0'"
echo "'PATH_PTB3' points to the PTB3 'TREEBANK_3' folder."
echo "Current settings are: $PYTHONPATH, $PATH_CORENLP, $PATH_PTB3"
read -p "Press any key to continue if they are all set well:" _TMP

# --
PATH_PTB3=$(readlink -f $PATH_PTB3)
# --

# step 1: get data
if [[ ! -d nombank.1.0 ]]; then
wget -nc https://nlp.cs.nyu.edu/meyers/nombank/nombank.1.0.tgz
tar -zxvf nombank.1.0.tgz
fi

# step 2: prepare them!
# convert
python3 -m msp2.tasks.zmtl3.scripts.nombank.prep_nb nombank.1.0 $PATH_PTB3 nb.all.json
# convert ud trees
python3 -m msp2.cli.annotate msp2.scripts.srl_pb.phrase2dep/p2d ann_batch_size:1000000 p2d_home:$PATH_CORENLP p2d_version:4.2.0 input_path:nb.all.json output_path:nb.all.ud.json
# split
python3 -m msp2.tasks.zmtl3.scripts.nombank.split_nb nb.all.ud.json _nb
for wset in train dev test; do
  mv _nb.${wset}.json nb.${wset}.ud.json
  python3 -m msp2.scripts.event.prep.sz_stat input_path:nb.${wset}.ud.json
done

# step 3: filter and map NB frames to PB
wget https://github.com/propbank/propbank-frames/archive/v3.1.tar.gz
tar -zxvf v3.1.tar.gz
python3 -m msp2.tasks.zmtl3.scripts.srl.s1_read_frames onto:pb output:f_pb.json dir:propbank-frames-3.1/frames/
python3 -m msp2.tasks.zmtl3.scripts.srl.s1_read_frames onto:nb output:f_nb.json dir:nombank.1.0/frames/
# map
python3 -m msp2.tasks.zmtl3.scripts.srl.s15_filter_nb onto_nb:f_nb.json onto_pb:f_pb.json input_files:nb.train.ud.json,nb.dev.ud.json,nb.test.ud.json "output_sub:nb,nb_f0" rm_reuse:0

# finished
ls -lh nb_f0.*.ud.json
echo "Finished, outputs are available at `pwd`/nb_f0.*.ud.json"
