#

# prepare PropBank data

# step 0: prepare
echo "Please make sure you have set the env variables of (use ABSolute path!!):"
echo "'PYTHONPATH' should point to the root dir of the msp code repo."
echo "'PATH_CORENLP' should point to a CORENLP path for tree conversion, we are using: 'stanford-corenlp-4.1.0'"
echo "'PATH_ONTO' points to the 'ontonotes-release-5.0' folder."
echo "'PATH_EWT' points to the 'eng_web_tbk' folder."
echo "You should also prepare a python2.7 conda env called 'py27'!!"
echo "Current settings are: $PYTHONPATH, $PATH_CORENLP, $PATH_ONTO, $PATH_EWT"
read -p "Press any key to continue if they are all set well:" _TMP

# --
PATH_CORENLP=$(readlink -f $PATH_CORENLP)
PATH_ONTO=$(readlink -f $PATH_ONTO)
PATH_EWT=$(readlink -f $PATH_EWT)
# --

# step 1: get PB data
git clone https://github.com/propbank/propbank-release/
cd propbank-release; git checkout a9accf68e210eb54dba7f9549b66e1d6ad5cc807; cd ..
# note: in our experiments, we are using the version of (20.10.03): a9accf68e210eb54dba7f9549b66e1d6ad5cc807

# step 2: run script
source "$(dirname `which conda`)/../etc/profile.d/conda.sh"
conda activate py27
cd propbank-release/docs/scripts
python map_all_to_conll.py --ontonotes $PATH_ONTO/ --ewt $PATH_EWT/
cd ../../../
conda deactivate
conda activate p21

# step 3: concat the datasets
python3 -m msp2.scripts.srl_pb.prepP_cat

# step 4: final convert (also for converting ud tree)
# change to our json format
for f in *.conll; do
  python3 -m msp2.cli.change_format R.input_path:$f R.input_format:conllpb R.use_multiline:1 W.output_path:${f}.json
done
# convert UD tree
for f in *.conll.json; do
python3 -m msp2.cli.annotate msp2.scripts.srl_pb.phrase2dep/p2d ann_batch_size:1000000 p2d_home:$PATH_CORENLP input_path:$f output_path:${f%.json}.ud.json
done
# merge sents into docs
mkdir -p pb_sents
mv *.conll.ud.json pb_sents
for ff in pb_sents/*.conll.ud.json; do
  python3 -m msp2.tasks.zmtl3.scripts.srl.s4_pbs2d $ff "$(basename $ff)"
done

# finished
ls -lh *.conll.ud.json
echo "Finished, outputs are available at `pwd`/*.conll.ud.json"
