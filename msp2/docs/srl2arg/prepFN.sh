#

# prepare FrameNet data

# step 0: prepare
echo "Please make sure you have set the env variables of (use ABSolute path!!):"
echo "'PYTHONPATH' should point to the root dir of the msp code repo."
echo "'PATH_FN17' points to the FN data 'fndata-1.7' folder."
echo "Current settings are: $PYTHONPATH, $PATH_FN17"
read -p "Press any key to continue if they are all set well:" _TMP

# --
PATH_FN17=$(readlink -f $PATH_FN17)
# --

# step 1: read FN data
python3 -m msp2.scripts.srl_fn.fn_reader $PATH_FN17 fn17

# step 2: some more processings
for ff in fn17; do
# filter examplars
python3 -m msp2.scripts.srl_fn.fn_filter_exemplars ${ff}/fulltext.json ${ff}/exemplars.json ${ff}/exemplars.filtered.json
# split data
python3 -m msp2.scripts.srl_fn.fn_split_ft ${ff}/fulltext.json ${ff}/fulltext
done

# step?: pos, lemma and depparse (assume pre-tokenized)
mkdir -p parsed
for ff in fn17; do
for ff2 in train dev test test1; do
python3 -m msp2.cli.annotate 'stanza' input_path:${ff}/fulltext.${ff2}.json output_path:parsed/${ff}_fulltext.${ff2}.json stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized stanza_use_gpu:0
done
python3 -m msp2.cli.annotate 'stanza' input_path:${ff}/exemplars.filtered.json output_path:parsed/${ff}_exemplars.filtered.json stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:32 stanza_use_gpu:0
done

# finished
ls -lh parsed/fn17_*.json
echo "Finished, outputs are available at `pwd`/parsed/fn17_*.json"
