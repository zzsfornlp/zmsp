#

# prepare QA data

# step 0: prepare
echo "Please make sure you have set the env variables of (use ABSolute path!!):"
echo "'PYTHONPATH' should point to the root dir of the msp code repo."
echo "Current settings are: $PYTHONPATH"
read -p "Press any key to continue if they are all set well:" _TMP

# --
# squad v2.0
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa data_type:squad input_file:train-v2.0.json output_file:en.squad.train.json
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa data_type:squad input_file:dev-v2.0.json output_file:en.squad.dev.json

# --
# qamr
git clone https://github.com/uwnlp/qamr
for wset in train dev test; do
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa data_type:qamr extra_input_file:qamr/data/wiki-sentences.tsv input_file:qamr/data/filtered/${wset}.tsv output_file:en.qamr.${wset}.json
done

# --
# qasrl
wget http://qasrl.org/data/qasrl-v2_1.tar
tar -x -f ./qasrl-v2_1.tar
for wset in train dev test; do
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa data_type:qasrl input_file:./qasrl-v2_1/orig/${wset}.jsonl.gz output_file:en.qasrl.${wset}.json
done

# --
# qanom
function download_gdrive_zip_file {
    ggID=$1
    archive=$2
    ggURL='https://drive.google.com/uc?export=download'
    echo "Downloading ${archive}"
    filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"
    getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
    curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${archive}"
}
download_gdrive_zip_file "1_cTOy9isFo2qglAXETD2rgDTkhxC_OZr" "qanom_dataset.zip"
unzip "qanom_dataset.zip" -d "qanom_dataset"
for wset in train dev test; do
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa data_type:qanom input_file:./qanom_dataset/annot.${wset}.csv output_file:en.qanom.${wset}.json
done

# parse them all
for ff in en.*.{train,dev,test}.json; do
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:en stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:${ff} "output_path:${ff%.json}.ud2.json"
done

# some final refinements
for dset in squad qamr; do
for wset in train dev test; do
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa data_type:json do_refind:1 input_file:en.$dset.$wset.ud2.json output_file:en.${dset}R.$wset.ud2.json
done; done

# final concat
cat en.qasrl.{train,dev,test}.ud2.json >en.qasrl.all.ud2.json
cat en.qanom.{train,dev,test}.ud2.json >en.qanom.all.ud2.json
cat en.qamrR.{train,dev,test}.ud2.json >en.qamrR.all.ud2.json

# finished
ls -lh en.*.ud2.json
echo "Finished, outputs are available at `pwd`/en.*.ud2.json"
