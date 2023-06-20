#

# --
# prepare CNER data

# --
# cner
git clone https://github.com/zliucr/CrossNER
mkdir -p _orig
for spec in CrossNER/ner_data/*; do
  domain=`basename $spec`
  for wset in train dev test; do
    python3 -m mspx.scripts.data.ner.prep_conll input_path:$spec/${wset}.txt output_path:_orig/cner.${domain}.${wset}.json
  done
  cat _orig/cner.${domain}.{train,dev,test}.json >_orig/cner.${domain}.all.json
done
rm -f _orig/cner.conll2003.*  # no need this!

# --
# predict with conll models on them and decide the mappings
#mkdir -p _pred
#for ff in _orig/cner.*.all.json; do
#  ff2=`basename $ff`
#  python3 -m mspx.tasks.zext.main fs:test allow_voc_unk:1 conf_sbase:bert_name:roberta-base vocab_load_dir:__vocabs/ner_en/ test0.group_files:$ff test0.output_file:_pred/$ff2 test0.tasks:ext0 model_load_name:zmodel.*.m
#done
#for ff in _orig/cner.*.all.json; do
#  ff2=`basename $ff`
#echo group fl "\"'N' if d.gold is None else d.gold.label, 'N' if d.pred is None else d.pred.label\"" | python3 -m mspx.cli.analyze frame frame_cate:ef gold:$ff preds:_preds/$ff2 auto_save_name:
#  python3 -m mspx.scripts.data.ner.map_label $ff '' cner_T
#done
# PER: researcher, person, writer, musicalartist, politician, scientist
# ORG: organisation, university, band, politicalparty
# LOC: country, location
# MISC: misc

# --
# parse
mkdir -p _parsed
for ff in _orig/cner.*.*.json; do
  ff2=`basename $ff`
  python3 -m mspx.cli.annotate anns:stanza stanza_lang:en input_path:$ff output_path:_parsed/${ff2%.json}.ud2.json
done

# --
# finally change label
mkdir -p _mapped
for ff in _parsed/cner.*.json; do
  ff2=`basename $ff`
  python3 -m mspx.scripts.data.ner.map_label $ff _mapped/$ff2 cner_T
done
# --

# --
# prepare unlabeled data
function download_gdrive_zip_file {
    ggID=$1
    archive=$2
    ggURL='https://drive.google.com/uc?export=download'
    echo "Downloading ${archive}"
    filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"
    getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
    curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${archive}"
}
mkdir -p _unlabel
download_gdrive_zip_file "1VfFUap_0can-cuXADbl3eUJ2KwBjKvXr" _unlabel/politics.txt
download_gdrive_zip_file "1oMkK8z9ajcQMZMrCxyx54jI3OnqmmtUM" _unlabel/science.txt
download_gdrive_zip_file "13Bm_bQ8bLQIc9tOkPkeKv1wobcMrCPmu" _unlabel/music.txt
download_gdrive_zip_file "1c9jfUA_Vz_izS8_8Svojb8OxbEAiwhpI" _unlabel/literature.txt
download_gdrive_zip_file "1PhCVyMoSONRlarkI4g0plY7lPet3wEo0" _unlabel/ai.txt
# down-sample (200K tokens) and convert
for ff in _unlabel/*.txt; do
  domain=`basename ${ff%.txt}`
  python3 -m mspx.scripts.tools.sample_shuffle shuffle_times:1 rate:200000 'sample_size_f:lambda x: len(x.split())' input_path:$ff output_path:_tmp.txt
  wc _tmp.txt
  python3 -m mspx.cli.change_format R.input_path:_tmp.txt R.input_format:plain_sent W.output_path:_unlabel/cner.${domain}.unlab.json
  rm -f _tmp.txt
done
# --
