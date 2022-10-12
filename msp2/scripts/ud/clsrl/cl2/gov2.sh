#

# prepare for clsrl2 (v2)

export PYTHONPATH="`readlink -f ../../src/`:${PYTHONPATH}"

# simply cp cl2's en set and shuffle zr/zh ones
for wset in train dev test; do
  for uset in '' '3'; do
    python3 sample_shuffle.py input:../pb12/en.${wset}.conll.ud${uset}.json output:en.${wset}.ud${uset}.json shuffle_times:1
    python3 sample_shuffle.py input:../pb12/ar.${wset}.conll.ud${uset}.json output:ar.${wset}.ud${uset}.json shuffle_times:1
    python3 sample_shuffle.py input:../pb12/zh.${wset}.conll.ud${uset}.json output:zh.${wset}.ud${uset}.json shuffle_times:1
    # use zh2 to overwrite ud ones!
    python3 sample_shuffle.py input:../pb12/zh2.${wset}.conll.ud${uset}.json output:zh.${wset}.ud${uset}.json shuffle_times:1
  done
done

# extra one: filter ar-nyuad
cat ../ud-treebanks-v2.7/UD_Arabic-NYUAD/ar_nyuad-ud-train.conllu >_ar.conllu
python3 filter_sents.py _ar.conllu ar_nyuad.train0.conllu ar.dev.ud.json ar.test.ud.json |& tee _logf.ar0
#e_key: 1923
#e_sent: 1953
#m_exclude: 1915
#m_sent: 15789
#m_survived: 13874
python3 filter_sents.py _ar.conllu ar_nyuad.train1.conllu ar.train.ud.json ar.dev.ud.json ar.test.ud.json |& tee _logf.ar1
#e_key: 8906
#e_sent: 9375
#m_exclude: 7995
#m_sent: 15789
#m_survived: 7794

# convert UD trees here (change to json)
python3 -m msp2.cli.change_format R.input_path:../ud-treebanks-v2.7/UD_English-EWT/en_ewt-ud-train.conllu R.input_format:conllu W.output_path:en_ewt.train.json
python3 -m msp2.cli.change_format R.input_path:../ud-treebanks-v2.7/UD_Arabic-PADT/ar_padt-ud-train.conllu R.input_format:conllu W.output_path:ar_padt.train.json
python3 -m msp2.cli.change_format R.input_path:../ud-treebanks-v2.7/UD_Chinese-GSDSimp/zh_gsdsimp-ud-train.conllu R.input_format:conllu W.output_path:zh_gsdsimp.train.json
python3 -m msp2.cli.change_format R.input_path:ar_nyuad.train0.conllu R.input_format:conllu W.output_path:ar_nyuad.train0.json
python3 -m msp2.cli.change_format R.input_path:ar_nyuad.train1.conllu R.input_format:conllu W.output_path:ar_nyuad.train1.json
