#

# prepare for clsrl2

export PYTHONPATH="`readlink -f ../../src/`:${PYTHONPATH}"

# simply cp cl2's en set and shuffle zr/zh ones
for wset in train dev test; do
  for uset in '' '3'; do
    cp ../pb12/en.${wset}.conll.ud${uset}.json en.${wset}.ud${uset}.json
    python3 sample_shuffle.py input:../pb12/ar.${wset}.conll.ud${uset}.json output:ar.${wset}.ud${uset}.json shuffle_times:1
    python3 sample_shuffle.py input:../pb12/zh.${wset}.conll.ud${uset}.json output:zh.${wset}.ud${uset}.json shuffle_times:1
  done
done

# extra one: check ar's texts
for wset in train dev test; do
  python3 check_pair_sents.py ../pb12/ar.${wset}.conll{.,.ud.}json 0 |& tee _logc.ar0.${wset}
  python3 check_pair_sents.py ../pb12/ar.${wset}.conll{.,.ud.}json 1 |& tee _logc.ar1.${wset}
done
# =>
#Counter({'tok': 242702, 'tok_mismatch': 57770, 'sent': 7422, 'sent_mismatch': 6868})  # train
#Counter({'tok': 242702, 'tok_mismatch': 44526, 'sent': 7422, 'sent_mismatch': 6720})  # strip_harakat
#Counter({'tok': 242702, 'tok_mismatch': 43994, 'sent': 7422, 'sent_mismatch': 6707})  # strip_tashkeel
#Counter({'tok': 28327, 'tok_mismatch': 6586, 'sent': 950, 'sent_mismatch': 904})  # dev
#Counter({'tok': 28327, 'tok_mismatch': 4891, 'sent': 950, 'sent_mismatch': 869})
#Counter({'tok': 28327, 'tok_mismatch': 4816, 'sent': 950, 'sent_mismatch': 863})
#Counter({'tok': 28371, 'tok_mismatch': 6780, 'sent': 1003, 'sent_mismatch': 880})  # test
#Counter({'tok': 28371, 'tok_mismatch': 5163, 'sent': 1003, 'sent_mismatch': 834})
#Counter({'tok': 28371, 'tok_mismatch': 5122, 'sent': 1003, 'sent_mismatch': 827})
# but seems ok if checking with translator!

# extra one: filter ar-nyuad
python3 filter_sents.py ../ud-treebanks-v2.7/UD_Arabic-NYUAD/ar_nyuad-ud-train.conllu ar_nyuad.train.conllu ar.dev.ud.json ar.test.ud.json |& tee _logf.ar
#e_key: 1923
#e_sent: 1953
#m_exclude: 1915
#m_sent: 15789
#m_survived: 13874
