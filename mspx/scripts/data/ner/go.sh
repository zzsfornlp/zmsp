#

# prepare NER data

# --
# conll 02/03
wget http://www.cnts.ua.ac.be/conll2002/ner.tgz -O ner_c02.tgz
tar -zxvf ner_c02.tgz
git clone https://github.com/UpasanaParashar/conll2003-tagger
git clone https://github.com/MaviccPRP/ger_ner_evals
# some pre-processing (make them into utf8)
mkdir -p data
for wset in train:train testa:dev testb:test; do
  IFS=: read -r p1 p2 <<< "$wset"
  cp conll2003-tagger/data/eng.${p1} data/en.${p2}.txt  # directly for EN
  iconv -f LATIN1 -t UTF-8 <ger_ner_evals/corpora/conll2003/deu.${p1} >data/de.${p2}.txt
  zcat ner/data/esp.${p1}.gz | iconv -f LATIN1 -t UTF-8 >data/es.${p2}.txt
  zcat ner/data/ned.${p1}.gz | iconv -f LATIN1 -t UTF-8 >data/nl.${p2}.txt
done
# convert
for f in data/*.txt; do
  python3 -m mspx.scripts.data.ner.prep_conll input_path:$f output_path:${f%.txt}.json
done |& tee data/_log
# de.dev.json: ({'tok': 51444, 'ef': 4833, 'sent': 2867, 'ef_PER': 1401, 'ef_ORG': 1241, 'ef_LOC': 1181, 'ef_MISC': 1010})
# de.test.json: ({'tok': 51943, 'ef': 3673, 'sent': 3005, 'ef_PER': 1195, 'ef_LOC': 1035, 'ef_ORG': 773, 'ef_MISC': 670})
# de.train.json: ({'tok': 206931, 'sent': 12152, 'ef': 11851, 'ef_LOC': 4363, 'ef_PER': 2773, 'ef_ORG': 2427, 'ef_MISC': 2288})
# en.dev.json: ({'tok': 51362, 'ef': 5942, 'sent': 3250, 'ef_PER': 1842, 'ef_LOC': 1837, 'ef_ORG': 1341, 'ef_MISC': 922})
# en.test.json: ({'tok': 46435, 'ef': 5648, 'sent': 3453, 'ef_LOC': 1668, 'ef_ORG': 1661, 'ef_PER': 1617, 'ef_MISC': 702})
# en.train.json: ({'tok': 203621, 'ef': 23499, 'sent': 14041, 'ef_LOC': 7140, 'ef_PER': 6600, 'ef_ORG': 6321, 'ef_MISC': 3438})
# es.dev.json: ({'tok': 52923, 'ef': 4352, 'sent': 1915, 'ef_ORG': 1700, 'ef_PER': 1222, 'ef_LOC': 985, 'ef_MISC': 445,})
# es.test.json: ({'tok': 51533, 'ef': 3559, 'sent': 1517, 'ef_ORG': 1400, 'ef_LOC': 1084, 'ef_PER': 735, 'ef_MISC': 340})
# es.train.json: ({'tok': 264715, 'ef': 18798, 'sent': 8323, 'ef_ORG': 7390, 'ef_LOC': 4914, 'ef_PER': 4321, 'ef_MISC': 2173})
# nl.dev.json: ({'tok': 37687, 'sent': 2895, 'ef': 2616, 'ef_MISC': 748, 'ef_PER': 703, 'ef_ORG': 686, 'ef_LOC': 479,})
# nl.test.json: ({'tok': 68875, 'sent': 5195, 'ef': 3941, 'ef_MISC': 1187, 'ef_PER': 1098, 'ef_ORG': 882, 'ef_LOC': 774})
# nl.train.json: ({'tok': 202644, 'sent': 15806, 'ef': 13344, 'ef_PER': 4716, 'ef_MISC': 3338, 'ef_LOC': 3208, 'ef_ORG': 2082})

# --
# parse with stanza
for cl in en de nl es; do
python3 -m mspx.cli.annotate anns:stanza stanza_lang:$cl input_path:data/${cl}.train.json output_path:data/${cl}.train.ud2.json
done
# --

# --
# wikigold
wget https://raw.githubusercontent.com/juand-r/entity-recognition-datasets/master/data/wikigold/CONLL-format/data/wikigold.conll.txt
python3 -m mspx.scripts.data.ner.prep_conll input_path:wikigold.conll.txt output_path:wikigold.all.json
python3 -m mspx.cli.annotate anns:stanza stanza_lang:en input_path:wikigold.all.json output_path:wikigold.all.ud2.json
# Counter({'c_word': 39007, 'c_Fef': 3558, 'c_doc': 1696, 'c_sent': 1696, 'c_Aef': 0})
# split them
python3 -m mspx.scripts.tools.sample_shuffle shuffle_times:1 split_sep:1000,350,350 split_names:train,test,dev input_path:wikigold.all.ud2.json output_path:wg.ZZKEYZZ.ud2.json
# for ff in wg.*; do echo -n "$ff: "; python3 -m mspx.scripts.tools.count_stat input_path:$ff |& grep -o "Finish.*"; done
#dev: Finish: Counter({'c_word': 7913, 'c_Fef': 700, 'c_doc': 346, 'c_sent': 346, 'c_Aef': 0})
#test: Finish: Counter({'c_word': 7851, 'c_Fef': 709, 'c_doc': 350, 'c_sent': 350, 'c_Aef': 0})
#train: Finish: Counter({'c_word': 23243, 'c_Fef': 2149, 'c_doc': 1000, 'c_sent': 1000, 'c_Aef': 0})

# --
# BTC
git clone https://github.com/GateNLP/broad_twitter_corpus
mkdir -p _btc
for s in a b e f g h; do
  python3 -m mspx.scripts.data.ner.prep_conll input_path:broad_twitter_corpus/$s.conll output_path:_btc/$s.json
done
# rm -f _btc/btc*
cp _btc/f.json _btc/btc.test.json
tail -n 1000 _btc/h.json >_btc/btc.dev.json
cat _btc/{a,b,e,g}.json >_btc/btc.train.json
head -n 1000 _btc/h.json >>_btc/btc.train.json
# parse
for ff in _btc/btc.{train,dev,test}.json; do
python3 -m mspx.cli.annotate anns:stanza stanza_lang:en input_path:$ff output_path:${ff%.json}.ud2.json
done
#dev: Finish: Counter({'c_word': 15000, 'c_Fef': 1722, 'c_doc': 1000, 'c_sent': 1000, 'c_Aef': 0})
#test: Finish: Counter({'c_word': 35428, 'c_Fef': 4376, 'c_doc': 2001, 'c_sent': 2001, 'c_Aef': 0})
#train: Finish: Counter({'c_word': 99933, 'c_Fef': 8779, 'c_doc': 6338, 'c_sent': 6338, 'c_Aef': 0})
# --
# random split to avoid further inner cross-domain
cat _btc/btc.{train,dev,test}.ud2.json >_btc/btc2.all.ud2.json
python3 -m mspx.scripts.tools.sample_shuffle shuffle_times:1 split_sep:2000,2000,10000 split_names:dev,test,train input_path:_btc/btc2.all.ud2.json output_path:_btc/btc2.ZZKEYZZ.ud2.json
#dev: Finish: Counter({'c_word': 32076, 'c_Fef': 3152, 'c_doc': 2000, 'c_sent': 2000, 'c_Aef': 0})
#test: Finish: Counter({'c_word': 31738, 'c_Fef': 3245, 'c_doc': 2000, 'c_sent': 2000, 'c_Aef': 0})
#train: Finish: Counter({'c_word': 86547, 'c_Fef': 8480, 'c_doc': 5339, 'c_sent': 5339, 'c_Aef': 0})
