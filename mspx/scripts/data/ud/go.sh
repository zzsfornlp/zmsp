#

# read UD (conllu) files

wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4758/ud-treebanks-v2.10.tgz
tar -zxvf ud-treebanks-v2.10.tgz
mkdir -p data
for ccl in en_ewt it_isdt cs_pdt fi_tdt; do
  for wset in train dev test; do
    python3 -m mspx.cli.change_format "R.input_path:ud-treebanks-v2.10/UD_*/${ccl}-ud-${wset}.conllu" R.input_format:conllu W.output_path:data/${ccl}.${wset}.json
  done
done |& tee data/_log
rm -rf ud-treebanks-v2.10  # rm to save space

# down-sample cs to make it smaller
for wset in train dev test; do
  python3 -m mspx.scripts.tools.sample_shuffle input_path:data/cs_pdt.${wset}.json output_path:data/cs_pdt0.${wset}.json shuffle_times:1 rate:0.2
done
#sample_shuffle: ['data/cs_pdt.train.json'] => data/cs_pdt0.train.json [split=[]+1]: Counter({'inst_orig': 68495, 'inst_final': 13699})
#sample_shuffle: ['data/cs_pdt.dev.json'] => data/cs_pdt0.dev.json [split=[]+1]: Counter({'inst_orig': 9270, 'inst_final': 1854})
#sample_shuffle: ['data/cs_pdt.test.json'] => data/cs_pdt0.test.json [split=[]+1]: Counter({'inst_orig': 10148, 'inst_final': 2030})

# --
# read gum/amalgum and split
wget https://github.com/amir-zeldes/gum/archive/refs/tags/V6.2.0.tar.gz -O gumv6.tar.gz
wget https://github.com/gucorpling/amalgum/archive/refs/tags/v0.2.tar.gz -O amalgumv02.tar.gz
tar -zxvf gumv6.tar.gz
tar -zxvf amalgumv02.tar.gz
# read
mkdir -p _gum
for wset in train dev test; do
  python3 -m mspx.scripts.data.ud.read_conllu deplab_level:1 store_in_doc:0 input_path:ud-treebanks-v2.10/UD_English-EWT/en_ewt-ud-${wset}.conllu output_path:_gum/en_ewt.${wset}.json
done
for domain in academic bio fiction interview news voyage whow; do
  # gum
  python3 -m mspx.scripts.data.ud.read_conllu deplab_level:1 store_in_doc:1 input_path:gum-6.2.0/dep/GUM_${domain}_*.conllu output_path:_gum/gum.${domain}.all.json
  # amalgum
  python3 -m mspx.scripts.data.ud.read_conllu deplab_level:1 store_in_doc:0 no_tree:1 input_path:amalgum-0.2/amalgum/${domain}/dep/*.conllu output_path:_gum/amalgum.${domain}.all.json
done |& tee _log.gum0
#to _gum/gum.academic.all.json: Counter({'tok': 15110, 'sent': 575, 'file': 16, 'doc': 16})
#to _gum/amalgum.academic.all.json: Counter({'tok': 500063, 'doc': 20729, 'sent': 20729, 'file': 664})
#to _gum/gum.bio.all.json: Counter({'tok': 17951, 'sent': 776, 'file': 20, 'doc': 20})
#to _gum/amalgum.bio.all.json: Counter({'tok': 498310, 'doc': 23541, 'sent': 23541, 'file': 598})
#to _gum/gum.fiction.all.json: Counter({'tok': 16307, 'sent': 1029, 'file': 18, 'doc': 18})
#to _gum/amalgum.fiction.all.json: Counter({'tok': 500067, 'doc': 24267, 'sent': 24267, 'file': 457})
#to _gum/gum.interview.all.json: Counter({'tok': 18042, 'sent': 1070, 'file': 19, 'doc': 19})
#to _gum/amalgum.interview.all.json: Counter({'tok': 500316, 'doc': 28298, 'sent': 28298, 'file': 821})
#to _gum/gum.news.all.json: Counter({'tok': 14094, 'sent': 645, 'file': 21, 'doc': 21})
#to _gum/amalgum.news.all.json: Counter({'tok': 500799, 'doc': 26435, 'sent': 26435, 'file': 685})
#to _gum/gum.voyage.all.json: Counter({'tok': 14958, 'sent': 771, 'file': 17, 'doc': 17})
#to _gum/amalgum.voyage.all.json: Counter({'tok': 500493, 'doc': 33710, 'sent': 33710, 'file': 452})
#to _gum/gum.whow.all.json: Counter({'tok': 16923, 'sent': 1095, 'file': 19, 'doc': 19})
#to _gum/amalgum.whow.all.json: Counter({'tok': 500340, 'doc': 34566, 'sent': 34566, 'file': 611})
# --
# split / sample
for domain in academic bio fiction interview news voyage whow; do
  # gum
  python3 -m mspx.scripts.tools.sample_shuffle shuffle_times:1 split_sep:7,7,100 split_names:train,test,dev input_path:_gum/gum.${domain}.all.json output_path:_gum/gum.${domain}.json
  # amalgum
  python3 -m mspx.scripts.tools.sample_shuffle shuffle_times:1 rate:200000 use_inst:1 'sample_size_f:lambda x: sum(len(z) for z in x.sents)' input_path:_gum/amalgum.${domain}.all.json output_path:_gum/amalgum.${domain}.unlab.json
done |& tee _log.gum1
# for ff in _gum/*.json; do echo -n "$ff: "; python3 -m mspx.scripts.tools.count_stat input_path:$ff |& grep -o "Finish.*"; done |& tee _log.gum_stat

# --
# read more for transfer
for wset in train dev test; do
  python3 -m mspx.scripts.data.ud.read_conllu deplab_level:1 store_in_doc:0 input_path:ud-treebanks-v2.10/UD_*/en_ewt-ud-${wset}.conllu output_path:en1_ewt.${wset}.json
  python3 -m mspx.scripts.data.ud.read_conllu deplab_level:1 store_in_doc:0 input_path:ud-treebanks-v2.10/UD_*/en_atis-ud-${wset}.conllu output_path:en1_atis.${wset}.json
  python3 -m mspx.scripts.data.ud.read_conllu deplab_level:1 store_in_doc:0 input_path:ud-treebanks-v2.10/UD_*/fi_tdt-ud-${wset}.conllu output_path:fi1_tdt.${wset}.json
done |& tee _log1
#to en1_ewt.train.json: Counter({'tok': 204578, 'doc': 12543, 'sent': 12543, 'file': 1})
#to en1_atis.train.json: Counter({'tok': 48655, 'doc': 4274, 'sent': 4274, 'file': 1})
#to fi1_tdt.train.json: Counter({'tok': 162814, 'doc': 12217, 'sent': 12217, 'file': 1})
#to en1_ewt.dev.json: Counter({'tok': 25149, 'doc': 2001, 'sent': 2001, 'file': 1})
#to en1_atis.dev.json: Counter({'tok': 6644, 'doc': 572, 'sent': 572, 'file': 1})
#to fi1_tdt.dev.json: Counter({'tok': 18308, 'doc': 1364, 'sent': 1364, 'file': 1})
#to en1_ewt.test.json: Counter({'tok': 25094, 'doc': 2077, 'sent': 2077, 'file': 1})
#to en1_atis.test.json: Counter({'tok': 6580, 'doc': 586, 'sent': 586, 'file': 1})
#to fi1_tdt.test.json: Counter({'tok': 21070, 'doc': 1555, 'sent': 1555, 'file': 1})

# --
# tweebank v2
git clone https://github.com/Oneplus/Tweebank
mkdir -p _tweet
for wset in train dev test; do
  python3 -m mspx.scripts.data.ud.read_conllu deplab_level:1 store_in_doc:0 input_path:Tweebank/en*${wset}.conllu output_path:_tweet/en1_tweet.${wset}.json
done
# train: Counter({'tok': 24753, 'doc': 1639, 'sent': 1639, 'file': 1})
# dev: Counter({'tok': 11759, 'doc': 710, 'sent': 710, 'file': 1})
# test: Counter({'tok': 19095, 'doc': 1201, 'sent': 1201, 'file': 1})
