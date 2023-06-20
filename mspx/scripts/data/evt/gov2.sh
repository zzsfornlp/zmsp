#

# processing English-ACE05 data

# --
set -e
ROOT_DIR=`pwd`
#PYTHONPATH=?? ONEIE=?? LDC=?? bash gov2.sh
# --

# step 0: prepare
echo "Please make sure you have set the env variables of (use ABSolute path!!):"
echo "'PYTHONPATH' should point to the root dir of the msp code repo."
echo "'ONEIE' points to the 'oneie_v0.4.8.tar.gz' file."
echo "'LDC' points to the 'LDC2006T06' dir."
echo "Current settings are: $PYTHONPATH, $ONEIE, $LDC"
read -p "Press any key to continue if they are all set well:" _TMP

# step 1: prepare data using oneie
# first remake splits since two dev files seem to be missing
tar -zxvf $ONEIE
mkdir -p _splits
cp oneie_v0.4.8/resource/splits/ACE05-E/train.doc.txt _splits/
for wset in dev test; do
  python3 -m mspx.scripts.data.evt.print_ace_splits ${wset} >_splits/${wset}.doc.txt
  diff _splits/${wset}.doc.txt oneie_v0.4.8/resource/splits/ACE05-E/${wset}.doc.txt
done
# process
mkdir -p _en_ace
python3 oneie_v0.4.8/preprocessing/process_ace.py -i ${LDC}/ace_2005_td_v7/data/ -o _en_ace -s _splits -l english

# step 2: convert and parse
# convert
mkdir -p data
for wset in train dev test; do
  python3 -m mspx.scripts.data.evt.conv_oneie2mine _en_ace/${wset}.oneie.json data/en.ace05.${wset}.json
done
# parse
for wset in train dev test; do
python3 -m mspx.cli.annotate anns:stanza stanza_lang:en input_path:data/en.ace05.${wset}.json output_path:data/en.ace05.${wset}.ud2.json
done
# check
wc _en_ace/*
wc data/*

# --
# check domain counts
#for wset in train dev test; do
#  python3 -m mspx.scripts.tools.split_domain input_path:data/en.ace05.${wset}.json 'domain_key_f:lambda x: judge_ace_genre(x.id)'
#done
#split_domain: ['data/en.ace05.train.json'] => : {'instC_bc': 55, 'instC_bn': 218, 'instC_cts': 39, 'instC_nw': 58, 'instC_un': 45, 'instC_wl': 114, 'inst_orig': 529}
#split_domain: ['data/en.ace05.dev.json'] => : {'instC_bc': 5, 'instC_bn': 8, 'instC_nw': 8, 'instC_un': 4, 'instC_wl': 5, 'inst_orig': 30}
#split_domain: ['data/en.ace05.test.json'] => : {'instC_nw': 40, 'inst_orig': 40}
# --

## step 3: final re-split and filter training sentences
## resplit
#python3 -m mspx.scripts.tools.sample_shuffle shuffle_times:1 split_sep:30 input_path:data/en.ace05.train.ud2.json output_path:data/en.ace05.resplit_train.ud2.json
## filter-sent (only changing training!)
#mkdir -p dataS
#python3 -m mspx.scripts.data.evt.filter_data input_path:data/en.ace05.resplit_train.ud2.json min_len:3 min_alpha_ratio:0.5 output_path:dataS/en.ace05.train.ud2.json output_ind_sents:1
#python3 -m mspx.scripts.data.evt.filter_data input_path:data/en.ace05.resplit_train.ud2.s0.json min_len:0 min_alpha_ratio:0. output_path:dataS/en.ace05.dev.ud2.json output_ind_sents:1
#python3 -m mspx.scripts.data.evt.filter_data input_path:data/en.ace05.dev.ud2.json min_len:0 min_alpha_ratio:0. output_path:dataS/en.ace05.test.ud2.json output_ind_sents:1

# step 3: overall resplit?
mkdir -p dataRS
cat data/en.ace05.{train,dev,test}.ud2.json >dataRS/docs.all.json
python3 -m mspx.scripts.tools.sample_shuffle shuffle_times:1 split_sep:60,120,10000 split_names:dev,test,train input_path:dataRS/docs.all.json output_path:dataRS/docs.ZZKEYZZ.json
# train: {'instC_bc': 42, 'instC_bn': 162, 'instC_cts': 26, 'instC_nw': 72, 'instC_un': 34, 'instC_wl': 83, 'inst_orig': 419}
# dev: {'instC_bc': 6, 'instC_bn': 19, 'instC_cts': 6, 'instC_nw': 9, 'instC_un': 6, 'instC_wl': 14, 'inst_orig': 60}
# test: {'instC_bc': 12, 'instC_bn': 45, 'instC_cts': 7, 'instC_nw': 25, 'instC_un': 9, 'instC_wl': 22, 'inst_orig': 120}
python3 -m mspx.scripts.data.evt.filter_data input_path:dataRS/docs.train.json min_len:3 min_alpha_ratio:0.5 output_path:dataRS/en.ace05.train.ud2.json output_ind_sents:1
python3 -m mspx.scripts.data.evt.filter_data input_path:dataRS/docs.dev.json min_len:0 min_alpha_ratio:0. output_path:dataRS/en.ace05.dev.ud2.json output_ind_sents:1
python3 -m mspx.scripts.data.evt.filter_data input_path:dataRS/docs.test.json min_len:0 min_alpha_ratio:0. output_path:dataRS/en.ace05.test.ud2.json output_ind_sents:1
#dev:({'c_word': 34483, 'c_Fef': 6007, 'c_doc': 2519, 'c_sent': 2519, 'c_Aef': 787, 'c_Aevt': 709, 'c_Fevt': 498})
#test:({'c_word': 61490, 'c_Fef': 10772, 'c_doc': 3975, 'c_sent': 3975, 'c_Aef': 1734, 'c_Aevt': 1687, 'c_Fevt': 1119})
#train:({'c_word': 205982, 'c_Fef': 35718, 'c_doc': 10831, 'c_sent': 10831, 'c_Aef': 6184, 'c_Aevt': 5703, 'c_Fevt': 3725})
