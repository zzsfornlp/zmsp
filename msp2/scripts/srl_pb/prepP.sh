#

# prepare recent PB release

# step 0: ontonotes and ewt
# ...
#PATH_ONTO=ontonotes-release-5.0
#PATH_EWT=eng_web_tbk

# step 1: get them
# frames
wget https://github.com/propbank/propbank-frames/archive/v3.1.tar.gz
tar -zxvf v3.1.tar.gz
# data
git clone https://github.com/propbank/propbank-release/
# (20.10.03): a9accf68e210eb54dba7f9549b66e1d6ad5cc807

# step 2: run script
# again python2 and goto "propbank-release/docs/scripts"
conda activate py27
python map_all_to_conll.py --ontonotes ../../../ontonotes-release-5.0/ --ewt ../../../eng_web_tbk/
conda deactivate

# step 3: concat the datasets
python3 prepP_cat.py
