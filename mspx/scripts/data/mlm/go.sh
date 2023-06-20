#

# download bookcorpus
#https://paperswithcode.com/dataset/bookcorpus
wget https://battle.shawwn.com/sdb/books1/books1.tar.gz
tar -zxvf books1.tar.gz

# download wiki
#wget -nc https://dumps.wikimedia.org/enwiki/20181120/enwiki-20181120-pages-articles.xml.bz2
# ok, use the current one
for cl in en cs de es fi fr it ja ru zh; do
WIKI_LANG=${cl} WIKI_VER=20220701 bash prep_wiki.sh
done |& tee _log.prep_wiki
# cat _log.prep_wiki | grep -E "Preparing|in "

# process
# import stanza; for cl in "en cs de es fi fr it ja ru zh".split(): stanza.download(cl)
# tokenize
#for cl in en; do  # note: only en takes ~5 days with 12 core ...
for cl in cs de es fi fr it ja ru zh; do
OMP_NUM_THREADS=1 python3 -m mspx.scripts.tools.run_para -n 12 -i ${cl}_ext/AA/wiki_?? -c "python3 -m mspx.scripts.data.mlm.proc_wiki cl:${cl} input_path:[[IN]] output_path:[[IN]].tok.bz2 stanza_processors:tokenize 2>&1 | tee [[IN]].tok.log"
#done |& tee _log.tok_enwiki
done |& tee _log.tok_ALLwiki

# preprocess
for cl in en; do
OMP_NUM_THREADS=1 python3 -m mspx.scripts.tools.run_para -n 12 -i ${cl}_ext/AA/wiki_?? -c "python3 -m mspx.scripts.data.mlm.preprocess input_path:[[IN]].tok.bz2 output_path:[[IN]].pp.bz2 toker:bert-base-cased 2>&1 | tee [[IN]].pp.log"
done |& tee _log.pp_enwiki
# for pp_enwiki: orig={doc=6.2M,sent=133.7M,tok=2.9B}, out={~,seq=30M,tok=2.8B,subtok=3.3B}
# Counter({'subtok_out': 3285286274, 'tok_orig': 2912511804, 'tok_out': 2817270439, 'sent_orig': 133690715, 'sent_out': 127922810, 'seq_out': 30167638, 'seq_out_L=20*5': 13913200, 'seq_out_L=20*6': 8949363, 'doc': 6187317, 'sent_miss': 5767905, 'seq_out_L=20*4': 5146001, 'seq_out_L=20*3': 2067302, '_time_doc': 231950.02000000005, 'seq_out_L=20*7': 39090, 'seq_out_L=20*8': 19720, 'seq_out_L=20*9': 10986, 'seq_out_L=20*10': 6444, 'seq_out_L=20*11': 4158, 'seq_out_L=20*12': 2858, 'seq_out_L=20*13': 1940, 'seq_out_L=20*14': 1364, 'seq_out_L=20*15': 1039, 'seq_out_L=20*16': 751, 'seq_out_L=20*17': 565, 'seq_out_L=20*18': 449, 'seq_out_L=20*19': 351, 'seq_out_L=20*20': 300, 'seq_out_L=20*22': 222, 'seq_out_L=20*21': 215, 'seq_out_L=20*23': 162, 'seq_out_L=20*24': 131, 'seq_out_L=20*26': 109, 'seq_out_L=20*25': 109})  # truncate count>100
# put together
for cl in en; do
python3 -m mspx.scripts.tools.sample_shuffle shuffle_times:1 input_path:${cl}_ext/AA/wiki_*.pp.bz2 output_path:${cl}_ext/${cl}wiki.pp.bz2 split_piece:4 split_sep:10000,10000
done |& tee _log.ss_enwiki
# for pp_enwiki: inst: 30167638
