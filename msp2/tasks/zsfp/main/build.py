#

# build dictionaries

from msp2.utils import zlog, zwarn, init_everything, Timer
from msp2.data.stream import MultiCatStreamer
from ..confs import OverallConf
from ..run import ZsfpVocabPackage

# --
def main(args):
    conf: OverallConf = init_everything(OverallConf(), args)
    dconf, tconf = conf.dconf, conf.tconf
    # data
    from .train import prepare_train_data
    train_streamers, dev_streamers, test_streamer, _ = prepare_train_data(dconf)
    extra_streamers = dev_streamers if test_streamer is None else dev_streamers + [test_streamer]
    # vocab
    vpack = ZsfpVocabPackage.build_from_stream(dconf, MultiCatStreamer(train_streamers), MultiCatStreamer(extra_streamers))
    vpack.save(dconf.dict_dir)
    zlog("The end of Building.")

if __name__ == '__main__':
    import sys
    with Timer(info=f"Building", print_date=True) as et:
        main(sys.argv[1:])

# example: for building vocab and filtering embeds
"""
# concat all training sources
cat ../fn_parsed/fn15_{exemplars.filtered,fulltext.train}.json >fn15_et_combined.json
# --
# get embeddings
# wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip
unzip wiki-news-300d-1M-subword.vec.zip  # get wiki-news-300d-1M-subword.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec
PYTHONPATH=../src/ python3 -m msp2.tasks.zsfp.main.build train:fn15_et_combined.json dev:../fn_parsed/fn15_fulltext.dev.json test:../fn_parsed/fn15_fulltext.test.json dict_dir:./ pretrain_hits_outf:hits.vec pretrain_wv_file:wiki-news-300d-1M-subword.vec |& tee _log.voc
PYTHONPATH=../src/ python3 -m msp2.tasks.zsfp.main.build train:fn15_et_combined.json dev:../fn_parsed/fn15_fulltext.dev.json test:../fn_parsed/fn15_fulltext.test.json dict_dir:./ pretrain_hits_outf:hits2.vec pretrain_wv_file:wiki.en.vec |& tee _log.voc2
PYTHONPATH=../src/ python3 -m msp2.tasks.zsfp.main.build train:fn15_et_combined.json dev:../fn_parsed/fn15_fulltext.dev.json test:../fn_parsed/fn15_fulltext.test.json dict_dir:./ pretrain_hits_outf:hits3.vec pretrain_wv_file:glove/glove.6B.300d.txt |& tee _log.voc3
# --
# filter for pb
PYTHONPATH=../src/ python3 -m msp2.tasks.zsfp.main.build train:../pb/conll05/train.conll.ud.json dev:../pb/conll05/dev.conll.ud.json,../pb/conll05/test.wsj.conll.ud.json,../pb/conll05/test.brown.conll.ud.json dict_dir:./ pretrain_hits_outf:hits_conll05.vec pretrain_wv_file:wiki-news-300d-1M-subword.vec |& tee _log.voc_conll05
"""
