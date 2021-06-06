#

# build dictionaries

from msp2.utils import zlog, zwarn, init_everything, Timer
from msp2.data.stream import MultiCatStreamer
from ..confs import OverallConf
from ..run import ZmtlVocabPackage

# --
def main(args):
    conf: OverallConf = init_everything(OverallConf(), args)
    dconf, tconf = conf.dconf, conf.tconf
    # data
    from .train import prepare_train_data
    train_streamers, dev_streamers, test_streamer, _ = prepare_train_data(dconf)
    extra_streamers = dev_streamers if test_streamer is None else dev_streamers + [test_streamer]
    # vocab
    vpack = ZmtlVocabPackage.build_from_stream(dconf, MultiCatStreamer(train_streamers), MultiCatStreamer(extra_streamers))
    vpack.save(dconf.dict_dir)
    zlog("The end of Building.")

if __name__ == '__main__':
    import sys
    with Timer(info=f"Building", print_date=True) as et:
        main(sys.argv[1:])

# example: for building vocab and filtering embeds
"""
# filter for pb
PYTHONPATH=../src/ python3 -m msp2.tasks.zmtl.main.build train:../pb/conll05/train.conll.ud.json dev:../pb/conll05/dev.conll.ud.json,../pb/conll05/test.wsj.conll.ud.json,../pb/conll05/test.brown.conll.ud.json dict_dir:./ pretrain_hits_outf:hits_conll05.vec pretrain_wv_file:wiki-news-300d-1M-subword.vec |& tee _log.voc_conll05
"""
