#

# building vocabs
# the initial part of training

from ..common.confs import init_everything
from ..common.data import get_data_reader, get_multisoure_data_reader
from ..common.vocab import ParserVocabPackage

def main(args):
    conf = init_everything(args+["partype:fp"])
    dconf = conf.dconf
    if dconf.multi_source:
        _reader_getter = get_multisoure_data_reader
    else:
        _reader_getter = get_data_reader
    train_streamer = _reader_getter(dconf.train, dconf.input_format, dconf.code_train, dconf.use_label0, cut=dconf.cut_train)
    vpack = ParserVocabPackage.build_from_stream(dconf, train_streamer, [])  # empty extra_stream
    vpack.save(dconf.dict_dir)

# SRC_DIR="../src/"
# PYTHONPATH=${SRC_DIR}/ python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.build_vocab train:[input] input_format:conllu dict_dir:[output_dir] init_from_pretrain:0 pretrain_file:?
# PYTHONPATH=${SRC_DIR}/ python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.build_vocab train:../data/UD_RUN/ud24/en_all.conllu input_format:conllu dict_dir:./vocab/ init_from_pretrain:1 pretrain_file:../data/UD_RUN/ud24/wiki.multi.en.filtered.vec pretrain_scale:1 pretrain_init_nohit:1
