#

#
from msp import utils
from msp.data import MultiCatStreamer, InstCacher

from ..common.confs import DepParserConf, init_everything, build_model
from ..common.data import get_data_reader
from ..common.vocab import DConf, ParserVocabPackage
from ..common.run import index_stream, batch_stream, ParserTrainingRunner

#
def main(args):
    conf = init_everything(args)
    dconf, pconf = conf.dconf, conf.pconf
    tconf = pconf.tconf
    iconf = pconf.iconf
    # data
    train_streamer = get_data_reader(dconf.train, dconf.input_format, dconf.code_train, dconf.use_label0)
    dev_streamer = get_data_reader(dconf.dev, dconf.input_format, dconf.code_dev, dconf.use_label0)
    test_streamer = get_data_reader(dconf.test, dconf.input_format, dconf.code_test, dconf.use_label0)
    # vocab
    if tconf.no_build_dict:
        vpack = ParserVocabPackage.build_by_reading(dconf)
    else:
        # include dev/test only for convenience of including words hit in pre-trained embeddings
        vpack = ParserVocabPackage.build_from_stream(dconf, train_streamer, MultiCatStreamer([dev_streamer, test_streamer]))
        vpack.save(dconf.dict_dir)
    # index the data
    to_cache = dconf.cache_data
    train_iter = batch_stream(index_stream(train_streamer, vpack, to_cache), tconf, True)
    dev_iter = batch_stream(index_stream(dev_streamer, vpack, to_cache), iconf, False)
    test_iter = batch_stream(index_stream(test_streamer, vpack, to_cache), iconf, False)
    # model
    model = build_model(conf.partype, conf, vpack)
    # training runner
    tr = ParserTrainingRunner(tconf, model, vpack, dev_outfs=dconf.output_file, dev_goldfs=[dconf.dev, dconf.test], dev_out_format=dconf.output_format)
    if tconf.load_model:
        tr.load(dconf.model_load_name, tconf.load_process)
    # go
    tr.run(train_iter, [dev_iter, test_iter])
    utils.zlog("The end of Training.")
