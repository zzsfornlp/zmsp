#

#
from msp import utils
from msp.data import MultiCatStreamer, InstCacher

from ..common.confs import DepParserConf, init_everything, build_model
from ..common.data import get_data_reader, get_multisoure_data_reader
from ..common.vocab import DConf, ParserVocabPackage
from ..common.run import index_stream, batch_stream, ParserTrainingRunner

#
def main(args):
    conf = init_everything(args)
    dconf, pconf = conf.dconf, conf.pconf
    tconf = pconf.tconf
    iconf = pconf.iconf
    # dev/test can be non-existing!
    if not dconf.dev and dconf.test:
        utils.zwarn("No dev but give test, actually use test as dev (for early stopping)!!")
    dt_golds, dt_codes, dt_aux_reprs, dt_aux_scores, dt_cuts = [], [], [], [], []
    for file, code, aux_repr, aux_score, one_cut in \
            [(dconf.dev, dconf.code_dev, dconf.aux_repr_dev, dconf.aux_score_dev, dconf.cut_dev),
             (dconf.test, dconf.code_test, dconf.aux_repr_test, dconf.aux_score_test, "")]:  # no cut for test!
        if len(file)>0:
            utils.zlog(f"Add file `{file}(code={code}, aux_repr={aux_repr}, aux_scpre={aux_score}, cut={one_cut})'"
                       f" as dt-file #{len(dt_golds)}.")
            dt_golds.append(file)
            dt_codes.append(code)
            dt_aux_reprs.append(aux_repr)
            dt_aux_scores.append(aux_score)
            dt_cuts.append(one_cut)
    # data
    if dconf.multi_source:
        _reader_getter = get_multisoure_data_reader
    else:
        _reader_getter = get_data_reader
    train_streamer = _reader_getter(dconf.train, dconf.input_format, dconf.code_train, dconf.use_label0,
                                     dconf.aux_repr_train, dconf.aux_score_train, cut=dconf.cut_train)
    dt_streamers = [_reader_getter(f, dconf.input_format, c, dconf.use_label0, aux_r, aux_s, cut=one_cut)
                    for f, c, aux_r, aux_s, one_cut in zip(dt_golds, dt_codes, dt_aux_reprs, dt_aux_scores, dt_cuts)]
    # vocab
    if tconf.no_build_dict:
        vpack = ParserVocabPackage.build_by_reading(dconf)
    else:
        # include dev/test only for convenience of including words hit in pre-trained embeddings
        vpack = ParserVocabPackage.build_from_stream(dconf, train_streamer, MultiCatStreamer(dt_streamers))
        vpack.save(dconf.dict_dir)
    # model
    model = build_model(conf.partype, conf, vpack)
    # index the data
    train_inst_preparer = model.get_inst_preper(True)
    test_inst_preparer = model.get_inst_preper(False)
    to_cache = dconf.cache_data
    to_cache_shuffle = dconf.to_cache_shuffle
    # todo(note): make sure to cache both train and dev to save time for cached computation
    train_iter = batch_stream(index_stream(train_streamer, vpack, to_cache, to_cache_shuffle, train_inst_preparer), tconf, True)
    dt_iters = [batch_stream(index_stream(z, vpack, to_cache, to_cache_shuffle, test_inst_preparer), iconf, False) for z in dt_streamers]
    # training runner
    tr = ParserTrainingRunner(tconf, model, vpack, dev_outfs=dconf.output_file, dev_goldfs=dt_golds, dev_out_format=dconf.output_format)
    if tconf.load_model:
        tr.load(dconf.model_load_name, tconf.load_process)
    # go
    tr.run(train_iter, dt_iters)
    utils.zlog("The end of Training.")
