#

#
from msp import utils
from msp.data import MultiCatStreamer, InstCacher

from ..run.confs import OverallConf, init_everything, build_model
from ..run.run import get_data_reader, PreprocessStreamer, index_stream, batch_stream, MltTrainingRunner
from ..run.vocab import MLMVocabPackage

#
def main(args):
    conf = init_everything(args)
    dconf, mconf = conf.dconf, conf.mconf
    # dev/test can be non-existing!
    if not dconf.dev and dconf.test:
        utils.zwarn("No dev but give test, actually use test as dev (for early stopping)!!")
    dt_golds, dt_cuts = [], []
    for file, one_cut in [(dconf.dev, dconf.cut_dev), (dconf.test, "")]:  # no cut for test!
        if len(file)>0:
            utils.zlog(f"Add file `{file}(cut={one_cut})' as dt-file #{len(dt_golds)}.")
            dt_golds.append(file)
            dt_cuts.append(one_cut)
    if len(dt_golds) == 0:
        utils.zwarn("No dev set, then please specify static lrate schedule!!")
    # data
    train_streamer = PreprocessStreamer(get_data_reader(dconf.train, dconf.input_format, cut=dconf.cut_train),
                                        lower_case=dconf.lower_case, norm_digit=dconf.norm_digit)
    dt_streamers = [PreprocessStreamer(get_data_reader(f, dconf.dev_input_format, cut=one_cut),
                                       lower_case=dconf.lower_case, norm_digit=dconf.norm_digit)
                    for f, one_cut in zip(dt_golds, dt_cuts)]
    # vocab
    if mconf.no_build_dict:
        vpack = MLMVocabPackage.build_by_reading(dconf.dict_dir)
    else:
        # include dev/test only for convenience of including words hit in pre-trained embeddings
        vpack = MLMVocabPackage.build_from_stream(dconf.vconf, train_streamer, MultiCatStreamer(dt_streamers))
        vpack.save(dconf.dict_dir)
    # aug2
    if mconf.aug_word2:
        vpack.aug_word2_vocab(train_streamer, MultiCatStreamer(dt_streamers), mconf.aug_word2_pretrain)
        vpack.save(mconf.aug_word2_save_dir)
    # model
    model = build_model(conf, vpack)
    # index the data
    train_inst_preparer = model.get_inst_preper(True)
    test_inst_preparer = model.get_inst_preper(False)
    to_cache = dconf.cache_data
    to_cache_shuffle = dconf.to_cache_shuffle
    # todo(note): make sure to cache both train and dev to save time for cached computation
    backoff_pos_idx = dconf.backoff_pos_idx
    train_iter = batch_stream(index_stream(train_streamer, vpack, to_cache, to_cache_shuffle, train_inst_preparer, backoff_pos_idx), mconf.train_batch_size, mconf, True)
    dt_iters = [batch_stream(index_stream(z, vpack, to_cache, to_cache_shuffle, test_inst_preparer, backoff_pos_idx), mconf.test_batch_size, mconf, False) for z in dt_streamers]
    # training runner
    tr = MltTrainingRunner(mconf.rconf, model, vpack, dev_outfs=dconf.output_file, dev_goldfs=dt_golds, dev_out_format=dconf.output_format)
    if mconf.train_preload_model:
        tr.load(dconf.model_load_name, mconf.train_preload_process)
    # go
    tr.run(train_iter, dt_iters)
    utils.zlog("The end of Training.")
