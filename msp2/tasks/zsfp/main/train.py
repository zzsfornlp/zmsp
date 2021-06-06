#

# train

from msp2.utils import zlog, zwarn, init_everything, Timer
from msp2.data.stream import MultiCatStreamer, MultiJoinStreamer
from msp2.proc import ScheduledValue
from ..confs import OverallConf, build_model
from ..run import ZsfpTrainingRunner, ZsfpVocabPackage, index_stream, train_prep_stream, batch_stream

# --
def prepare_train_data(dconf):
    train_files = dconf.train
    dev_files = dconf.dev
    # get the streamers
    train_streamers = [dconf.R.get_reader(input_path=f, **ps) for f,ps in zip(train_files, dconf.get_train_props())]
    dev_streamers = [dconf.R.get_reader(input_path=f) for f in dev_files]
    # note: test_streamer is only used for looking up word embeddings when building vocabs for convenience
    test_streamer = dconf.R.get_reader(input_path=dconf.test) if dconf.test else None
    zlog(f"Prepare for training: train={train_files}, dev={dev_files}, (test={dconf.test})")
    return train_streamers, dev_streamers, test_streamer, dev_files

# --
def main(args):
    conf: OverallConf = init_everything(OverallConf(), args)
    dconf, tconf = conf.dconf, conf.tconf
    # data
    train_streamers, dev_streamers, test_streamer, dev_golds = prepare_train_data(dconf)
    # vocab
    if tconf.no_build_dict:  # read
        vpack = ZsfpVocabPackage.build_by_reading(dconf)
    else:
        # include dev/test only for convenience of including words hit in pre-trained embeddings
        extra_streamers = dev_streamers if test_streamer is None else dev_streamers + [test_streamer]
        vpack = ZsfpVocabPackage.build_from_stream(dconf, MultiCatStreamer(train_streamers), MultiCatStreamer(extra_streamers))
        vpack.save(dconf.dict_dir)
    # model
    model = build_model(conf, vpack=vpack)
    train_inst_preparer = model.get_inst_preper(True)
    test_inst_preparer = model.get_inst_preper(False)
    # actual streams
    prepared_train_streamers = [
        train_prep_stream(index_stream(z, vpack, tconf.train_use_cache, tconf.cache_shuffle_times, train_inst_preparer), tconf)
        for z in train_streamers]
    if len(prepared_train_streamers) > 1:  # ms_train
        ms_budgets = [ScheduledValue(f"ms_budget{i}", c) for i,c in enumerate(dconf.get_ms_train_budgets())]
        joined_train_streamer = MultiJoinStreamer(prepared_train_streamers, dconf.ms_stop_idx, ratios=ms_budgets)
    else:
        ms_budgets = []
        joined_train_streamer = prepared_train_streamers[0]
    train_iter, train_batch_f = batch_stream(joined_train_streamer, tconf, True)
    dev_iters = [batch_stream(index_stream(
        z, vpack, tconf.dev_use_cache, 0, test_inst_preparer), tconf, False)[0] for z in dev_streamers]
    # training runner
    tr = ZsfpTrainingRunner.create(model, train_iter, train_batch_f, conf, dev_iters,
                                   [dconf.output+f".dev{i}" for i in range(len(dev_golds))], dev_golds)
    for mv in ms_budgets:  # add them for scheduling!
        tr.add_scheduled_value(mv)
    # load?
    if tconf.load_model:
        tr.load(dconf.model_load_name, tconf.load_process, load_strict=dconf.model_load_strict)
    # go
    tr.run()
    zlog("The end of Training.")

if __name__ == '__main__':
    import sys
    with Timer(info=f"Training", print_date=True) as et:
        main(sys.argv[1:])
