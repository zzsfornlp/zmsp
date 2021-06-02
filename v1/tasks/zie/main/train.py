#

#
from msp import utils
from msp.data import MultiCatStreamer, InstCacher
from msp.zext.process_train import ScheduledValue

from ..common.confs import OverallConf, init_everything, build_model, get_berter
from ..common.data import get_data_reader, BerterDataAuger
from ..common.data_multi import MultiSpecialJoinStream
from ..common.vocab import IEVocabPackage
from ..common.run import index_stream, batch_stream, MyIETrainingRunner

#
def main(args):
    conf: OverallConf = init_everything(args)
    dconf, mconf = conf.dconf, conf.mconf
    tconf = mconf.tconf
    iconf = mconf.iconf
    #
    # dev/test can be non-existing!
    if not dconf.dev and dconf.test:
        utils.zwarn("No dev but give test, actually use test as dev (for early stopping)!!")
    dt_golds, dt_aux_reprs = [], []
    for file, aux_repr in [(dconf.dev, dconf.aux_repr_dev), (dconf.test, dconf.aux_repr_test)]:
        if len(file) > 0:
            utils.zlog(f"Add file `{file}(aux_repr={aux_repr})' as dt-file #{len(dt_golds)}.")
            dt_golds.append(file)
            dt_aux_reprs.append(aux_repr)
    # data
    if len(dconf.ms_train)>0:
        # do ms train, ignore dconf.train
        train_streamers = [get_data_reader(f, dconf.input_format, dconf.use_label0, dconf.noef_link0, dconf.aux_repr_train, max_evt_layers=dconf.max_evt_layers) for f in dconf.ms_train]
        train_streamer = MultiCatStreamer(train_streamers)  # simple concat for building vocab
        ms_budgets = [ScheduledValue(f"ms_budget{i}", c) for i,c in enumerate(dconf.ms_train_budget_list[:len(train_streamers)])]
        assert len(ms_budgets) == len(train_streamers)
        utils.zlog(f"Multi-source training with inputsL {dconf.ms_train}")
    else:
        train_streamers = ms_budgets = None
        train_streamer = get_data_reader(dconf.train, dconf.input_format, dconf.use_label0, dconf.noef_link0, dconf.aux_repr_train, max_evt_layers=dconf.max_evt_layers)
    dt_streamers = [get_data_reader(f, dconf.input_format, dconf.use_label0, dconf.noef_link0, aux_r)
                    for f, aux_r in zip(dt_golds, dt_aux_reprs)]
    # vocab
    if tconf.no_build_dict:
        vpack = IEVocabPackage.build_by_reading(conf)
    else:
        # include dev/test only for convenience of including words hit in pre-trained embeddings
        vpack = IEVocabPackage.build_from_stream(conf, train_streamer, MultiCatStreamer(dt_streamers))
        vpack.save(dconf.dict_dir)
    # model
    model = build_model(conf.model_type, conf, vpack)
    # use bert? todo(note): pre-compute here?
    if dconf.use_bert:
        bmodel = get_berter(dconf.bconf)
        train_streamer = BerterDataAuger(train_streamer, bmodel, "aux_repr")
        dt_streamers = [BerterDataAuger(z, bmodel, "aux_repr") for z in dt_streamers]
    # index the data
    train_inst_preparer = model.get_inst_preper(True)
    test_inst_preparer = model.get_inst_preper(False)
    to_cache = dconf.cache_data
    to_cache_shuffle = dconf.cache_shuffle
    # -----
    if ms_budgets is None:
        train_iter = batch_stream(index_stream(train_streamer, vpack, to_cache, to_cache_shuffle, train_inst_preparer), tconf, True)
    else:
        indexes_streamers = [index_stream(s, vpack, to_cache, to_cache_shuffle, train_inst_preparer) for s in train_streamers]
        multi_streamer = MultiSpecialJoinStream(indexes_streamers, ms_budgets, dconf.ms_stop_idx)
        train_iter = batch_stream(multi_streamer, tconf, True)
    # -----
    dt_iters = [batch_stream(index_stream(z, vpack, to_cache, to_cache_shuffle, test_inst_preparer), iconf, False)
                for z in dt_streamers]
    # training runner
    tr = MyIETrainingRunner(tconf, model, vpack, dev_outfs=dconf.output_file, dev_goldfs=dt_golds,
                            dev_out_format=dconf.output_format, eval_conf=dconf.eval_conf)
    # -----
    if ms_budgets is not None:
        tr.add_scheduled_values(ms_budgets)  # add s-values
    # -----
    if tconf.load_model:
        tr.load(dconf.model_load_name, tconf.load_process)
    # go
    tr.run(train_iter, dt_iters)
    utils.zlog("The end of Training.")
