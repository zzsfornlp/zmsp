#

# special decoding with arg-aug

from msp.utils import Helper, zlog, zwarn, Timer, StatRecorder, zopen
from msp.data import MultiHelper, WordVectors, VocabBuilder

from ..common.confs import OverallConf, init_everything, build_model, get_berter
from ..common.data import get_data_reader, BerterDataAuger, get_data_writer
from ..common.vocab import DConf, IEVocabPackage
from ..common.run import index_stream, batch_stream, MyIETestingRunner
from ..common.eval import MyIEEvaler, MyIEEvalerConf

from ..models2.decoderA import ArgAugConf, ArgAugDecoder

#
class DecodeAConf(OverallConf):
    def __init__(self, model_type, args):
        super().__init__(model_type, None)
        self.aconf = ArgAugConf()
        self.verbose = True
        self.do_eval = True
        #
        if args is not None:
            self.update_from_args(args)
            self.validate()

def main(args):
    conf: DecodeAConf = init_everything(args, DecodeAConf)
    dconf, mconf = conf.dconf, conf.mconf
    iconf = mconf.iconf
    # vocab
    vpack = IEVocabPackage.build_by_reading(conf)
    # prepare data
    test_streamer = get_data_reader(dconf.test, dconf.input_format, dconf.use_label0, dconf.noef_link0, dconf.aux_repr_test, max_evt_layers=dconf.max_evt_layers)
    # model
    model = build_model(conf.model_type, conf, vpack)
    model.load(dconf.model_load_name)
    # use bert?
    if dconf.use_bert:
        bmodel = get_berter(dconf.bconf)
        test_streamer = BerterDataAuger(test_streamer, bmodel, "aux_repr")
    # finally prepare iter (No Cache!!, actually no batch_stream)
    test_inst_preparer = model.get_inst_preper(False)
    test_iter = index_stream(test_streamer, vpack, False, False, test_inst_preparer)
    # =====
    # run
    decoder = ArgAugDecoder(conf.aconf, model)
    all_docs = []
    stat_recorder = StatRecorder(False)
    with Timer(tag="Decode", info="Decoding", print_date=True):
        with zopen(dconf.output_file, 'w') as fd:
            data_writer = get_data_writer(fd, dconf.output_format)
            for one_doc in test_iter:
                info = decoder.decode(one_doc)
                stat_recorder.record(info)
                if conf.verbose:
                    zlog(f"Decode one doc, id={one_doc.doc_id} info={info}")
                # release resources
                for one_sent in one_doc.sents:
                    one_sent.extra_features["aux_repr"] = None  # todo(note): special name!
                # write output
                data_writer.write([one_doc])
                #
                all_docs.append(one_doc)
    if conf.verbose:
        zlog(f"Finish decoding, overall: {stat_recorder.summary()}")
    # eval?
    if conf.do_eval:
        evaler = MyIEEvaler(MyIEEvalerConf())
        result = evaler.eval(all_docs, all_docs)
        Helper.printd(result)
    zlog("The end.")
