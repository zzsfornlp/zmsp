#

#
from typing import List, Iterable, Dict
import numpy as np

from msp.utils import zlog, zopen, GLOBAL_RECORDER, Helper, zcheck, Random
from msp.data import FAdapterStreamer, BatchArranger, InstCacher, FListAdapterStream, ShuffleStreamer
from msp.zext.process_train import TrainingRunner, RecordResult
from msp.zext.process_test import TestingRunner, ResultManager

from .data import DocInstance, Sentence, get_data_writer
from .vocab import IEVocabPackage
from .eval import MyIEEvaler

#
class IndexerHelper:
    def __init__(self, vpack: IEVocabPackage):
        #
        self.word_vocab = vpack.get_voc("word")
        self.lemma_vocab = vpack.get_voc("lemma")
        self.char_vocab = vpack.get_voc("char")
        self.upos_vocab = vpack.get_voc("upos")
        self.ulabel_vocab = vpack.get_voc("ulabel")
        #
        self.hl_evt = vpack.get_voc('hl_evt')
        self.hl_ef = vpack.get_voc('hl_ef')
        self.hl_arg = vpack.get_voc('hl_arg')

    def index_doc(self, inst: DocInstance):
        for sent in inst.sents:
            self.index_sent(sent)
        # build targets
        if inst.entity_fillers is not None:
            for one_ef in inst.entity_fillers:
                one_ef.type_idx = self.hl_ef.val2idx(one_ef.type)
        if inst.events is not None:
            for one_evt in inst.events:
                one_evt.type_idx = self.hl_evt.val2idx(one_evt.type)
                if one_evt.links is not None:
                    for one_arg in one_evt.links:
                        one_arg.role_idx = self.hl_arg.val2idx(one_arg.role)

    def index_sent(self, sent: Sentence):
        # todo(warn): remember to norm word; replace singleton at model's input, not here
        sent.words.build_idxes(self.word_vocab)
        sent.lemmas.build_idxes(self.lemma_vocab)
        sent.chars.build_idxes(self.char_vocab)
        sent.uposes.build_idxes(self.upos_vocab)
        sent.ud_labels.build_idxes(self.ulabel_vocab)
        # sort the lists in each sentence according to (position, type-freq)
        if sent.entity_fillers is not None:
            sent.entity_fillers.sort(key=lambda x: (x.mention.hard_span.position(), -self.hl_ef.val2count(x.type)))
        if sent.events is not None:
            sent.events.sort(key=lambda x: (x.mention.hard_span.position(), -self.hl_evt.val2count(x.type)))
        # todo(note): currently we do not need to sort args?

# for indexing instances
class IndexerStreamer(FAdapterStreamer):
    def __init__(self, in_stream, vpack, inst_preparer):
        super().__init__(in_stream, self._go_index, False)
        #
        self.index_helper = IndexerHelper(vpack=vpack)
        self.inst_preparer = inst_preparer

    def _go_index(self, inst: DocInstance):
        self.index_helper.index_doc(inst)
        #
        if self.inst_preparer is not None:
            inst = self.inst_preparer(inst)
        # in fact, inplaced if not wrapping model specific preparer
        return inst

#
def index_stream(in_stream, vpack, cached, cache_shuffle, inst_preparer):
    i_stream = IndexerStreamer(in_stream, vpack, inst_preparer)
    if cached:
        return InstCacher(i_stream, shuffle=cache_shuffle)
    else:
        return i_stream

# for arrange batches:
# todo(warn): batch_size means number of sents, no sorting by #sent
_BS_sample_stream = Random.stream(Random.random_sample)
#
def batch_stream(in_stream, ticonf, training):
    # =====
    def _count_train_sents(d):
        # todo(note): here, we include only "hit" sents for training, but maybe the alternative can be also ok, especially for doc-hint?
        valid_sents = [x for x in d.sents if x.length<ticonf.train_skip_length and x.length>=ticonf.train_min_length]
        return len(valid_sents) - len([x for x in valid_sents if len(x.events)==0]) * ticonf.train_skip_noevt_rate
    # =====
    if ticonf.no_15E78:
        in_stream = FListAdapterStream(in_stream, lambda d: ([] if d.dataset.startswith("LDC2015E78") else [d]))
    if training:
        MBS = ticonf.maxibatch_size
        if ticonf.train_sent_based:
            assert False, "this mode (sent-based) should be deprecated"
            # todo(note): this will not be cached since caching is before this at index_stream
            sent_stream = FListAdapterStream(in_stream, lambda d: [x for x in d.sents if x.length<ticonf.train_skip_length and x.length>=ticonf.train_min_length and (len(x.events)>0 or next(_BS_sample_stream)>ticonf.train_skip_noevt_rate)])
            if ticonf.train_sent_shuffle:
                sent_stream = ShuffleStreamer(sent_stream)
            b_stream = BatchArranger(sent_stream, batch_size=ticonf.batch_size, maxibatch_size=MBS, batch_size_f=None,
                                     dump_detectors=None, single_detectors=None, sorting_keyer=lambda x: x.length,
                                     shuffling=ticonf.shuffle_train)
        elif ticonf.train_msent_based:
            msent_stream = FListAdapterStream(in_stream, lambda d: [x.preps["ms"] for x in d.sents if x.length<ticonf.train_skip_length and x.length>=ticonf.train_min_length and (len(x.events)>0 or next(_BS_sample_stream)>ticonf.train_skip_noevt_rate)])
            assert not ticonf.train_sent_shuffle
            # use subword size to sort, which should be similar to word size
            b_stream = BatchArranger(msent_stream, batch_size=ticonf.batch_size, maxibatch_size=MBS, batch_size_f=None,
                                     dump_detectors=None, single_detectors=None, sorting_keyer=lambda x: x.subword_size,
                                     shuffling=ticonf.shuffle_train)
        else:
            _sent_counter = _count_train_sents
            b_stream = BatchArranger(in_stream, batch_size=ticonf.batch_size, maxibatch_size=MBS, batch_size_f=_sent_counter,
                                     dump_detectors=None, single_detectors=lambda x: _sent_counter(x)>=ticonf.batch_size,
                                     sorting_keyer=_sent_counter, shuffling=ticonf.shuffle_train)
    else:
        _sent_counter = lambda d: len(d.sents)
        b_stream = BatchArranger(in_stream, batch_size=ticonf.batch_size, maxibatch_size=1, batch_size_f=_sent_counter,
                                 dump_detectors=None, single_detectors=None, sorting_keyer=None, shuffling=False)
    return b_stream

# =====

# manage results
class MyIEResultManager(ResultManager):
    def __init__(self, vpack, outf, goldf, out_format, eval_conf, release_resources):
        self.insts = []
        #
        self.vpack = vpack
        self.outf = outf
        assert goldf is None, "Currently for convenience, assuming that the gold is already in the insts."
        self.goldf = goldf
        self.out_format = out_format
        self.eval_conf = eval_conf
        self.release_resources = release_resources

    def add(self, ones: List[DocInstance]):
        self.insts.extend(ones)
        if self.release_resources:
            for one_doc in ones:
                for one_sent in one_doc.sents:
                    one_sent.extra_features["aux_repr"] = None  # todo(note): special name!

    def _set_type(self, insts: List[DocInstance]):
        for one_doc in insts:
            for one_ef in one_doc.pred_entity_fillers:
                one_ef.type = str(one_ef.type_idx)
            for one_evt in one_doc.pred_events:
                one_evt.type = str(one_evt.type_idx)
                for one_arg in one_evt.links:
                    one_arg.role = str(one_arg.role_idx)

    # write, eval & summary
    def end(self):
        # sorting by idx of reading
        self.insts.sort(key=lambda x: x.inst_idx)
        # todo(+1): write other output file
        # self._set_type(self.insts)  # todo(note): no need for this step
        if self.outf is not None:
            with zopen(self.outf, "w") as fd:
                data_writer = get_data_writer(fd, self.out_format)
                data_writer.write(self.insts)
        # evaluation
        evaler = MyIEEvaler(self.eval_conf)
        res = evaler.eval(self.insts, self.insts)
        # the criterion will be average of U/L-evt/arg; now using only labeled results
        # all_results = [res["event"][0], res["event"][1], res["argument"][0], res["argument"][1]]
        # all_results = [res["event"][1], res["argument"][1]]
        all_results = [res[z][1] for z in self.eval_conf.res_list]
        res["res"] = float(np.average([float(z) for z in all_results]))
        # make it loadable by json
        for k in ["event", "argument", "argument2", "entity_filler"]:
            res[k] = str(res.get(k))
        zlog("zzzzzevent: %s" % res["res"], func="result")
        # =====
        # clear pred ones for possible reusing
        for one_doc in self.insts:
            for one_sent in one_doc.sents:
                one_sent.pred_events.clear()
                one_sent.pred_entity_fillers.clear()
        return res

# testing runner
class MyIETestingRunner(TestingRunner):
    def __init__(self, model, vpack, outf, goldf, out_format, eval_conf, release_resources):
        super().__init__(model)
        self.res_manager = MyIEResultManager(vpack, outf, None, out_format, eval_conf, release_resources)

    # run and record for one batch
    def _run_batch(self, insts):
        res = self.model.inference_on_batch(insts)
        self.res_manager.add(insts)
        return res

    # eval & report
    def _run_end(self):
        x = self.test_recorder.summary()
        res = self.res_manager.end()
        x.update(res)
        Helper.printd(x, sep=" ")
        return x

#
class MyIEDevResult(RecordResult):
    def __init__(self, result_ds: List[Dict]):
        self.all_results = result_ds
        rr = {}
        for idx, vv in enumerate(result_ds):
            rr["zres"+str(idx)] = vv  # record all the results for printing
        # todo(note): use the first dev result
        res = result_ds[0]["res"]
        super().__init__(rr, res)

# training runner
class MyIETrainingRunner(TrainingRunner):
    def __init__(self, rconf, model, vpack, dev_outfs, dev_goldfs, dev_out_format, eval_conf):
        super().__init__(rconf, model)
        self.vpack = vpack
        self.dev_out_format = dev_out_format
        #
        self.dev_goldfs = dev_goldfs
        if isinstance(dev_outfs, (list, tuple)):
            zcheck(len(dev_outfs) == len(dev_goldfs), "Mismatched number of output and gold!")
            self.dev_outfs = dev_outfs
        else:
            self.dev_outfs = [dev_outfs] * len(dev_goldfs)
        self.eval_conf = eval_conf

    def _run_fb(self, annotated_insts, loss_factor: float):
        res = self.model.fb_on_batch(annotated_insts, loss_factor=loss_factor)
        return res

    def _run_train_report(self):
        x = self.train_recorder.summary()
        y = GLOBAL_RECORDER.summary()
        # todo(warn): get loss/tok
        # todo(note): too complex to div here, only accumulating the sums.
        # x["loss_tok"] = x.get("loss_sum", 0.)/x["tok"]
        if len(y) > 0:
            x.update(y)
        Helper.printd(x, " || ")
        return RecordResult(x)

    def _run_update(self, lrate: float, grad_factor: float):
        self.model.update(lrate, grad_factor)

    def _run_validate(self, dev_streams):
        if not isinstance(dev_streams, Iterable):
            dev_streams = [dev_streams]
        dev_results = []
        zcheck(len(dev_streams) == len(self.dev_goldfs), "Mismatch number of streams!")
        dev_idx = 0
        for one_stream, one_dev_outf, one_dev_goldf in zip(dev_streams, self.dev_outfs, self.dev_goldfs):
            # todo(+2): overwrite previous ones?
            rr = MyIETestingRunner(self.model, self.vpack, one_dev_outf+".dev"+str(dev_idx), one_dev_goldf,
                                   self.dev_out_format, self.eval_conf, release_resources=False)
            x = rr.run(one_stream)
            dev_results.append(x)
            dev_idx += 1
        return MyIEDevResult(dev_results)
