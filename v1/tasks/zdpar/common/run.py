#

#
from typing import List, Iterable, Dict

from msp.utils import zlog, zopen, GLOBAL_RECORDER, Helper, zcheck
from msp.data import FAdapterStreamer, BatchArranger, InstCacher
from msp.zext.process_train import TrainingRunner, RecordResult
from msp.zext.process_test import TestingRunner, ResultManager
from msp.zext.dpar import ParserEvaler

from .data import ParseInstance, get_data_writer
from .vocab import ParserVocabPackage

# for indexing instances
class IndexerStreamer(FAdapterStreamer):
    def __init__(self, in_stream, vpack: ParserVocabPackage, inst_preparer):
        super().__init__(in_stream, self._go_index, False)
        #
        self.word_normer = vpack.word_normer
        self.w_vocab = vpack.get_voc("word")
        self.c_vocab = vpack.get_voc("char")
        self.p_vocab = vpack.get_voc("pos")
        self.l_vocab = vpack.get_voc("label")
        self.inst_preparer = inst_preparer

    def _go_index(self, inst: ParseInstance):
        # word
        # todo(warn): remember to norm word; replace singleton at model's input, not here
        if inst.words.has_vals():
            w_voc = self.w_vocab
            inst.words.set_idxes([w_voc.get_else_unk(w) for w in self.word_normer.norm_stream(inst.words.vals)])
        # others
        if inst.chars.has_vals():
            inst.chars.build_idxes(self.c_vocab)
        if inst.poses.has_vals():
            inst.poses.build_idxes(self.p_vocab)
        if inst.labels.has_vals():
            inst.labels.build_idxes(self.l_vocab)
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

# for arrange batches
def batch_stream(in_stream, ticonf, training):
    if training:
        b_stream = BatchArranger(in_stream, batch_size=ticonf.batch_size, maxibatch_size=20, batch_size_f=None,
                                 dump_detectors=lambda one: len(one)>=ticonf.train_skip_length or len(one)<ticonf.train_min_length,
                                 single_detectors=None, sorting_keyer=len, shuffling=ticonf.shuffle_train)
    else:
        b_stream = BatchArranger(in_stream, batch_size=ticonf.batch_size, maxibatch_size=-1, batch_size_f=None,
                                 dump_detectors=None, single_detectors=lambda one: len(one)>=ticonf.infer_single_length,
                                 sorting_keyer=len, shuffling=False)
    return b_stream

# =====
# decoding results
class ParserResultMannager(ResultManager):
    def __init__(self, vpack, outf, goldf, out_format):
        self.insts = []
        #
        self.vpack = vpack
        self.outf = outf
        self.goldf = goldf  # todo(note): goldf is only for reporting which file to compare?
        self.out_format = out_format

    def add(self, ones: List[ParseInstance]):
        self.insts.extend(ones)

    # write, eval & summary
    def end(self):
        # sorting by idx of reading
        self.insts.sort(key=lambda x: x.inst_idx)
        # todo(+1): write other output file
        if self.outf is not None:
            with zopen(self.outf, "w") as fd:
                data_writer = get_data_writer(fd, self.out_format)
                data_writer.write(self.insts)
        #
        evaler = ParserEvaler()
        # evaler2 = ParserEvaler(ignore_punct=True, punct_set={"PUNCT", "SYM"})
        eval_arg_names = ["poses", "heads", "labels", "pred_poses", "pred_heads", "pred_labels"]
        for one_inst in self.insts:
            # todo(warn): exclude the ROOT symbol; the model should assign pred_*
            real_values = one_inst.get_real_values_select(eval_arg_names)
            evaler.eval_one(*real_values)
            # evaler2.eval_one(*real_values)
        report_str, res = evaler.summary()
        # _, res2 = evaler2.summary()
        #
        zlog("Results of %s vs. %s" % (self.outf, self.goldf), func="result")
        zlog(report_str, func="result")
        res["gold"] = self.goldf            # record which file
        # res2["gold"] = self.goldf            # record which file
        zlog("zzzzztest: testing result is " + str(res))
        # zlog("zzzzztest2: testing result is " + str(res2))
        zlog("zzzzzpar: %s" % res["res"], func="result")
        return res

# testing runner
class ParserTestingRunner(TestingRunner):
    def __init__(self, model, vpack, outf, goldf, out_format):
        super().__init__(model)
        self.res_manager = ParserResultMannager(vpack, outf, goldf, out_format)

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
class ParsingDevResult(RecordResult):
    def __init__(self, result_ds: List[Dict]):
        self.all_results = result_ds
        rr = {}
        for idx, vv in enumerate(result_ds):
            rr["zres"+str(idx)] = vv                        # record all the results for printing
        # res = 0. if (len(result_ds)==0) else result_ds[0]["res"]        # but only use the first one as DEV-res
        # todo(note): use the first dev result, and use UAS if res<=0.
        if len(result_ds) == 0:
            res = 0.
        else:
            # todo(note): the situation of totally unlabeled parsing
            if result_ds[0]["res"] <= 0.:
                res = result_ds[0]["tok_uas"]
            else:
                res = result_ds[0]["res"]
        super().__init__(rr, res)

# training runner
class ParserTrainingRunner(TrainingRunner):
    def __init__(self, rconf, model, vpack, dev_outfs, dev_goldfs, dev_out_format):
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

    # to be implemented
    def _run_fb(self, annotated_insts, loss_factor: float):
        res = self.model.fb_on_batch(annotated_insts, loss_factor=loss_factor)
        return res

    def _run_train_report(self):
        x = self.train_recorder.summary()
        y = GLOBAL_RECORDER.summary()
        # todo(warn): get loss/tok
        x["loss_tok"] = x.get("loss_sum", 0.)/x["tok"]
        if len(y) > 0:
            x.update(y)
        # zlog(x, "report")
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
            rr = ParserTestingRunner(self.model, self.vpack, one_dev_outf+".dev"+str(dev_idx), one_dev_goldf, self.dev_out_format)
            x = rr.run(one_stream)
            dev_results.append(x)
            dev_idx += 1
        return ParsingDevResult(dev_results)
