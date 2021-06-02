#

# some processing tools on the data stream

from typing import List, Iterable, Dict
from collections import OrderedDict
from copy import deepcopy

from msp.utils import zlog, zwarn, zopen, GLOBAL_RECORDER, Helper, zcheck
from msp.data import FAdapterStreamer, BatchArranger, InstCacher
from msp.zext.process_train import TrainingRunner, RecordResult
from msp.zext.process_test import TestingRunner, ResultManager
from msp.zext.dpar import ParserEvaler
from msp.data import WordNormer

from ..data.insts import GeneralSentence
from ..data.io import BaseDataWriter, BaseDataReader, ConlluParseReader, PlainTextReader, ConllNerReader
from .vocab import MLMVocabPackage, UD2_POS_UNK_MAP

from .seqeval import get_prf_scores

# =====
# preparing data

# pre-processing (lower_case, norm_digit)
class PreprocessStreamer(FAdapterStreamer):
    def __init__(self, in_stream, lower_case: bool, norm_digit: bool):
        super().__init__(in_stream, self._go_prep, True)
        self.normer = WordNormer(lower_case, norm_digit)

    def _go_prep(self, inst: GeneralSentence):
        inst.word_seq.reset(self.normer.norm_stream(inst.word_seq.vals))

# for indexing instances
class IndexerStreamer(FAdapterStreamer):
    def __init__(self, in_stream, vpack: MLMVocabPackage, inst_preparer, backoff_pos_idx: int):
        super().__init__(in_stream, self._go_index, False)
        # -----
        self.w_vocab = vpack.get_voc("word")
        self.w2_vocab = vpack.get_voc("word2")  # extra set
        self.c_vocab = vpack.get_voc("char")
        self.p_vocab = vpack.get_voc("pos")
        self.l_vocab = vpack.get_voc("deplabel")
        self.n_vocab = vpack.get_voc("ner")
        self.inst_preparer = inst_preparer
        self.backoff_pos_idx = backoff_pos_idx

    def _go_index(self, inst: GeneralSentence):
        # word
        # todo(warn): remember to norm word; replace singleton at model's input, not here
        if inst.word_seq.has_vals():
            w_voc = self.w_vocab
            word_idxes = [w_voc.get_else_unk(w) for w in inst.word_seq.vals]  # todo(note): currently unk-idx is large
            if self.backoff_pos_idx >= 0:  # when using this mode, there must be backoff strings in word vocab
                zwarn("The option of 'backoff_pos_idx' is deprecated, do not mix things in this way!")
                word_backoff_idxes = [w_voc[UD2_POS_UNK_MAP[z]] for z in inst.pos_seq.vals]
                word_idxes = [(zb if z>=self.backoff_pos_idx else z) for z, zb in zip(word_idxes, word_backoff_idxes)]
            inst.word_seq.set_idxes(word_idxes)
        # =====
        # aug extra word set!!
        if self.w2_vocab is not None:
            w2_voc = self.w2_vocab
            inst.word2_seq = deepcopy(inst.word_seq)
            word_idxes = [w2_voc.get_else_unk(w) for w in inst.word2_seq.vals]  # todo(note): currently unk-idx is large
            inst.word2_seq.set_idxes(word_idxes)
        # =====
        # others
        char_seq, pos_seq, dep_tree, ner_seq = [getattr(inst, z, None) for z in ["char_seq", "pos_seq", "dep_tree", "ner_seq"]]
        if char_seq is not None and char_seq.has_vals():
            char_seq.build_idxes(self.c_vocab)
        if pos_seq is not None and pos_seq.has_vals():
            pos_seq.build_idxes(self.p_vocab)
        if dep_tree is not None and dep_tree.has_vals():
            dep_tree.build_label_idxes(self.l_vocab)
        if ner_seq is not None and ner_seq.has_vals():
            ner_seq.build_idxes(self.n_vocab)
        if self.inst_preparer is not None:
            inst = self.inst_preparer(inst)
        # in fact, inplaced if not wrapping model specific preparer
        return inst

#
def index_stream(in_stream, vpack, cached, cache_shuffle, inst_preparer, backoff_pos_idx):
    i_stream = IndexerStreamer(in_stream, vpack, inst_preparer, backoff_pos_idx)
    if cached:
        return InstCacher(i_stream, shuffle=cache_shuffle)
    else:
        return i_stream

# for arrange batches
def batch_stream(in_stream, batch_size, ticonf, training):
    MIN_SENT_LEN_FOR_BSIZE = 10
    if training:
        batch_size_f = (lambda x: max(len(x), MIN_SENT_LEN_FOR_BSIZE)) if ticonf.train_batch_on_len else None
        b_stream = BatchArranger(in_stream, batch_size=batch_size, maxibatch_size=ticonf.train_maxibatch_size, batch_size_f=batch_size_f,
                                 dump_detectors=lambda one: len(one)>=ticonf.train_max_length or len(one)<ticonf.train_min_length,
                                 single_detectors=None, sorting_keyer=len, shuffling=ticonf.train_shuffle)
    else:
        batch_size_f = (lambda x: max(len(x), MIN_SENT_LEN_FOR_BSIZE)) if ticonf.test_batch_on_len else None
        b_stream = BatchArranger(in_stream, batch_size=batch_size, maxibatch_size=ticonf.test_maxibatch_size, batch_size_f=batch_size_f,
                                 dump_detectors=None, single_detectors=lambda one: len(one)>=ticonf.test_single_length,
                                 sorting_keyer=len, shuffling=False)
    return b_stream

# =====
# data io

def get_data_reader(file_or_fd, input_format, **kwargs):
    if input_format == "conllu":
        r = ConlluParseReader(file_or_fd, **kwargs)
    elif input_format == "ner":
        r = ConllNerReader(file_or_fd, **kwargs)
    elif input_format == "json":
        r = BaseDataReader(GeneralSentence, file_or_fd)
    elif input_format == "plain":
        r = PlainTextReader(file_or_fd, **kwargs)
    else:
        raise NotImplementedError(f"Unknown input_format {input_format}")
    return r

def get_data_writer(file_or_fd, output_format, **kwargs):
    if output_format == "json":
        return BaseDataWriter(file_or_fd, **kwargs)
    else:
        raise NotImplementedError(f"Unknown output_format {output_format}")

# =====
# runner

class MltResultMannager(ResultManager):
    def __init__(self, vpack, outf, goldf, out_format):
        self.insts = []
        #
        self.vpack = vpack
        self.outf = outf
        self.goldf = goldf  # todo(note): goldf is only for reporting which file to compare?
        self.out_format = out_format

    def add(self, ones: List[GeneralSentence]):
        self.insts.extend(ones)

    # write, eval & summary
    def end(self):
        # sorting by idx of reading
        self.insts.sort(key=lambda x: x.sid)
        # todo(+1): write other output file
        if self.outf is not None:
            with zopen(self.outf, "w") as fd:
                data_writer = get_data_writer(fd, self.out_format)
                data_writer.write_list(self.insts)
        # eval for parsing & pos & ner
        evaler = ParserEvaler()
        ner_golds, ner_preds = [], []
        has_syntax, has_ner = False, False
        for one_inst in self.insts:
            gold_pos_seq = getattr(one_inst, "pos_seq", None)
            pred_pos_seq = getattr(one_inst, "pred_pos_seq", None)
            gold_ner_seq = getattr(one_inst, "ner_seq", None)
            pred_ner_seq = getattr(one_inst, "pred_ner_seq", None)
            gold_tree = getattr(one_inst, "dep_tree", None)
            pred_tree = getattr(one_inst, "pred_dep_tree", None)
            if gold_pos_seq is not None and gold_tree is not None:
                # todo(+N): only ARTI_ROOT in trees
                gold_pos, gold_heads, gold_labels = gold_pos_seq.vals, gold_tree.heads[1:], gold_tree.labels[1:]
                pred_pos = pred_pos_seq.vals if (pred_pos_seq is not None) else [""]*len(gold_pos)
                pred_heads, pred_labels = (pred_tree.heads[1:], pred_tree.labels[1:]) if (pred_tree is not None) \
                    else ([-1]*len(gold_heads), [""]*len(gold_labels))
                evaler.eval_one(gold_pos, gold_heads, gold_labels, pred_pos, pred_heads, pred_labels)
                has_syntax = True
            if gold_ner_seq is not None and pred_ner_seq is not None:
                ner_golds.append(gold_ner_seq.vals)
                ner_preds.append(pred_ner_seq.vals)
                has_ner = True
        # -----
        zlog("Results of %s vs. %s" % (self.outf, self.goldf), func="result")
        if has_syntax:
            report_str, res = evaler.summary()
            zlog(report_str, func="result")
        else:
            res = {}
        if has_ner:
            for _n, _v in zip("prf", get_prf_scores(ner_golds, ner_preds)):
                res[f"ner_{_n}"] = _v
        res["gold"] = self.goldf  # record which file
        zlog("zzzzztest: testing result is " + str(res))
        # note that res['res'] will not be used
        if 'res' in res:
            del res['res']
        # -----
        return res

# testing runner
class MltTestingRunner(TestingRunner):
    def __init__(self, model, vpack, outf, goldf, out_format):
        super().__init__(model)
        self.res_manager = MltResultMannager(vpack, outf, goldf, out_format)

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
        MltDevResult.calc_acc(x)
        Helper.printd(x, sep=" || ")
        return x

#
class MltDevResult(RecordResult):
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
            res = result_ds[0]["res"]
        super().__init__(rr, res)

    @staticmethod
    def calc_acc(x):
        # put results for MLM/ORP
        addings = OrderedDict()
        for comp_name, comp_loss_names in zip(["mlm", "plm", "orp", "dp", "pos", "ner"],
                                              [["word", "pos"], ["l2r", "r2l"], ["d-1", "d-2"], ["head", "label"],
                                               ["slab"], ["slab", "crf"]]):
            for c in comp_loss_names:
                addings[f"res_{c}.acc"] = x.get(f"loss:{comp_name}.{c}_corr", 0.) / (x.get(f"loss:{comp_name}.{c}_count", 0.) + 1e-5)
                addings[f"res_{c}.nll"] = x.get(f"loss:{comp_name}.{c}_sum", 0.) / (x.get(f"loss:{comp_name}.{c}_count", 0.) + 1e-5)
        for k,v in addings.items():
            if v>0.:  # only care active ones
                x[k] = v
        # -----
        # set "res"
        res_cands = [-addings["res_word.nll"], -addings["res_l2r.nll"], -addings["res_r2l.nll"],
                     -addings["res_d-1.nll"], -addings["res_d-2.nll"], x.get("tok_las", 0.),
                     addings["res_slab.acc"], x.get("ner_f", 0.)]
        x["res"] = sum(z for z in res_cands if z!=0.)

# training runner
class MltTrainingRunner(TrainingRunner):
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
        if len(y) > 0:
            x.update(y)
        MltDevResult.calc_acc(x)
        Helper.printd(x, " || ")
        return RecordResult(x, score=x.get("res", 0.))

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
            rr = MltTestingRunner(self.model, self.vpack, one_dev_outf+".dev"+str(dev_idx), one_dev_goldf, self.dev_out_format)
            x = rr.run(one_stream)
            dev_results.append(x)
            dev_idx += 1
        return MltDevResult(dev_results)
