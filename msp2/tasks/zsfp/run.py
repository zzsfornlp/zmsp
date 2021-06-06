#

# other running components

from typing import Dict, Callable, Union, List
import os
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.data.vocab import VocabPackage, SimpleVocab, WordVectors
from msp2.data.inst import yield_sents, Sent, SeqField, Doc, InputCharSeqField, set_ee_heads
from msp2.data.stream import Streamer, FWrapperStreamer, CacheStreamer, WrapperStreamer, FListWrapperStreamer, \
    ShuffleStreamer, BatchArranger
from msp2.utils import default_pickle_serializer, default_json_serializer, zlog, Conf, OtherHelper, mkdir_p, Random, wrap_color
from msp2.proc import TrainingRunner, TestingRunner, ZModel, FrameEvaler, \
    MyFNEvaler, MyFNEvalConf, MyPBEvaler, MyPBEvalConf, ResultRecord
from .confs import DConf, TConf, TRConf

# =====
# vocabs

class ZsfpVocabPackage(VocabPackage):
    def __init__(self, vocabs: Dict=None, embeds: Dict=None, dconf: DConf = None):
        super().__init__(vocabs, embeds)
        self.dconf = dconf

    # =====
    # why not save them all at one place? (harder to view?)
    def load(self, prefix="./"):
        fname = prefix + "zsfp.voc.pkl"
        self.vocabs, self.embeds = default_pickle_serializer.from_file(fname)

    def save(self, prefix="./"):
        fname = prefix + "zsfp.voc.pkl"
        mkdir_p(os.path.dirname(fname))  # try to make dir if not there!
        default_pickle_serializer.to_file([self.vocabs, self.embeds], fname)

    # =====
    # loading or building

    @staticmethod
    def build_by_reading(dconf: DConf):
        zlog(f"Load vocabs from vocab dir: {dconf.dict_dir}")
        one = ZsfpVocabPackage({}, {}, dconf)
        one.load(dconf.dict_dir)
        return one

    @staticmethod
    def build_from_stream(dconf: DConf, stream, extra_stream):
        zlog("Build vocabs from streams.")
        # here, collect them all
        # -- basic inputs
        voc_word = SimpleVocab.build_empty("word")
        voc_lemma = SimpleVocab.build_empty("lemma")
        voc_upos = SimpleVocab.build_empty("upos")
        voc_char = SimpleVocab.build_empty("char")
        voc_deplab = SimpleVocab.build_empty("deplab")
        # -- frame ones
        voc_evt, voc_ef, voc_arg = SimpleVocab.build_empty("evt"), SimpleVocab.build_empty("ef"), SimpleVocab.build_empty("arg")
        voc_collections = {"word": voc_word, "lemma": voc_lemma, "upos": voc_upos, "char": voc_char, "deplab": voc_deplab,
                           "evt": voc_evt, "ef": voc_ef, "arg": voc_arg}
        # read all and build
        for sent in yield_sents(stream):
            # -- basic inputs
            if sent.seq_word is not None:
                voc_word.feed_iter(sent.seq_word.vals)
                for w in sent.seq_word.vals:
                    voc_char.feed_iter(w)
            if sent.seq_lemma is not None:
                voc_lemma.feed_iter(sent.seq_lemma.vals)
            if sent.seq_upos is not None:
                voc_upos.feed_iter(sent.seq_upos.vals)
            if sent.tree_dep is not None and sent.tree_dep.seq_label is not None:
                voc_deplab.feed_iter(sent.tree_dep.seq_label.vals)
            # -- frames
            if sent.entity_fillers is not None:
                voc_ef.feed_iter((ef.type for ef in sent.entity_fillers))
            if sent.events is not None:
                voc_evt.feed_iter((evt.type for evt in sent.events))
                for evt in sent.events:
                    if evt.args is not None:
                        voc_arg.feed_iter((arg.role for arg in evt.args))
        # sort everyone!
        for voc in voc_collections.values():
            voc.build_sort()
        # extra for evt/arg
        if dconf.dict_frame_file:
            frames = default_json_serializer.from_file(dconf.dict_frame_file)
            for one_f in frames.values():  # no count, simply feed!!
                if len(one_f["lexUnit"]) > 0:  # todo(+W): currently ignore non-lex frames
                    voc_evt.feed_one(one_f["name"], c=0)
                    for one_fe in one_f["FE"]:
                        voc_arg.feed_one(one_fe["name"], c=0)
            zlog(f"After adding frames from {dconf.dict_frame_file}, evt={voc_evt}, arg={voc_arg}")
        # -----
        # deal with pre-trained word embeddings
        w2vec = None
        if dconf.pretrain_wv_file:
            # todo(warn): for convenience, extra vocab (usually dev&test) is only used for collecting pre-train vecs
            # collect extra words and lemmas
            extra_word_counts = {}
            extra_lemma_counts = {}
            for sent in yield_sents(extra_stream):
                if sent.seq_word is not None:
                    for w in sent.seq_word.vals:
                        extra_word_counts[w] = extra_word_counts.get(w, 0) + 1
                if sent.seq_lemma is not None:
                    for w in sent.seq_lemma.vals:
                        extra_lemma_counts[w] = extra_lemma_counts.get(w, 0) + 1
            # must provide dconf.pretrain_file
            w2vec = WordVectors.load(dconf.pretrain_wv_file)
            # first filter according to thresholds
            _filter_f = lambda ww, rank, val: (val >= dconf.word_fthres and rank <= dconf.word_rthres) or \
                                              w2vec.find_key(ww) is not None
            voc_word.build_filter(_filter_f)
            voc_lemma.build_filter(_filter_f)
            # then add extra ones
            for w in sorted(extra_word_counts.keys(), key=lambda z: (-extra_word_counts[z], z)):
                if w2vec.find_key(w) is not None and (w not in voc_word):
                    voc_word.feed_one(w)
            for w in sorted(extra_lemma_counts.keys(), key=lambda z: (-extra_lemma_counts[z], z)):
                if w2vec.find_key(w) is not None and (w not in voc_lemma):
                    voc_lemma.feed_one(w)
            # by-product of filtered output pre-trained embeddings for later faster processing
            if dconf.pretrain_hits_outf:
                # find all keys again!!
                w2vec.clear_hits()
                for vv in [voc_word, voc_lemma]:
                    for _idx in range(*(vv.non_special_range())):
                        w2vec.find_key(vv.idx2word(_idx))
                w2vec.save_hits(dconf.pretrain_hits_outf)
            # embeds
            word_embed1 = voc_word.filter_embed(w2vec, scale=dconf.pretrain_scale)
            lemma_embed1 = voc_lemma.filter_embed(w2vec, scale=dconf.pretrain_scale)
        else:
            voc_word.build_filter_thresh(rthres=dconf.word_rthres, fthres=dconf.word_fthres)
            voc_lemma.build_filter_thresh(rthres=dconf.word_rthres, fthres=dconf.word_fthres)
            word_embed1 = lemma_embed1 = None
        # return
        ret = ZsfpVocabPackage(voc_collections, {"word": word_embed1, "lemma": lemma_embed1}, dconf)
        return ret

# =====
# stream data

# my reading (wrapper!)
class MyDataReaderConf(Conf):
    def __init__(self):
        self.br = ReaderGetterConf()  # basic reader
        # extra functions
        self.wl_use_lc = False  # use lower case words and lemmas?
        self.deplab_use_label0 = True  # using only first-level ud label
        self.sent_loss_weight_non = 1.0  # sent level loss_weight_non, default is 1.
        self.assume_frame_lu = False  # special mode, assume that we have the input frame's LUs
        self.set_ee_heads = True  # assign heads for evt and args (by default simply assign them!!)
        self.do_unlabeled_arg = False  # simply give all args the same role (as unlabeled system!)
        # --

    def get_reader(self, input_path: str, _clone=True, **kwargs):  # directly get a reader
        conf = MyDataReaderConf.direct_conf(self.copy() if _clone else self, **kwargs)
        # --
        assert conf.br.input_path == "", "Input path not specified there!"
        basic_reader = conf.br.get_reader(input_path=input_path)  # basic reader
        wrapped_reader = _MyDataReaderWrapper(basic_reader, conf)  # wrapped reader
        return wrapped_reader

class _MyDataReaderWrapper(FWrapperStreamer):
    def __init__(self, base_streamer: Streamer, conf: MyDataReaderConf):
        super().__init__(base_streamer, self._f, inplaced=True)
        self.conf = conf
        # --
        from .extract.constrainer import FN2UD_POS_MAP
        self._FN_POS_MAP = FN2UD_POS_MAP.copy()
        # --

    # go
    def _f(self, inst):
        conf: MyDataReaderConf = self.conf
        wl_use_lc, deplab_use_label0, sent_loss_weight_non, assume_frame_lu = \
            conf.wl_use_lc, conf.deplab_use_label0, conf.sent_loss_weight_non, conf.assume_frame_lu
        for sent in yield_sents([inst]):
            if wl_use_lc:
                if sent.seq_word is not None:
                    sent.seq_word.set_vals([w.lower() for w in sent.seq_word.vals])
                if sent.seq_lemma is not None:
                    sent.seq_lemma.set_vals([w.lower() for w in sent.seq_lemma.vals])
            if deplab_use_label0:
                sent_tree = sent.tree_dep
                if sent_tree is not None and sent_tree.seq_label is not None:  # use first-level label!!
                    sent_tree.seq_label.set_vals([s.split(":")[0] for s in sent_tree.seq_label.vals])
            if sent_loss_weight_non != 1.0:  # set the special property
                sent._loss_weight_non = sent_loss_weight_non
            if assume_frame_lu:  # special assumption!!
                for evt in sent.events:
                    lu_lemma, lu_pos = evt.info.get("luName").split(".")
                    assign_idx = evt.mention.wridx-1  # note: simply put at the ending token!
                    sent.seq_lemma.vals[assign_idx] = lu_lemma
                    if lu_pos in self._FN_POS_MAP:
                        sent.seq_upos.vals[assign_idx] = self._FN_POS_MAP[lu_pos]
            if conf.set_ee_heads:  # set head using dep-parse tree
                sent_tree = sent.tree_dep
                if sent_tree is not None:
                    set_ee_heads([sent])
            if conf.do_unlabeled_arg:
                for evt in sent.events:
                    for arg in list(evt.args):  # note: simply remove "V" and "C-V" here!!
                        if arg.role in ["V", "C-V"]:
                            arg.delete_self()
                        arg.role = "ZARG"  # just a placeholder!
        # --

#
class IndexerHelper:
    def __init__(self, vpack: ZsfpVocabPackage):
        # basic
        self.voc_word = vpack.get_voc("word")
        self.voc_lemma = vpack.get_voc("lemma")
        self.voc_upos = vpack.get_voc("upos")
        self.voc_char = vpack.get_voc("char")
        self.voc_deplab = vpack.get_voc("deplab")
        # frame
        self.voc_evt = vpack.get_voc('evt')
        self.voc_ef = vpack.get_voc('ef')
        self.voc_arg = vpack.get_voc('arg')

    def index_seq(self, seq: SeqField, voc: SimpleVocab, allow_unk: bool):
        if seq is not None:
            seq_idxes = [voc.get_else_unk(z) for z in seq.vals] if allow_unk else [voc[z] for z in seq.vals]
            seq.set_idxes(seq_idxes)

    def index_items(self, items, voc: SimpleVocab, allow_unk: bool):
        for item in items:
            item.set_label_idx(voc.get_else_unk(item.label) if allow_unk else voc[item.label])

    def index_char_seq(self, seq: InputCharSeqField, voc: SimpleVocab, allow_unk: bool):
        if seq is not None:
            seq_idxes = [[voc.get_else_unk(c) for c in z] for z in seq.vals] if allow_unk else [[voc[c] for c in z] for z in seq.vals]
            seq.set_idxes(seq_idxes)

    def index_sent(self, sent: Sent):
        # basic
        self.index_seq(sent.seq_word, self.voc_word, True)
        self.index_seq(sent.seq_lemma, self.voc_lemma, True)
        self.index_seq(sent.seq_upos, self.voc_upos, True)
        self.index_char_seq(sent.seq_word.get_char_seq(), self.voc_char, True)  # note: char needs special building
        if sent.tree_dep is not None:
            self.index_seq(sent.tree_dep.seq_label, self.voc_deplab, True)
        # frames
        if sent.entity_fillers is not None:
            self.index_items(sent.entity_fillers, self.voc_ef, True)
        if sent.events is not None:
            self.index_items(sent.events, self.voc_evt, True)
            self.index_items((a for evt in sent.events for a in evt.args), self.voc_arg, True)

# for indexing instances
class IndexerStreamer(FWrapperStreamer):
    def __init__(self, in_stream: Streamer, vpack: ZsfpVocabPackage, inst_preparer: Callable):
        super().__init__(in_stream, self._go_index, False)
        #
        self.index_helper = IndexerHelper(vpack=vpack)
        self.inst_preparer = inst_preparer

    def _go_index(self, inst: Union[Doc, Sent]):
        for sent in yield_sents([inst]):  # make it iterable
            self.index_helper.index_sent(sent)
        # set read_idx!!
        inst.set_read_idx(self.count())
        # inst_preparer
        if self.inst_preparer is not None:
            inst = self.inst_preparer(inst)
        # in fact, inplaced if not wrapping model specific preparer
        return inst

# function to get IndexerStreamer (before possibly join_stream)
def index_stream(in_stream: Streamer, vpack: ZsfpVocabPackage, use_cache: bool, cache_shuffle_times: int, inst_preparer: Callable):
    i_stream = IndexerStreamer(in_stream, vpack, inst_preparer)
    c_stream = i_stream  # cache?
    if use_cache:
        c_stream = CacheStreamer(i_stream, shuffle_times=cache_shuffle_times)
    return c_stream

# especially for training
def train_prep_stream(in_stream: Streamer, tconf: TConf):
    # for training, we get all the sentences!
    assert tconf.train_stream_mode == "sent", "Currently we only support sent training!"
    sent_stream = FListWrapperStreamer(
        in_stream, lambda d: [x for x in yield_sents([d]) if len(x) <= tconf.train_max_length and len(x) >= tconf.train_min_length and (len(x.events) > 0 or next(_BS_sample_stream) > tconf.train_skip_noevt_rate)])  # filter out certain sents!
    if tconf.train_stream_reshuffle_times > 0:  # reshuffle for sents
        sent_stream = ShuffleStreamer(sent_stream, shuffle_bsize=tconf.train_stream_reshuffle_bsize,
                                      shuffle_times=tconf.train_stream_reshuffle_times)
    return sent_stream

# function to get BatchArranger
_BK_gen = Random.get_generator("train")
_BS_sample_stream = Random.stream(_BK_gen.random_sample)
def batch_stream(in_stream: Streamer, tconf: TConf, training: bool):
    _sent_counter = lambda d: len(list(yield_sents([d])))
    _tok_counter = lambda d: sum(len(s) for s in yield_sents([d]))
    _frame_counter = lambda d: sum(len(s.events) for s in yield_sents([d]))
    _ftok_counter = lambda d: sum(max(1, len(s.events))*len(s) for s in yield_sents([d]))
    batch_size_f_map = {"sent": _sent_counter, "tok": _tok_counter, "frame": _frame_counter, "ftok": _ftok_counter}
    if training:
        batch_size_f = batch_size_f_map[tconf.train_count_mode]
        b_stream = BatchArranger(in_stream, bsize=tconf.train_batch_size, maxi_bsize=tconf.train_maxibatch_size,
                                 batch_size_f=batch_size_f, dump_detectors=None, single_detectors=None, sorting_keyer=lambda x: len(x),
                                 shuffle_batches_times=tconf.train_batch_shuffle_times)
    else:
        batch_size_f = batch_size_f_map[tconf.test_count_mode]
        b_stream = BatchArranger(in_stream, bsize=tconf.test_batch_size, maxi_bsize=1, batch_size_f=batch_size_f,
                                 dump_detectors=None, single_detectors=None, sorting_keyer=None, shuffle_batches_times=0)
    return b_stream, batch_size_f

# =====
# runners for train & test

class ZsfpTestingRunner(TestingRunner):
    def __init__(self, model: ZModel, test_stream: Streamer, conf, outf: str, goldf: str):
        super().__init__(model, test_stream)
        # --
        self.all_insts = []  # tmp store for all predictions
        self.conf = conf
        self.outf, self.goldf = outf, goldf
        # --
        self._evaler = None

    def get_evaler(self, reset=True):
        if self._evaler is None:
            eval_conf = self.conf.dconf.eval_conf
            if isinstance(eval_conf, MyFNEvalConf):  # todo(+N): ugly!
                evaler = MyFNEvaler(eval_conf)
            elif isinstance(eval_conf, MyPBEvalConf):
                evaler = MyPBEvaler(eval_conf)
            else:
                evaler = FrameEvaler(eval_conf)
            self._evaler = evaler
        # --
        if reset:
            self._evaler.reset()  # clear history!
        return self._evaler

    # end the testing & eval & return sumarized results
    def _run_end(self):
        x = self.test_recorder.summary()
        zlog(f"Test-Info: {OtherHelper.printd_str(x, sep=' ')}")
        res = self._go_end()  # use eval results instead!
        res.results['info'] = x  # add the infos
        self.test_recorder.reset()  # reset recorder!
        return res

    # run and record for one batch
    def _run_batch(self, insts: List):
        info = self.model.predict_on_batch(insts)
        self._go_add(insts)
        return info

    # =====
    # after each batch
    def _go_add(self, insts):
        self.all_insts.extend(insts)

    # write, eval & summary
    def _go_end(self):
        dconf: DConf = self.conf.dconf
        tconf: TConf = self.conf.tconf
        # --
        # sorting by idx of reading
        self.all_insts.sort(key=lambda x: x.read_idx)
        # write results first
        if self.outf:
            with dconf.W.get_writer(output_path=self.outf) as writer:
                writer.write_insts(self.all_insts)
        # eval
        gold_insts = list(dconf.R.get_reader(input_path=self.goldf))  # read golds again
        res = self.get_evaler().eval(gold_insts, self.all_insts)
        res_summary = res.get_summary()
        res_brief_str = res.get_brief_str()
        res_detailed_str = "\n".join(["\t"+s for s in res.get_detailed_str().split("\n")])
        res_result = res.get_result()
        res_summary.update({"fpred": self.outf, "fgold": self.goldf})  # add file info
        zlog(f"{wrap_color('Building Detailed Results')} ({self.outf} vs {self.goldf}):\n{res_detailed_str}", func="result")
        zlog(f"{wrap_color('Building Brief Results')}: {res_brief_str}", func="result")
        # zlog(f"zzzzzresultsummary: {res_summary}", func="result")
        zlog(f"zzzzzresult: {res_result}", func="result")
        # remember to clear!
        self.all_insts.clear()
        # --
        return ResultRecord(results=res_summary, description=res_brief_str, score=res.get_result())

class ZsfpTrainingRunner(TrainingRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def create(model: ZModel, train_stream: Streamer, train_batch_f: Callable, conf,
               dev_streams: List[Streamer], dev_outfs: List[str], dev_goldfs: List[str]):
        # make dev-runners
        dev_runners = []
        for dev_stream, dev_outf, dev_goldf in zip(dev_streams, dev_outfs, dev_goldfs):
            one_dev_runner = ZsfpTestingRunner(model, dev_stream, conf, dev_outf, dev_goldf)
            dev_runners.append(one_dev_runner)
        # make TrainingRunner
        discard_batch_f = lambda xs: sum(len(z.events) for z in xs)<=0 if conf.tconf.train_skip_batch_noevt else None
        runner = ZsfpTrainingRunner(conf.tconf.tr_conf, model, train_stream, train_batch_f, discard_batch_f, dev_runners)
        return runner
