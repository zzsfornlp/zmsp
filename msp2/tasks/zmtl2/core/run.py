#

# input batch of input_item (msent) => run_time data
# -- together with task and other info (mixin of data + task, but do not need to know too much details)
"""
General modes:
1. context: no-sep, seg-all0/seg-center1, has-center
2. pairwise: has-sep, seg-01, no-center: mode0 must input doc with two sents, mode1 similar to 'context'
3. paragraph: no-sep, seg-all0, no-center
InputItem from different perspectives:
For input: [cls] sent0 ... ([sep]) sent1 ... ([sep]) ... sentN [sep]
For output: [cls] sent0 ... sent1 ... sentN
"""

__all__ = [
    "InputSeqInfo", "InputItem", "InputBatch",
    "ZIConverter", "ZIConverterConf", "ZIBatcher", "ZIBatcherConf"
]

from typing import List
from collections import OrderedDict
import numpy as np
from msp2.data.inst import Doc, Sent, Frame, yield_sents, yield_frames, InputSubwordSeqField, DataPadder
from msp2.data.stream import IterStreamer, BatchArranger
from msp2.utils import Conf, Random, Constants
from msp2.nn import BK

# --
# run_time input item

class InputSeqInfo:
    def __init__(self, enc_input_ids: List, enc_input_segids: List, dec_sel_idxes: List, dec_sel_lens: List, dec_offsets: List):
        self.enc_input_ids = enc_input_ids  # List(len_enc)[int] for enc's input
        self.enc_input_segids = enc_input_segids  # List(len_enc)[int] for enc's segid
        self.dec_sel_idxes = dec_sel_idxes  # List(len_dec)[int], select from enc's seq to dec's seq
        self.dec_sel_lens = dec_sel_lens  # List(len_dec)[int], number of subtoks for the full words
        self.dec_offsets = dec_offsets  # List(len(sents))[int], offset of each sent's start token in dec's seq

    @classmethod
    def create_from_subtoks(cls, item: 'InputItem', subs: List[InputSubwordSeqField], IDX_CLS: int, IDX_SEP: int):
        _enc_input_ids, _enc_input_segids, _dec_sel_idxes, _dec_sel_lens, _dec_offsets = [IDX_CLS], [0], [0], [1], []
        _cur_tok_offset = 1
        _cur_sub_offset = 1
        for sidx, sent in enumerate(item.sents):
            sub_idxes = subs[sidx].idxes
            if item.add_seps[sidx]:  # add here!!
                sub_idxes = sub_idxes + [IDX_SEP]
            _enc_input_ids.extend(sub_idxes)
            _enc_input_segids.extend([item.seg_ids[sidx]] * len(sub_idxes))
            # --
            # for enc->dec, simply use the begin one (remember to add offset!!)
            _dec_sel_idxes.extend([(_cur_sub_offset+z) for z in subs[sidx].align_info.orig2begin])
            _dec_sel_lens.extend(subs[sidx].align_info.split_sizes)  # add each one's sizes
            _cur_sub_offset += len(sub_idxes)
            # --
            _dec_offsets.append(_cur_tok_offset)
            _cur_tok_offset += len(sent)
        # --
        ret = InputSeqInfo(_enc_input_ids, _enc_input_segids, _dec_sel_idxes, _dec_sel_lens, _dec_offsets)
        return ret

class InputItem:
    def __init__(self):
        # data
        self.sents: List[Sent] = []  # (continuous) sentences
        self.inst = None  # original instance, like Doc or Sent or Frame or ...
        # settings
        self.add_seps: List[int] = []  # where to add [sep] (after each sent.), must add [cls] and final [sep]
        self.seg_ids: List[int] = []  # each sent get one id
        # -- note: if sidx!=None, then both loss and pred are restricted there!
        self.center_sidx: int = None  # center sent idx (inside local list) if there are, None means nope
        # run_time info (usually with prep_item)
        self.seq_info: InputSeqInfo = None  # the very basic seq info for enc/dec, usually assigned by Encoder
        self.info = {}  # extra info
        self._batch_len = None  # length used for batching

    def __repr__(self):
        return f"Item(L={len(self.sents)}): {self.sents}"

    @classmethod
    def create(cls, sents: List[Sent], inst=None, add_seps=None, seg_ids=None, center_sidx=None, **extra_info):
        ret = InputItem()
        ret.sents = sents
        ret.inst = inst
        # --
        _nsent = len(sents)
        if add_seps is None:  # by default no seg
            add_seps = [0] * _nsent
        add_seps[-1] = 1  # but the last one surely needs it!!
        ret.add_seps = add_seps
        if seg_ids is None:  # by default all zero
            seg_ids = [0] * _nsent
        ret.seg_ids = seg_ids
        ret.center_sidx = center_sidx
        ret.info.update(extra_info)
        return ret

    def assign_batch_len(self, length: int):
        self._batch_len = length

    # item len, utilized when batching!!
    def __len__(self):
        if self._batch_len is not None:
            return self._batch_len
        # we have a default one!
        return 1+sum(map(len, self.sents))  # here we use dec's length (since easier to obtain)!!

    @property
    def center_sent(self):
        return None if self.center_sidx is None else self.sents[self.center_sidx]

# --
# from insts to items

class ZIConverterConf(Conf):
    def __init__(self):
        self.input_strategy = "none"  # none/sent/frame
        self.convert_strategy = "context"  # context/pairwise0or1/paragraph
        # -- context mode
        self.left_extend_nsent = 0  # number of sent to extend to the left
        self.right_extend_nsent = 0  # number of sent to extend to the right
        self.extend_word_budget = Constants.INT_PRAC_MAX  # overall budget
        self.center_special_id = False  # use seg=1 for center sentence?

class ZIConverter:
    def __init__(self, conf: ZIConverterConf):
        self.conf = conf
        # --
        self._input_f = {"none": lambda x: x, "sent": lambda x: yield_sents(x),
                         "sent_evt": lambda x: (z for z in yield_sents(x) if len(z.events)>0),  # sents with evts!
                         "frame": lambda x: yield_frames(x)}[conf.input_strategy]
        self._convert_f = {"context": self._convert_context, "pairwise0": self._convert_pairwise0,
                           "pairwise1": self._convert_pairwise1}[conf.convert_strategy]
        # --

    # --
    # detailed strategies

    # center_sent + surrounding_context!
    def _convert_context(self, stream_inst):
        conf: ZIConverterConf = self.conf
        _left_extend_nsent, _right_extend_nsent, _center_special_id = \
            conf.left_extend_nsent, conf.right_extend_nsent, conf.center_special_id
        _extend_word_budget = conf.extend_word_budget
        # --
        for inst in stream_inst:
            for sent in yield_sents([inst]):
                _cur_words = len(sent)
                _cur_left, _cur_right = sent.prev_sent, sent.next_sent
                left_sents, right_sents = [], []
                while _cur_words < _extend_word_budget and (_cur_left is not None or _cur_right is not None):
                    # expand left? note: prefer previous!
                    if _cur_left is not None:
                        _one_len = len(_cur_left)
                        if len(left_sents) < _left_extend_nsent and (_one_len + _cur_words) <= _extend_word_budget:
                            left_sents.append(_cur_left)
                            _cur_left = _cur_left.prev_sent
                            _cur_words += _one_len
                        else:
                            _cur_left = None
                    # expand right?
                    if _cur_right is not None:
                        _one_len = len(_cur_right)
                        if len(right_sents) < _right_extend_nsent and (_one_len + _cur_words) <= _extend_word_budget:
                            right_sents.append(_cur_right)
                            _cur_right = _cur_right.next_sent
                            _cur_words += _one_len
                        else:
                            _cur_right = None
                # final one
                left_sents.reverse()
                cur_sents = left_sents + [sent] + right_sents
                seg_ids = [0] * len(cur_sents)
                center_sidx = len(left_sents)
                if _center_special_id:  # especially set center as 1
                    seg_ids[center_sidx] = 1
                ret = InputItem.create(cur_sents, inst=inst, add_seps=None, seg_ids=seg_ids, center_sidx=center_sidx)
                yield ret
        # --

    # Doc with two sentences (for sent-pair task)
    def _convert_pairwise0(self, stream_inst):
        for inst in stream_inst:
            assert isinstance(inst, Doc) and len(inst.sents) == 2
            ret = InputItem.create(list(inst.sents), inst=inst, add_seps=[1,1], seg_ids=[0,1], center_sidx=None)
            yield ret
        # --

    # special pairwise to mimic pairwise0
    def _convert_pairwise1(self, stream_inst):
        for inst in stream_inst:
            for sent in yield_sents([inst]):
                if sent.next_sent is None:
                    continue
                ret = InputItem.create([sent, sent.next_sent], inst=None, add_seps=[1,1], seg_ids=[0,1], center_sidx=None)
                yield ret
        # --

    def convert(self, stream_inst):
        yield from self._convert_f(self._input_f(stream_inst))

# --
# input batch

class InputBatch:
    def __init__(self, items: List[InputItem], dataset):
        self.items = items
        self.dataset = dataset  # mainly for extra information!
        # --
        self.seq_info: InputBatchedSeqInfo = None  # to be set by Encoder
        self.info = {}  # extra info

    def set_seq_info(self, **kwargs):
        self.seq_info = InputBatchedSeqInfo(self, **kwargs)

    def __len__(self):
        return len(self.items)

# --
# batched input seq (with runtime tensors)

class InputBatchedSeqInfo:
    def __init__(self, ibatch: InputBatch, IDX_PAD: int):
        # preps
        self.bsize = len(ibatch)
        self.arange1_t = BK.arange_idx(self.bsize)  # [bsize]
        self.arange2_t = self.arange1_t.unsqueeze(-1)  # [bsize, 1]
        self.arange3_t = self.arange2_t.unsqueeze(-1)  # [bsize, 1, 1]
        # batched them
        all_seq_infos = [z.seq_info for z in ibatch.items]
        # enc: [*, len_enc]: ids(pad IDX_PAD), masks, segids(pad 0)
        self.enc_input_ids = BK.input_idx(DataPadder.go_batch_2d([z.enc_input_ids for z in all_seq_infos], int(IDX_PAD)))
        self.enc_input_masks = BK.input_real(DataPadder.lengths2mask([len(z.enc_input_ids) for z in all_seq_infos]))
        self.enc_input_segids = BK.input_idx(DataPadder.go_batch_2d([z.enc_input_segids for z in all_seq_infos], 0))
        # dec: [*, len_dec]: sel_idxes(pad 0), sel_lens(pad 1), masks, sent_idxes(pad ??)
        self.dec_sel_idxes = BK.input_idx(DataPadder.go_batch_2d([z.dec_sel_idxes for z in all_seq_infos], 0))
        self.dec_sel_lens = BK.input_idx(DataPadder.go_batch_2d([z.dec_sel_lens for z in all_seq_infos], 1))
        self.dec_sel_masks = BK.input_real(DataPadder.lengths2mask([len(z.dec_sel_idxes) for z in all_seq_infos]))
        _max_dec_len = BK.get_shape(self.dec_sel_masks, 1)
        _dec_offsets = BK.input_idx(DataPadder.go_batch_2d([z.dec_offsets for z in all_seq_infos], _max_dec_len))
        # note: CLS as -1, then 0,1,2,..., PAD gets -2!
        self.dec_sent_idxes = \
            (BK.arange_idx(_max_dec_len).unsqueeze(0).unsqueeze(-1) >= _dec_offsets.unsqueeze(-2)).sum(-1).long() - 1
        self.dec_sent_idxes[self.dec_sel_masks<=0.] = -2
        # dec -> enc: [*, len_enc] (calculated on needed!)
        # note: require 1-to-1 mapping (except pads)!!
        self._enc_back_hits = None
        self._enc_back_sel_idxes = None
        # --

    def _calc_enc_back(self):
        hits = BK.zeros(self.enc_input_ids.shape)  # [*, len_enc]
        hits[self.arange2_t, self.dec_sel_idxes] = 1.
        back_idxes = hits.cumsum(-1).long() - 1  # [*, len_enc]
        return hits, back_idxes

    @property
    def enc_back_hits(self):
        if self._enc_back_hits is None:
            self._enc_back_hits, self._enc_back_sel_idxes = self._calc_enc_back()
        return self._enc_back_hits

    @property
    def enc_back_sel_idxes(self):
        if self._enc_back_sel_idxes is None:
            self._enc_back_hits, self._enc_back_sel_idxes = self._calc_enc_back()
        return self._enc_back_sel_idxes

    # --
# input batcher

class ZIBatcherConf(Conf):
    def __init__(self):
        # general
        self.batch_size = 1024
        self.batch_size_f = 'tok'  # or "lambda x: 1" / "lambda x: len(x)*len(x.events)"
        self.batch_maxi_bsize = 10
        # simple len constraint
        self.filter_min_length = 0
        self.filter_max_length = Constants.INT_PRAC_MAX
        # bucket
        self.bucket_interval = 20  # (len(x)//interval)
        self.bucket_shuffle_times = 1

class ZIBatcher:
    _PREBUILT_BSF = {
        'sent': (lambda x: 1),
        'tok': (lambda x: len(x)),
        'ftok': (lambda x: len(x)*max(
            1, len(x.center_sent.events) if x.center_sent is not None else sum(len(z.events) for z in x.sents))),
    }

    def __init__(self, conf: ZIBatcherConf):
        self.conf = conf
        self.batch_size_f = ZIBatcher._PREBUILT_BSF.get(conf.batch_size_f)
        if self.batch_size_f is None:
            self.batch_size_f = eval(conf.batch_size_f)
        # --

    def _put_buckets(self, stream_item):
        conf = self.conf
        _bucket_interval = conf.bucket_interval
        _filter_min_length = conf.filter_min_length
        _filter_max_length = conf.filter_max_length
        # --
        buckets = {}
        for item in stream_item:
            _len = len(item)
            if _len<_filter_min_length or _len>_filter_max_length:
                continue  # simple length filter here!
            _idx = _len // _bucket_interval
            if _idx not in buckets:
                buckets[_idx] = [item]
            else:
                buckets[_idx].append(item)
        # make a list
        ret = [buckets[k] for k in sorted(buckets.keys())]
        return ret

    def yield_batches(self, stream_item, loop: bool, filter_f=None):
        conf = self.conf
        _gen = Random.get_generator('stream')
        _bucket_shuffle_times = conf.bucket_shuffle_times
        if filter_f is None:
            filter_f = lambda x: True  # no drop
        # --
        # prepare
        buckets = self._put_buckets(stream_item)
        orig_counts = [len(b) for b in buckets]
        pvals = np.asarray(orig_counts) / sum(orig_counts)  # for sample!
        arrangers = []
        for b_items in buckets:
            # first shuffle
            for _ in range(_bucket_shuffle_times):
                _gen.shuffle(b_items)
            # get arranger
            input_stream = IterStreamer(b_items, restartable=True)
            arranger = BatchArranger(input_stream, bsize=conf.batch_size, maxi_bsize=conf.batch_maxi_bsize,
                                     batch_size_f=self.batch_size_f, dump_detectors=(lambda x: not filter_f(x)),
                                     sorting_keyer=(lambda x: len(x)), shuffle_batches_times=_bucket_shuffle_times)
            arranger.restart()
            arrangers.append(arranger)
        # go!!
        _len_buckets = len(buckets)
        while True:
            choice = _gen.choice(_len_buckets, p=pvals)  # choose a bucket
            chosen_arranger = arrangers[choice]
            items, _eos = chosen_arranger.next_and_check()
            if _eos:
                if loop:  # simply restart it
                    chosen_arranger.restart()
                else:  # pval clear; note: not change pvals every batch, but maybe does not matter!
                    pvals[choice] = 0.
                    _remain = pvals.sum().item()
                    if _remain <= 0.:
                        break  # finished!!
                    pvals = pvals / _remain
            else:
                yield items
        # --

# --
# b msp2/tasks/zmtl2/core/run:320
