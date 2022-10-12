#

# plain DataBatcher
# mainly in testing mode!!

from msp2.data.inst import yield_sents, yield_frames
from msp2.data.stream import IterStreamer, BatchArranger
from msp2.utils import Conf, Constants, Random
import numpy as np
from .common import *

# --

class PlainBatcherConf(Conf):
    def __init__(self):
        # how to iter dataset insts
        self.inst_f = "sent"  # how to yield instances?
        # batch size
        self.batch_size = 512
        self.batch_size_f = 'len'  # ... or "lambda x: 1" / "lambda x: len(x)*len(x.events)"
        self.batch_maxi_bsize = 10
        # simple len constraint
        self.filter_min_length = 0
        self.filter_max_length = Constants.INT_PRAC_MAX
        # bucket
        self.bucket_interval = 20  # (len(x)//interval)
        self.bucket_shuffle_times = 1
        # --

    # note: record this when first met!!
    def special_inst_f(self, x):
        for sent in yield_sents(x):
            inc = sent.info.get("batcher_inc")
            if inc is None:
                inc = len(sent.events) > 0
                sent.info["batcher_inc"] = inc
            if inc:
                yield sent
        # --

    def get_inst_f(self):
        ret = {
            "none": lambda x: [x], "sent": lambda x: yield_sents(x),
            "sent_evt": lambda x: (z for z in yield_sents(x) if len(z.events) > 0),  # sents with evts!
            "sent_evt2": self.special_inst_f,  # special mode!!
            "frame": lambda x: yield_frames(x)
        }.get(self.inst_f)
        if ret is None:
            ret = eval(self.inst_f)
        return ret

    def get_size_f(self):
        ret = {
            'num': (lambda x: 1),
            'len': (lambda x: len(x)),
        }.get(self.batch_size_f)
        if ret is None:
            ret = eval(self.batch_size_f)
        return ret

    def get_batcher(self):
        return PlainBatcher(self)

class PlainBatcher:
    def __init__(self, conf: PlainBatcherConf):
        self.conf = conf
        self.inst_f = conf.get_inst_f()
        self.batch_size_f = conf.get_size_f()
        # --
        self.len_f = (lambda x: len(x.sent))  # get item length
        # --

    def _put_buckets(self, stream_item):
        conf = self.conf
        _bucket_interval = conf.bucket_interval
        _filter_min_length = conf.filter_min_length
        _filter_max_length = conf.filter_max_length
        # --
        buckets = {}
        for item in stream_item:
            _len = self.len_f(item)
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

    def yield_batches(self, stream, loop: bool):
        conf = self.conf
        _gen = Random.get_generator('stream')
        _bucket_shuffle_times = conf.bucket_shuffle_times
        # --
        # prepare instances
        all_items = [DataItem(one) for inst in stream for one in self.inst_f(inst)]  # simply get them all!
        buckets = self._put_buckets(all_items)
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
                                     batch_size_f=self.batch_size_f, sorting_keyer=self.len_f,
                                     shuffle_batches_times=_bucket_shuffle_times)
            arranger.restart()
            arrangers.append(arranger)
        # go!!
        _len_buckets = len(buckets)
        if _len_buckets > 0:
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
