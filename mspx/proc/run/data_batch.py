#

__all__ = [
    "DataItem", "InputBatch",
    "DataBatcherConf", "DataBatcher", "BatcherPlainConf", "BatcherPlain",
]

from typing import List, Union
import numpy as np
from mspx.data.inst import Sent, Frame, yield_sents, yield_frames
from mspx.utils import Registrable, Configurable, Conf, Constants, Random, ZHelper

# --
# common helper

class DataItem:
    def __init__(self, inst: Union[Sent, Frame], **kwargs):
        self.inst = inst
        if isinstance(inst, Sent):
            self.sent = inst
            self.frame = None
        elif isinstance(inst, Frame):
            self.sent = inst.sent
            self.frame = inst
        else:
            raise NotImplementedError()
        # --
        self.cache = {}
        self.cache.update(kwargs)

    def __repr__(self):
        return f"Frame({self.frame})" if self.frame is not None else f"Sent({self.sent})"

class InputBatch:
    def __init__(self, items: List, dataset=None):
        self.items = [z if isinstance(z, DataItem) else DataItem(z) for z in items]
        self.dataset = dataset  # mainly for extra information!
        self.info = {}
        # --

    def __len__(self):
        return len(self.items)

# --
# batchers

@Registrable.rd('DB')
class DataBatcherConf(Conf):
    @classmethod
    def get_base_conf_type(cls): return DataBatcherConf
    @classmethod
    def get_base_node_type(cls): return DataBatcher

@Registrable.rd('_DB')
class DataBatcher(Configurable):
    # yield data insts with input inst stream!
    def batch_insts(self, stream, no_cache=False):
        # yield from stream
        raise NotImplementedError()

# --
# plain batcher

@DataBatcherConf.rd('plain')
class BatcherPlainConf(DataBatcherConf):
    def __init__(self):
        super().__init__()
        # --
        # how to iter dataset insts
        self.inst_f = "sent"  # how to yield instances?
        self.len_f = 'word'  # word/subword:??
        # batch size
        self.batch_size = 512
        self.batch_size_f = ''  # by default the same as 'len_f'
        self.cache_mul = 0   # cache mul: at least collect mul*batch_size, zero for caching all!
        # simple len constraint
        self.filter_min_length = 0
        self.filter_max_length = Constants.INT_PRAC_MAX
        # bucket: (min(bml,len(x))//interval)
        self.bucket_max_length = Constants.INT_PRAC_MAX
        self.bucket_interval = 20  # note: a very large interval means only one bucket
        self.bucket_shuffle_times = 1
        # sampling rather than itering: note: ONLY use this for training!
        self.sample_f = ""  # weighting f; lambda inst: ... or special
        self.sample_repeat = 1.  # how much to sample?
        # --

@BatcherPlainConf.conf_rd()
class BatcherPlain(DataBatcher):
    def __init__(self, conf: BatcherPlainConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: BatcherPlainConf = self.conf
        self.inst_f = self.get_inst_f()
        self.len_f = self.get_len_f()
        self.batch_size_f = self.get_batch_size_f()
        self.sample_f = None
        if conf.sample_f:  # also feed self for helper!
            self.sample_f = ZHelper.eval_ff(conf.sample_f, default_args='inst,self')
        # --
        self._cached_buckets = None

    def get_inst_f(self):
        from .data_prep import yield_aggr_sents
        # --
        _inst_f = self.conf.inst_f
        _args = None
        if _inst_f.startswith("frame"):
            _inst_f, *_args = _inst_f.split(":")
        elif _inst_f.startswith("sentA"):
            _inst_f, *_args = _inst_f.split(":")
        elif _inst_f.startswith("sentF"):
            _inst_f, *_args = _inst_f.split(":")
        elif _inst_f.startswith("sentK"):
            _inst_f, *_args = _inst_f.split(":")
        _cates = list(_args) if _args else None  # if interpreted in this way
        ret = {
            "none": lambda x: [x],  # put it as it is
            "sent": lambda x: yield_sents(x),  # yield sents
            "sentA": lambda x: yield_aggr_sents([x], *_args, len_f=self.len_f),  # concatenate sents
            "sentF": lambda x: (z for z in yield_sents(x) if len([z2 for z2 in z.get_frames(cates=_cates) if not z2.info.get('is_pred', False)])>0),  # sents with valid frames
            "sentFA": lambda x: (z for z in yield_sents(x) if len([z3 for z2 in z.get_frames(cates=_cates) if not z2.info.get('is_pred', False) for z3 in z2.args if not z3.info.get('is_pred', False)])>0),  # sents with valid frame args
            "sentDP": lambda x: (z for z in yield_sents(x) if (z.tree_dep is not None and any(z2>=0 for z2 in z.tree_dep.seq_head.vals))),  # sents with valid dpar-edge
            "sentK": lambda x: (z for z in yield_sents(x) if any(z.info.get(kk) for kk in _args)),  # sents with key-marks
            "frame": lambda x: yield_frames(x, False, cates=_cates),  # frames
            "frameS": lambda x: yield_frames(x, True, cates=_cates),  # frames in sents
        }.get(_inst_f)
        if ret is None:
            ret = ZHelper.eval_ff(self.conf.inst_f, 'inst')
        return ret

    def get_len_f(self):
        _len_f = self.conf.len_f
        if _len_f.startswith('subword'):
            from mspx.data.vocab import TokerPretrained
            _len_f, sub_name = _len_f.split(":", 1)
            sub_toker = TokerPretrained(bert_name=sub_name)
        else:
            sub_toker = None
        ret = {
            'word': (lambda x: len(x)),  # word len
            'subword': (lambda x: len(x.sent.seq_word.get_sf(sub_toker=sub_toker))),
        }.get(_len_f)
        if ret is None:
            ret = ZHelper.eval_ff(self.conf.len_f, 'inst')
        return ret

    def get_batch_size_f(self):
        _bs_f = self.conf.batch_size_f
        if _bs_f == '': return None
        _args = None
        if _bs_f.startswith("lenF"):
            _bs_f, *_args = _bs_f.split(":")
        ret = {
            'num': (lambda x: 1),
            'len': (lambda x: self.len_f(x)),
            'lenF': (lambda x: self.len_f(x) * max(1, len(x.get_frames(cates=_args)))),  # sent-LEN * frame-NUM
        }.get(_bs_f)
        if ret is None:
            ret = ZHelper.eval_ff(self.conf.batch_size_f, 'inst')
        return ret

    # helper for sampling weighting
    def sw_fratio(self, inst, cates, minv=0., maxv=1.):
        inst = inst.sent  # must be with sent!
        ret0 = sum(z.mention.wlen for z in inst.yield_frames(cates=cates))/len(inst)
        ret = max(min(ret0, maxv), minv)  # force range
        return ret
    # --

    def _iter_mulbatch(self, stream, trg_size: int):
        conf: BatcherPlainConf = self.conf
        _filter_min_length = conf.filter_min_length
        _filter_max_length = conf.filter_max_length
        _inst_f = self.inst_f
        # --
        cur_storage = []
        cur_size = 0
        for inst in stream:
            for _inst in _inst_f(inst):
                _len = self.len_f(_inst)
                if _len < _filter_min_length or _len > _filter_max_length:
                    continue  # simple length filter here!
                cur_storage.append((_inst, _len))
                cur_size += _len
                if trg_size > 0 and cur_size >= trg_size:
                    yield cur_storage
                    cur_storage = []
                    cur_size = 0
        if len(cur_storage) > 0:
            yield cur_storage

    def _put_buckets(self, pairs):
        conf: BatcherPlainConf = self.conf
        _bi, _bml = conf.bucket_interval, conf.bucket_max_length
        # --
        buckets = {}
        for _inst, _len in pairs:
            _len = min(_len, _bml)
            _idx = _len // _bi  # note: simply do //
            item = (DataItem(_inst), _len)
            if _idx not in buckets:
                buckets[_idx] = [item]
            else:
                buckets[_idx].append(item)
        # make a list
        ret = [buckets[k] for k in sorted(buckets.keys())]
        return ret

    def _batch_from_buckets(self, buckets):
        conf: BatcherPlainConf = self.conf
        _gen = Random.get_generator('stream')
        _batch_size = conf.batch_size
        # --
        if self.sample_f is not None:  # do weighted sampling
            good_buckets = []
            good_weights = []
            for b_idx, b_insts in enumerate(buckets):  # filter positive ones!
                b_ws = [self.sample_f(z[0],self) for z in buckets[b_idx]]
                _good_ws = [z2 for z2 in b_ws if z2>0]
                if len(_good_ws) > 0:  # no adding if no good ones!
                    good_buckets.append([z1 for z1,z2 in zip(buckets[b_idx], b_ws) if z2>0])
                    good_weights.append(_good_ws)
            # weighting arrs
            arr_bucket = np.asarray([sum(z) for z in good_weights])
            arr_bucket = arr_bucket / arr_bucket.sum()  # [B]
            arr_weights = [np.asarray(z) / sum(z) for z in good_weights]  # B x [??]
            _good_size = sum(z2[1] for z1 in good_buckets for z2 in z1)
            sample_times = int(max(1., _good_size/_batch_size) * conf.sample_repeat)
            avg_len = [sum(z2[1] for z2 in z1)/len(z1) for z1 in good_buckets]
            _idxes0, _idxes1 = np.asarray(range(len(good_buckets))), [np.asarray(range(len(z))) for z in good_buckets]
            for _ in range(max(1, sample_times)):
                _ii = _gen.choice(_idxes0, p=arr_bucket)
                curr_bucket = good_buckets[_ii]
                e_num = max(1, min(int(_batch_size/avg_len[_ii]), len(curr_bucket)))
                iidxes = _gen.choice(_idxes1[_ii], e_num, replace=False, p=arr_weights[_ii])
                yield from self._batch_from_items([curr_bucket[z] for z in iidxes], _batch_size, max_time=1)
        else:  # simply do itering
            for _ in range(conf.bucket_shuffle_times):  # note: inplace shuffling is fine!
                _gen.shuffle(buckets)
                for one_bucket in buckets:
                    _gen.shuffle(one_bucket)
            for _items in buckets:
                yield from self._batch_from_items(_items, _batch_size)
        # --

    def _batch_from_items(self, items, batch_size: int, max_time=-1):
        cur_storage = []
        cur_size = 0
        _bs_f = self.batch_size_f
        for _item, _len in items:
            cur_storage.append(_item)
            cur_size += (_len if _bs_f is None else _bs_f(_item.inst))
            if cur_size >= batch_size:
                yield cur_storage
                max_time -= 1
                if max_time == 0:
                    return  # no budgets left
                cur_storage = []
                cur_size = 0
        if len(cur_storage) > 0:
            yield cur_storage

    def batch_insts(self, stream, no_cache=False):
        conf: BatcherPlainConf = self.conf
        _bucket_shuffle_times = conf.bucket_shuffle_times
        _cache_size = conf.cache_mul * conf.batch_size
        # --
        # prepare instances
        if _cache_size <= 0 and (not no_cache):  # cache it all!
            _buckets = self._cached_buckets
            if _buckets is None:
                pairs = sum(self._iter_mulbatch(stream, _cache_size), [])
                _buckets = self._put_buckets(pairs)
                self._cached_buckets = _buckets
            _biter = [_buckets]
        else:
            _biter = (self._put_buckets(pairs) for pairs in self._iter_mulbatch(stream, _cache_size))
        # --
        for _bs in _biter:
            yield from self._batch_from_buckets(_bs)
        # --

# --
# b mspx/proc/run/data_batch:
