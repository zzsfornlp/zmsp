# --

from typing import List
from collections import OrderedDict
from mspx.utils import Conf, Configurable, Constants, ZHelper, Random

class InputBatch:
    def __init__(self, items: List, dataset=None):
        self.items = items
        self.dataset = dataset  # mainly for extra information!
        self.info = {}
        self.cache = {}
        # --

    def __len__(self):
        return len(self.items)

class BatcherConf(Conf):
    def __init__(self):
        # how to iter dataset insts
        self.inst_f = "[inst]"  # how to yield instances?
        # batch size
        self.batch_size = 8
        self.batch_size_f = '1'  # by default 1 per inst
        self.cache_mul = 0   # cache mul: at least collect mul*batch_size, zero for caching all!
        # bucket
        self.bucket_f = '1'  # split buckets?
        self.bucket_shuffle_times = 1

class Batcher(Configurable):
    def __init__(self, conf: BatcherConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: BatcherConf = self.conf
        # --
        self.inst_f = ZHelper.eval_ff(conf.inst_f, 'inst', locals=locals(), globals=globals())
        self.batch_size_f = ZHelper.eval_ff(conf.batch_size_f, 'inst', locals=locals(), globals=globals())
        self.bucket_f = ZHelper.eval_ff(conf.bucket_f, 'inst', locals=locals(), globals=globals())
        # --
        self._cached_buckets = None

    def _iter_mulbatch(self, stream, trg_size: int):
        conf: BatcherConf = self.conf
        _inst_f, _batch_size_f = self.inst_f, self.batch_size_f
        # --
        cur_storage = []
        cur_size = 0
        for inst in stream:
            for _inst in _inst_f(inst):
                _size = _batch_size_f(_inst)
                cur_storage.append(_inst)
                cur_size += _size
                if trg_size > 0 and cur_size >= trg_size:
                    yield cur_storage
                    cur_storage = []
                    cur_size = 0
        if len(cur_storage) > 0:
            yield cur_storage

    def _put_buckets(self, insts):
        conf: BatcherConf = self.conf
        _bucket_f = self.bucket_f
        # --
        buckets = {}
        for _inst in insts:
            _idx = _bucket_f(_inst)
            if _idx not in buckets:
                buckets[_idx] = [_inst]
            else:
                buckets[_idx].append(_inst)
        # --
        ret = OrderedDict()
        for kk in sorted(buckets.keys()):
            ret[kk] = buckets[kk]
        return ret

    def _batch_from_buckets(self, buckets):
        conf: BatcherConf = self.conf
        _gen = Random.get_generator('stream')
        _batch_size = conf.batch_size
        _all_buckets = list(buckets.values())
        for _ in range(conf.bucket_shuffle_times):  # note: inplace shuffling is fine!
            _gen.shuffle(_all_buckets)
            for one_bucket in _all_buckets:
                _gen.shuffle(one_bucket)
        for _insts in _all_buckets:
            yield from self._batch_from_insts(_insts, _batch_size)

    def _batch_from_insts(self, items, batch_size: int, max_time=-1):
        cur_storage = []
        cur_size = 0
        _bs_f = self.batch_size_f
        for _inst in items:
            cur_storage.append(_inst)
            cur_size += _bs_f(_inst)
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
        conf: BatcherConf = self.conf
        _cache_size = conf.cache_mul * conf.batch_size
        # --
        # prepare instances
        if _cache_size <= 0 and (not no_cache):  # cache it all!
            _buckets = self._cached_buckets
            if _buckets is None:
                insts = sum(self._iter_mulbatch(stream, _cache_size), [])
                _buckets = self._put_buckets(insts)
                self._cached_buckets = _buckets
            _biter = [_buckets]
        else:
            _biter = (self._put_buckets(insts) for insts in self._iter_mulbatch(stream, _cache_size))
        # --
        for _bs in _biter:
            yield from self._batch_from_buckets(_bs)
        # --

# --
# b mspx/znew/prompt/data/batch:
