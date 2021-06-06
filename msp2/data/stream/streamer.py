#

from typing import Union, Iterable, List, Callable, SupportsFloat
from msp2.utils import zfatal, Random, Constants, zlog

# =====
# basic

STREAMER_RANDOM_GEN = Random.get_generator('stream')

# basic streamer
class Streamer:
    def __init__(self):
        self.eos = None  # by default, EOS is None
        # status
        self._count = 0
        self._max_count = 0
        self._restart_times = 0
        self._active = False
        self._stack = []

    def __repr__(self):
        return f"{self.__class__.__name__}(A={self._active},R={self._restart_times},C={self._count})"

    def __iter__(self):
        self.restart()      # for convenience
        return self

    def __next__(self):
        one = self.next()
        if self._active:
            return one
        else:
            raise StopIteration("EOS: End of Stream")

    def is_active(self): return self._active
    def is_inactive(self): return not self._active
    def is_eos(self, one: object): return one == self.eos
    def count(self): return self._count
    def max_count(self): return self._max_count

    def set_eos(self, eos: object):
        self.eos = eos

    def restart(self):
        # first restart the possible base ones
        self._restart()
        #
        self._max_count = max(self._max_count, self._count)
        self._count = 0
        self._restart_times += 1
        self._active = True  # state
        self._stack.clear()

    def put(self, item: object, d_count: int = 0):  # usually use d_count=-1 for putting back
        assert self._active, "Cannot put item to an inactive stream!"
        self._stack.append(item)
        self._count += d_count

    def next(self):
        if self._active:
            if len(self._stack) > 0:
                x = self._stack.pop()
            else:
                x = self._next()
            # check output
            if self.is_eos(x):
                self._active = False
            else:
                self._count += 1
            return x
        else:  # always return eos when not active
            return self.eos

    # return both next one and check whether the return _iseos
    def next_and_check(self):
        return self.next(), not self._active

    def has_next(self):
        x = self.next()
        if not self._active:
            return False  # no need to put eos
        self.put(x, -1)  # put back to stack and minus one for the count

    # =====
    # to be implemented
    def _next(self):
        raise NotImplementedError()

    def _restart(self):
        raise NotImplementedError()

# =====
# composition ones

# Streamer wrapper/decorator, stacked streamers, driven by the ended Streamer
class WrapperStreamer(Streamer):
    def __init__(self, base_streamer: Streamer):
        super().__init__()
        self._base_streamer: Streamer = base_streamer

    def _next(self):
        _base_streamer = self._base_streamer
        x, _iseos = _base_streamer.next_and_check()
        if _iseos:
            return self.eos
        else:
            return x

    def _restart(self):
        self._base_streamer.restart()

# infinite wrap-around loop streamer, return EOS only at empty loop
class LoopStreamer(WrapperStreamer):
    def __init__(self, base_streamer: Streamer):
        super().__init__(base_streamer)

    def _next(self):
        _base_streamer = self._base_streamer
        one, _iseos = _base_streamer.next_and_check()
        if _iseos:
            _base_streamer.restart()
            # todo(warn): can return None (stop) if empty loop
            two, _iseos = _base_streamer.next_and_check()
            if _iseos:
                return self.eos
            else:
                return two
        else:
            return one

    def _restart(self):
        pass  # no need to restart here!

# from iterable, but can only iter once (can be combined with Cacher)
class IterStreamer(Streamer):
    def __init__(self, src: Iterable, restartable=False):
        super().__init__()
        self.restartable = restartable
        self._src = src
        self._iter = None

    def _next(self):
        try:
            one = next(self._iter)
            return one
        except StopIteration:
            return self.eos

    def _restart(self):
        if self._iter is None or self.restartable:
            self._iter = iter(self._src)
        else:
            zfatal("Cannot restart a non-repeatable IterStreamer")

# from function as iterable src
class FIterStreamer(Streamer):
    def __init__(self, src_f: Callable):
        super().__init__()
        self._src_f = src_f
        self._iter = None

    def _next(self):
        try:
            one = next(self._iter)
            return one
        except StopIteration:
            return self.eos

    def _restart(self):
        self._iter = iter(self._src_f())

# map: modify the instance with F
class FWrapperStreamer(WrapperStreamer):
    def __init__(self, base_streamer: Streamer, func: Callable, inplaced=False):
        super().__init__(base_streamer)
        self.func = func
        self.inplaced = inplaced

    def _next(self):
        one, _iseos = self._base_streamer.next_and_check()
        if _iseos:
            return self.eos
        z = self.func(one)
        return one if self.inplaced else z

# map + flatten
class FListWrapperStreamer(WrapperStreamer):
    def __init__(self, base_streamer: Streamer, func: Callable):
        super().__init__(base_streamer)
        self.f = func
        self.cache = []
        self.cache_idx = 0

    def _next(self):
        while self.cache_idx >= len(self.cache):
            one, _iseos = self._base_streamer.next_and_check()
            if _iseos:
                return self.eos
            self.cache.clear()
            self.cache_idx = 0
            self.cache.extend(self.f(one))
        r = self.cache[self.cache_idx]
        self.cache_idx += 1
        return r

    def _restart(self):
        super()._restart()
        self.cache.clear()
        self.cache_idx = 0

# filter
class FilterWrapperStreamer(WrapperStreamer):
    def __init__(self, base_streamer: Streamer, func_filter: Callable):
        super().__init__(base_streamer)
        self.func_filter = func_filter

    def _next(self):
        while True:
            one, _iseos = self._base_streamer.next_and_check()
            if _iseos:
                return self.eos
            if self.func_filter(one):
                return one

# truncate
class TruncateStreamer(WrapperStreamer):
    def __init__(self, base_streamer: Streamer, k: int = -1):
        super().__init__(base_streamer)
        self.k = k

    def _next(self):
        _base_streamer = self._base_streamer
        x, _iseos = _base_streamer.next_and_check()
        if _iseos or self.count() >= self.k:
            return self.eos
        else:
            return x

# =====
# Multi Streamers

# input from multiple streams
class MultiStreamer(Streamer):
    def __init__(self, base_streams: List[Streamer]):
        super().__init__()
        #
        self._base_streamers = list(base_streams)
        self._num_streamers = len(self._base_streamers)

    def _restart(self):
        for one in self._base_streamers:
            one.restart()

# simple concat multi-streamer
class MultiCatStreamer(MultiStreamer):
    def __init__(self, base_streamers: List[Streamer]):
        super().__init__(base_streamers)
        self._cur_idx = 0

    def _next(self):
        cur_idx = self._cur_idx
        while True:
            if cur_idx >= self._num_streamers:
                return self.eos
            _streamer = self._base_streamers[cur_idx]
            one, _iseos = _streamer.next_and_check()
            if _iseos:
                cur_idx += 1
                self._cur_idx = cur_idx
            else:
                return one

    def _restart(self):
        super()._restart()
        self._cur_idx = 0

# like MultiCat, but join the streams (horizontally) rather than concat (vertically)
# -- ratios: for each time, what ratio to put for each stream, stop_sidx: stop on which streamer (-1 means None)
# note: will not restart non-stop ones automatically in the middle, may combine with 'LoopStreamer' to do that!
class MultiJoinStreamer(MultiStreamer):
    def __init__(self, base_streamers: List[Streamer], stop_sidx=-1, ratios: List[SupportsFloat]=None, verbose=True):
        super().__init__(base_streamers)
        # --
        if ratios is None:  # by default all 1
            ratios = [1.] * self._num_streamers
        assert self._num_streamers>0 and self._num_streamers == len(ratios)
        self._ratios = ratios
        self._stop_sidx = stop_sidx
        self._random_sampler = Random.stream(STREAMER_RANDOM_GEN.random_sample)
        # status
        self._cur_idx = self._num_streamers-1
        self._cur_ratio = 0.
        self._cur_counts = [0] * self._num_streamers
        self.verbose = verbose

    def _next(self):
        ret = self.eos  # by default, EOS if no one can return anything
        cur_idx, cur_ratio = self._cur_idx, self._cur_ratio
        # at most travel one round + 1
        for _ in range(self._num_streamers+1):
            flag_ok = False
            if cur_ratio<=0.:
                pass  # no budget
            elif cur_ratio<1. and next(self._random_sampler) >= cur_ratio:
                pass  # not hit with random ratio
            else:
                # try to get one
                cur_streamer = self._base_streamers[cur_idx]
                one, _iseos = cur_streamer.next_and_check()
                if _iseos:
                    if self._stop_sidx == cur_idx:
                        if self.verbose:
                            zlog(f"End for one, count = {self._cur_counts}", func='report')
                        return self.eos  # directly EOS since SIDX returns EOS
                else:
                    self._cur_counts[cur_idx] += 1  # record
                    ret = one
                    flag_ok = True
            # --
            if flag_ok:
                self._cur_idx, self._cur_ratio = cur_idx, cur_ratio - 1  # cost one for the current step
                break
            else:  # forward one streamer
                cur_idx = (cur_idx + 1) % self._num_streamers
                cur_ratio = float(self._ratios[cur_idx])
        return ret

    def _restart(self):
        super()._restart()
        self._cur_idx = self._num_streamers - 1
        self._cur_ratio = 0.
        self._cur_counts = [0] * self._num_streamers

# zip-like multi-streamer, with various modes, return a list of instances
class MultiZipStreamer(MultiStreamer):
    def __init__(self, base_streamers: List[Streamer], stop_sidx=-1, auto_mdoe="any", padding: object = None):
        super().__init__(base_streamers)
        self._stop_sidx = stop_sidx
        self._padding = padding
        self._stop_f = {
            "any": lambda valid_num: valid_num<self._num_streamers,
            "all": lambda valid_num: valid_num==0,
        }[auto_mdoe]

    def _next(self):
        rets = [self._padding] * self._num_streamers
        valid_num = 0
        for idx, ss in enumerate(self._base_streamers):
            one, _iseos = ss.next_and_check()
            if _iseos:
                if idx == self._stop_sidx:
                    return self.eos  # directly EOS since SIDX returns EOS
            else:
                valid_num += 1
                rets[idx] = one
        if self._stop_f(valid_num):
            return self.eos
        return rets

    # use the default one
    # def _restart(self):
    #     super()._restart()

# =====
# Cache and Shuffle

class Cache:
    def __len__(self): raise NotImplementedError()
    def put(self, one: object): raise NotImplementedError()
    def get(self): raise NotImplementedError()
    def restart(self): raise NotImplementedError()
    def clear(self): raise NotImplementedError()

# Cache that stores everything in Memory
class InMemCache(Cache):
    def __init__(self, eos: object, shuffle_times: int):
        self.c = []
        self.ptr = 0
        self.eos = eos
        self.shuffle_times = shuffle_times

    def __len__(self):
        return len(self.c)

    def put(self, one): # todo(note): always append at the end
        self.c.append(one)

    def get(self):
        if self.ptr >= len(self.c):
            return self.eos  # no more, return EOS
        else:
            self.ptr += 1
            return self.c[self.ptr-1]

    def restart(self):
        self.ptr = 0
        for _ in range(self.shuffle_times):
            STREAMER_RANDOM_GEN.shuffle(self.c)

    def clear(self):
        self.c.clear()
        self.ptr = 0

    def steal_cache(self):
        # todo(note): special operation to change things inplace!!
        return self.c

# read from base and store at first restart, later stream from cache
class CacheStreamer(WrapperStreamer):
    def __init__(self, base_streamer: Streamer, cache_builder=InMemCache, shuffle_times=0):
        super().__init__(base_streamer)
        self.cache = cache_builder(eos=self.eos, shuffle_times=shuffle_times)

    def _next(self):
        return self.cache.get()

    def _restart(self):
        if self._restart_times==0:
            for one in self._base_streamer:
                self.cache.put(one)
        self.cache.restart()

# shuffling; shuffle_bsize=-1 means read all and shuffle
class ShuffleStreamer(WrapperStreamer):
    def __init__(self, base_streamer: Streamer, shuffle_bsize=-1, shuffle_times=2):
        super().__init__(base_streamer)
        self.shuffle_bsize = shuffle_bsize
        self.shuffle_times = shuffle_times
        self.storage = []  # actually an InMemCache

    def _next(self):
        _storage = self.storage
        if len(_storage) == 0:
            # read base: todo(note): maybe not efficient enough?
            _base = self._base_streamer
            _count, _trg = 0, self.shuffle_bsize
            _storage.clear()
            while _count != _trg:
                new_one, _iseos = _base.next_and_check()
                if _iseos:
                    break
                _storage.append(new_one)
                _count += 1
            if _count > 0:
                _storage.reverse()
                for _ in range(self.shuffle_times):
                    STREAMER_RANDOM_GEN.shuffle(_storage)
                x = _storage.pop()
                return x
            else:  # no more!
                return self.eos
        else:
            x = _storage.pop()
            return x

    # use the default one
    def _restart(self):
        super()._restart()
        self.storage.clear()

# =====
# Classes for Batching

# some helping functions
class BatchHelper:
    @staticmethod
    def group_buckets(input_insts: List, thresh_all: int = None, thresh_diff: int = None,
                      size_f: Callable = (lambda x: len(x)), sort_key: Callable = None):
        if thresh_all is None:
            thresh_all = Constants.INT_PRAC_MAX
        if thresh_diff is None:
            thresh_diff = Constants.INT_PRAC_MAX
        # sort inputs?
        if sort_key is not None:
            input_insts = sorted(input_insts, key=sort_key)
        # prepare buckets
        buckets = []
        cur_size_all, cur_size_start = 0, None
        tmp_bucket = []
        for one in input_insts:
            one_size = size_f(one)
            if len(tmp_bucket) == 0:  # always add when empty
                cur_size_all = cur_size_start = one_size
                tmp_bucket = [one]  # todo(+w): here also need to check thresh_all!!
            elif one_size-cur_size_start >= thresh_diff:  # a new start with current
                buckets.append(tmp_bucket)
                cur_size_all = cur_size_start = one_size
                tmp_bucket = [one]
            else:
                cur_size_all += one_size  # add to cur_size
                tmp_bucket.append(one)  # add one
                if cur_size_all >= thresh_all:  # a new start after current
                    buckets.append(tmp_bucket)
                    cur_size_all, cur_size_start = 0, None
                    tmp_bucket = []
        # --
        if len(tmp_bucket) > 0:
            buckets.append(tmp_bucket)
        return buckets

# handling the batching of instances, also filtering, sorting, recording, etc.
# streamer: base, batch_size: sum(batch_size_f(z) for z), maxibatch_size: read-in bs*mbs every time,
# dump_detectors, single_detectors, sorting_keyer: sort bs*mbs, shuffling: shuffle on buckets in bs*mbs
class BatchArranger(WrapperStreamer):
    def __init__(self, base_streamer: Streamer, bsize: int, maxi_bsize: int, batch_size_f: Callable = None,
                 dump_detectors: Union[Callable, List[Callable]] = None, single_detectors: Union[Callable, List[Callable]] = None,
                 sorting_keyer: Callable = None, shuffle_batches_times=0):
        super().__init__(base_streamer)
        self.batch_size = bsize
        self.batch_size_f = (lambda one: 1) if batch_size_f is None else batch_size_f
        # todo(note): if <=0 then read all at one time and possibly sort all
        self.maxibatch_size = maxi_bsize if maxi_bsize>0 else Constants.INT_PRAC_MAX
        if dump_detectors is not None and not isinstance(dump_detectors, Iterable):
            dump_detectors = [dump_detectors]
        self.dump_detectors = [] if dump_detectors is None else dump_detectors
        if single_detectors is not None and not isinstance(single_detectors, Iterable):
            single_detectors = [single_detectors]
        self.single_detectors = [] if single_detectors is None else single_detectors
        self.sorting_keyer = sorting_keyer  # default(None) no sorting
        self.shuffle_batches_times = shuffle_batches_times  # shuffle batches inside the maxi-batches?
        # caches
        self.buffered_bsize_ = 0
        self.buffer_ = []           # list of instances
        self.buckets_ = []          # list of already prepared batch of instances

    @property
    def k(self):
        # (nearly) sort size or cache size
        return self.batch_size * self.maxibatch_size

    # get the next mini-batch
    def _next(self):
        # buffered read
        if len(self.buckets_) == 0:
            # read into buffer
            while self.buffered_bsize_ < self.k:
                one, _iseos = self._base_streamer.next_and_check()
                if _iseos:
                    break
                # dump instances (like short or long instances)
                dump_instance = any(f_(one) for f_ in self.dump_detectors)
                if dump_instance:
                    continue
                # single instances
                single_instance = any(f_(one) for f_ in self.single_detectors)
                if single_instance:
                    # immediate arrange this special one
                    return [one]
                # add this instance to buffer
                self.buffer_.append(one)
                self.buffered_bsize_ += self.batch_size_f(one)
            # prepare buffering
            if len(self.buffer_) > 0:
                buckets = BatchHelper.group_buckets(self.buffer_, thresh_all=self.batch_size,
                                                    size_f=self.batch_size_f, sort_key=self.sorting_keyer)
                # another shuffle?
                if self.shuffle_batches_times > 0:
                    for _ in range(self.shuffle_batches_times):
                        STREAMER_RANDOM_GEN.shuffle(buckets)
                else:
                    # todo(note): to keep sorting-order if sorting else original-order BECAUSE-OF later POP
                    buckets.reverse()
                # clear here
                self.buckets_ = buckets
                self.buffer_ = []
                self.buffered_bsize_ = 0
        # return buckets
        if len(self.buckets_) > 0:
            ret = self.buckets_.pop()
            return ret
        else:
            return None

    # restart the status including the streamer
    def _restart(self):
        super()._restart()
        self.buffered_bsize_ = 0
        self.buffer_ = []
        self.buckets_ = []

# =====
# a bucketed version: [0, B1], (B1, B2], ..., (B_{n-1}, Bn], larger as singleton
# todo(+N): not tested yet!!
class BucketedBatchArranger(WrapperStreamer):
    def __init__(self, base_streamer: Streamer, bsize: int, batch_size_f: Callable = None,
                 bucket_bounds=tuple([10, 20, 30, 40, 50, 60, 80, 100]), bucket_f: Callable = None, shuffle_batches_times=0):
        super().__init__(base_streamer)
        # --
        self.bsize = bsize
        self.batch_size_f = (lambda x: len(x)) if batch_size_f is None else batch_size_f
        self.bucket_bounds = bucket_bounds
        self.bucket_f = (lambda x: len(x)) if bucket_f is None else bucket_f
        self.shuffle_batches_times = shuffle_batches_times
        # --
        self.num_buckets = len(bucket_bounds)
        self.bmap = [0]  # for 0
        lasti, cur_idx = 0, 0
        for ii in range(len(bucket_bounds)):
            self.bmap.extend((bucket_bounds[ii]-lasti) * [cur_idx])
            lasti = bucket_bounds[ii]
            cur_idx += 1
        assert len(self.bmap) == max(bucket_bounds) + 1 and self.bmap[-1] == len(bucket_bounds) - 1
        # --
        self._reset_buckets()

    def _reset_buckets(self):
        self._cur_buffer_sizes = [0] * self.num_buckets
        self._buffers = [[] for _ in range(self.num_buckets)]

    # get the next mini-batch
    def _next(self):
        bucket_f = self.bucket_f
        bsize_f = self.batch_size_f
        # first read some more data until one bucket hit it!
        while True:
            one, _iseos = self._base_streamer.next_and_check()
            if _iseos:
                break  # no more data
            # put it into one bucket
            bucket_res = bucket_f(one)
            if bucket_res > len(self.bmap):
                # directly return that inst!
                return [one]
            # put it into buffers
            cur_bidx = self.bmap[bucket_res]
            cur_bsize = bsize_f(one)
            self._cur_buffer_sizes[cur_bidx] += cur_bsize
            self._buffers[cur_bidx].append(one)
            if self._cur_buffer_sizes[cur_bidx] >= self.bsize:
                # let it go
                _insts = self._buffers[cur_bidx]
                self._buffers[cur_bidx] = []
                self._cur_buffer_sizes[cur_bidx] = 0
                return _insts
        # at here, we simply try to push all buffers out one by one
        for ii in range(self.num_buckets):
            _insts = self._buffers[ii]
            if len(_insts) > 0:
                self._buffers[ii] = []
                self._cur_buffer_sizes[ii] = 0
                return _insts
        # finally nothing left
        return self.eos

    # restart the status including the streamer
    def _restart(self):
        super()._restart()
        self._reset_buckets()
