#

from msp.utils import zopen, zcheck, zfatal
from msp.utils import Constants, Random
from typing import Iterable, Sequence

# =====
# Basic Streamers

# continuing returning this one after ended without restart
# todo(warn): assume all regard None as EOS, that is, is_eos is finalized!
STREAMER_EOS = None

# Simplified Item Streamed Obtainer
# todo(warn): should always restart() or iter() before usage
class Streamer(object):
    def __init__(self):
        self.count_ = 0
        self.max_count_ = 0
        self.restart_times_ = 0
        self.active_ = False

    def __iter__(self):
        self.restart()      # for convenience
        return self

    def __next__(self):
        one = self.next()
        if self.active_:
            return one
        else:
            raise StopIteration("End of stream: get END -> should restart!")

    def is_active(self):
        return self.active_

    def is_eos(self, one):
        return one is None

    def restart(self):
        # first restart the possible base ones
        self._restart()
        #
        self.max_count_ = max(self.max_count_, self.count_)
        self.count_ = 0
        self.restart_times_ += 1
        self.active_ = True

    def count(self):
        return self.count_

    def next(self):
        x = self._next()
        # todo(warn): assume instances can never be None!
        if self.is_eos(x):
            self.active_ = False
        else:
            self.count_ += 1
        return x

    # ===============
    # to be implemented
    def _next(self):
        raise NotImplementedError()

    def _restart(self):
        raise NotImplementedError()

# Item adapter/modifier, stacked streamers, driven by the ended Streamer
class AdapterStreamer(Streamer):
    def __init__(self, base_streamer):
        super().__init__()
        self.base_streamer_ = base_streamer

    def _restart(self):
        self.base_streamer_.restart()

# infinite wrap-around loop streamer, return EOS only at empty loop
class LoopStreamer(AdapterStreamer):
    def __init__(self, base_streamer):
        super().__init__(base_streamer)

    def _next(self):
        one = self.base_streamer_.next()
        if self.base_streamer_.is_eos(one):
            self.base_streamer_.restart()
            # todo(warn): can return None (stop) if empty loop
            two = self.base_streamer_.next()
            return two
        else:
            return one

# from iterable, but can only iter once (can be combined with Cacher)
class IterStreamer(Streamer):
    def __init__(self, src, restartable=False):
        super().__init__()
        self.restartable = restartable
        self.src_ = src
        self.it_ = None

    def _restart(self):
        if self.it_ is None or self.restartable:
            self.it_ = iter(self.src_)
        else:
            zfatal("Cannot restart a non-repeatable stream")

    def _next(self):
        try:
            one = next(self.it_)
            return one
        except StopIteration:
            return None

#
class FileOrFdStreamer(Streamer):
    def __init__(self, file_or_fd):
        super().__init__()
        self.file = file_or_fd
        self.fd = None

    def __del__(self):
        if self.fd is not None:
            self.fd.close()

    def _restart(self):
        if isinstance(self.file, str):
            if self.fd is not None:
                self.fd.close()
            self.fd = zopen(self.file)
        else:
            zcheck(self.restart_times_==0, "Cannot restart a FdStreamer")

# inplace modify the instance
class FAdapterStreamer(AdapterStreamer):
    def __init__(self, base_streamer, f, inplaced):
        super().__init__(base_streamer)
        self.f = f
        self.inplaced = inplaced

    def _next(self):
        one = self.base_streamer_.next()
        if self.base_streamer_.is_eos(one):
            return None
        z = self.f(one)
        if self.inplaced:
            return one
        else:
            return z

# f: Inst -> List[Inst] (augment or filter)
class FListAdapterStream(AdapterStreamer):
    def __init__(self, base_streamer, f):
        super().__init__(base_streamer)
        self.f = f
        self.cache = []
        self.cache_idx = 0

    def _next(self):
        while self.cache_idx >= len(self.cache):
            one = self.base_streamer_.next()
            if self.base_streamer_.is_eos(one):
                return None
            self.cache.clear()
            self.cache_idx = 0
            self.cache.extend(self.f(one))
        r = self.cache[self.cache_idx]
        self.cache_idx += 1
        return r

# =====
# Random Streams

# inf stream of random numbers
# todo(warn): f will be: {lambda size: Random.?(?, size=size)}
class RandomStreamer(Streamer):
    def __init__(self, f, batch_size=1024):
        super().__init__()
        #
        self.f_ = f
        self.bs_ = batch_size
        self.cache_ = None
        self.idx_ = batch_size

    # can never end!
    def _next(self):
        bs = self.bs_
        idx = self.idx_
        if idx >= bs:
            self.cache_ = self.f_(bs)
            idx = 0
        x = self.cache_[idx]
        self.idx_ = idx + 1
        return x

    def _restart(self):
        pass

    # typical usage
    @staticmethod
    def get_random_bool_streamer(true_rate, batch_size=1024):
        return RandomStreamer(lambda size: Random.random_bool(true_rate, size, "data"), batch_size)

# randomly drop instances
# todo(warn): be careful not to use with Cache
class DropStreamer(AdapterStreamer):
    def __init__(self, base_streamer, sample_rate, batch_size=1024):
        super().__init__(base_streamer)
        #
        self.rs_ = RandomStreamer.get_random_bool_streamer(sample_rate, batch_size)
        self.rs_.restart()

    def _next(self):
        while True:
            one = self.base_streamer_.next()
            if self.base_streamer_.is_eos(one):
                return None
            if self.rs_.next():
                return one

# =====
# Multi Streamers

# input from multiple streams
class MultiStreamer(Streamer):
    def __init__(self, base_streams: Sequence):
        super().__init__()
        #
        self.base_streamers_ = list(base_streams)
        self.num_streamers_ = len(self.base_streamers_)

#
# simple concat multi-streamer
# -- DropStreamer + MultiCat + ShuffleStreamer can be a good mixer
class MultiCatStreamer(MultiStreamer):
    def __init__(self, base_streamers: Sequence):
        super().__init__(base_streamers)
        self._mc_set_cur(0)

    def _mc_set_cur(self, i):
        self.cur_idx_ = i
        self.cur_streamer_ = None if self.cur_idx_>=self.num_streamers_ else self.base_streamers_[self.cur_idx_]

    def _next(self):
        while True:
            if self.cur_streamer_ is None:
                return None
            one = self.cur_streamer_.next()
            if self.cur_streamer_.is_eos(one):
                self._mc_set_cur(self.cur_idx_+1)
            else:
                return one

    def _restart(self):
        for one in self.base_streamers_:
            one.restart()
        self._mc_set_cur(0)

#
# like MultiCat, but join the streams (horizontally) rather than concat (vertically)
# todo(note): currently mixing them 1 by 1; maybe not efficient enough if there are too many streamers
class MultiJoinStreamer(MultiStreamer):
    def __init__(self, base_streamers: Sequence):
        super().__init__(base_streamers)
        self.cur_pointer = 0
        self.ended = [False] * len(self.base_streamers_)

    def _next(self):
        # find the next active streamer
        starting_point = self.cur_pointer
        num_streamers = self.num_streamers_
        cur_point = starting_point
        flag_success = False
        one = None
        for i in range(num_streamers):  # at most travel one round
            if not self.ended[cur_point]:
                new_streamer = self.base_streamers_[cur_point]
                one = new_streamer.next()
                if new_streamer.is_eos(one):
                    self.ended[cur_point] = True
                else:
                    flag_success = True
            cur_point = (cur_point + 1) % num_streamers
            if flag_success:
                break
        self.cur_pointer = cur_point
        if flag_success:
            return one
        else:
            return None

    def _restart(self):
        for one in self.base_streamers_:
            one.restart()
        self.cur_pointer = 0
        self.ended = [False] * len(self.base_streamers_)

#
# zip-like multi-streamer, with various modes, return a list of instances
# -- can be useful to combine with LoopStreamer for wrapping-around
# todo(warn): change IO-shape
# stop_sidx: stopped based on which base-streamer, -1 means auto
# auto_mode: 'any' or 'all'
class MultiZipStreamer(MultiStreamer):
    def __init__(self, base_streamers: Sequence, stop_sidx=-1, auto_mdoe="any", padding=None):
        super().__init__(base_streamers)
        self.stop_sidx = stop_sidx
        self.padding = padding
        if self.stop_sidx > 0:
            auto_mdoe = "all"
        self.auto_stop_f = {"any": (lambda: self.active_num<self.num_streamers_),
                            "all": (lambda: self.active_num==0)}[auto_mdoe]
        #
        self.active_num = 0

    def _next(self):
        rets = [self.padding] * self.num_streamers_
        for idx, ss in enumerate(self.base_streamers_):
            if ss.is_active():
                one = ss.next()
                if ss.is_eos(one):
                    self.active_num -= 1
                else:
                    rets[idx] = one
        if self.stop_sidx > 0:
            if not self.base_streamers_[self.stop_sidx].is_active():
                return None
        else:
            if self.auto_stop_f():
                return None
        # todo(+2): did not put-back the instances of the un-eos stream, may bring troubles if continue to use them!
        return rets

    def _restart(self):
        for one in self.base_streamers_:
            one.restart()
        self.active_num = self.num_streamers_

# =====
# Cacher

class Cache(object):
    def put(self, one):
        raise NotImplementedError()

    def get(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def clear(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

class InplacedCache(Cache):
    def __init__(self, shuffle=False):
        self.c = []
        self.ptr = 0
        self.shuffle = shuffle

    def __len__(self):
        return len(self.c)

    def put(self, one):
        # todo(warn): always append at the end
        self.c.append(one)

    def get(self):
        if self.ptr >= len(self.c):
            # todo(warn): None as special sentinel
            return None
        else:
            self.ptr += 1
            return self.c[self.ptr-1]

    def clear(self):
        self.c.clear()
        self.ptr = 0

    def reset(self):
        self.ptr = 0
        if self.shuffle:
            Random.shuffle(self.c, "data")

# read from src at the first pass, later stream from cache
# always being eager reader
class InstCacher(Streamer):
    def __init__(self, src_stream, cache_builder=InplacedCache, shuffle=False):
        super().__init__()
        self.src = src_stream
        self.cache = cache_builder(shuffle=shuffle)

    def _next(self):
        return self.cache.get()

    def _restart(self):
        if self.restart_times_==0:
            for one in self.src:
                self.cache.put(one)
        self.cache.reset()

# todo(+1): partial shuffle to avoid read all?
class ShuffleStreamer(AdapterStreamer):
    # -1 means read all and shuffle
    def __init__(self, src_stream, cache_builder=InplacedCache):
        super().__init__(src_stream)
        self.src = src_stream
        self.cache = cache_builder(shuffle=True)

    def _next(self):
        return self.cache.get()

    def _restart(self):
        # todo(note): rebuild cache each time (do not need to call super, since for-loop will trigger the src.restart)
        self.cache.clear()
        for one in self.src:
            self.cache.put(one)
        self.cache.reset()

# =====
# BatchHelper

# handling the batching of instances, also filtering, sorting, recording, etc.
# streamer: base, batch_size: sum(batch_size_f(z) for z), maxibatch_size: read-in bs*mbs every time,
# dump_detectors, single_detectors, sorting_keyer: sort bs*mbs, shuffling: shuffle on buckets in bs*mbs
# todo(warn): change IO-shape
class BatchArranger(AdapterStreamer):
    def __init__(self, streamer, batch_size, maxibatch_size, batch_size_f, dump_detectors, single_detectors, sorting_keyer, shuffling):
        super(BatchArranger, self).__init__(streamer)
        self.batch_size = batch_size
        self.batch_size_f = (lambda one: 1) if batch_size_f is None else batch_size_f
        # todo(notice): if <=0 then read all at one time and possibly sort all
        self.maxibatch_size = maxibatch_size if maxibatch_size>0 else Constants.INT_PRAC_MAX
        if dump_detectors is not None and not isinstance(dump_detectors, Iterable):
            dump_detectors = [dump_detectors]
        self.dump_detectors = [] if dump_detectors is None else dump_detectors
        if single_detectors is not None and not isinstance(single_detectors, Iterable):
            single_detectors = [single_detectors]
        self.single_detectors = [] if single_detectors is None else single_detectors
        self.sorting_keyer = sorting_keyer  # default(None) no sorting
        self.shuffling = shuffling  # shuffle inside the maxi-batches?
        # caches
        self.buffered_bsize_ = 0
        self.buffer_ = []           # list of instances
        self.buckets_ = []          # list of already prepared batch of instances

    @property
    def k(self):
        # (nearly) sort size or cache size
        return self.batch_size * self.maxibatch_size

    # getter and checker for batch_size (might have unknown effects if changed in the middle)
    def bsize(self, bs=None):
        if bs is None:
            return self.batch_size
        else:
            self.batch_size = int(bs)
            return self.batch_size

    # restart the status including the streamer
    def _restart(self):
        super()._restart()
        self.buffered_bsize_ = 0
        self.buffer_ = []
        self.buckets_ = []

    # get the next mini-batch
    def _next(self):
        # buffered read
        if len(self.buckets_) == 0:
            # read into buffer
            while self.buffered_bsize_ < self.k:
                one = self.base_streamer_.next()
                if self.base_streamer_.is_eos(one):
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
                # sorting
                sorted_buffer = self.buffer_
                if self.sorting_keyer is not None:
                    sorted_buffer.sort(key=self.sorting_keyer)      # small first
                # prepare buckets
                buckets = []
                tmp_bsize = 0
                tmp_bucket = []
                for one in sorted_buffer:
                    tmp_bsize += self.batch_size_f(one)
                    tmp_bucket.append(one)
                    if tmp_bsize >= self.batch_size:
                        buckets.append(tmp_bucket)
                        tmp_bsize = 0
                        tmp_bucket = []
                if len(tmp_bucket) > 0:
                    buckets.append(tmp_bucket)
                # another shuffle?
                if self.shuffling:
                    Random.shuffle(buckets, "data")
                else:
                    # todo(warn): to keep sorting-order if sorting else original-order BECAUSE-OF later POP
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
