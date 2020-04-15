# misc
import sys, re
from collections import Iterable
import numpy as np
from .log import zopen, zlog
from .check import zcheck, zfatal

class Constants(object):
    # number ranges (although not actual ranges for python, but should be ok practically)
    PRAC_NUM_ = 12345678
    INT_PRAC_MAX = int(PRAC_NUM_)
    INT_PRAC_MIN = -INT_PRAC_MAX
    REAL_PRAC_MAX = float(PRAC_NUM_)
    REAL_PRAC_MIN = -REAL_PRAC_MAX
    #
    # INT_MAX = 2**31-1
    # INT_MIN = -2**31+1
    # REAL_MAX = sys.float_info.max
    # REAL_MIN = -REAL_MAX

class StrHelper(object):
    NUM_PATTERN = re.compile(r"[0-9]+|[0-9]+\.[0-9]*|[0-9]+[0-9,]+|[0-9]+\\/[0-9]+|[0-9]+/[0-9]+")

    # todo(warn): heurist simple rule
    @staticmethod
    def norm_num(w, norm_to="1", leave_single=True):
        if (leave_single and len(w) <= 1) or (not re.fullmatch(StrHelper.NUM_PATTERN, w)):
            return w
        else:
            return norm_to

class NumHelper:
    # truncate float
    @staticmethod
    def truncate_float(f, digits):
        factor = 10 ** digits
        return int(f * factor) / factor

class Helper(object):
    # -- List Helper
    @staticmethod
    def join_list(list_iter):
        ret = []
        for one in list_iter:
            ret.extend(one)
        return ret

    @staticmethod
    def split_list(one, num_split):
        rets = []
        n_each = (len(one) + num_split - 1) // num_split
        cur_idx = 0
        for i in range(num_split):
            # todo(0): in python, does not care about over-boundary for slicing
            rets.append(one[cur_idx:cur_idx+n_each])
            cur_idx += n_each
        return rets

    @staticmethod
    def get_index(one, them, fail=-1):
        try:
            pos = them.index(one)      # position
        except ValueError:
            pos = fail
        return pos

    @staticmethod
    def argmin_and_min(them):
        if len(them) == 0:
            return -1, Constants.REAL_PRAC_MAX
        idx = int(np.argmin(them))
        return idx, them[idx]

    @staticmethod
    def argmax_and_max(them):
        if len(them) == 0:
            return -1, Constants.REAL_PRAC_MIN
        idx = int(np.argmax(them))
        return idx, them[idx]

    @staticmethod
    def accu(i0, ones, func):
        x = i0
        for one in ones:
            x = func(x, one)
        return x

    @staticmethod
    def accu_inp(i0, ones, func):
        for one in ones:
            func(i0, one)

    # -- Dictionary helper
    # input: Iterable of values, return: {value: counts}
    @staticmethod
    def count_freq(stream):
        word_freqs = {}
        # read
        for w in stream:
            if w not in word_freqs:
                word_freqs[w] = 1
            else:
                word_freqs[w] += 1
        return word_freqs

    # rank-highest freq(value) of a dictionary, return keys
    @staticmethod
    def rank_key(v):
        words = [w for w in v]
        words.sort(key=lambda x: (v[x], x), reverse=True)
        return words

    # {key->idx} -> {idx->key}
    @staticmethod
    def reverse_idx(v):
        # reverse
        final_words = [None] * len(v)
        for tok in v:
            idx = v[tok]
            zcheck(final_words[idx] is None, "Repeated idx at idx=%d, old=%s, new=%s." % (idx, final_words[idx], tok))
            final_words[idx] = tok
        return final_words

    # on sorted values, return list of [highest-rank, lowest-rank] (ignoring values by filter)
    @staticmethod
    def build_ranges(values, ignore_f=lambda x: x is None, ignore_return=(-1,-1)):
        ranges = []
        equal_idxes = []    # current idxes for equal values
        #
        def end_group():
            if len(equal_idxes)>0:
                one_range = [equal_idxes[0], equal_idxes[-1]]
                for one_idx in equal_idxes:
                    ranges[one_idx] = one_range
                equal_idxes.clear()
        #
        for idx, val in enumerate(values):
            if ignore_f(val):   # not concerned
                ranges.append(ignore_return)
            else:
                if len(equal_idxes) > 0:
                    prev_val = values[equal_idxes[0]]
                    if val == prev_val:
                        pass
                    elif val < prev_val:
                        end_group()
                    else:
                        zfatal("Unsorted value list for building ranges.")
                equal_idxes.append(idx)
                ranges.append(None)     # placeholder
        end_group()
        return ranges

    # light-weighted stat
    @staticmethod
    def stat_addone(x, k, v=1, d=0):
        c = x.get(k, d)
        x[k] = c + v

    @staticmethod
    def stat_addv(x, kv):
        for k, v in kv.items():
            Helper.stat_addone(x, k, v)

    # pretty print
    @staticmethod
    def printd(d, sep="\n"):
        zlog(Helper.printd_str(d, sep))

    @staticmethod
    def printl(vs, sep="\n"):
        zlog(sep.join([str(v) for v in vs]))

    @staticmethod
    def printd_str(d, sep="\n"):
        thems = []
        for k in sorted(list(d.keys())):
            thems.append("%s: %s" % (k, d[k]))
        return sep.join(thems)

    # -- General Helper: can be recursive
    @staticmethod
    def stream_on_file(fd, tok=(lambda x: x.strip().split())):
        for line in fd:
            for w in tok(line):
                yield w

    @staticmethod
    def stream_rec(obj, leaf=(lambda x: False)):
        if leaf(obj):
            yield obj
        elif isinstance(obj, Iterable):
            for x in obj:
                for y in Helper.stream_rec(x, leaf):
                    yield y
        else:
            yield obj

    @staticmethod
    def flatten(obj, leaf=(lambda x: False)):
        return list(Helper.stream_rec(obj, leaf=leaf))

    @staticmethod
    def argmax(v, lowest):
        if isinstance(v, dict):
            keys = v.keys()
        else:
            keys = range(len(v))
        midx = None
        mval = lowest
        for k in keys:
            cur_val = v[k]
            if cur_val > mval:
                midx = k
                mval = cur_val
        return midx

    @staticmethod
    def apply_keys(v, keys):
        cur_v = v
        for k in keys:
            cur_v = cur_v[k]
        return cur_v

    # ===== check end of Iterator
    @staticmethod
    def check_end_iter(iter):
        try:
            x = next(iter)
            zfatal("Failed end-check of iter: get %s." % str(x))
        except StopIteration:
            pass

    # =====
    @staticmethod
    def check_is_range(idxes, length):
        return len(idxes) == length and all(i == v for i, v in enumerate(idxes))

class ZObject(object):
    def __init__(self, _m=None, **kwargs):
        if _m is not None:
            self.update(_m)
        self.update(kwargs)

    def update(self, m):
        if isinstance(m, ZObject):
            m = m.__dict__
        for k, v in m.items():
            setattr(self, k, v)

    def __repr__(self):
        return str(self.__dict__)
