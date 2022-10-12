#

# misc of utils

__all__ = [
    "Constants", "StrHelper", "DictHelper", "NumHelper", "OtherHelper", "ZObject", "ZIndexer", "ZCachedValue",
    "ZRuleFilter", "AllInSet",
]

from typing import Dict, Callable, Type, Union, List
import sys, re
import numpy as np
import pandas as pd
from collections import Counter
from .log import zlog

# some useful constant values
class Constants:
    # number ranges (although not actual ranges for python, but should be ok practically for most of the cases)
    PRAC_NUM_ = 10000000  # 10M
    INT_PRAC_MAX = int(PRAC_NUM_)
    INT_PRAC_MIN = -INT_PRAC_MAX
    REAL_PRAC_MAX = float(PRAC_NUM_)
    REAL_PRAC_MIN = -REAL_PRAC_MAX
    #
    # INT_MAX = 2**31-1
    # INT_MIN = -2**31+1
    # REAL_MAX = sys.float_info.max
    # REAL_MIN = -REAL_MAX

# helper for str
class StrHelper:
    NUM_PATTERN = re.compile(r"[0-9]+|[0-9]+\.[0-9]*|[0-9]+[0-9,]+|[0-9]+\\/[0-9]+|[0-9]+/[0-9]+")

    # todo(note): heurist simple rule
    @staticmethod
    def norm_num(w: str, norm_to="1", leave_single=True):
        if (leave_single and len(w) <= 1) or (not re.fullmatch(StrHelper.NUM_PATTERN, w)):
            return w
        else:
            return norm_to

    # find the char positions of all the pieces
    @staticmethod
    def index_splits(s: str, pieces: List[str]):
        cur_start = 0
        rets = []
        for p in pieces:
            # todo(+N): to handle index error?
            posi_start = s.index(p, cur_start)
            rets.append((posi_start, len(p)))
            cur_start = posi_start + len(p)
        return rets

    # delete all space
    @staticmethod
    def delete_spaces(s: str):
        return re.sub(r'\s', '', s)

    # add prefix for each split
    @staticmethod
    def split_prefix_join(s: str, prefix: str, sep: str = None):
        fileds = [prefix+z for z in s.split(sep)]
        if sep is None:
            sep = " "  # make a default one
        return sep.join(fileds)

# helper on dict
class DictHelper:
    # rank-highest freq(value) of a dictionary, return keys
    @staticmethod
    def sort_key(v: Dict, reverse=False):
        words = list(v.keys())
        words.sort(key=lambda x: (v[x], x), reverse=reverse)
        return words

    # update but assert non-existing
    @staticmethod
    def update_dict(v: Dict, v2: Dict, key_prefix="", assert_nonexist=True):
        for _k, _v in v2.items():
            _k2 = key_prefix + _k
            if assert_nonexist:
                assert _k2 not in v
            v[_k2] = _v

    # show info counts
    @staticmethod
    def get_counts_info_table(counts: Dict, key=None):
        sorted_keys = sorted(counts.keys(), key=key)
        # --
        res = []
        accu_counts = 0
        for i, k in enumerate(sorted_keys):
            accu_counts += counts[k]
            res.append([i, k, counts[k], 0., accu_counts, 0.])
        d = pd.DataFrame(res, columns=["Idx", "Key", "Count", "Perc.", "ACount", "APerc."])
        d["Perc."] = d["Count"] / accu_counts
        d["APerc."] = d["ACount"] / accu_counts
        return d

# helper on number
class NumHelper:
    # truncate float
    @staticmethod
    def truncate_float(f: float, digits: int):
        factor = 10 ** digits
        return int(f * factor) / factor

# other helper
class OtherHelper:
    @staticmethod
    def printd(d, sep="\n", **kwargs):
        zlog(OtherHelper.printd_str(d, sep, **kwargs))

    @staticmethod
    def printl(vs, sep="\n"):
        zlog(sep.join([str(v) for v in vs]))

    @staticmethod
    def printd_str(d, sep="\n", try_div=False, **kwargs):
        thems = []
        for k in sorted(list(d.keys())):
            k2, _divs = k, []
            if try_div:  # try to div by sth.
                while "_" in k2:
                    k2 = "_".join(k2.split("_")[:-1])
                    _div = d.get(k2)
                    if _div is not None:
                        _divs.append(_div)
            _res_divs = ", ".join([f"{d[k]/z:.2f}" if z!=0 else 'nan' for z in _divs])
            thems.append(f"{k}: {d[k]}" + (f" ({_res_divs})" if _res_divs else ""))
        return sep.join(thems)

    @staticmethod
    def get_module(obj: object):
        return sys.modules[obj.__module__]

    @staticmethod
    def take_first_samples(insts: List, ratio: float):
        _s = ratio
        if _s <= 1.0:
            _s = int(len(insts)*_s+0.99)
        else:
            _s = int(_s)
        return insts[:_s]

# general object
class ZObject:
    def __init__(self, _m: Dict = None, **kwargs):
        self.update(_m, **kwargs)

    def update(self, _m: Dict = None, **kwargs):
        if _m is not None:
            if isinstance(_m, ZObject):
                _m = _m.__dict__
            for k, v in _m.items():
                setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __repr__(self):
        return str(self.__dict__)

# indexer
class ZIndexer:
    def __init__(self):
        self._chs_map = {}  # k -> object
        self._counter_map = Counter()  # DataType -> count (for get new id)

    def reset(self):
        self._chs_map.clear()
        self._counter_map.clear()

    def register(self, id: str, v: object, new_name: str = None):
        if id is None:  # make a new name
            # todo(+N): workable but not a very elegant solution?
            new_idx = self._counter_map.get(new_name, 0)  # starting with 0
            id = f"{new_name}{new_idx}"
            while id in self._chs_map:  # until find a new name!
                new_idx += 1
                id = f"{new_name}{new_idx}"
            self._counter_map[new_name] += 1
        assert id not in self._chs_map, f"ID already exists in the Index: {id} -> {self._chs_map[id]}"
        self._chs_map[id] = v
        return id

    # remove all names with certain prefix and reset the counter!
    def clear_name(self, name: str):
        if name in self._counter_map:
            upper_idx = self._counter_map[name]
            del self._counter_map[name]
            for i in range(upper_idx):
                self.remove(f"{name}{i}")

    def lookup(self, id: str, df=None):  # search
        return self._chs_map.get(id, df)

    # return whether removed successfully
    def remove(self, id: str):
        if id in self._chs_map:
            del self._chs_map[id]  # todo(note): notice that we do not revert _counter_map
            return True
        return False

# cached value
class ZCachedValue:
    def __init__(self, f):
        self.f = f
        self._evaled = False
        self._v = None

    @property
    def value(self):
        if not self._evaled:
            self._v = self.f()
        return self._v

# --
# Rule Filter

class AllInSet(set):
    def __contains__(self, item):
        return True

class ZRuleFilter:
    def __init__(self, filter_rules: List[str], name_book: Dict, default_value: bool, name2keys=(lambda x: [x])):
        self.rules = []  # (+/-, set)
        assert '*' not in name_book
        name_book['*'] = AllInSet()  # shortcut for ALL
        for r in filter_rules:
            _judge = {'+': True, '-': False}[r[0]]
            _set = name_book.get(r[1:], {r[1:]})
            self.rules.append((_judge, _set))
        self.default_value = default_value
        self.name2keys = name2keys
        # --

    def filter_by_name(self, name: str):
        for _judge, _set in self.rules:
            for n in self.name2keys(name):
                if n in _set:
                    return _judge
        return self.default_value
