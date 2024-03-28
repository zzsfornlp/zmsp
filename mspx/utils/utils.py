#

# misc of utils

__all__ = [
    "Constants", "ZHelper", "ZObject", "ZResult", "AllIncSet", "ZArr",
]

from typing import Dict, Callable, Type, Union, List, SupportsFloat, Iterable, Sequence
import sys, re, os
from collections import Counter
from .log import zlog, zwarn
from .reg import Registrable
from .seria import Serializable
from .system import zglob

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

# misc helper
class ZHelper:
    NUM_PATTERN = re.compile(r"[0-9]+|[0-9]+\.[0-9]*|[0-9]+[0-9,]+|[0-9]+\\/[0-9]+|[0-9]+/[0-9]+")

    # todo(note): heurist simple rule
    @staticmethod
    def norm_num(w: str, norm_to="1", leave_single=True):
        if (leave_single and len(w) <= 1) or (not re.fullmatch(ZHelper.NUM_PATTERN, w)):
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

    # rank-highest freq(value) of a dictionary, return keys
    @staticmethod
    def sort_key(v: Dict, reverse=False):
        words = list(v.keys())
        words.sort(key=lambda x: (v[x], x), reverse=reverse)
        return words

    # update but assert non-existing
    @staticmethod
    def update_dict(v: Dict, v2: Dict, key_prefix="", assert_nonexist=True, adding_init=None):
        for _k, _v in v2.items():
            _k2 = key_prefix + _k
            if assert_nonexist:
                assert _k2 not in v
            if adding_init is not None:
                v[_k2] = v.get(_k2, adding_init) + _v
            else:
                v[_k2] = _v

    # show info counts
    @staticmethod
    def get_counts_info_table(counts: Dict, key=None, keys=None):
        import pandas as pd
        sorted_keys = sorted(counts.keys(), key=key) if keys is None else keys
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

    @staticmethod
    def printd(d, sep="\n", **kwargs):
        zlog(ZHelper.printd_str(d, sep, **kwargs))

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
            _s = int(len(insts)*_s+0.99999)
        else:
            _s = int(_s)
        return insts[:_s]

    # get a new key
    @staticmethod
    def get_new_key(d: Dict, prefix: str):
        ii = len(d)  # note: simply start from len
        while True:
            ret = f"{prefix}{ii}"
            if ret not in d:
                return ret
            ii += 1
        # --

    @staticmethod
    def yield_batches(stream, batch_size: int, batch_size_f=(lambda x: 1)):
        cur_bs = 0
        ret = []
        for one in stream:
            if cur_bs >= batch_size:
                if len(ret) > 0:
                    yield ret
                cur_bs = 0
                ret = []
            cur_bs += batch_size_f(one)
            ret.append(one)
        if len(ret) > 0:  # remaining ones
            yield ret

    @staticmethod
    def pad_strings(items: Iterable, pad: str, pad_left=True):
        items = list(items)
        if len(items) == 0:
            return 0
        items = [str(z) for z in items]
        pad = str(pad)
        assert len(pad) == 1
        mm = max(len(z) for z in items)
        ret = []
        for z in items:
            _p = pad * (mm-len(z))
            ret.append((_p+z) if pad_left else (z+_p))
        return ret

    @staticmethod
    def eval_ff(ff_str: str, default_args="", globals=None, locals=None):
        ff_str = ff_str.strip()
        if default_args:
            if not ff_str.startswith("lambda "):
                ff_str = f"lambda {default_args}: {ff_str}"
        ret = eval(ff_str, globals, locals)
        return ret

    @staticmethod
    def insert_path(path: str, s: str, position=-1, sep='.'):
        dir_name, file_name = os.path.dirname(path), os.path.basename(path)
        fields = file_name.split(None if sep=='' else sep)
        if position < 0:  # note: make the semantics similar to torch!
            if position == -1:
                fields.append(s)
            else:
                fields.insert(position+1, s)
        else:
            fields.insert(position, s)
        file2 = (sep if sep is not None else "").join(fields)
        ret = os.path.join(dir_name, file2)
        return ret

    @staticmethod
    def resort_dict(d: dict, key=None, recursive=True):
        if key is None:
            key = lambda x: x[0]  # use key
        sorted_items = sorted(d.items(), key=key)
        ret = {}  # note: after py3.7, dict are sorted by insertion!
        for kk, vv in sorted_items:
            if recursive and isinstance(vv, dict):
                vv = ZHelper.resort_dict(vv, key=key, recursive=recursive)
            ret[kk] = vv
        return ret

    @staticmethod
    def rec_getattr(obj, path, df=None):
        # recursively get attribute
        curr = obj
        for p in path:
            if curr is None:
                return df
            curr = getattr(curr, p, None)
        return curr

    @staticmethod
    def get_choice(val, *choices):
        res = tuple([val==z for z in choices])
        assert sum(res) == 1, f"Error for not choosing only one option: {choices} -> {res}"
        return res

    @staticmethod
    def get_check_all(vals, df):
        if len(vals) == 0: return df
        assert all(z==vals[0] for z in vals), f"Error for not consistent vals: {vals}"
        return vals[0]

    @staticmethod
    def iter_fd(fd, use_tqdm=True, update_interval=1, description=""):
        if use_tqdm:
            from tqdm import tqdm
            start = fd.tell()
            end = fd.seek(0, os.SEEK_END)
            fd.seek(start)
            _accu = 0
            with tqdm(total=end-start, disable=(not tqdm)) as pbar:
                pbar.set_description(description)
                _last = fd.tell()
                line = fd.readline()
                while line:
                    yield line
                    line = fd.readline()
                    # --
                    _accu += 1
                    if _accu >= update_interval:
                        _accu = 0
                        _now = fd.tell()
                        pbar.update(_now-_last)
                        _last = _now
                    # --
        else:
            yield from fd

    @staticmethod
    def get_np_type_from_name(name: str, df: Union[str, int]):
        import numpy as np
        # --
        def _find_dtype():
            for one in ["uint16", "uint32", "uint64", "int16", "int32", "int64"]:
                for f0 in reversed(os.path.split(name)):
                    for f1 in reversed(f0.split(".")):
                        if one in f1:
                            return one
            return None
        # --
        _dtype = _find_dtype()
        if _dtype is None:
            _dtype = df if isinstance(df, str) else ("uint32" if df>=2**16 else "uint16")
            dtype = getattr(np, _dtype)
            zlog(f"Use default dtype {dtype} from hit={_dtype}, name={name}")
        else:
            dtype = getattr(np, _dtype)
            zlog(f"Get dtype {dtype} from hit={_dtype}, name={name}")
        return dtype

    @staticmethod
    def iter_from_fd(fd, tqdm_enabled=True):
        if tqdm_enabled:
            from tqdm import tqdm
            from pathlib import Path
            with tqdm(total=Path(fd.name).stat().st_size, disable=(not tqdm_enabled)) as pbar:
                while True:
                    line = fd.readline()
                    if not line:
                        break
                    pbar.update(fd.tell() - pbar.n)
                    yield line
        else:
            yield from fd

    @staticmethod
    def recursive_apply(item, f_target=None, f_modify=None, inplace=False):
        if f_target is None:
            f_target = (lambda x: False)
        if f_modify is None:
            f_modify = (lambda x: x)
        # --
        if f_target(item):
            return f_modify(item)  # note: f_modify should always return things even if inplaced!
        elif isinstance(item, dict):
            if inplace:
                for k in list(item.keys()):
                    item[k] = ZHelper.recursive_apply(item[k], f_target=f_target, f_modify=f_modify, inplace=inplace)
            else:
                return {k: ZHelper.recursive_apply(v, f_target=f_target, f_modify=f_modify, inplace=inplace) for k, v in item.items()}
        elif isinstance(item, (list, tuple)):
            if inplace:
                for ii in range(len(item)):
                    item[ii] = ZHelper.recursive_apply(item[ii], f_target=f_target, f_modify=f_modify, inplace=inplace)  # will fail if tuple
            else:
                return type(item)([ZHelper.recursive_apply(z, f_target=f_target, f_modify=f_modify, inplace=inplace) for z in item])
        else:
            return item

# useful general object
@Registrable.rd('zobj')
class ZObject(Serializable):
    def __init__(self, _m: Dict = None, **kwargs):
        self.update(_m, **kwargs)

    def update(self, _m: Dict = None, **kwargs):
        if _m is not None:
            if isinstance(_m, ZObject):
                _m = _m.__dict__
            for k, v in _m.items():
                setattr(self, str(k), v)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def assign_trg(self, trg, use_trg_names=True):
        if isinstance(trg, (Callable, type)):
            ret = trg()
        else:  # directly use as instance
            ret = trg
        if use_trg_names:
            for k in trg.__dict__.keys():
                if hasattr(self, k):
                    setattr(trg, k, getattr(self, k))
        else:
            for k, v in self.__dict__.items():
                setattr(trg, k, v)
        return ret

    def __getitem__(self, item):
        _sep = '...'  # note: easier accessing!
        if _sep in item:
            item0, item1 = item.split(_sep, 1)
            return self.__dict__[item0][item1]
        else:
            return self.__dict__[item]

    def __setitem__(self, key, value): self.__dict__[key] = value
    def __contains__(self, item): return item in self.__dict__
    def __repr__(self): return str(self.__dict__)
    def keys(self): return self.__dict__.keys()
    def items(self): return self.__dict__.items()
    def get(self, key, df=None): return self.__dict__.get(key, df)

@Registrable.rd('zres')
class ZResult(ZObject):
    RES_KEY = "zres"  # result (float)
    DES_KEY = "zdes"  # description (str)

    def __init__(self, _m: Union[Dict, SupportsFloat] = None, res: SupportsFloat = None, des: str = None, **kwargs):
        if isinstance(_m, (float, int)):
            res = _m  # allow this!
            _m = {}
        super().__init__(_m, **kwargs)
        for k, v in zip([ZResult.RES_KEY, ZResult.DES_KEY], [res, des]):
            if v is not None:
                if hasattr(self, k):
                    zwarn(f"ZResult's {k} already exists, rewrite it: {getattr(self, k)} -> {v}")
                setattr(self, str(k), v)
        # --

    def __repr__(self):
        return self.to_str()

    def to_str(self):
        if self.description is not None:
            des = self.description
        else:
            des0 = self.to_dict()
            for kk in list(des0.keys()):
                vv = getattr(self, kk, None)
                if vv is not None and isinstance(vv, ZResult):  # handle the recursive
                    des0[kk] = vv.to_str()
            des = ", ".join([f"\"{k}\"={v}" for k,v in des0.items()])
        return f"ZResult({float(self):.4f}): {{{des}}}"

    @property
    def result(self): return getattr(self, ZResult.RES_KEY, Constants.REAL_PRAC_MIN)  # the larger the better
    @property
    def description(self): return getattr(self, ZResult.DES_KEY, None)
    def __float__(self): return float(self.result)
    def __lt__(self, other): return float(self) < float(other)
    def __le__(self, other): return float(self) <= float(other)
    def __gt__(self, other): return float(self) > float(other)
    def __ge__(self, other): return float(self) >= float(other)

    @staticmethod
    def stack(results, keys=None, weights=None):
        final_rr, final_res, final_weight = {}, 0., 0.
        for ii, rr in enumerate(results):
            kk = keys[ii] if keys is not None else f"res{ii}"
            ww = weights[ii] if weights is not None else 1.
            final_rr[kk] = rr
            final_res += float(rr)
            final_weight += ww
        ret = ZResult(final_rr, res=final_res/max(1., final_weight))
        return ret

# an all-including set!
class AllIncSet(set):
    def __contains__(self, item):
        return True

# wrapper for np.ndarray[Num]
@Registrable.rd('zarr')
class ZArr(Serializable):
    def __init__(self, arr=None):
        self.arr = arr

    def __repr__(self):
        return f"Zarr{self.arr}"

    def spec_to_dict(self):
        return {'arr': self.arr2obj(self.arr)}

    def spec_from_dict(self, data: Dict):
        return {'arr': self.obj2arr(data['arr'])}

    @staticmethod
    def arr2obj(arr):
        if arr is None:
            return None
        # return arr.tolist()
        import base64
        ret = (arr.shape, str(arr.dtype), base64.b64encode(arr.tobytes()).decode('ascii'))
        return ret

    @staticmethod
    def obj2arr(obj):
        if obj is None:
            return obj
        import numpy as np
        # return np.asarray(obj)
        import base64
        _shape, _dtype, _bytes = obj
        _bytes = base64.b64decode(_bytes)
        arr0 = np.frombuffer(_bytes, dtype=_dtype).reshape(_shape)
        return arr0
