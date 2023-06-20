#

# data instance

import math
from typing import Dict, List, Union, Type, Callable
from collections import Counter
from mspx.utils import Serializable, Conf, zglob1, default_json_serializer, zlog, Random, get_global_conf

class DataInst(dict, Serializable):
    # get or set
    def get_or_set_field(self, name, cls):
        ret = self.get(name, None)
        if ret is None:
            ret = cls()
            self[name] = ret
        return ret

    @property
    def cache(self): return self.get_or_set_field('_cache', dict)  # no store!

    def has_cache(self, key):
        return key in self.cache

    def get_cache(self, key, df=None):
        return self.cache.get(key, df)

    def set_cache(self, key, value):
        self.cache[key] = value

    def del_cache(self, key):
        ret = None
        if key in self.cache:
            ret = self.cache[key]
            del self.cache[key]
        return ret

    # --
    def to_dict(self, store_type=False, store_all_fields=False):
        assert not store_type and not store_all_fields
        if '_cache' in self:
            ret = self.copy()
            del ret['_cache']
        else:
            ret = self
        return ret

    def from_dict(self, data: Dict):
        self.update(data)  # note: simply directly update!

class DataConf(Conf):
    def __init__(self):
        self.path = ""  # path
        self.shuffle_times = 0  # shuffle data?
        self.shuffle_seed = 0
        # sampling options
        self.sample_balance = 'label'  # balance which for sampling, '' means nope
        self.sample_start = 0
        self.sample_count = -1

def decide_range(x, length):
    if x in ['', 'None']:
        return None
    x = float(x)
    if abs(x) >= 1:
        assert float(int(x)) == x, "Must be an integer"
        return int(x)
    else:  # note: 0 is still 0!
        sign = -1 if x<0 else 1
        vv = int(length * abs(x))
        return sign*vv

# loading + simple sampling
def obtain_data(conf: DataConf, **kwargs):
    if kwargs:
        conf = DataConf.direct_conf(conf, copy=True, **kwargs)
    # --
    # loading
    path = conf.path
    if path.startswith(":"):  # use datasets
        import datasets
        _path, _name, _split = path.split(":")[1:]
        cache_dir = zglob1(get_global_conf(['utils', 'global_cache_dir']))
        _load = datasets.load_dataset(_path, _name, split=_split, cache_dir=cache_dir)
    else:
        _path = zglob1(path)
        _load = default_json_serializer.yield_iter(_path)
    ret = _load
    zlog(f"Load data from {path}: Instances = {len(ret) if hasattr(ret, '__len__') else 'UNK'}")
    # --
    if conf.shuffle_times > 0:
        ret = list(ret)  # note: must get all!
        _gen = Random.get_np_generator(conf.shuffle_seed)
        for _ in range(conf.shuffle_times):
            _gen.shuffle(ret)
        zlog(f"Shuffling data: seed = {conf.shuffle_seed}")
    # --
    # simple sampling
    if conf.sample_count > 0:
        _bkey = conf.sample_balance
        _all_insts = list(ret)  # note: must get all!
        _balance_cc = Counter([z.get(_bkey, None) for z in _all_insts])
        _label_budget = int(math.ceil(conf.sample_count / len(_balance_cc)))  # budget for each label
        _label_cc = Counter()
        # --
        ret = []
        for _inst in _all_insts[conf.sample_start:]:
            if len(ret) >= conf.sample_count:
                break
            _ikey = _inst.get(_bkey, None)
            if _label_cc[_ikey] >= _label_budget:
                continue
            _label_cc[_ikey] += 1
            ret.append(_inst)
        zlog(f"Sample data[{conf.sample_start} + {conf.sample_count}]: "
             f"{_balance_cc}[{sum(_balance_cc.values())}] -> {_label_cc}[{sum(_label_cc.values())}]")
    # --
    final_ret = (DataInst(z) for z in ret)
    yield from final_ret
