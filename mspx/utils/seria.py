#

# seria: save & load

__all__ = [
    "Serializable", "Serializer",
    "default_json_serializer", "default_pickle_serializer", "get_json_serializer",
    "JsonHelper", "default_json_helper",
]

from typing import Type, Dict, Set, Union, Iterable, IO
import json
import pickle
from .reg import Registrable
from .file import zopen

# =====
# A class that is Serializable (to/from dict)

class Serializable(Registrable):
    """
    A class that supports to_dict and from_dict.
    """

    REC_MAX_LEVEL = 1  # max recursive level
    KEY_TYPE = "__t"  # key for cls-key to store

    # return the fields for seria save
    def seria_fields(self, store_all_fields=False) -> Iterable[str]:
        if store_all_fields:
            return list(self.__dict__.keys())
        else:
            # note: by default, all fields that does not start with "_"
            return [z for z in self.__dict__.keys() if not z.startswith("_")]

    # specific ones to be overridden
    def spec_to_dict(self):
        return {}

    def spec_from_dict(self, data: Dict):
        return {}

    # outside functions
    def to_dict(self, store_type=True, store_all_fields=False):  # inst to dict
        ret = {}
        if store_type:
            ret[Serializable.KEY_TYPE] = self.cls2key()
        spec = self.spec_to_dict()
        for k in self.seria_fields(store_all_fields=store_all_fields):
            if k in spec:  # directly use special ones!
                ret[k] = spec[k]
            else:
                v = getattr(self, k)
                ret[k] = self.rec_to_obj(v, store_type=store_type, store_all_fields=store_all_fields)
        return ret

    def from_dict(self, data: Dict):  # dict to instance (modify/update inplace if items)
        spec = self.spec_from_dict(data)
        for k, d in data.items():  # note: load those in the data
            if k.startswith("__"):
                continue  # note: ignore special ones!
            if k in spec:  # directly use special ones!
                setattr(self, k, spec[k])
            else:
                v0 = getattr(self, k, None)
                if v0 is not None and isinstance(v0, Serializable):
                    v0.from_dict(d)
                else:
                    v1 = Serializable.rec_from_obj(d)
                    setattr(self, k, v1)  # directly set it!
        self.finish_from_dict()
        return self

    def to_file(self, file: str, **kwargs):
        d = self.to_dict(**kwargs)
        return default_json_serializer.to_file(d, file)

    # last step for building in "from_dict"
    def finish_from_dict(self):
        pass

    def from_other(self, other: 'Serializable'):
        return self.from_dict(other.to_dict())

    def copy(self):
        ret = self.__class__()
        ret.from_other(self)
        return ret

    # creating with cls
    @classmethod
    def create_from_dict(cls, data: Dict):
        # if it happens we have a default constructor
        ret = cls()  # note: require a constructor that knows how to build this class at the first place!!
        ret.from_dict(data)
        return ret

    @classmethod
    def create_from_file(cls, file: str):  # shortcut
        d = default_json_serializer.from_file(file)
        if isinstance(d, dict):
            ret = cls.create_from_dict(d)
        else:  # already ok using default_json_serializer
            ret = d
        return ret

    @staticmethod
    def create(data):
        cls_key = data.get(Serializable.KEY_TYPE, None)
        if cls_key is None:
            return data  # no cls!
        else:
            cls = Registrable.key2cls(cls_key)
            return cls.create_from_dict(data)

    # --
    # seria helpers
    @staticmethod
    def rec_to_obj(obj, curr_level=0, **kwargs):
        next_level = curr_level + 1
        if isinstance(obj, Serializable):
            return obj.to_dict(**kwargs)
        elif curr_level >= Serializable.REC_MAX_LEVEL:
            return obj  # no further rec!
        elif isinstance(obj, (list, tuple)):
            return type(obj)([Serializable.rec_to_obj(z, curr_level=next_level) for z in obj])
        elif isinstance(obj, dict):
            return type(obj)([(k, Serializable.rec_to_obj(z, curr_level=next_level)) for k, z in obj.items()])
        else:  # simply return the object itself!
            return obj

    @staticmethod
    def rec_from_obj(obj, curr_level=0):
        next_level = curr_level + 1
        if isinstance(obj, dict):
            key = obj.get(Serializable.KEY_TYPE, None)
            if key is not None:
                cls = Registrable.key2cls(key)
                return cls.create_from_dict(obj)
            elif curr_level >= Serializable.REC_MAX_LEVEL:
                return obj
            else:
                return type(obj)([(k, Serializable.rec_from_obj(z, curr_level=next_level)) for k, z in obj.items()])
        elif curr_level >= Serializable.REC_MAX_LEVEL:
            return obj
        elif isinstance(obj, (list, tuple)):
            return type(obj)([Serializable.rec_from_obj(z, curr_level=next_level) for z in obj])
        else:
            return obj

# =====
# io with json/pickle/...

class _BaseRW:
    def open(self, file: str, mode: str): raise NotImplementedError()
    def from_fd(self, fd, **kwargs): raise NotImplementedError()
    def to_fd(self, one: object, fd, **kwargs): raise NotImplementedError()
    def from_obj(self, s: object, **kwargs): raise NotImplementedError()
    def to_obj(self, one: object, **kwargs): raise NotImplementedError()
    def from_fd_one(self, fd, **kwargs): raise NotImplementedError()
    def to_fd_one(self, one: object, fd, **kwargs): raise NotImplementedError()

# builtin types will not use this!
class _MyJsonEncoder(json.JSONEncoder):
    def default(self, one: object):
        if isinstance(one, Serializable):
            return one.to_dict()
        else:
            return json.JSONEncoder.default(self, one)

class _JsonRW(_BaseRW):
    def __init__(self, cls: Type = None):
        assert cls is None or issubclass(cls, Serializable)
        self.cls = cls

    def _load(self, d):
        if self.cls is not None:
            return self.cls.create_from_dict(d)
        else:  # note: use "create" shortcut
            return Serializable.create(d)

    # mainly forwarding
    def open(self, file: str, mode: str): return zopen(file, mode)  # plain text mode
    def from_fd(self, fd, **kwargs): return self._load(json.load(fd, **kwargs))
    def to_fd(self, one: object, fd, **kwargs): return json.dump(one, fd, cls=_MyJsonEncoder, ensure_ascii=False, **kwargs)
    def from_obj(self, s: str, **kwargs): return self._load(json.loads(s, **kwargs))
    def to_obj(self, one: object, **kwargs): return json.dumps(one, cls=_MyJsonEncoder, ensure_ascii=False, **kwargs)
    def to_fd_one(self, one: object, fd, **kwargs): fd.write(self.to_obj(one, **kwargs) + "\n")

    def from_fd_one(self, fd, **kwargs):
        line = fd.readline()
        if len(line) == 0:
            raise EOFError()
        else:
            return self.from_obj(line, **kwargs)

class _PickleRW(_BaseRW):
    # mainly forwarding
    def open(self, file: str, mode: str): return zopen(file, mode+'b')  # binary mode
    def from_fd(self, fd, **kwargs): return pickle.load(fd, **kwargs)
    def to_fd(self, one: object, fd, **kwargs): return pickle.dump(one, fd, **kwargs)
    def from_obj(self, s: bytes, **kwargs): return pickle.loads(s, **kwargs)
    def to_obj(self, one: object, **kwargs): return pickle.dumps(one, **kwargs)
    def from_fd_one(self, fd, **kwargs): return pickle.load(fd, **kwargs)
    def to_fd_one(self, one: object, fd, **kwargs): return pickle.dump(one, fd, **kwargs)

class Serializer:
    def __init__(self, rw: _BaseRW):
        self.rw = rw

    # open
    def open(self, file: str, mode: str):
        return self.rw.open(file, mode)

    # from or to files
    def from_file(self, fn_or_fd: Union[str, IO], **kwargs):
        if isinstance(fn_or_fd, str):
            with self.rw.open(fn_or_fd, 'r') as fd:
                return self.from_file(fd, **kwargs)
        else:
            return self.rw.from_fd(fn_or_fd, **kwargs)

    def to_file(self, one: object, fn_or_fd: Union[str, IO], **kwargs):
        if isinstance(fn_or_fd, str):
            with self.rw.open(fn_or_fd, 'w') as fd:
                return self.to_file(one, fd, **kwargs)
        else:
            return self.rw.to_fd(one, fn_or_fd, **kwargs)

    # from or to objects: directly forward
    def from_obj(self, s: object, **kwargs): return self.rw.from_obj(s, **kwargs)
    def to_obj(self, one: object, **kwargs): return self.rw.to_obj(one, **kwargs)

    # iterative ones
    def yield_iter(self, fn_or_fd: Union[str, IO], max_num=-1, **kwargs):
        if isinstance(fn_or_fd, str):
            with self.rw.open(fn_or_fd, 'r') as fd:
                yield from self.yield_iter(fd, max_num, **kwargs)
        else:
            c = 0
            while c != max_num:
                try:
                    yield self.rw.from_fd_one(fn_or_fd, **kwargs)
                    c += 1
                except EOFError:
                    break

    def save_iter(self, ones: Iterable, fn_or_fd: Union[str, IO], **kwargs):
        if isinstance(fn_or_fd, str):
            with self.rw.open(fn_or_fd, 'w') as fd:
                self.save_iter(ones, fd, **kwargs)
        else:
            for one in ones:
                self.rw.to_fd_one(one, fn_or_fd, **kwargs)

    def load_list(self, fn_or_fd: Union[str, IO], max_num=-1, **kwargs):
        return list(self.yield_iter(fn_or_fd, max_num, **kwargs))

# useful instances
default_json_serializer = Serializer(_JsonRW(None))
default_pickle_serializer = Serializer(_PickleRW())
def get_json_serializer(cls: Type): return Serializer(_JsonRW(cls))

# simpler json helpers
class JsonHelper:
    def from_jsonl(self, file):
        with open(file) as fd:
            for line in fd:
                yield json.loads(line)

    def from_json(self, file):
        with open(file) as fd:
            yield from json.load(fd)

    def from_auto(self, file):
        if file.endswith("jsonl"):
            yield from self.from_jsonl(file)
        elif file.endswith("json"):
            yield from self.from_json(file)
        else:
            raise RuntimeError(f"UNK file format {file}")

    def to_jsonl(self, file, ds):
        with open(file, 'w') as fd:
            for one in ds:
                fd.write(f"{json.dumps(one, ensure_ascii=False)}\n")

    def to_json(self, file, ds):
        with open(file, 'w') as fd:
            json.dump(ds, fd, ensure_ascii=False, indent=2)

    def to_auto(self, file, ds):
        if file.endswith("jsonl"):
            self.to_jsonl(file, ds)
        elif file.endswith("json"):
            self.to_json(file, ds)
        else:
            raise RuntimeError(f"UNK file format {file}")

default_json_helper = JsonHelper()
