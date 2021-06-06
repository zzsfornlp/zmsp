#

# for Serialization

__all__ = [
    "get_class_id", "JsonSerializable", "Serializer",
    "default_json_serializer", "default_pickle_serializer", "get_json_serializer",
]

from typing import Type, Dict, Union, Iterable, IO
import json
import pickle
from copy import deepcopy
from .file import zopen

# =====
# using json:
# class: cls_from_json; inst: to_json, from_json;

# get unique class id
def get_class_id(cls: Type, use_mod=False):
    if use_mod:
        return f"{cls.__module__}-{cls.__name__}"
    else:
        return cls.__name__

# A class that is Serializable
class JsonSerializable:
    def valid_json_fields(self):  # by default, all fields are valid
        return list(self.__dict__.keys())

    @classmethod
    def cls_from_json(cls, data: Dict, **kwargs):  # json-dict to instance (creation)
        # if it happens we have a default constructor
        ret = cls(**kwargs)  # todo(note): require a constructor that knows how to build this class at the first place!!
        ret.from_json(data)
        return ret

    def to_json(self) -> Dict:  # inst to json-dict
        # simple default version
        ret = {}
        for k in self.valid_json_fields():
            v = getattr(self, k)
            if isinstance(v, JsonSerializable):
                ret[k] = v.to_json()
            else:
                ret[k] = v  # should be simple type
        return ret

    def from_json(self, data: Dict):  # json-dict to instance (modify/update inplace)
        # simple default version
        for k in self.valid_json_fields():  # only load what we have
            if k in data:
                v = getattr(self, k)
                v2 = data[k]
                if isinstance(v, JsonSerializable):
                    v.from_json(v2)
                else:  # todo(+1): here we replace instead of modify
                    setattr(self, k, v2)
        # rebuild after loading
        self.finish_build()

    # actual deep copy!!
    def copy(self):
        # simple default version
        inst = self.__class__()  # todo(note): require a default init
        for k in self.valid_json_fields():  # only load what we have
            v = getattr(self, k)
            if isinstance(v, JsonSerializable):
                v2 = v.copy()
            else:
                v2 = deepcopy(v)
            setattr(inst, k, v2)
        inst.finish_build()
        return inst

    # used to finish the building self after creation or loading
    def finish_build(self):
        pass

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
        if isinstance(one, JsonSerializable):
            return one.to_json()
        else:
            return json.JSONEncoder.default(self, one)

class _JsonRW(_BaseRW):
    def __init__(self, cls: Type = None):
        assert cls is None or issubclass(cls, JsonSerializable)
        self.cls = cls

    def _load(self, v):
        if self.cls is not None:
            return self.cls.cls_from_json(v)
        else:
            return v

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
