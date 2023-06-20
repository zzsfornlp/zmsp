#

# seria: save & load

__all__ = [
    "Serializable", "InfoField", "DEFAULT_INFO_FIELD",
    "Serializer", "default_json_serializer", "default_pickle_serializer", "get_json_serializer",
]

from typing import Type, Dict, Set, Union, Iterable, IO
import json
import pickle
from .reg import Registrable
from .file import zopen

# =====
# A class that is Serializable (to/from dict)
# note: allow fine-grained control over fields
# note: complex classes should specify save/load by themselves!

"""
# information for DataInstance's sub-components
# - inner_type: the inner-most type for that field
# - wrapper_type: the out type for that field, current only supports (list, dict)
# - is_rec: is recursive?
# - no_store_f: (lambda v, par: bool) if True, then no storage for "Serializable.to_dict"
# - to_f/from_f: (lambda v: ...) -- further functions for (inner_)to/from
"""
class InfoField:
    def __init__(self, inner_type: Type = None, wrapper_type: Type = None, is_rec=False,
                 no_store_f=None, to_f=None, from_f=None):
        assert inner_type is None or issubclass(inner_type, Serializable)
        self.inner_type = inner_type  # actual type
        assert wrapper_type is None or (wrapper_type in [list, dict])
        self.wrapper_type = wrapper_type  # wrapping type
        assert (not is_rec) or (wrapper_type is None)
        self.is_rec = is_rec  # special recursive flag
        if no_store_f == 'len0':  # a shortcut for len==0
            no_store_f = (lambda v,p: len(v)==0)
        self.no_store_f, self.to_f, self.from_f = no_store_f, to_f, from_f

    # --
    # inst -> dict

    def to_dict(self, v):
        # 1) field is None
        if v is None:
            return v
        # 2) wrap/rec
        if self.wrapper_type is not None:
            return self.wrap_to_dict(v)
        if self.is_rec:
            return self.rec_to_dict(v)
        # 3) obj
        return self.inner_to_dict(v)

    def inner_to_dict(self, v, add_end=False):
        if self.to_f is not None:
            v = self.to_f(v)
        if isinstance(v, Serializable):
            store_type = (self.inner_type is None) or (not isinstance(v, self.inner_type))
            d = v.to_dict(store_type=store_type)
            if add_end and (not store_type):
                d['__e'] = 1  # extra ending mark!
        else:
            d = v
        return d

    def wrap_to_dict(self, v):
        _wt = self.wrapper_type
        if _wt is list:
            d = [self.inner_to_dict(z) for z in v]
        elif _wt is dict:
            d = {k: self.inner_to_dict(z) for k,z in v.items()}
        else:
            raise NotImplementedError(f"Unsupported wrapper_type = {_wt}")
        return d

    def rec_to_dict(self, v):
        if isinstance(v, list):
            d = [self.rec_to_dict(z) for z in v]
        elif isinstance(v, dict):
            d = {k: self.rec_to_dict(z) for k,z in v.items()}
        else:
            d = self.inner_to_dict(v, add_end=True)
        return d

    # --
    # inst <- dict

    def from_dict(self, d, v0=None):
        # 1) None
        if d is None:
            return d
        # 2) wrap/vec
        if self.wrapper_type is not None:
            return self.wrap_from_dict(d, v0=v0)
        if self.is_rec:
            return self.rec_from_dict(d, v0=v0)
        # 3) obj
        return self.inner_from_dict(d, v0=v0)

    def inner_from_dict(self, d, v0):
        _cls = None
        if isinstance(d, dict) and '__t' in d:  # first check stored type
            _cls = Registrable.key2cls(d['__t'])
            assert issubclass(_cls, Serializable)
        elif self.inner_type is not None:  # then field-info
            _cls = self.inner_type
        elif isinstance(v0, Serializable):  # finally existing one
            _cls = type(v0)
        if _cls is None:
            ret = d  # don't know the type!
        else:
            ret = _cls.create_from_dict(d)
        if self.from_f is not None:
            ret = self.from_f(ret)
        return ret

    def wrap_from_dict(self, d, v0):
        _wt = self.wrapper_type
        if _wt is list:
            v = [self.inner_from_dict(z, v0) for z in d]
        elif _wt is dict:
            v = {k: self.inner_from_dict(z, v0) for k, z in d.items()}
        else:
            raise NotImplementedError(f"Unsupported wrapper_type = {_wt}")
        return v

    def rec_from_dict(self, d, v0):
        v = d
        if isinstance(d, dict) and ('__t' in d or '__e' in d):
            v = self.inner_from_dict(d, v0)
        else:
            if isinstance(d, list):
                v = [self.rec_from_dict(z, v0) for z in d]
            elif isinstance(d, dict):
                v = {k: self.rec_from_dict(z, v0) for k,z in d.items()}
        return v

# --
# a default one
DEFAULT_INFO_FIELD = InfoField()
# --

class Serializable(Registrable):
    # specify field types; note: cls-method!
    @classmethod
    def info_fields(cls):
        _cls_info = cls.cls2info()
        _info = _cls_info.get("info_fields", None)  # check cached ones (safe to cache since cls-specific)
        if _info is None:
            _info = {}
            for cl in reversed(cls.__mro__):  # collect in reverse order
                if issubclass(cl, Serializable) and hasattr(cl, "_info_fields"):
                    _info.update(cl._info_fields())
            _cls_info["info_fields"] = _info  # store the cache
        return _info

    # to be overridden
    @classmethod
    def _info_fields(cls):
        return {}  # str(field) -> InfoField

    # return the fields for seria save; note: obj-method!
    def seria_fields(self, store_all_fields=False) -> Iterable[str]:
        if store_all_fields:
            return list(self.__dict__.keys())
        else:
            # note: by default, all fields that does not start with "_"
            return [z for z in self.__dict__.keys() if not z.startswith("_")]

    def to_dict(self, store_type=True, store_all_fields=False):  # inst to dict
        ret = {}
        if store_type:
            ret["__t"] = self.cls2key()
        info_fields = self.info_fields()
        for k in self.seria_fields(store_all_fields=store_all_fields):
            _if = info_fields.get(k, DEFAULT_INFO_FIELD)
            v = getattr(self, k)
            if _if.no_store_f is not None:
                if _if.no_store_f(v, self):
                    continue  # no storage!
            ret[k] = _if.to_dict(v)
        return ret

    def from_dict(self, data: Dict):  # dict to instance (modify/update inplace)
        info_fields = self.info_fields()
        for k, d in data.items():  # note: load those in the data
            if k.startswith("__"):
                continue  # note: ignore special ones!
            _if = info_fields.get(k, DEFAULT_INFO_FIELD)
            v0 = getattr(self, k, None)
            v = _if.from_dict(d, v0)
            setattr(self, k, v)  # directly set it!
        self.finish_from_dict()
        return self

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
        ret = cls.create_from_dict(d)
        return ret

    # shortcut!
    @staticmethod
    def create(data):
        return DEFAULT_INFO_FIELD.from_dict(data)

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
        else:  # note: use "create" instead of "rec_from_dict" here
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
