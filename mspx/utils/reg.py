#

# reg: storing class specific information

__all__ = [
    "Registrable", "IdAssignable",
]

from typing import Type, Union, Callable, Dict, TypeVar
import importlib

# =====
# mainly storing extra info for each cls (allowing short cuts!)

class Registrable:
    # global storage
    _cls2info: Dict[Union[Type, Callable], Dict] = {}  # key -> info
    _key2cls: Dict[str, Union[Type, Callable]] = {}  # shortcuts

    # --
    # class methods

    # register a class
    @classmethod
    def reg(base_cls, T: Union[Type, Callable], key: str = None, **kwargs):
        if isinstance(T, type):
            assert issubclass(T, base_cls)
        else:  # try to create one!
            assert isinstance(T(), base_cls)
        _key = None if key is None else Registrable._form_key(base_cls, key)  # form the current key
        if _key is not None:
            assert _key not in Registrable._key2cls, f"Key already exists when adding {T}: {_key}"
            assert ":" not in _key, f"Should not contain ':' in shortcut key: {_key}"
            Registrable._key2cls[_key] = T  # add shortcut!
        # --
        _info = Registrable.cls2info(T)
        if _key is not None:
            _info['Ks'].append(_key)
        _info.update(kwargs)
        if isinstance(T, type) and issubclass(T, Registrable):
            T.special_reg(_key, **kwargs)
        return T

    @classmethod
    def special_reg(cls, key: str, **kwargs):
        pass  # by default nothing to do

    # reg as decorator: directly use class type as constructor
    @classmethod
    def rd(base_cls, key: str = None, **kwargs):
        _TV = TypeVar('_TV', bound=Type)
        def _decorator(_T: _TV) -> _TV:
            # assert issubclass(_T, base_cls)
            base_cls.reg(_T, key=key, **kwargs)
            return _T
        # --
        return _decorator

    # --
    # looking up methods

    @classmethod
    def cls2info(cls, T: Type = None, field=None, df=None):
        if T is None:
            T = cls  # for convenience!
        _info = Registrable._cls2info.get(T)
        if _info is None:  # add it
            _info = {'Ks': []}
            Registrable._cls2info[T] = _info
        if field is None:
            return _info  # return full
        else:
            return _info.get(field, df)

    @classmethod
    def cls2key(cls, T: Type = None, relative=True):
        if T is None or T is cls:
            T = cls  # for convenience!
            relative = False
        _keys = Registrable.cls2info(T, field='Ks')
        if _keys is not None and len(_keys) > 0:
            ret = _keys[0]  # note: use the first registered one!
            if relative:  # remove base_cls prefix!
                prefix = cls.cls2key()
                assert ret.startswith(prefix), f"Not subclass: {prefix} vs {ret}"
                ret = ret[len(prefix):].lstrip('/')
        else:  # fall back to the full name!
            ret = Registrable.get_cid(T)
        return ret

    @classmethod
    def key2cls(base_cls, key: str, df=None):  # shortcut
        if ':' in key:  # full name
            return Registrable.cid2cls(key)
        else:
            # --
            # try to import module first
            if '///' in key:
                mod, key = key.split("///")
                _m = importlib.import_module(mod)  # try to load that module to allow register!
            # --
            _key = Registrable._form_key(base_cls, key)
            return Registrable._key2cls.get(_key, df)

    # --
    # global static ones

    # a default naming scheme
    @staticmethod
    def get_cid(cls: Type):  # default class id
        return f"{cls.__module__}:{cls.__name__}"

    @staticmethod
    def cid2cls(cid: str):
        _mod, _name = cid.split(":")
        _m = importlib.import_module(_mod)
        cls = getattr(_m, _name)
        return cls

    @staticmethod
    def _form_key(base_cls: Type, key: str):
        if base_cls is Registrable:
            base_key = ''
        else:
            base_key = base_cls.cls2key() + "/"
        _key = base_key + key
        _key = _key.rstrip('/')
        return _key

class IdAssignable:
    _counts = {}  # cls_id(base_cls) -> int

    @classmethod
    def get_new_id(cls):
        base_cls_id = Registrable.get_cid(cls)  # key0
        if base_cls_id not in IdAssignable._counts:
            IdAssignable._counts[base_cls_id] = 0
        ret = IdAssignable._counts[base_cls_id]
        IdAssignable._counts[base_cls_id] += 1
        return ret

    @classmethod
    def clear(cls):
        base_cls_id = Registrable.get_cid(cls)  # key0
        IdAssignable._counts[base_cls_id] = 0

# --
# b mspx/utils/reg:99
