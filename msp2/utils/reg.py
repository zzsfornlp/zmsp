#

# Registrable
# mix-in class for registering

__all__ = [
    "Registrable", "IdAssignable",
]

from typing import Type, Union, Callable, Dict, Tuple
from collections import defaultdict
from .seria import get_class_id
from .utils import ZObject

class Registrable:
    # global one: cls_id(base_cls) -> key -> ZObject
    _reg_index: Dict[str, Dict] = {}

    # -----
    # the basic one
    @classmethod
    def reg(base_cls, key: str, T: Union[Type, Callable], **kwargs):
        # base
        _index = Registrable._reg_index
        base_cls_id = get_class_id(base_cls, use_mod=True)  # key0
        if base_cls_id not in _index:
            _index[base_cls_id] = {}
        _base_index = _index[base_cls_id]
        # current
        assert key not in _base_index, f"Repeated key {key} for {base_cls_id}!"
        _base_index[key] = ZObject(T=T, **kwargs)
        return T

    # -----
    # as decorator: directly use class type as constructor
    @classmethod
    def reg_decorator(base_cls, key: str, **kwargs):
        def _decorator(_T: Union[Type, Callable]):
            # assert issubclass(_T, base_cls)
            base_cls.reg(key, _T, **kwargs)
            return _T
        # --
        return _decorator

    # -----
    # lookup
    @classmethod
    def lookup(base_cls, key: str, df=None):
        _index = Registrable._reg_index
        base_cls_id = get_class_id(base_cls, use_mod=True)  # key0
        _base_index = _index.get(base_cls_id, {})
        return _base_index.get(key, df)

    @classmethod
    def try_load_and_lookup(base_cls, key: str, df=None, ret_name=False):
        try:
            mod, name = key.split("/")
            import importlib
            module = importlib.import_module(mod)  # try to load that module to allow register!
        except:
            name = key
        ret = base_cls.lookup(name, df=df)
        if ret_name:
            return ret, name
        else:
            return ret

    @classmethod
    def keys(base_cls):
        _index = Registrable._reg_index
        base_cls_id = get_class_id(base_cls, use_mod=True)  # key0
        _base_index = _index.get(base_cls_id, {})
        return _base_index.keys()

class IdAssignable:
    _counts = {}  # cls_id(base_cls) -> int

    @classmethod
    def get_new_id(cls):
        base_cls_id = get_class_id(cls, use_mod=True)  # key0
        if base_cls_id not in IdAssignable._counts:
            IdAssignable._counts[base_cls_id] = 0
        ret = IdAssignable._counts[base_cls_id]
        IdAssignable._counts[base_cls_id] += 1
        return ret

    @classmethod
    def clear(cls):
        base_cls_id = get_class_id(cls, use_mod=True)  # key0
        IdAssignable._counts[base_cls_id] = 0
