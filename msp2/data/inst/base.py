#

# the basic one

__all__ = [
    "DataInstance", "DataInstanceComposite", "SubInfo", "get_global_indexer", "reset_global_indexer",
]

from typing import Dict, List, Union, Type, Callable
from copy import deepcopy
from collections import Counter, defaultdict, namedtuple
from msp2.utils import JsonSerializable, get_class_id, zwarn, ZIndexer, zlog

# How the data is represented?
# todo(+N): is this really good design or making things complex?
# todo(+3): currently we cannot move things (mv or rm), needed?
"""
In one phrase: a tree with flying reference links.
1) A DataInstance is only STORED at one place by its par, thus, the STORED version should be a tree with refs
2) A DataInstance can contain sub-fields (which can be STORED or REF), which can be further DataInstance
3) For REF ones, they are indicated in the SubInfo map by "is_ref", and will be replaced by relative-id(str)
4) For REFed ones, they should be contained in DataInstanceComposite, which can index items as in a file system
5) reg is only needed for those may be REFed
*) "/" is forbidden in id, since it is preserved for path sep!!
"""

"""
# information for DataInstance's sub-components
# - inner_type: the inner-most type for that field
# - wrapper_type: the out type for that field, current only supports "list"
# - is_ref: is this field only a reference (not stored here)
# - needs_reg: not is_ref and needs to reg (and thus get id to be foundable) inside this Composite
# - reg_sname: reg short-name for id if there are not provided
# - df_val: default value
"""
class SubInfo:
    def __init__(self, inner_type: Type, wrapper_type: Type = None, is_ref=False, needs_reg=False, reg_sname='', df_val=None):
        self.inner_type = inner_type
        self.wrapper_type = wrapper_type
        self.is_ref = is_ref
        self.needs_reg = needs_reg
        self.reg_sname = reg_sname
        self.df_val = df_val

# todo(note): protocol for fields
"""
1) no SubInfo: must be simple types that are usually directly set or json or deepcopy
2) v is None: allowed, directly use or pass None
3) three cases: 3.1) DataInstance: ref or not, reg or not; 3.2) JsonSerializable; 3.3) simple ones with df_val
"""

# DataInstance Traverser
class DataInstanceTraverser:
    # template method for traversing
    def traverse(self, inst: 'DataInstance'):
        # todo(+5): where to split the details, init/keys/deal/ret
        raise NotImplementedError()

# Basic Data Instance
class DataInstance(JsonSerializable):
    # =====
    # class method for sub-components info
    _InstSubMaps = {}  # cache

    # use cache
    @classmethod
    def get_sub_map(cls) -> Dict[str, SubInfo]:
        _map = DataInstance._InstSubMaps
        k = get_class_id(cls, use_mod=True)
        if k not in _map:
            r = {}
            for cl in reversed(cls.__mro__):  # collect in reverse order
                if issubclass(cl, DataInstance) and hasattr(cl, "_get_sub_map"):
                    r.update(cl._get_sub_map())
            _map[k] = r
        return _map[k]

    # to be overridden
    @classmethod
    def _get_sub_map(cls) -> Dict[str, SubInfo]:
        # return mapping: FieldName -> SubInfo
        return {"info": SubInfo(dict, df_val={})}

    # =====
    # overriding JsonSerializable

    def valid_json_fields(self):  # by default, all fields are valid
        # not "_*" fields which can be tmp fields, but do include "_id"
        ret = [z for z in self.__dict__.keys() if not z.startswith("_")]
        if self._id is not None:  # no export id if it is None
            ret.append("_id")
        return ret

    # helper function to deal with wrapper
    def _deal_with_wrapper(self, v: Union[object, List], k_info: SubInfo, func: Callable, **kwargs):
        wrapper_type = None if k_info is None else k_info.wrapper_type
        if wrapper_type is None:  # no wrapper
            return func(v, k_info, **kwargs)
        elif v is None:  # specific for case2
            return None
        else:
            assert isinstance(v, wrapper_type), f"Unmatched type to the wrapper type, {type(v)} vs. {wrapper_type}"
            if isinstance(v, list):  # simply going through
                return [func(vv, k_info, **kwargs) for vv in v]
            else:
                raise NotImplementedError()

    def _func_from_json(self, v: object, k_info: SubInfo):
        if k_info is None or v is None:  # case1/case2
            return v
        inner_type = k_info.inner_type
        if issubclass(inner_type, DataInstance):  # case3.1
            if k_info.is_ref:
                assert isinstance(v, str), f"Ref is not a str, but a {type(v)}"
                return v  # leave it here for later deref
            else:
                to_assign = inner_type.cls_from_json(v, _par=self)  # directly set par here!
                if k_info.needs_reg:
                    assert to_assign.id is not None and isinstance(self, DataInstanceComposite)
                    self.register_inst(to_assign)
                return to_assign
        elif issubclass(inner_type, JsonSerializable):  # case3.2
            return inner_type.cls_from_json(v)
        else:  # case3.3
            return v

    def from_json(self, data: Dict):
        # load according to input instead, todo(note): note that we are re-creating new ones
        cur_sub_map = self.__class__.get_sub_map()
        for k, v in data.items():  # todo(note): set according to input
            k_info = cur_sub_map.get(k)
            to_assign = self._deal_with_wrapper(v, k_info, self._func_from_json)
            setattr(self, k, to_assign)
        # rebuild after loading
        self.finish_build()
        return self

    def _func_to_json(self, v: object, k_info: SubInfo):
        if k_info is None or v is None:  # case1/case2
            return v
        inner_type = k_info.inner_type
        if issubclass(inner_type, DataInstance):  # case3.1
            if k_info.is_ref:
                return v.get_rel_path(self)
            else:
                return v.to_json()
        elif issubclass(inner_type, JsonSerializable):  # case3.2
            return v.to_json()
        else:  # case3.3
            return v

    # very similar to default, except special handling for Reference fields
    def to_json(self) -> Dict:
        cur_sub_map = self.__class__.get_sub_map()
        ret = {}
        for k in self.valid_json_fields():
            v = getattr(self, k)
            k_info = cur_sub_map.get(k)
            to_assign = self._deal_with_wrapper(v, k_info, self._func_to_json)
            if k_info is not None and to_assign == k_info.df_val:
                pass  # todo(note): put nothing to make it brief if match df_val
            else:
                ret[k] = to_assign
        return ret

    def _func_deref(self, v: object, k_info: SubInfo):
        if k_info is None or v is None:  # case1/case2
            return v
        inner_type = k_info.inner_type
        if issubclass(inner_type, DataInstance):  # case3.1
            if k_info.is_ref:
                if isinstance(v, str):
                    return self.find_inst(v)
                else:
                    return v
            else:
                v.deref()
        return v  # final fallback

    # actually 'finish_build' but use a new one since this should be called top-down after loading all
    # -> should be explicitly called by top-level DataInstance after loading!!
    def deref(self):
        # handle chs
        cur_sub_map = self.__class__.get_sub_map()
        for k in self.valid_json_fields():
            v = getattr(self, k)
            k_info = cur_sub_map.get(k)
            to_assign = self._deal_with_wrapper(v, k_info, self._func_deref)
            if k_info is not None and k_info.is_ref:
                setattr(self, k, to_assign)
        return self

    def _func_copy(self, v: object, k_info: SubInfo, inst: 'DataInstance'):
        if k_info is None or v is None:  # case1/case2
            return deepcopy(v)  # here we need to deepcopy things!!
        inner_type = k_info.inner_type
        if issubclass(inner_type, DataInstance):  # case3.1
            if k_info.is_ref:
                return v.get_rel_path(self)
            else:
                v_copy = v.copy()
                v_copy.set_par(inst)
                if k_info.needs_reg:
                    assert v_copy.id is not None and isinstance(inst, DataInstanceComposite)
                    inst.register_inst(v_copy)
                return v_copy
        elif issubclass(inner_type, JsonSerializable):  # case3.2
            return v.copy()
        else:  # case3.3
            return v

    # actual deep copy
    def copy(self, ignore_fields=None):
        inst = self.__class__(_id=self._id)  # copy one!!
        cur_sub_map = self.__class__.get_sub_map()
        for k in self.valid_json_fields():
            if ignore_fields is not None and k in ignore_fields:
                continue  # ignore these fields
            v = getattr(self, k)
            k_info = cur_sub_map.get(k)
            to_assign = self._deal_with_wrapper(v, k_info, self._func_copy, inst=inst)
            setattr(inst, k, to_assign)
        inst.finish_build()
        return inst

    # =====
    # simple traverse with func: no return is needed
    def traverse_simple(self, func: Callable):
        cur_sub_map = self.__class__.get_sub_map()
        for k in self.valid_json_fields():
            v = getattr(self, k)
            k_info = cur_sub_map.get(k)
            self._deal_with_wrapper(v, k_info, func)

    # complex traverse with Traverser
    def traverse_complex(self, traverser: DataInstanceTraverser):
        traverser.traverse(self)

    # =====
    # actual own methods

    # __init__ should not be called from the outside, but only used for internal init
    # inside here, we should setup some default values, then two ways for creating: 1) cls.create, 2) self.from_json
    def __init__(self, _id: str = None, _par: 'DataInstance' = None):
        # todo(note): id(reg)/par setup are mostly delayed to 'add_*' by its real par, rather than create time
        self._id = _id  # id
        self._par = _par  # Parent Node
        self.info = {}  # other info
        self._cache = {}  # other info not to be save/load

    # def __getattr__(self, item):
    #     info = self.info
    #     if item in info:
    #         return info[item]
    #     else:
    #         raise AttributeError()

    @property
    def id(self):
        return self._id

    def set_id(self, id: str):
        self._id = id

    @property
    def read_idx(self):  # todo(+N): here assign read idx to support in-order writing
        return self._read_idx

    def set_read_idx(self, idx: int):
        self._read_idx = idx

    @property
    def par(self):
        return self._par

    def set_par(self, par: 'DataInstance'):
        self._par = par

    # create is the real creating method
    @classmethod
    def create(cls, id: str = None, par: 'DataInstance' = None):
        inst = cls(_id=id, _par=par)
        return inst

    def __repr__(self):
        # return f"{self.__class__.__name__}(id={self.id},par={self.par})"
        return f"{self.__class__.__name__}({self.id})"

    # check par, add par
    def add_inst(self, inst: 'DataInstance'):
        # todo(note): 'par' should be not set or already self!
        inst_par = inst.par
        assert inst_par is None or inst_par is self, f"Cannot add {inst_par}'s inst {inst}!"
        inst.set_par(self)
        return inst

    # register self to global
    def reg_self_globally(self, global_indexer_name: str = None):
        # reg self if no par?
        if self.par is None and global_indexer_name is not None:
            get_global_indexer(global_indexer_name).register_inst(self)
        return self

    # =====
    # relating with path

    # get parent path
    def get_par_spine(self, inc_self=False, top_down=True):
        path: List['DataInstance'] = []
        c: DataInstance = self if inc_self else self._par
        while c is not None:
            path.append(c)
            c = c._par
        if top_down:
            path.reverse()
        return path

    # search up
    def search_up_for_type(self, trg_type: Type, inc_self=False):
        c = self if inc_self else self._par
        while c is not None:
            if isinstance(c, trg_type):
                return c
            c = c._par
        return None

    # find inst by path name
    def find_inst(self, path: str, df=None, raise_err=True):
        if path.startswith("/"):  # search from the root with the abs path
            current_point = self
            while current_point.par is not None:
                current_point = current_point.par
            return current_point.find_inst(path.lstrip("/"))  # make sure not abs path
        path = path.strip("/")
        fields = path.split("/")
        starting_point = self
        while starting_point is not None:  # up to the upmost level
            try:  # try to find it starting from this one
                current_point = starting_point
                for f in fields:
                    if current_point is None:
                        break
                    if f=="." or f=="":
                        continue
                    if f=="..":
                        current_point = current_point.par
                    else:  # must be a DataInstanceComposite
                        assert isinstance(current_point, DataInstanceComposite)
                        current_point = current_point.search_id(f)
            except:
                current_point = None
            if current_point is not None:
                return current_point  # success
            starting_point = starting_point.par  # start from an upper level
        if raise_err:
            raise RuntimeError("Cannot find ref!")
        return df  # failed, return default one

    # get abs path
    def get_abs_path(self):
        spine = self.get_par_spine(inc_self=True, top_down=True)
        paths = [z.id for z in spine]
        ret = "/" + "/".join(paths)
        return ret

    # get rel path (starting from 'cur')
    def get_rel_path(self, cur: 'DataInstance'):
        self_spine = self.get_par_spine(inc_self=True, top_down=True)
        cur_spine = cur.get_par_spine(inc_self=True, top_down=True)
        diverge_idx = 0
        while diverge_idx<len(self_spine) and diverge_idx<len(cur_spine) and self_spine[diverge_idx] is cur_spine[diverge_idx]:
            diverge_idx += 1  # require "is": the same object
        self_path = [z.id for z in self_spine[diverge_idx:]]  # go down for self(trg)
        # cur_path = [".."] * len(cur_spine[diverge_idx:])  # go up from cur(src)
        if len(self_path) == 0:  # if purely going up
            cur_path = [".."] * len(cur_spine[diverge_idx:])
        else:
            # todo(note): for brevity, ignore the ".."s, since we will go upwards, but be careful!!
            #   currently things are fine since there will not be strange name conflicts...
            cur_path = []
        return "/".join(cur_path+self_path)

    # get rel path until certain Type
    def get_path_until(self, t: Type):
        trg = self.search_up_for_type(t)
        if trg is not None:  # get rel path
            return self.get_rel_path(trg)
        else:  # otherwise return abs path
            return self.get_abs_path()

    # pretty printing
    def pp(self, method: str, printing=False, **kwargs):
        from .helper2 import MyPrettyPrinter
        ff = getattr(MyPrettyPrinter, method)
        ss = ff(self, **kwargs)
        if printing:
            zlog(ss)
        return ss

# -----
# data composite (as containers with indexers, plain DataInstance can also be container but cannot index)
class DataInstanceComposite(DataInstance):
    # --
    def __init__(self, _id: str = None, _par: 'DataInstance' = None):
        super().__init__(_id, _par)
        # --
        self._indexer = ZIndexer()

    def reset_indexer(self):  # clear all records
        self._indexer.reset()

    def register_inst(self, inst: DataInstance, sname_prefix: str = None):  # register
        inst_id = inst.id
        if sname_prefix is None:
            sname_prefix = get_class_id(inst.__class__)
        new_id = self._indexer.register(inst_id, inst, sname_prefix)
        if inst_id is None:  # also set the new id
            inst.set_id(new_id)

    def clear_insts(self, sname_prefix: str):  # clean certain instances
        self._indexer.clear_name(sname_prefix)

    def search_id(self, id: str, df=None):  # search
        return self._indexer.lookup(id, df=df)

    def remove_id(self, id: str):  # remove entry
        return self._indexer.remove(id)

    # =====
    def add_and_reg_inst(self, inst: 'DataInstance', sname_prefix: str = None):
        super().add_inst(inst)
        self.register_inst(inst, sname_prefix)  # todo(note): register this one to self to assign id and enable index!!
        return inst

    def del_inst(self, inst: 'DataInstance'):
        self.remove_id(inst.id)

# global (root) singleton indexer pools
_GLOBAL_INDEXERS = defaultdict(DataInstanceComposite)
def get_global_indexer(name=""): return _GLOBAL_INDEXERS[name]
def reset_global_indexer(name=""): _GLOBAL_INDEXERS[name].reset_indexer()
