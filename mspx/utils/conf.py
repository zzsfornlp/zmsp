#

# conf: configuration

__all__ = [
    "Conf", "GlobalConf", "get_singleton_global_conf", "get_global_conf", "ConfItem",
    "ConfEntry", "ConfEntryGlob", "ConfEntryChoices", "ConfEntryCallback",
    "ConfEntryList", "ConfEntryDict", "ConfEntryTyped",
    "Configurable",
]

import os
import numpy as np
from typing import Dict, Type, List, Iterable, Tuple, Callable, Union, TypeVar
from collections import OrderedDict, defaultdict
from .reg import Registrable
from .seria import Serializable
from .system import zglob, zglob1, zopen
from .log import zlog, zwarn

# =====
# conf composite

# note: Conf is a special class that allow non-Serializable (un-resolved) components initially!!
class Conf(Serializable):
    # --
    # common values
    SEP_NV = ":"  # sep for name/key & value
    SEP_HIER = "."  # sep for hierarchies
    SEP_HIER2 = "__"  # marks of SEP in kwargs' keys
    SEP_LIST = ","  # splitting for list-entry
    SEP_DICT_F = "::"  # splitting for dict-entry field
    SEP_DICT_NV = ":"  # splitting for dict-entry key/value
    MARK_CC = "--"  # marks for cmd-like later args & ending
    # --

    def good_names(self):
        return self.seria_fields()

    # ===== final checking
    # check error or force settings; note: only need to be called for the very top one!!
    def validate(self):
        # realize all ConfEntries & recursive validate!
        for n in self.good_names():  # directly on all
            v = getattr(self, n)
            if isinstance(v, ConfEntry):  # replace with the real value
                setattr(self, n, v.get())
            # recursively: post-order
            if isinstance(v, Conf):
                v.validate()
        # then self
        self._validate()
        # --

    # actual validation for the current object
    def _validate(self):
        pass  # to be overridden

    # =====
    # updates

    # update for one item
    def _update_one(self, k: str, val: Union[str, object]):
        old_v = getattr(self, k)
        # first if use ConfEntry, directly use it
        if isinstance(old_v, ConfEntry):
            tmp_old_v = old_v.get()
            old_v.set(val)
            new_v = old_v.get()
            old_v = tmp_old_v  # set to original one!
        elif isinstance(val, type(old_v)):  # ok to directly set!
            new_v = val
        else:
            assert isinstance(val, str)
            new_v = None
            # case 1: specially dealing with List/Dict
            if new_v is None:
                # note: avoid the case for eval with a simple heurist: checking ends_pair
                ends_pair = (val[0] + val[-1]) if len(val) > 0 else ""
                needs_eval = (ends_pair in ["()", "[]", "{}"])
                if not needs_eval:
                    if isinstance(old_v, list):
                        # get or guess element's type
                        _list_item_type = (str if len(old_v)==0 else type(old_v[0]))
                        new_v = ConfEntryList.list_convert(val, _list_item_type)
                    elif isinstance(old_v, dict):  # simply make them str and convert specifically!
                        _dict_item_type = (str if len(old_v)==0 else type(next(iter(old_v.values()))))
                        new_v = ConfEntryDict.dict_convert(val, _dict_item_type)
            # otherwise
            if new_v is None:  # must be not assigned successfully by case 1
                trg_type = type(old_v)
                # case 2.-1: special one
                if old_v is None:
                    new_v = val
                # case 2: special types for which directly using eval
                elif issubclass(trg_type, (List, Dict)):
                    new_v = eval(val)
                # case 3: otherwise use trg_type
                else:  # otherwise use trg_type as constructor
                    new_v = ConfEntryTyped.typed_convert(val, trg_type)
                if not isinstance(new_v, trg_type):
                    zwarn(f"Possible type error when updating {self}/{k}={val}, trg={trg_type}, real={type(new_v)}")
        # set it!!
        self.__setattr__(k, new_v)
        return (old_v, new_v)

    # collecting all names, including shortcuts (all k-last suffixes + zero/one previous one)
    # todo(+N): the number will explode if there are too many layers? -> using simpler lookup only by tokens??
    def _collect_all_names(self):
        ret_map: Dict[str, List[ConfItem]] = defaultdict(list)  # partial_name -> List[ConfItem]
        ret_list: List[ConfItem] = []  # List of all entries
        _SEP = Conf.SEP_HIER
        # --
        def _add_rec(cur_conf: Conf, path: List[str]):
            for n in cur_conf.good_names():
                path.append(n)
                one = getattr(cur_conf, n)
                if isinstance(one, Conf):  # recursively adding
                    _add_rec(one, path)
                else:  # add a concrete one
                    full_path = _SEP.join(path)
                    # one item inside conf
                    item = ConfItem(full_path=full_path, par_conf=cur_conf, key=n)
                    ret_list.append(item)
                    # --
                    for i in range(len(path)):
                        short_name = _SEP.join(path[i:])  # suffix
                        ret_map[short_name].append(item)  # add short name
                        # further allow combine two: (n^3)
                        for j, mid_name in enumerate(path[:max(0, i-1)]):  # i-1 to avoid repeating
                            ret_map[_SEP.join([mid_name, short_name])].append(item)
                            for lead_name in path[:j]:  # join three
                                ret_map[_SEP.join([lead_name, mid_name, short_name])].append(item)
                path.pop()
        # --
        _add_rec(self, [])
        return ret_map, ret_list

    # update for all: directly from iters of (k,v)
    def _update(self, iters: Iterable, _quite=True, _quite_all=False, _check=False, _update_all=False):
        # first loop to solve all the "ConfEntryChoices"
        cur_loop = 0
        inputs: List[Tuple[str, str]] = list(iters)
        while True:
            name_map, citem_list = self._collect_all_names()  # remember to re-calculate all
            # first deal with all choices
            choice_ones = []
            remaining_inputs = []  # those not related with choices currently
            for n, v in inputs:
                items = name_map.get(n, None)
                if items is not None and len(items)==1 and items[0].is_precheck_entry():
                    old_v, new_v = items[0].do_update(v)
                    choice_ones.append(f"Update(Loop={cur_loop}) precheck_entry '{n}={v}':"
                                       f" {items[0].full_path} = {old_v} -> {new_v}")
                else:  # only the remaining ones are needed!
                    remaining_inputs.append((n,v))
            if not _quite:
                for one in choice_ones:
                    zlog(one, func="config")
            # --
            # then expand all other choices at current!
            realized_num = 0
            for item in citem_list:
                if item.is_precheck_entry():  # set to default/current one
                    realized_num += 1
                    item.realize_entry_choice()
            # next or break
            inputs = remaining_inputs
            if len(choice_ones)==0 and realized_num==0: break  # no left to assign
            cur_loop += 1
        # --
        name_map, citem_list = self._collect_all_names()  # remember to re-calculate all
        # --
        # then deal with the rest
        good_ones, warn_ones, bad_ones = [], [], []
        hit_full_name = defaultdict(list)
        for n, v in inputs:
            item_list = name_map.get(n, None)
            if item_list is None or len(item_list)==0:
                bad_ones.append(f"Unknown config {n}={v}")
                continue
            if len(item_list) != 1 and not _update_all:
                bad_ones.append(f"Bad(ambiguous or non-exist) config {n}={v}, -> {[z.full_path for z in item_list]}")
                continue
            for _item in item_list:  # _update_all or _update_one (otherwise err before!)
                full_name = _item.full_path
                hit_full_name[full_name].append(n)
                if len(hit_full_name[full_name]) >= 2:
                    warn_ones.append(f"Repeated config with different different names: {full_name}: {hit_full_name[full_name]}")
                old_v, new_v = _item.do_update(v)
                if old_v != new_v:  # real update
                    good_ones.append(f"Update config '{n}={v}': {full_name} = {old_v} -> {new_v}")
        # report
        if not _quite:
            for one in good_ones:
                zlog(one, func="config")
            for one in warn_ones:
                zwarn(f"WARNING: {one}")
        if not _quite_all:  # need extra flag!
            for one in bad_ones:
                zwarn(f"ERROR: {one}")
        # check
        if _check:
            assert len(bad_ones) == 0, "Bad confs!!"
        return self

    # various methods
    def update_from_dict(self, d: Dict, **kwargs):
        return self._update(d.items(), **kwargs)

    def update_from_iters(self, iters: Iterable, **kwargs):
        return self._update(iters, **kwargs)

    def update_from_kwargs(self, **kwargs):
        return self.update_from_dict(kwargs)

    # directly write fields (no fancy updates!)
    def direct_update(self, _assert_exists=False, _rm_names=None, _finish=False, **kwargs):
        if _rm_names is not None:
            for n in _rm_names:  # rm names
                delattr(self, n)
        for k, v in kwargs.items():
            ks = k.split(Conf.SEP_HIER2)
            # go down the hierarchy
            cc = self
            for kk in ks[:-1]:
                cc = getattr(cc, kk)
            kk = ks[-1]
            # --
            if _assert_exists:
                assert hasattr(cc, kk), f"No-exist of attr: {k}"
            # if _try_convert_type:  # note: no converting, since this is mostly used in programes!
            #     try:
            #         vtmp = type(getattr(cc, kk))(v)
            #         v = vtmp
            #     except:
            #         pass
            setattr(cc, kk, v)  # note: directly set!!
        if _finish:
            self._update([])  # finish up!
        return self

    # convenient ones
    @classmethod
    def direct_conf(cls, conf: 'Conf' = None, validate=False, copy=False, **kwargs):
        if conf is None:
            conf = cls()
        elif copy:
            conf = conf.copy()
        conf = conf.direct_update(**kwargs)
        if validate:
            conf.validate()
        return conf

    def direct_update_from_other(self, conf: 'Conf'):
        names0 = self.good_names()
        names1 = set(conf.good_names())
        names = [z for z in names0 if z in names1]
        for n in names:
            v0, v1 = getattr(self, n), getattr(conf, n)
            if isinstance(v0, Conf) and isinstance(v1, Conf):
                v0.direct_update_from_other(v1)
            else:  # simply directly update!
                setattr(self, n, v1)
        return self

    # get values for keys from 1) kwargs, 2) self
    def obtain_values(self, keys, **kwargs):
        ret = []
        for k in keys:
            v0 = getattr(self, k)
            if k in kwargs:
                one = kwargs[k]
                if v0 is not None:  # also convert type!
                    one = ConfEntryTyped.typed_convert(one, type(v0))
            else:
                one = v0
            ret.append(one)
        return ret

    # =====
    # some external methods

    @staticmethod
    def extend_args(args: List[str], quite=False):
        sep = Conf.SEP_NV
        # check first one
        args = list(args)
        if len(args) > 0 and len(args[0].split(sep)) == 1:
            arg_file = args[0]
            if not quite:
                zlog(f"Try to read config file from {args[0]}.", func="config")
            f_args = []
            if not os.path.isfile(arg_file):
                zwarn(f"Input config file error: {arg_file}")
            else:
                with zopen(arg_file) as fd:
                    for line in fd:
                        line = line.strip()
                        if len(line)>0 and line[0]!='#':
                            f_args.append(line)
            # cmd configs are at the end
            args = f_args + args[1:]
        # --
        # processing: remove quotes
        quotes_to_remove = ["''", "\"\""]
        new_args = []
        for a in args:
            while True:
                if len(a) > 2 and any(a[0] + a[-1] == z for z in quotes_to_remove):
                    a = a[1:-1]
                else:
                    break
            new_args.append(a)
        args = new_args
        # --
        # final processing
        ii = 0  # next one to process
        argv = OrderedDict()
        _mark_cc = Conf.MARK_CC
        while ii < len(args):
            a = args[ii]
            ii += 1
            if a == _mark_cc: continue  # skip only mark_cc!
            fields = a.split(sep, 1)  # only split the first one
            assert len(fields) == 2, "Strange config updating value"
            k0, v0 = fields
            # --
            # special formats for list "--p1.p2.key: ... --"
            if k0.startswith(_mark_cc):
                assert v0 == ""
                k0 = k0[len(_mark_cc):]
                v0s = []
                while ii < len(args) and (not args[ii].startswith(_mark_cc)):
                    v0s.append(args[ii])
                    ii += 1
                v0 = Conf.SEP_LIST.join(v0s)  # note: simply join!
            # --
            if k0 in argv and not quite:
                zwarn(f"Overwrite with config {k0} = {v0}")
            argv[k0] = v0
        return argv

    # from list of strings (the first one can be conf file)
    def update_from_args(self, args: List[str], quite=False, check=True, add_global_key="G", validate=True):
        if not quite:
            zlog(f"Update conf from args: {args}.", func="config")
        argv = Conf.extend_args(args, quite=quite)
        if add_global_key:
            assert not hasattr(self, add_global_key)
            setattr(self, add_global_key, get_singleton_global_conf())  # put in global conf to update!!
        self.update_from_dict(argv, _quite=quite, _quite_all=quite, _check=check)
        if validate:
            self.validate()
        return argv

    # --
    # target obj related!
    def make_node(self, *args, **kwargs):  # note: a default protocol!
        layer_type = self.cls2info(field='RC')  # reverse-of-conf
        return layer_type(self, *args, **kwargs)

    # --
    # note: name:type:args, :type:args, type
    def _callbak_parse(self, s: str):
        _sep = Conf.SEP_DICT_NV
        _parts = s.strip().split(_sep)
        if len(_parts) == 0:
            ret = []
        elif len(_parts) == 1:  # type
            ret = [''] + _parts
        else:  # name:type...
            ret = _parts
        return ret

    # callback for a list of entries
    def callback_entries(self, s: str, ff=None, T=None, **kwargs):
        specs = ConfEntryList.list_convert(s, str)
        ret = []
        if T is None:
            T = Registrable
        if ff is None:  # ff to get conf
            ff = lambda *zargs: T.key2cls(zargs[0])(*zargs[1:])
        for spec in specs:
            _parts = self._callbak_parse(spec)
            if _parts[0] == "":
                _parts[0] = _parts[1]  # must be there!
            _name = _parts[0]
            # add it
            assert not hasattr(self, _name)
            cc = ff(*_parts[1:])
            assert isinstance(cc, Conf)
            if kwargs:
                cc.direct_update(**kwargs)
            setattr(self, _name, cc)
            # --
            ret.append(_parts)
        return ret

    # one entry
    def callback_entry(self, s: str, ff=None, T=None, df=None, **kwargs):
        _sep = Conf.SEP_DICT_NV
        if T is None:
            T = Registrable
        if ff is None:  # ff to get conf
            ff = lambda *zargs: T.key2cls(zargs[0])(*zargs[1:])
        # --
        _parts = self._callbak_parse(s)
        if len(_parts) == 0:
            return df
        _name, _args = _parts[0], _parts[1:]
        cc = ff(*_args)
        assert isinstance(cc, Conf)
        if kwargs:
            cc.direct_update(**kwargs)
        if _name == '':  # if no name, then simply this one itself!
            return cc
        else:  # add a new one
            assert not hasattr(self, _name)
            setattr(self, _name, cc)
            return _parts
        # --

    # shortcut decorator (using Conf to do rd!)
    @classmethod
    def conf_rd(cls, **kwargs):  # conf's reg decorator for target!
        _TV = TypeVar('_TV', bound=Type)
        def _decorator(_T: _TV) -> _TV:
            base_conf_type = cls.get_base_conf_type()
            base_node_type = cls.get_base_node_type()
            assert issubclass(_T, base_node_type)
            base_conf_type.reg(cls, RC=_T)  # conf -> obj
            _key = base_conf_type.cls2key(cls)
            if ":" in _key:
                _key = None  # note: no reg if no short name!
            base_node_type.reg(_T, key=_key, C=cls, **kwargs)  # obj -> conf
            return _T
        # --
        return _decorator

    @classmethod
    def get_base_conf_type(cls): return Conf
    @classmethod
    def get_base_node_type(cls): return Registrable
    # --

# =====
# global conf pools
class GlobalConf(Conf):
    def __init__(self):
        # --
        pass

    def add_subconf(self, name: str, value: Conf):
        assert not hasattr(self, name)
        setattr(self, name, value)

    def get_or_add_subconf(self, name: str, value: Conf):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            self.add_subconf(name, value)
            return value

# singleton
_singleton_global_conf = None
def get_singleton_global_conf():
    global _singleton_global_conf
    if _singleton_global_conf is None:
        _singleton_global_conf = GlobalConf()
    return _singleton_global_conf

def get_global_conf(path, df=None):
    if isinstance(path, str):
        path = [path]
    curr = get_singleton_global_conf()
    for p in path:
        if curr is None:
            return df
        curr = getattr(curr, p, None)
    return curr

# =====
# helper class for updating
class ConfItem:
    def __init__(self, full_path: str, par_conf: Conf, key: str):
        self.full_path = full_path
        self.par_conf = par_conf
        self.key = key

    # note: special entries to be pre-checked!
    def is_precheck_entry(self):
        return isinstance(getattr(self.par_conf, self.key), (ConfEntryChoices, ConfEntryCallback))

    def realize_entry_choice(self):
        setattr(self.par_conf, self.key, getattr(self.par_conf, self.key).get())

    def do_update(self, val):
        return self.par_conf._update_one(self.key, val)

# =====
# Conf Entries

# basic one
class ConfEntry:
    def __init__(self, df_val=None):
        self.val = df_val

    def get(self): return self.val
    def set(self, x: object): self.val = self.convert(x)
    def __repr__(self): return f"{self.__class__.__name__}(cur={self.val})"
    def convert(self, x: object): return x  # convert input to val

# typed entry
class ConfEntryTyped(ConfEntry):
    def __init__(self, type: Type, df_val: object = None):
        super().__init__(df_val=df_val)
        self.type = type

    def convert(self, x: str):
        return ConfEntryTyped.typed_convert(x, self.type)

    @staticmethod
    def typed_convert(x: str, T: Type):
        if issubclass(T, bool):  # todo(note): especially make "0" be False
            return T(int(x))
        else:
            return T(x)

# special glob with climbing up dirs
class ConfEntryGlob(ConfEntry):
    _ZGLOB_ITER = 10  # this should be enough

    def __init__(self, zglob1: bool, **kwargs):
        super().__init__(df_val="")
        self.zglob1: bool = zglob1
        self.glob_kwargs = {"check_prefix": "..", "check_iter": ConfEntryGlob._ZGLOB_ITER}
        self.glob_kwargs.update(kwargs)

    def convert(self, pathname: str):
        if self.zglob1:
            return zglob1(pathname, **self.glob_kwargs)
        else:
            return zglob(pathname, **self.glob_kwargs)

class ConfEntryList(ConfEntry):
    def __init__(self, item_type: Type, default: List):
        super().__init__(default)
        self.item_type = item_type

    def convert(self, x: str):
        return ConfEntryList.list_convert(x, self.item_type)

    @staticmethod
    def list_convert(x: str, T: Type):
        # try split and assign
        try:
            ret = [ConfEntryTyped.typed_convert(z, T) for z in x.split(Conf.SEP_LIST)] if len(x) > 0 else []
        except:
            ret = eval(x)
        return ret

class ConfEntryDict(ConfEntry):
    def __init__(self, item_type: Type, default: Dict):
        super().__init__(default)
        self.item_type = item_type

    def convert(self, x: str):
        return ConfEntryDict.dict_convert(x, self.item_type)

    @staticmethod
    def dict_convert(x: str, T: Type):
        # try split and assign
        try:
            ret = OrderedDict()  # note: preserve the order!!
            if len(x) > 0:
                for z in x.split(Conf.SEP_DICT_F):
                    k, v = z.split(Conf.SEP_DICT_NV, 1)
                    ret[k] = ConfEntryTyped.typed_convert(v, T)
        except:
            ret = eval(x)
        return ret

# special entry with choices
class ConfEntryChoices(ConfEntry):
    def __init__(self, choices: Dict[str, Conf], default_choice=''):
        super().__init__(df_val=choices.get(default_choice))
        self.choices: Dict[str, object] = choices
        self.default_choice = default_choice

    def convert(self, x: str):
        cc = self.choices.get(x)
        if isinstance(cc, Conf):
            cc._choice = x  # save it here!
        return cc

# special entry with callback
class ConfEntryCallback(ConfEntry):
    def __init__(self, callback: Callable, default_s: str = None):
        super().__init__()
        self.callback = callback
        if default_s is not None:  # allow a default one!
            self.set(default_s)

    def convert(self, s: str):
        ret = self.callback(s)  # note: specific args!
        return ret

# --
class Configurable(Registrable):
    def __init__(self, conf: Conf, **kwargs):
        self.conf = self.setup_conf(conf, **kwargs)
        # --

    # setup conf
    def setup_conf(self, conf: Conf, _no_copy=False, **kwargs):
        if conf is None:  # make a default one!
            conf_type = self.cls2info(field='C')
            conf = conf_type()
        elif not _no_copy:  # todo(note): always copy to local!
            from copy import deepcopy
            conf = deepcopy(conf)
        conf.direct_update(**kwargs)
        conf.validate()  # note: here we do full validate to setup the private conf!
        return conf

    # update conf
    def update_conf(self, _direct=False, **kwargs):
        conf = self.conf
        if _direct:
            conf.direct_update(**kwargs)
        else:
            conf.update_from_dict(kwargs)
        conf.validate()
