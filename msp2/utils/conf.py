#

# configuration class

__all__ = [
    "Conf", "GlobalConf", "get_singleton_global_conf", "ConfItem",
    "ConfEntry", "ConfEntryGlob", "ConfEntryChoices", "ConfEntryList", "ConfEntryTyped",
]

from typing import List, Tuple, Dict, Iterable, Type, Union
from collections import defaultdict, OrderedDict
from .log import zlog, zwarn, zopen
from .file import zglob, zglob1
from .seria import JsonSerializable, default_json_serializer, get_class_id

# =====
# conf composite
class Conf(JsonSerializable):
    NV_SEP = ":"  # sep for name/key & value
    HIERARCHICAL_SEP = "."  # sep for hierarchies
    LIST_SEP = ","  # sep for list

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def valid_json_fields(self):
        return self.get_good_names()

    # by default, export everything
    def get_good_names(self):
        return super().valid_json_fields()

    # update from another Conf
    def update_from_conf(self, cc: 'Conf'):
        self.from_json(cc.to_json())  # simply do save and load

    # get type hints, otherwise use default type (default value's type)
    # also indicating type inside containers, like List
    # also including special ones (as str): like zglob*, zglob1
    # -- to be overridden
    _ConfTypeHints = {}  # cache

    # use cache
    @classmethod
    def get_type_hints(cls) -> Dict[str, Type]:
        _cache = Conf._ConfTypeHints
        k = get_class_id(cls, use_mod=True)
        if k not in _cache:
            r = {}
            # first try to copy super ones!
            for cl in reversed(cls.__mro__):  # collect in reverse order
                if issubclass(cl, Conf) and hasattr(cl, "_get_type_hints"):
                    r.update(cl._get_type_hints())
            _cache[k] = r
        return _cache[k]

    # to be overridden
    @classmethod
    def _get_type_hints(cls) -> Dict[str, Type]:
        return {}  # no hints for default one

    # ===== final checking
    # check error or force settings; note: only need to be called for the very top one!!
    def validate(self):
        # first realize all ConfEntries
        for n in self.__dict__:  # directly on all
            v = getattr(self, n)
            if isinstance(v, ConfEntry):  # replace with the real value
                setattr(self, n, v.realize())
        # recursively: post-order
        for n in self.get_good_names():
            v = getattr(self, n)  # must be there
            if isinstance(v, Conf):
                v.validate()
        # then self
        self._do_validate()

    # actual validation for the current object
    # -- to be overridden
    def _do_validate(self):
        pass

    # =====
    # updates

    # collecting all names, including shortcuts (all k-last suffixes + zero/one previous one)
    # todo(warn): the number will explode if there are too many layers?
    def _collect_all_names(self):
        ret_map: Dict[str, List[ConfItem]] = defaultdict(list)  # partial_name -> List[ZObject]
        ret_list: List[ConfItem] = []  # List of all entries
        _SEP = Conf.HIERARCHICAL_SEP
        # --
        def _add_rec(cur_conf: Conf, path: List[str]):
            for n in cur_conf.get_good_names():
                path.append(n)
                one = cur_conf.__dict__[n]
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

    # do update for one conf
    def _do_update(self, k: str, value_str: str):
        old_v = self.__dict__[k]
        # first if use ConfEntry, directly use it
        if isinstance(old_v, ConfEntry):
            tmp_old_v = old_v.val
            old_v.set(value_str)
            new_v = old_v.realize()
            old_v = tmp_old_v  # set to original one!
        else:
            # check type mapping, otherwise use default value's type
            sub_conf_type_hints = self.__class__.get_type_hints()  # class method
            trg_type = type(old_v) if old_v is not None else str  # note: by default str if old_v is None
            hint_type = sub_conf_type_hints.get(k)
            if hint_type is not None:  # if there are type hints, use that one as target!!
                trg_type = hint_type
            new_v = None
            # case 0: special hint type!!
            if hint_type == "zglob*":  # find list of files
                new_v = ConfEntryGlob(False).getv(value_str)
            elif hint_type == "zglob1":  # find one file
                new_v = ConfEntryGlob(True).getv(value_str)
            # ...
            if new_v is None:
                # case 1: specially dealing with List
                # todo(note): avoid the case for eval with a simple heurist: checking ends_pair
                ends_pair = (value_str[0]+value_str[-1]) if len(value_str)>0 else ""
                needs_eval = (ends_pair in ["()", "[]", "{}"])
                if isinstance(old_v, list) and not needs_eval:
                    # get or guess element's type
                    _list_item_type = (str if len(old_v)==0 else type(old_v[0])) if hint_type is None else hint_type
                    new_v = ConfEntryList.list_getv(value_str, _list_item_type)
            # otherwise
            if new_v is None:  # must be not assigned successfully by case 1
                # case 2: special types for which directly using eval
                if issubclass(trg_type, (List, Tuple, Dict)):
                    new_v = eval(value_str)
                # case 3: otherwise use trg_type
                else:  # otherwise use trg_type as constructor
                    new_v = ConfEntryTyped.typed_getv(value_str, trg_type)
                if not isinstance(new_v, trg_type):
                    zwarn(f"Possible type error when updating {self}/{k}={value_str}, trg={trg_type}, real={type(new_v)}")
        # set it!!
        self.__setattr__(k, new_v)
        # return (old_v, new_v)
        return (old_v, new_v)

    # directly from iters of (k,v)
    def _update(self, iters: Iterable, _quite=True, _check=False):
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
                if items is not None and len(items)==1 and items[0].is_entry_choice():
                    old_v, new_v = items[0].do_update(v)
                    choice_ones.append(f"Update(Loop={cur_loop}) choice '{n}={v}': {items[0].full_path} = {old_v} -> {new_v}")
                else:  # only the remaining ones are needed!
                    remaining_inputs.append((n,v))
            if not _quite:
                for one in choice_ones:
                    zlog(one, func="config")
            # --
            # then expand all choices at current!
            realized_num = 0
            for item in citem_list:
                if item.is_entry_choice():  # set to default/current one
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
            if len(item_list) != 1:
                bad_ones.append(f"Bad(ambiguous or non-exist) config {n}={v}, -> {[z.full_path for z in item_list]}")
                continue
            full_name = item_list[0].full_path
            hit_full_name[full_name].append(n)
            if len(hit_full_name[full_name]) >= 2:
                warn_ones.append(f"Repeated config with different different names: {full_name}: {hit_full_name[full_name]}")
            old_v, new_v = item_list[0].do_update(v)
            if old_v != new_v:  # real update
                good_ones.append(f"Update config '{n}={v}': {full_name} = {old_v} -> {new_v}")
        # report
        if not _quite:
            for one in good_ones:
                zlog(one, func="config")
            for one in warn_ones:
                zwarn(f"WARNING: {one}")
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

    # directly write fields (only one level)
    def direct_update(self, _assert_exists=False, _try_converse_type=True, **kwargs):
        for k, v in kwargs.items():
            if _assert_exists:
                assert hasattr(self, k), f"No-exist of attr: {k}"
            if _try_converse_type:
                try:
                    vtmp = type(getattr(self, k))(v)
                    v = vtmp
                except:
                    pass
            setattr(self, k, v)
        return self

    # =====
    # some external methods

    @staticmethod
    def extend_args(args: List[str], quite=False):
        sep = Conf.NV_SEP
        # check first one
        args = list(args)
        if len(args) > 0 and len(args[0].split(sep)) == 1:
            if not quite:
                zlog(f"Read config file from {args[0]}.", func="config")
            f_args = []
            with zopen(args[0]) as fd:
                for line in fd:
                    line = line.strip()
                    if len(line)>0 and line[0]!='#':
                        f_args.append(line)
            # cmd configs are at the end
            args = f_args + args[1:]
        argv = OrderedDict()
        for a in args:
            fields = a.split(sep, 1)        # only split the first one
            assert len(fields) == 2, "Strange config updating value"
            if fields[0] in argv and not quite:
                zwarn(f"Overwrite with config {a}")
            argv[fields[0]] = fields[1]
        return argv

    # from list of strings (the first one can be conf file)
    def update_from_args(self, args: List[str], quite=False, check=True, add_global_key="G", validate=True):
        if not quite:
            zlog(f"Update conf from args: {args}.", func="config")
        argv = Conf.extend_args(args, quite=quite)
        if add_global_key:
            assert not hasattr(self, add_global_key)
            setattr(self, add_global_key, get_singleton_global_conf())  # put in global conf to update!!
        self.update_from_dict(argv, _quite=quite, _check=check)
        if validate:
            self.validate()
        return argv

    # convenient ones
    @classmethod
    def direct_conf(cls, conf: 'Conf' = None, **kwargs):
        if conf is None:
            conf = cls()
        conf = conf.direct_update(**kwargs)
        return conf

# =====
# helper class for updating
class ConfItem:
    def __init__(self, full_path: str, par_conf: Conf, key: str):
        self.full_path = full_path
        self.par_conf = par_conf
        self.key = key

    def is_entry_choice(self):
        return isinstance(getattr(self.par_conf, self.key), ConfEntryChoices)

    def realize_entry_choice(self):
        setattr(self.par_conf, self.key, getattr(self.par_conf, self.key).realize())

    def do_update(self, value_str: str):
        return self.par_conf._do_update(self.key, value_str)

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
_singleton_global_conf = GlobalConf()
def get_singleton_global_conf(): return _singleton_global_conf

# =====
# Conf Entries

# basic one
class ConfEntry:
    def __init__(self, df_val=None):
        self.val = df_val

    def realize(self): return self.val
    def set(self, x: object): self.val = self.getv(x)
    def __repr__(self): return f"{self.__class__.__name__}(cur={self.val})"

    # to be implemented
    def getv(self, x: object): raise NotImplementedError()

# typed entry
class ConfEntryTyped(ConfEntry):
    def __init__(self, type: Type, df_val: object = None):
        super().__init__(df_val=df_val)
        self.type = type

    def getv(self, x: str):
        return ConfEntryTyped.typed_getv(x, self.type)

    @staticmethod
    def typed_getv(x: str, T: Type):
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

    def getv(self, pathname: str):
        if self.zglob1:
            return zglob1(pathname, **self.glob_kwargs)
        else:
            return zglob(pathname, **self.glob_kwargs)

# special entry with choices
class ConfEntryChoices(ConfEntry):
    def __init__(self, choices: Dict[str, Conf], default_choice: str):
        super().__init__(df_val=choices[default_choice])
        self.choices: Dict[str, object] = choices
        self.default_choice = default_choice

    def getv(self, choice: str):
        ret = self.choices[choice]
        try:  # try to set the choice
            ret._choice = choice
        except:
            pass
        return ret

class ConfEntryList(ConfEntry):
    def __init__(self, item_type: Type, default: List):
        super().__init__(default)
        self.item_type = item_type

    def getv(self, x: str):
        return ConfEntryList.list_getv(x, self.item_type)

    @staticmethod
    def list_getv(x: str, T: Type):
        # try split and assign
        try:
            ret = [ConfEntryTyped.typed_getv(z, T) for z in x.split(Conf.LIST_SEP)] if len(x) > 0 else []
        except:
            ret = eval(x)
        return ret
