#

from collections import Iterable, OrderedDict, defaultdict
import json

from .log import printing, zopen
from .check import zcheck, zfatal, zwarn
from .utils import Constants

# todo(+N): need to provide more detailed names if there are ambiguity, anyway to set default ones?
class Conf(object):
    NV_SEP = ":"
    HIERARCHICAL_SEP = "."
    LIST_SEP = ","

    #
    def good_name(self, n, d=None):
        # todo(+2): need special rules?
        # good_start = not n.startswith("_")
        good_start = True
        if d is None:
            return good_start
        else:
            return good_start and (n in d)

    # Conf <-> Recursive-Dict
    def to_builtin(self):
        ret = {}
        for n in self.__dict__:
            if self.good_name(n):
                one = self.__dict__[n]
                if isinstance(one, Conf):
                    ret[n] = one.to_builtin()
                else:
                    ret[n] = one
        return ret

    def from_builtin(self, v):
        for n in v:
            if self.good_name(n):
                one = self.__dict__[n]
                if isinstance(one, Conf):
                    one.from_builtin(v[n])
                else:
                    self.__dict__[n] = v[n]

    #
    # from Conf, no returns
    def update_from_conf(self, cc):
        for n, v in cc.__dict__.items():
            if self.good_name(n, self.__dict__):
                if isinstance(self.__dict__[n], Conf):
                    zcheck(isinstance(v, Conf), "Not Sub-Conf.")
                    self.__dict__[n].update_from_conf(v)
                else:
                    self.__dict__[n] = type(self.__dict__[n])(v)

    # =====
    # collecting shortcuts (all k-last suffixes * zero/one previous random one)
    # todo(warn): the number will explode if there are too many layers?
    def collect_all_names(self):
        def _add_rec(v, cur_conf, path):
            for n in cur_conf.__dict__:
                if cur_conf.good_name(n) and not n.startswith("_"):  # todo(note): here does not include "_"-starters
                    path.append(n)
                    one = cur_conf.__dict__[n]
                    if isinstance(one, Conf):
                        _add_rec(v, one, path)
                    else:
                        full_path = Conf.HIERARCHICAL_SEP.join(path)
                        for i in range(len(path)):
                            short_name = Conf.HIERARCHICAL_SEP.join(path[i:])
                            # further combination: (n^2): single previous
                            for lead_name in [""] + path[:max(0, i-1)]:  # i-1 to avoid repeating for continuous one
                                if lead_name:
                                    final_name = Conf.HIERARCHICAL_SEP.join([lead_name, short_name])
                                else:
                                    final_name = short_name
                                #
                                if final_name not in v:
                                    v[final_name] = []
                                v[final_name].append(full_path)
                    path.pop()
        #
        ret = {}    # post_name -> full_name set
        _add_rec(ret, self, [])
        return ret

    # do update for one
    def _do_update(self, n, v):
        #
        def _getv(tt, vv):
            if tt == bool:
                # todo(warn): make "0" be False
                return bool(int(vv))
            else:
                return tt(vv)
        #
        ks = n.split(Conf.HIERARCHICAL_SEP)
        # get to the last layer
        sub_conf = self
        for k in ks[:-1]:
            sub_conf = sub_conf.__dict__[k]
        # update the last name
        last_name = ks[-1]
        old_v = sub_conf.__dict__[last_name]
        item_type = type(old_v)
        # todo(+1): can also use python's "eval" function
        if isinstance(old_v, list):
            try:
                v_toassign = json.loads(v)
                if not isinstance(v_toassign, list):
                    raise ValueError("Not a list!")
            # except json.JSONDecodeError:
            except ValueError:
                v_toassign = v.split(Conf.LIST_SEP) if len(v)>0 else []
            # todo(warn): for default empty ones, also special treating for bool
            ele_type = type(old_v[0]) if len(old_v) > 0 else str
            sub_conf.__dict__[last_name] = [_getv(ele_type, z) for z in v_toassign]
        else:
            sub_conf.__dict__[last_name] = _getv(item_type, v)
        # return (name, old value, new value)
        return (n, old_v, sub_conf.__dict__[last_name])

    # directly from dict
    def update_from_dict(self, d):
        name_maps = self.collect_all_names()
        good_ones, bad_ones = [], []
        hit_full_name = defaultdict(list)
        for n, v in d.items():
            name_list = name_maps.get(n, None)
            if name_list is None:
                bad_ones.append(f"Unknown config {n}={v}")
                continue
            if len(name_list) != 1:
                bad_ones.append(f"Bad(ambiguous or non-exist) config {n}={v}, -> {name_list}")
                continue
            full_name = name_list[0]
            hit_full_name[full_name].append(n)
            if len(hit_full_name[full_name]) >= 2:
                bad_ones.append(f"Repeated config with different short names: {full_name}: {hit_full_name[full_name]}")
                continue
            _, old_v, new_v = self._do_update(full_name, v)
            good_ones.append(f"Update config '{n}={v}': {full_name} = {old_v} -> {new_v}")
        return good_ones, bad_ones

    def update_from_kwargs(self, **kwargs):
        return self.update_from_dict(kwargs)

    # used for init
    def init_from_kwargs(self, **kwargs):
        _, bad_ones = self.update_from_dict(kwargs)
        zcheck(len(bad_ones)==0, "Bad conf: %s" % bad_ones)
        return self

    # ===== final checking
    # check error or force settings:
    # todo(warn): should be called from the upmost Conf
    def validate(self):
        # recursively: post-order
        for n in self.__dict__:
            if self.good_name(n):
                one = self.__dict__[n]
                if isinstance(one, Conf):
                    one.validate()
        # then self
        self.do_validate()

    # todo(warn): to be overridden!!
    def do_validate(self):
        pass

    # =====
    # extra reading from conf file
    @staticmethod
    def extend_args(args, quite=False):
        sep = Conf.NV_SEP
        # check first one
        if len(args) > 0 and len(args[0].split(sep)) == 1:
            if not quite:
                printing("Read config file from %s." % args[0], func="config")
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
            assert len(fields) == 2, "strange config updating value"
            if fields[0] in argv:
                zwarn("Overwrite config %s." % a)  # TODO(+N): a tricky bug when using different names for the same conf!!
            argv[fields[0]] = fields[1]
        return argv

    # used when need to specify some key args at the very start
    @staticmethod
    def search_args(args, targets, types, defaults):
        all_argv = Conf.extend_args(args, True)
        basic_argv = {}
        for idx, k in enumerate(targets):
            ff, dd = types[idx], defaults[idx]
            if k in all_argv:
                zz = ff(all_argv[k])
            else:
                zz = dd
            basic_argv[k] = zz
        return all_argv, basic_argv

    # from list of strings (the first one can be conf file)
    def update_from_args(self, args, quite=False, fatal=True):
        if not quite:
            printing("Update conf from args %s." % (args,), func="config")
        #
        if not isinstance(args, dict):
            argv = Conf.extend_args(args, quite=quite)
        else:
            argv = args
        good_ones, bad_ones = self.update_from_dict(argv)
        if not quite:
            for one in good_ones:
                printing(one, func="config")
            for one in bad_ones:
                zwarn(one)
        if fatal and len(bad_ones) > 0:
            zfatal("Config Err above!")
        return argv
