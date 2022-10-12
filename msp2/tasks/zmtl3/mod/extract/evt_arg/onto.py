#

__all__ = [
    "Frame", "Role", "Onto",
]

# similar to data.vocab.frames
# ok, another set of onto(-like) utils?; note: okay, still simply use pickle to serialize!
# note: simply rebuild vocab and over-write if updated, since things will likely be fully external!!

from typing import List, Union, Dict, Tuple
from collections import Counter
from msp2.utils import default_json_serializer, default_pickle_serializer, zlog

# --
# items
class _Element:
    def __init__(self, name: str, **kwargs):
        self.name = name  # the original label name
        self.info = {}
        self.info.update(kwargs)

    def __getattr__(self, item):
        for z in [item, "_" + item]:
            if z in self.info:
                return self.info[z]
        raise AttributeError(f"Unknown item of {item}")

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def to_json(self):
        ret = self.__dict__.copy()
        del ret['info']  # no putting info!
        for k in list(ret.keys()):
            if k.startswith("_"):
                del ret[k]
        for k, v in ret.items():
            if isinstance(v, (list, tuple)):
                ret[k] = [(z.name if isinstance(z, _Element) else z) for z in v]
        for k, v in self.info.items():
            if not k.startswith("_") and 'builtin' in type(v).__module__:
                ret[k] = v
        return ret

    @staticmethod
    def sort_key(element: '_Element'):
        return element.name  # sort with the name

    def simple_name2phrase(self, name: str):
        ret = name.lower()
        for sep in "_:.-":
            ret = " ".join(ret.split(sep))
        return ret

# frame
class Frame(_Element):
    def __init__(self, name: str, vp: str = None, core_roles: List['Role'] = None, noncore_roles: List['Role'] = None,
                 template: str = None, **kwargs):
        super().__init__(name, **kwargs)
        self.vp = self.simple_name2phrase(name) if vp is None else vp  # verb-like phrase
        self.core_roles = [] if core_roles is None else core_roles  # List[Role]
        self.noncore_roles = [] if noncore_roles is None else noncore_roles  # List[Role]
        # note: two options, "<R1> ... <R2> ..." or [[RoleName, [Preps]], [...], [None, []], [...], ...]
        self.template: Union[str, List] = template
        # --
        self._role_map: Dict[str, Tuple[Role, bool]] = None  # (active) role maps

    @property
    def active_noncore_roles(self):
        _m = self.role_map
        return [z for z in self.noncore_roles if z.name in _m]

    @property
    def role_map(self):
        if self._role_map is None:
            self.build_role_map()
        return self._role_map

    def build_role_map(self, nc_filter=None, force_rebuild=False):
        if nc_filter is None:
            nc_filter = lambda x: True
        assert self._role_map is None or force_rebuild
        # --
        cc = 0
        _rmap = {}
        for c_role in self.core_roles:  # always add core roles
            _rmap[c_role.name] = (c_role, True)
            cc += 1
        for nc_role in self.noncore_roles:
            _name = nc_role.name
            if nc_filter(_name):
                _rmap[_name] = (nc_role, False)
                cc += 1
        assert cc == len(_rmap)
        self._role_map = _rmap
        return

    def find_role(self, name: str, df=None):  # return Role, IsCore?
        if name in self.role_map:
            return self.role_map[name]
        return df, False  # by default not "IsCore"

    def self_check(self):
        # check no repeat roles
        all_role_names = [z.name for z in self.core_roles + self.noncore_roles]
        assert len(set(all_role_names)) == len(all_role_names)
        # check template
        if self.template is not None:
            self.check_template(self.template)
        # --

    def check_template(self, template):
        # --
        if isinstance(template, str):
            import re
            pat = re.compile(r"<[_a-zA-Z0-9]+>")  # PartialSeq._PATTERN
            vars = [z[1:-1] for z in pat.findall(template)]
        else:
            vars = [z[0] for z in template if z[0] is not None]
        # --
        set_vars = set(vars)
        assert len(set_vars) == len(vars)  # no repeat
        assert all(c.name in set_vars for c in self.core_roles)  # all core-roles should be there
        set_vars.difference_update([c.name for c in self.core_roles])
        set_vars.difference_update([c.name for c in self.noncore_roles])
        assert len(set_vars) == 0  # no other roles!
        # --

    # pretty_print
    def pp_str(self, tpl_include_nc=None):
        a_map_ss = f' (a_map={self.info["a_map"]})' if 'a_map' in self.info else ''
        cr_ss = lambda z: f'{z.name}({z.np}/{z.info.get("np_vn")}/{z.info.get("np_fn")})<{"/".join(z.qwords) if z.qwords is not None else ""}>'
        rets = [
            f"{self.name} ({self.vp}){a_map_ss}",
            f"\tCore: {[cr_ss(z) for z in self.core_roles]}",
            f"\tNonCore: {[z.name for z in self.noncore_roles]}",
        ]
        if self.template is not None:
            if isinstance(self.template, str):
                rets.append(f"\tTemplate(S): {self.template}")
            else:
                ones = []
                core_map = {z.name: True for z in self.core_roles}
                core_map.update({z.name: False for z in self.noncore_roles})
                # --
                if tpl_include_nc is None:
                    inc_nc = core_map  # include all
                else:
                    inc_nc = set(tpl_include_nc)
                # --
                for name, pps in self.template:
                    if name is None:
                        ones.append("V")
                    else:
                        is_core = core_map[name]
                        if not is_core and name not in inc_nc:
                            continue  # ignore this one
                        _ps = '' if len(pps)==0 else f"({'/'.join(pps)})"
                        ones.append(f"{_ps}<{name}>")
                rets.append(f"\tTemplate(L): {' '.join(ones)}")
        # --
        ret = "\n".join(rets)
        return ret

# role
class Role(_Element):
    def __init__(self, name: str, np: str = None, qwords: List = None, **kwargs):
        super().__init__(name, **kwargs)
        self.np = self.simple_name2phrase(name) if np is None else np  # noun phrase
        self.qwords = qwords  # List[question-words], note: first one is the default one!
        # --

# --
# onto (currently only have frame-role relations!)
class Onto:
    def __init__(self, frames: List[Frame], roles: List[Role]):
        self.frames = frames  # all frames
        self.roles = roles  # all roles
        # --
        # caches: None if not built
        self._idx_offset = None
        self._frame_map: Dict[str, Frame] = None

    def refresh_cache(self):
        if self._idx_offset is not None:
            self.build_idxes(self._idx_offset, force_rebuild=True)  # rebuild!
        self._frame_map = None
        # --

    def build_idxes(self, idx_offset: int, force_rebuild=False):
        assert force_rebuild or self._idx_offset is None
        self._idx_offset = idx_offset
        for ii, ff in enumerate(self.frames, idx_offset):
            ff.info['_idx'] = ii
        for ii, rr in enumerate(self.roles, idx_offset):
            rr.info['_idx'] = ii
        # --

    @property
    def frame_map(self):
        if self._frame_map is None:
            self._frame_map = {f.name: f for f in self.frames}
            assert len(self._frame_map) == len(self.frames)
        return self._frame_map

    def find_frame(self, name: str, df=None):
        return self.frame_map.get(name, df)

    def __repr__(self):
        return f"Onto:frames={len(self.frames)},roles={len(self.roles)},core={sum(len(z.core_roles) for z in self.frames)}"

    @staticmethod
    def create_simple_onto(frame_names: List[str], role_names: List[str]):
        # simply create a vocab like flatten one
        roles = [Role(r, None) for r in role_names]
        frames = [Frame(f, None, roles, []) for f in frame_names]  # allow every role into every frame!
        ret = Onto(frames, roles)
        return ret

    @staticmethod
    def create_from_json(d):
        roles = [Role(**d2) for d2 in d['roles']]
        r_map = {r.name: r for r in roles}
        assert len(r_map) == len(roles)
        frames = [Frame(**d2) for d2 in d['frames']]
        for f in frames:  # replace with real roles
            f.core_roles = [r_map[z] for z in f.core_roles]
            f.noncore_roles = [r_map[z] for z in f.noncore_roles]
        ret = Onto(frames, roles)
        for r in ret.roles:
            r.name = r.name.rsplit("^N^", 1)[0]
        return ret

    def to_json(self):
        # deal with repeat names
        cc_role_names = Counter()
        for r in self.roles:
            _orig_name = r.name
            cc = cc_role_names[_orig_name]
            if cc > 0:  # not the first one
                r.name = f"{r.name}^N^{cc}"
            cc_role_names[_orig_name] += 1
        # copy out
        ret = {"frames": [f.to_json() for f in self.frames], "roles": [r.to_json() for r in self.roles]}
        # revert back
        for r in self.roles:
            r.name = r.name.rsplit("^N^", 1)[0]
        return ret

    @staticmethod
    def load_onto(s: str):
        from .onto_rs import get_predefined_onto
        d = get_predefined_onto(s)
        if d is not None:
            ret = Onto.create_from_json(d)
            load_type = 'predefined'
        else:
            try:
                ret = default_pickle_serializer.from_file(s)
                load_type = 'pkl'
            except:
                d = default_json_serializer.from_file(s)
                ret = Onto.create_from_json(d)
                load_type = 'json'
        for f in ret.frames:
            f.self_check()
        zlog(f"Load ({load_type}) from {s}: {ret}")
        return ret

    def pp_str(self, *args, **kwargs):
        rets = []
        for f in self.frames:
            rets.append(f.pp_str(*args, **kwargs))
        ret = "\n\n".join(rets)
        return ret
