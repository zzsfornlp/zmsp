#

# lexical frame knowledge

from typing import List, Dict, Union
from collections import defaultdict, Counter
import numpy as np
from msp2.utils import zwarn, zlog
from msp2.data.inst import yield_frames

# --

# Frame
class ZFrame:
    def __init__(self, name: str, descr: str = None):
        self.name = name
        self.lexicons: List['ZLexicon'] = []
        self.roles: List['ZRole'] = []
        self.descr = descr
        self.info = {}

    def add_lexicon(self, lex: 'ZLexicon'):
        self.lexicons.append(lex)

    def add_role(self, role: 'ZRole'):
        self.roles.append(role)

    def __repr__(self):
        return f"{self.name}({self.roles})"

    def to_string(self):
        rets = [f"Name: {self.name}"]
        rets.append(f"Descr: {self.descr}")
        rets.append(f"Lexicon: {self.lexicons}")
        if len(self.info) > 0:
            rets.append(f"Info: {self.info}")
        rets.append(f"Roles: ")
        for role in self.roles:
            rets.append(f"\t{role.to_string()}")
        return "\n".join(rets)

# Lexicon, lemma+pos, LU
class ZLexicon:
    def __init__(self, lemma: str, pos: str):
        self.lemma = lemma
        self.pos = pos
        self.info = {}

    def __repr__(self):
        return f"{self.lemma}.{self.pos}"

# role, argument, FE
class ZRole:
    def __init__(self, name: str, category: str, descr: str = None):
        self.name = name
        self.category = category
        self.descr = descr
        self.info = {}

    def __repr__(self):
        return f"{self.name}"

    def to_string(self):
        ret = f"{self.name}({self.category}): {self.descr}"
        if len(self.info) > 0:
            ret += f" ({self.info})"
        return ret

# Frame Collections
class ZFrameCollection:
    def __init__(self, frames: List[ZFrame]):
        self.frames = frames

    def add_frame(self, frame: ZFrame):
        self.frames.append(frame)

# --
# Specific helpers
class ZFrameCollectionHelper:
    # note: most of the time, if splitting, using the first token will be fine?
    @staticmethod
    def build_lu_map(collection: ZFrameCollection, split_lu=''):  # LU -> List[Frame]
        # --
        def _score(_fname: str, _lemma: str):  # check prefix overlap and get iou
            _i = 0
            while _i < len(_fname) and _i < len(_lemma) and _fname[_i]==_lemma[_i]:
                _i += 1
            _ret = _i / len(_fname)
            if _ret < 0.5:  # set a thresh
                _ret = 0.
            return _ret
        # --
        ret0 = defaultdict(set)
        for f in collection.frames:
            frame_name = f.name
            for lu in f.lexicons:
                lemma = lu.lemma
                lemma_pos = f"{lu.lemma}.{lu.pos}"
                ret0[lemma].add(frame_name)
                ret0[lemma_pos].add(frame_name)
                if split_lu != '':
                    lemma_fileds = lemma.split(split_lu)
                    # decide which one by check prefix overlap!
                    f0 = frame_name.lower().split(".")[0]
                    best_score, best_one = _score(f0, lemma_fileds[0]), lemma_fileds[0]
                    for field in lemma_fileds[1:]:
                        _ss = _score(f0, field.lower())
                        if _ss > best_score:
                            best_score, best_one = _ss, field
                    ret0[best_one].add(frame_name)
        ret = {f: sorted(v) for f,v in ret0.items()}
        return ret

    @staticmethod
    def build_role_map(collection: ZFrameCollection):  # Frame -> List[Role]
        ret0 = defaultdict(set)
        for f in collection.frames:
            frame_name = f.name
            for role in f.roles:
                ret0[frame_name].add(role.name)
        ret = {f: sorted(v) for f, v in ret0.items()}
        return ret

    @staticmethod
    def build_constraint_arrs(m: Dict[str, Union[List[str], Dict]], voc_trg, voc_src=None, warning=True):
        # first build targets
        trg_len = len(voc_trg)
        arr_m = {}
        cc = Counter()
        for s, ts in m.items():
            trg_arr = np.zeros(trg_len, dtype=np.float32)
            hit_t = 0
            for t in ts:  # ts can be either List[str] or Dict[str,??]
                t_idx = voc_trg.get(t)
                if t_idx is not None:
                    trg_arr[t_idx] = 1.
                    hit_t += 1
                else:
                    cc["miss_t"] += 1  # miss one t
            if hit_t == 0:
                if warning:
                    zwarn(f"No trgs for src: {s}({ts})")
                cc["miss_ts"] += 1  # miss full ts
            arr_m[s] = trg_arr
        # then for src if providing voc
        if voc_src is None:
            zlog(f"Build constraint_arrs with trg: {len(arr_m)} x {trg_len}; {cc}")
            return arr_m
        else:
            arr_m2 = np.zeros([len(voc_src), trg_len], dtype=np.float32)
            hit_s = 0
            for s, arr in arr_m.items():
                s_idx = voc_src.get(s)
                if s_idx is not None:
                    arr_m2[s_idx] = arr
                    hit_s += 1
                else:
                    cc["miss_s"] += 1
            zlog(f"Build constraint_arrs with src/trg: {arr_m2.shape}; hit={hit_s}/{len(arr_m)}={hit_s/len(arr_m):.4f}; {cc}")
            return arr_m2
        # --

# --
# role budget helper
class RoleBudgetHelper:
    @staticmethod
    def build_role_budgets_from_data(data_stream, max_budget=1000):
        ret = {}
        for f in yield_frames(data_stream):
            f_type = f.type
            _tmp_budget = {}
            for a in f.args:
                a_role = a.role
                _tmp_budget[a_role] = _tmp_budget.get(a_role, 0) + 1
            if f_type not in ret:
                ret[f_type] = {}
            for rr, cc in _tmp_budget.items():
                if cc > 0:
                    ret[f_type][rr] = min(max_budget, max(ret[f_type].get(rr, 0), cc))
        return ret

    @staticmethod
    def build_role_budgets_from_collection(collection: ZFrameCollection, default_budget=1):
        ret = {}
        for f in collection.frames:
            if f.name not in ret:
                ret[f.name] = {}
            for r in f.roles:
                ret[f.name][r.name] = default_budget
        return ret

# --
# b msp2/data/vocab/frames:141
