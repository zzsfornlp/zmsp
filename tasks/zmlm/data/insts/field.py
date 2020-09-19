#

from typing import List
import numpy as np
from msp.utils import zwarn
from msp.data import Vocab
from .base import BaseDataItem, data_type_reg

# =====
# fields are those that must specify IO by themselves
#   todo(note): Field need to deal with empties (usually denoted as None in init)

# base class
@data_type_reg
class DataField(BaseDataItem):
    pass

# general sequence field: words, poses, ...
@data_type_reg
class SeqField(DataField):
    def __init__(self, vals: List, idxes: List[int] = None):
        self.vals = vals
        self.idxes = idxes

    def __len__(self):
        return len(self.vals)

    def has_vals(self):
        return self.vals is not None

    # reset the vals
    def reset(self, vals: List, idxes: List[int] = None):
        self.vals = vals
        self.idxes = idxes

    # set indexes (special situations)
    def set_idxes(self, idxes: List[int]):
        assert len(idxes)==len(self.vals), "Unmatched length of input idxes."
        self.idxes = idxes

    # two directions: val <=> idx
    # init-val -> idx
    def build_idxes(self, voc: Vocab):
        self.idxes = [voc.get_else_unk(w) for w in self.vals]

    # set idx & val
    def build_vals(self, idxes: List[int], voc: Vocab):
        self.idxes = idxes
        self.vals = [voc.idx2word(i) for i in idxes]

    def __repr__(self):
        return str(self.vals)

    # =====
    # io: no idxes, need to rebuild if needed

    def to_builtin(self, *args, **kwargs):
        return self.vals

    @classmethod
    def from_builtin(cls, vals: List):
        return SeqField(vals)

# initiated from words (built from words)
@data_type_reg
class InputCharSeqField(SeqField):
    def __init__(self, words: List[str]):
        super().__init__(words)

    def build_idxes(self, c_voc: Vocab):
        self.idxes = [[c_voc.get_else_unk(c) for c in w] for w in self.vals]

    # set idx & val
    def build_vals(self, idxes: List[int], voc: Vocab):
        raise RuntimeError("No need to build vals for CharSeq.")

    def __repr__(self):
        return str(self.vals)

# NumpyArrField
@data_type_reg
class NpArrField(DataField):
    def __init__(self, arr: np.ndarray, float_decimal=None):
        self.arr = arr
        self.float_decimal: int = float_decimal

    @staticmethod
    def _round_float(x, d):
        r = 10 ** d
        return int(r*x) / r

    def __repr__(self):
        return f"Array: type={self.arr.dtype}, shape={self.arr.shape}"

    def to_builtin(self, *args, **kwargs):
        flattened_arr = self.arr.flatten()
        flattened_list = flattened_arr.tolist()
        if self.float_decimal is not None:
            flattened_list = [NpArrField._round_float(x, self.float_decimal) for x in flattened_list]
        return (self.arr.shape, self.arr.dtype.name, flattened_list)

    @classmethod
    def from_builtin(cls, v):
        shape, dtype, vlist = v
        return np.asarray(vlist, dtype=dtype).reshape(shape)

# DepTreeField: todo(note): remember everything is considering offset=1 for ARTI_ROOT
@data_type_reg
class DepTreeField(DataField):
    def __init__(self, heads: List[int], labels: List[str], label_idxes: List[int] = None):
        # todo(note): ARTI_ROOT as 0 should be included here, added from the outside!
        self.heads = heads
        self.labels = labels
        self.label_idxes = label_idxes
        # == some cache values
        self._dep_dists = None  # m-h
        self._label_matrix = None  # Arr[m,h] of int
        # -----
        if self.heads is not None:
            if self.heads[0] != 0 or self.labels[0] != "":
                zwarn("Bad values for ARTI_ROOT!!")
            self._build_tree()  # build and check

    def __len__(self):
        return len(self.heads)

    def __repr__(self):
        return f"DepTree of len={len(self)}"

    def has_vals(self):
        return self.heads is not None

    # build the extra info for the trees
    def _build_tree(self):
        # TODO(!)
        pass

    # For labels: two directions: val <=> idx
    # init-val -> idx
    def build_label_idxes(self, l_voc: Vocab):
        self.label_idxes = [l_voc.get_else_unk(w) for w in self.labels]

    # set idx & val
    @staticmethod
    def build_label_vals(label_idxes: List[int], l_voc: Vocab):
        labels = [l_voc.idx2word(i) for i in label_idxes]
        labels[0] = ""  # ARTI_ROOT
        return labels

    # =====

    def to_builtin(self, *args, **kwargs):
        return (self.heads, self.labels, self.label_idxes)

    @classmethod
    def from_builtin(cls, v):
        heads, labels, label_idxes = v
        return DepTreeField(heads, labels, label_idxes)

    # =====
    @property
    def dep_dists(self):
        if self._dep_dists is None:
            self._dep_dists = [m-h for m, h in enumerate(self.heads)]
        return self._dep_dists

    @property
    def label_matrix(self):
        if self._label_matrix is None:
            # todo(+N): currently only mh_dep, may be extended to other relations or multi-hop relations
            cur_heads, cur_lidxs = self.heads, self.label_idxes  # [Alen=1+Rlen]
            cur_alen = len(cur_heads)
            _mat = np.zeros([cur_alen, cur_alen], dtype=np.long)  # [Alen, Alen]
            _mat[np.arange(cur_alen), cur_heads] = cur_lidxs  # assign mh_dep
            _mat[0] = 0  # no head for ARTI_ROOT
            self._label_matrix = _mat
        return self._label_matrix
