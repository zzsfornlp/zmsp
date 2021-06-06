#

# some Basic Fields

__all__ = [
    "SeqField", "PlainSeqField", "InputCharSeqField", "InputSubwordSeqField",
]

from typing import List, Dict
from .base import DataInstance, DataInstanceComposite, SubInfo
from .helper import InSentInstance, SubwordTokenizer, SplitAlignInfo

# =====
# basic seq of tokens
class SeqField(InSentInstance):
    def __init__(self, _id: str = None, _par: 'DataInstance' = None):
        super().__init__(_id, _par)
        # --
        self.vals: List = None
        self._idxes: List[int] = None  # no savings for _idxes

    @classmethod
    def create(cls, vals: List, idxes: List[int] = None, id: str = None, par: 'DataInstance' = None):
        inst = super().create(id, par)
        inst.vals = vals
        inst._idxes = idxes
        return inst

    def __repr__(self):
        return f"{self.__class__.__name__}: {str(self.vals)}"

    def __len__(self):
        return len(self.vals)

    def set_vals(self, vals: List):
        self.vals = vals

    def has_vals(self):
        return self.vals is not None

    @property
    def idxes(self):
        return self._idxes

    # set indexes (special situations)
    def set_idxes(self, idxes: List[int]):
        assert len(idxes)==len(self.vals), "Unmatched length of input idxes."
        self._idxes = idxes

# =====
# Plain one
class PlainSeqField(SeqField):
    def __init__(self, _id: str = None, _par: 'DataInstance' = None):
        super().__init__(_id, _par)
        # --
        self._char_seq: InputCharSeqField = None
        self._subword_seq: InputSubwordSeqField = None

    # compute on demand
    def get_char_seq(self) -> 'InputCharSeqField':
        if self._char_seq is None:
            self._char_seq = InputCharSeqField.create(self.vals, None, par=self)
        return self._char_seq

    def get_subword_seq(self, toker: SubwordTokenizer) -> 'InputSubwordSeqField':
        if self._subword_seq is None:
            self._subword_seq = InputSubwordSeqField.create(self.vals, toker, par=self)
        return self._subword_seq

# =====
# initiated from words (built from words)
class InputCharSeqField(SeqField):
    def __init__(self, _id: str = None, _par: 'DataInstance' = None):
        super().__init__(_id, _par)

    # set indexes (special situations)
    def set_idxes(self, idxes: List[List[int]]):
        assert len(idxes)==len(self.vals), "Unmatched length of input idxes."
        self._idxes = idxes

# =====
# initiated from words (built from words)
# also include mapper from words to subwords
class InputSubwordSeqField(SeqField):
    @classmethod
    def _get_sub_map(cls) -> Dict[str, SubInfo]:
        return {
            "align_info": SubInfo(SplitAlignInfo),
        }

    def __init__(self, _id: str = None, _par: 'DataInstance' = None):
        super().__init__(_id, _par)
        # --
        self.align_info: SplitAlignInfo = None

    @classmethod
    def create(cls, vals: List, toker: SubwordTokenizer, id: str = None, par: 'DataInstance' = None):
        split_vals, split_idxes, align_info = toker.sub_vals(vals)
        inst: InputSubwordSeqField = super().create(split_vals, split_idxes, id, par)
        inst.align_info = align_info
        return inst
