#

# some Basic Fields

__all__ = [
    "SeqMAlignInfo", "SeqField",
]

from typing import List, Dict, Union, Sequence, Iterable
from .base import DataInst

# =====

# monotonic alignments
@DataInst.rd('malign')
class SeqMAlignInfo(DataInst):
    def __init__(self, counts: Sequence[int] = ()):
        super().__init__()
        # -- [start, end) of the maligns
        self.o2n_start, self.o2n_end, self.n2o_start, self.n2o_end = SeqMAlignInfo.build_malign(counts)
        # --

    def get_reverse(self):
        # note: simply reverse the maps by directly assigning!
        ret = SeqMAlignInfo()
        ret.o2n_start, ret.o2n_end, ret.n2o_start, ret.n2o_end = \
            self.n2o_start, self.n2o_end, self.o2n_start, self.o2n_end
        return ret

    @staticmethod
    def build_malign(counts: Sequence[int]):
        # counts: (>0 split, <0 merge) len(fractions)==len(old)
        o2n_start, o2n_end = [], []  # len(old)
        n2o_start, n2o_end = [], []  # len(new)
        if counts:  # build them!
            cur_ii, cur_jj = 0, 0  # next-idx for (old, new)
            while cur_ii < len(counts):
                vv = counts[cur_ii]
                if vv >= 0:  # one to many (vv)
                    o2n_start.append(cur_jj)
                    o2n_end.append(cur_jj+vv)
                    n2o_start.extend([cur_ii] * vv)
                    n2o_end.extend([cur_ii+1] * vv)
                    cur_ii += 1
                    cur_jj += vv
                else:  # many (-vv) to one
                    rvv = -vv
                    assert all(counts[z]==vv for z in range(cur_ii, cur_ii+rvv))
                    o2n_start.extend([cur_jj] * rvv)
                    o2n_end.extend([cur_jj+1] * rvv)
                    n2o_start.append(cur_ii)
                    n2o_end.append(cur_ii+rvv)
                    cur_ii += rvv
                    cur_jj += 1
        return o2n_start, o2n_end, n2o_start, n2o_end

    @staticmethod
    def combine(infos: Iterable['SeqMAlignInfo']):
        infos = list(infos)
        o2n_start, o2n_end, n2o_start, n2o_end = [], [], [], []
        # --
        old_offset, new_offset = 0, 0
        for info in infos:
            o2n_start.extend([z+new_offset for z in info.o2n_start])
            o2n_end.extend([z+new_offset for z in info.o2n_end])
            n2o_start.extend([z+old_offset for z in info.n2o_start])
            n2o_end.extend([z+old_offset for z in info.n2o_end])
            old_offset += len(info.o2n_start)  # +=len(old)
            new_offset += len(info.n2o_start)  # +=len(new)
        # --
        ret = SeqMAlignInfo()
        ret.o2n_start, ret.o2n_end, ret.n2o_start, ret.n2o_end = o2n_start, o2n_end, n2o_start, n2o_end
        return ret

# seq of tokens
@DataInst.rd('seq')
class SeqField(DataInst):
    def __init__(self, vals: List = None, idxes: List[int] = None, par: DataInst = None):
        super().__init__(par=par)
        # --
        self.vals = vals
        self._idxes = idxes  # no savings for _idxes
        # --  # note: these are not stored currently!
        self._ma_info: SeqMAlignInfo = None  # par -> self!
        self._ma_fields: Dict[str, 'SeqField'] = {}  # sig -> SeqField (self -> children)
        # --

    def __repr__(self): return f"{self.__class__.__name__}: {str(self.vals)}"
    def __len__(self): return len(self.vals)
    def __getitem__(self, item): return self.vals[item]
    @property
    def idxes(self): return self._idxes
    @property
    def ma_fields(self): return self._ma_fields
    @property
    def ma_info(self): return self._ma_info

    def clear_cached_vals(self):
        super().clear_cached_vals()
        self._idxes = None
        self._ma_fields.clear()

    # set indexes (special situations)
    def set_idxes(self, idxes: List[int]):
        assert len(idxes)==len(self.vals), "Unmatched length of input idxes."
        self._idxes = idxes

    def set_ma_info(self, ma_info):
        self._ma_info = ma_info

    # --
    # get sub-field with sub-tokenizer
    def get_sf(self, sub_toker=None, sig=None):
        _sig = sub_toker.get_sig() if sig is None else sig
        _sf = self._ma_fields.get(_sig)
        if _sf is None:
            _svals, _sidxes, _sinfo = sub_toker.sub_vals(self.vals)
            _sf = SeqField(_svals, _sidxes, par=self)
            _sf.set_ma_info(_sinfo)  # set info
            self._ma_fields[_sig] = _sf  # set cache
        return _sf

    @staticmethod
    def combine(seqs: Iterable['SeqField']):
        seqs = list(seqs)
        # combine main
        all_vals = sum([s.vals for s in seqs], [])
        all_idxes = []
        for s in seqs:
            if s.idxes is None:
                all_idxes = None
                break  # no idxes if any is None
            else:
                all_idxes.extend(s.idxes)
        ret = SeqField(all_vals, all_idxes)  # note: par is None
        # combine sf
        all_fs = {}
        if len(seqs) > 0:
            for k in seqs[0].ma_fields.keys():
                _all_subs = [z.ma_fields.get(k) for z in seqs]
                if all(z is not None for z in _all_subs):
                    all_fs[k] = SeqField.combine(_all_subs)
                    all_fs[k].set_par(ret)
                    all_fs[k].set_ma_info(SeqMAlignInfo.combine([z.ma_info for z in _all_subs]))
        ret.ma_fields.update(all_fs)
        # --
        return ret
