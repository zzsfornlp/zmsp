#

# Trees for a sentence

__all__ = [
    "DepTree", "PhraseTree", "HeadFinder",
]

# =====
# Trees

from typing import List, Dict, Iterable, Union, Tuple
import numpy as np
from .base import DataInstance, DataInstanceComposite, SubInfo
from .helper import InDocInstance, InSentInstance
from .field import PlainSeqField

# =====
# dependency tree
# todo(note): inputs should not include the ARTI_ROOT, but "heads" should already assume it and start from 1
class DepTree(InSentInstance):
    @classmethod
    def _get_sub_map(cls) -> Dict[str, SubInfo]:
        return {"seq_head": SubInfo(PlainSeqField), "seq_label": SubInfo(PlainSeqField)}

    def __init__(self, _id: str = None, _par: 'DataInstance' = None):
        super().__init__(_id, _par)
        # --
        self.seq_head: PlainSeqField = None
        self.seq_label: PlainSeqField = None
        # == some cache values
        self._depths: List[int] = None  # From Arti-ROOT, where Depth(Real-Root)==1
        self._dep_dists: List[int] = None  # m-h
        self._label_matrix: np.ndarray = None  # Arr[m,h] of int
        self._chs_lists: List[List[int]] = None  # List(len==m+1) of List[int]
        self._ranges: List = None  # List[(left, right)], len==m

    def __len__(self):
        return len(self.seq_head)

    def __repr__(self):
        return f"DepTree(len={len(self)})"

    def _clear_caches(self):
        self._depths = None
        self._dep_dists = None
        self._label_matrix = None
        self._chs_lists = None
        self._ranges = None

    @classmethod
    def create(cls, heads: List[int] = None, labels: List[str] = None, sent: 'Sent' = None,
               id: str = None, par: 'DataInstance' = None):
        inst: DepTree = super().create(id, par)
        if sent is not None:  # directly set Sent
            inst.set_sent(sent)
        if heads is not None:
            inst.build_heads(heads)
        if labels is not None:
            inst.build_labels(labels)
        return inst

    def build_heads(self, heads: List[int]):
        sent = self.sent
        if sent is not None:
            assert len(sent) == len(heads), "Error: length mismatch!"
        self.seq_head = PlainSeqField.create(vals=heads, par=self)
        return self.seq_head

    def build_labels(self, labels: List[str]):
        assert len(labels) == len(self.seq_head), "Error: length mismtach"  # directly match with heads, must be there
        self.seq_label = PlainSeqField.create(vals=labels, par=self)
        return self.seq_label

    # =====
    @property
    def dep_dists(self):
        if self._dep_dists is None:  # note: +1 as extra offset!
            self._dep_dists = [m+1-h for m, h in enumerate(self.seq_head.vals)]
        return self._dep_dists

    @property
    def label_matrix(self):
        if self._label_matrix is None:
            # todo(+N): currently only mh_dep, may be extended to other relations or multi-hop relations
            cur_heads, cur_lidxs = self.seq_head.vals, self.seq_label.idxes  # [Alen=1+Rlen]
            mlen = len(cur_heads)
            hlen = mlen+1
            _mat = np.zeros([mlen, hlen], dtype=np.long)  # [m,h], by default 0!
            _mat[np.arange(mlen), cur_heads] = cur_lidxs  # assign mh_dep
            self._label_matrix = _mat
        return self._label_matrix

    @property
    def chs_lists(self):
        if self._chs_lists is None:
            cur_heads = self.seq_head.vals
            # --
            chs = [[] for _ in range(len(cur_heads)+1)]
            for m,h in enumerate(cur_heads):  # note: already sorted in left-to-right
                chs[h].append(m)  # note: key is hidx, value is midx
            self._chs_lists = chs
        return self._chs_lists  # note: include +1 offset since ROOT can also have chs

    @property
    def depths(self):
        if self._depths is None:
            cur_heads = self.seq_head.vals
            # --
            depths = [-1] * len(cur_heads)
            for m in range(len(cur_heads)):
                path = []
                cur_idx = m
                while cur_idx>=0 and depths[cur_idx] < 0:  # not ArtiRoot and not visited
                    path.append(cur_idx)
                    cur_idx = cur_heads[cur_idx] - 1  # offset -1
                up_dist = depths[cur_idx] if cur_idx>=0 else 0
                for i, idx in enumerate(reversed(path)):
                    depths[idx] = up_dist + i + 1  # +1 for offset of ArtiRoot
            self._depths = depths
        return self._depths

    @property
    def ranges(self):
        if self._ranges is None:
            # todo(+2): go recursion could be more efficient!
            cur_heads = self.seq_head.vals
            ranges = [[z,z] for z in range(len(cur_heads))]
            for m in range(len(cur_heads)):
                cur_idx = m
                while cur_idx >= 0:
                    ranges[cur_idx][0] = min(m, ranges[cur_idx][0])
                    ranges[cur_idx][1] = max(m, ranges[cur_idx][1])
                    cur_idx = cur_heads[cur_idx] - 1  # offset -1
            self._ranges = ranges
        return self._ranges

    # other info
    # note: here all idxes has no +1 offset!! by default no including of the connecting point (common ancestor=0)!
    def get_spine(self, widx: int):
        cur_heads = self.seq_head.vals
        ret = []
        while widx >= 0:
            ret.append(widx)
            widx = cur_heads[widx] - 1
        return ret

    def get_path(self, idx0: int, idx1: int, inc_common=0):
        spine0 = self.get_spine(idx0)
        spine1 = self.get_spine(idx1)
        # remove common ones
        i0, i1 = len(spine0)-1, len(spine1)-1
        while i0>=0 and i1>=0 and spine0[i0]==spine1[i1]:
            i0 -= 1
            i1 -= 1
        spine0 = spine0[:i0+1+inc_common]  # include how many common ancesters
        spine1 = spine1[:i1+1+inc_common]
        return spine0, spine1

# =====
# phrase-based (constituency) tree
# TODO(!)
class PhraseTree(InSentInstance):
    pass

# =====
# head word/span finder (based on dependency tree)

class HeadFinder:
    NOUN_HEAD_SCORES = {"NOUN": 0, "PROPN": -1, "NUM": -2, "VERB": -3, "PRON": -4, "ADJ": -7, "ADV": -10, "SYM": -11, "X": -12}
    VERB_HEAD_SCORES = {"VERB": 1, "NOUN": 0, "PROPN": -1, "NUM": -2, "PRON": -4, "ADJ": -7, "ADV": -10, "SYM": -11, "X": -12}

    def __init__(self, upos_score_map: Union[Dict[str, float], str]):
        self.upos_score_map = self._get_upos_score_map(upos_score_map)

    def _get_upos_score_map(self, upos_score_map: Union[Dict[str, float], str]):
        if isinstance(upos_score_map, str):
            return {"NOUN": HeadFinder.NOUN_HEAD_SCORES, "VERB": HeadFinder.VERB_HEAD_SCORES}[upos_score_map]
        else:
            assert isinstance(upos_score_map, dict)
            return upos_score_map

    # return single head_idx on hspan
    def find_shead_from_span(self, sent, hspan_widx: int, hspan_wlen: int):
        # todo(+W): by default simply return rightmost!
        return hspan_widx + hspan_wlen - 1

    # return single head_idx
    def find_shead(self, sent, widx: int, wlen: int):
        hspan_widx, hspan_wlen = self.find_hspan(sent, widx, wlen)
        return self.find_shead_from_span(sent, hspan_widx, hspan_wlen)

    # retrun list(idxes)
    def get_mindepth_list(self, sent, widx: int, wlen: int):
        dt_depths = sent.tree_dep.depths
        # pass 1: first pass by min depth
        min_depth = min(dt_depths[z] for z in range(widx, widx+wlen))
        cand_pass1 = [z for z in range(widx, widx+wlen) if dt_depths[z]<=min_depth]
        assert len(cand_pass1) > 0
        return cand_pass1

    # return (head_widx, head_wlen)
    def find_hspan(self, sent, widx: int, wlen: int):
        dt: DepTree = sent.tree_dep
        dt_depths = dt.depths
        idx_start, idx_end = widx, widx + wlen
        # pass 1: first pass by min depth
        min_depth = min(dt_depths[z] for z in range(idx_start, idx_end))
        cand_pass1 = [z for z in range(idx_start, idx_end) if dt_depths[z]<=min_depth]
        assert len(cand_pass1) > 0
        if len(cand_pass1) == 1:
            return (cand_pass1[0], 1)
        # pass 2: second pass by POS
        upos_score_map = self.upos_score_map
        uposes = sent.seq_upos.vals
        upos_scores = [upos_score_map.get(uposes[z], -100) for z in cand_pass1]
        max_upos_score = max(upos_scores)
        cand_pass2 = [v for i,v in enumerate(cand_pass1) if upos_scores[i]>=max_upos_score]
        assert len(cand_pass2) > 0
        if len(cand_pass2) == 1:
            return (cand_pass2[0], 1)
        # pass 3: by default since get the span
        ret_widx = min(cand_pass2)
        ret_wlen = max(cand_pass2) - ret_widx + 1
        return (ret_widx, ret_wlen)

    # shortcut
    def set_head_for_mention(self, mention, force_refind=False, skip_if_shead=True):
        sent = mention.sent
        widx, wlen = mention.get_span()
        # --
        if not force_refind and skip_if_shead and mention.shead_widx is not None:
            return  # no need to set!
        # --
        # first for hspan
        hspan_widx, hspan_wlen = mention.get_span(hspan=True)
        if force_refind or hspan_widx is None:
            hspan_widx, hspan_wlen = self.find_hspan(sent, widx, wlen)
            mention.set_span(hspan_widx, hspan_wlen, hspan=True)
        # then for shead
        shead_widx, _ = mention.get_span(shead=True)
        if force_refind or shead_widx is None:
            shead_widx = self.find_shead_from_span(sent, hspan_widx, hspan_wlen)
            mention.set_span(shead_widx, 1, shead=True)
        # --
