#

# Trees for a sentence

__all__ = [
    "DepTree", "PhraseTree", "HeadFinder",
]

# =====
# Trees

from typing import List, Dict, Iterable, Union, Tuple
import numpy as np
from collections import defaultdict
from mspx.utils import zwarn, Conf, InfoField
from .base import DataInst
from .field import SeqField

# =====
# dependency tree
# note: inputs should not include the ARTI_ROOT, but "heads" should already assume it and start from 1

@DataInst.rd('dt')
class DepTree(DataInst):
    def __init__(self, heads: List[int] = None, labels: List[str] = None, par: 'Sent' = None):
        super().__init__(par=par)
        # --
        self.seq_head: SeqField = None if heads is None else SeqField(heads, par=par)
        self.seq_label: SeqField = None if labels is None else SeqField(labels, par=par)
        # == some cache values
        self._depths: List[int] = None  # From Arti-ROOT, where Depth(Real-Root)==1, len==m
        self._dep_dists: List[int] = None  # m-h, len==m
        self._label_matrix: np.ndarray = None  # Arr[m,h] of int
        self._chs_lists: List[List[int]] = None  # List(len==m+1) of List[int]
        self._ranges: List = None  # List[(left, right)], len==m
        # --

    def __len__(self):
        return len(self.seq_head)

    def __repr__(self):
        return f"DepTree(len={len(self)})"

    def clear_cached_vals(self):
        super().clear_cached_vals()
        self._depths = None
        self._dep_dists = None
        self._label_matrix = None
        self._chs_lists = None
        self._ranges = None

    @classmethod
    def _info_fields(cls):
        return {'seq_head': InfoField(inner_type=SeqField),
                'seq_label': InfoField(inner_type=SeqField)}

    def finish_from_dict(self):
        # special handling!
        for z in [self.seq_head, self.seq_label]:
            if z is not None:
                z.set_par(self)
        # --

    def build_heads(self, heads: List[int]):
        sent = self.sent
        if sent is not None:
            assert len(sent) == len(heads), "Error: length mismatch!"
        self.seq_head = SeqField(heads, par=self)
        return self.seq_head

    def build_labels(self, labels: List[str]):
        assert len(labels) == len(self.seq_head), "Error: length mismtach"  # directly match with heads, must be there
        self.seq_label = SeqField(labels, par=self)
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
            _mat = np.zeros([mlen, hlen], dtype=np.int64)  # [m,h], by default 0!
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
        hits = set()
        ret = []
        while widx >= 0:
            if widx in hits:
                zwarn(f"Find a loop in {cur_heads}")
                break
            hits.add(widx)
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

    def get_path_between_mentions(self, m0, m1, level=1, inc_common=0, return_joint_lab=True):
        _widx0, _wlen0 = m0.get_span()
        _widx1, _wlen1 = m1.get_span()
        hf = HeadFinder()
        _hidx0 = hf.find_head(m0.sent, _widx0, _wlen0)
        _hidx1 = hf.find_head(m1.sent, _widx1, _wlen1)
        best_one = self.get_path(_hidx0, _hidx1, inc_common=inc_common)
        if return_joint_lab:
            _labs = self.get_labels(level=level)
            ret = [f"^{_labs[z]}" for z in best_one[0]] + [f"{_labs[z]}" for z in best_one[1]]
            return ret
        else:
            return best_one
        # --

    def get_labels(self, level: int = None):
        ret = list(self.seq_label.vals)
        if level is not None:
            ret = [':'.join(z.split(':')[:level]) for z in ret]
        return ret

# =====
# phrase tree (constituency tree)

class _TreeNode:
    def __init__(self):
        self.tag: str = None
        self.word: str = None
        self.chs: List[_TreeNode] = []
        self.widx: int = None
        self.wlen: int = None

    def reindex(self, offset=0):
        self.widx = offset
        if self.is_leaf():
            self.wlen = 1
        else:
            ii = offset
            for ch in self.chs:
                ch.reindex(offset)
                ii += ch.wlen
            self.wlen = ii - offset
        # --

    def add_ch(self, ch: '_TreeNode'):
        self.chs.append(ch)

    def is_leaf(self):  # note: no chs is a leaf! (disallow empty phrases!)
        return len(self.chs) == 0

    # change them all!
    def to_string(self):
        if self.is_leaf():
            fs = [self.tag, self.word]
        else:
            str_chs = " ".join([ch.to_string() for ch in self.chs])
            fs = [self.tag, str_chs]
        return f'({" ".join([z for z in fs if z is not None])})'
        # --

class _TreeReader:
    # tokenize: input stream of chars
    def _tokenize(self, char_stream):
        cur = []
        for c in char_stream:
            is_space = str.isspace(c)
            is_bracket = (c in "()")
            if is_space or is_bracket:  # end of a token
                if len(cur) > 0:
                    yield ''.join(cur)
                    cur.clear()
            if is_space:
                continue
            elif is_bracket:
                yield c
            else:
                cur.append(c)
            # --
        if len(cur) > 0:
            yield ''.join(cur)
        # --

    # parse: input stream of tokens
    def _parse(self, tok_stream):
        stack = []
        for tok in tok_stream:
            if tok == '(':  # open one
                node = _TreeNode()
                stack.append(node)
            elif tok == ')':
                ch = stack.pop()
                if len(stack) == 0:
                    ch.reindex()
                    yield ch  # already top level
                else:  # add ch
                    stack[-1].add_ch(ch)
            else:
                node = stack[-1]
                if node.tag is None:
                    node.tag = tok
                else:  # leaf
                    assert len(node.chs) == 0
                    node.word = tok
        assert len(stack) == 0
        # --

    def yield_parses(self, char_stream=None, tok_stream=None):
        if tok_stream is None:
            tok_stream = self._tokenize(char_stream)
        parse_stream = self._parse(tok_stream)
        yield from parse_stream

@DataInst.rd('pt')
class PhraseTree(DataInst):
    def __init__(self, parse_str: str = None, par: 'Sent' = None):
        super().__init__(par=par)
        # --
        self._root: _TreeNode = None
        if parse_str is not None:
            self.build_parse(parse_str)
            if par is not None:
                assert len(par) == self.root.wlen
        # --

    @property
    def root(self): return self._root

    def build_parse(self, parse_str: str):
        self._root = list(_TreeReader().yield_parses(char_stream=parse_str))[0]

    def to_dict(self, store_type=True):
        ret = super().to_dict(store_type)
        if self._root is not None:
            ret.update({'_root': self._root.to_string()})  # only store id!!
        return ret

    def from_dict(self, data: Dict):
        super().from_dict(data)
        _rr = data.get('_root')
        if _rr is not None:
            self.build_parse(data['_root'])
        # --

# =====
# head word/span finder (based on dependency tree)

class HeadFinder:
    DEFAULT_SCORES = {
        'noun': {"NOUN": 0, "PROPN": -1, "NUM": -2, "VERB": -3, "PRON": -4, "ADJ": -7, "ADV": -10, "SYM": -11, "X": -12},
        'verb': {"VERB": 0, "NOUN": -1, "PROPN": -2, "NUM": -3, "PRON": -4, "ADJ": -7, "ADV": -10, "SYM": -11, "X": -12},
        'auto': {"VERB": 0, "NOUN": 0},
    }

    def __init__(self, upos_score_map='auto', prefer_right=True):
        self.upos_score_map = self._get_upos_score_map(upos_score_map)
        self.prefer_right = prefer_right
        # --

    def _get_upos_score_map(self, upos_score_map: Union[Dict[str, float], str]):
        if isinstance(upos_score_map, str):
            return HeadFinder.DEFAULT_SCORES[upos_score_map]
        else:
            assert isinstance(upos_score_map, dict)
            return upos_score_map

    def find_head(self, sent, widx: int, wlen: int):
        dt: DepTree = sent.tree_dep
        dt_depths = dt.depths
        idx_start, idx_end = widx, widx + wlen
        # pass 1: first pass by min depth
        min_depth = min(dt_depths[z] for z in range(idx_start, idx_end))
        cand_pass1 = [z for z in range(idx_start, idx_end) if dt_depths[z]<=min_depth]
        assert len(cand_pass1) > 0
        if len(cand_pass1) == 1:
            return cand_pass1[0]
        # pass 2: second pass by POS
        upos_score_map = self.upos_score_map
        uposes = sent.seq_upos.vals
        upos_scores = [upos_score_map.get(uposes[z], -100) for z in cand_pass1]
        max_upos_score = max(upos_scores)
        cand_pass2 = [v for i,v in enumerate(cand_pass1) if upos_scores[i]>=max_upos_score]
        assert len(cand_pass2) > 0
        if len(cand_pass2) == 1:
            return cand_pass2[0]
        # pass 3: select an end
        if self.prefer_right:
            return cand_pass2[-1]
        else:
            return cand_pass2[0]

    # shortcut
    def set_head_for_mention(self, mention, force_refind=False):
        sent = mention.sent
        widx, wlen = mention.get_span()
        # --
        if mention.wlen == 1:
            return  # no need
        if not force_refind and mention.shead_widx is not None:
            return  # already existing!
        # --
        shead_widx = self.find_head(sent, widx, wlen)
        mention.set_span(shead_widx, 1, shead=True)
        # --

# --
# b mspx/data/inst/tree:?
