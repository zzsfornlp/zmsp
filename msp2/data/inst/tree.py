#

# Trees for a sentence

__all__ = [
    "DepTree", "PhraseTree", "HeadFinder", "DepSentStruct",
]

# =====
# Trees

from typing import List, Dict, Iterable, Union, Tuple
import numpy as np
from collections import defaultdict
from msp2.utils import zwarn, Conf
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
        self._fnode = None  # FNode

    def __len__(self):
        return len(self.seq_head)

    def __repr__(self):
        return f"DepTree(len={len(self)})"

    def clear_caches(self):
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

    @property
    def fnode(self):
        if self._fnode is None:
            self._fnode = FNodeReader.get_reader().read_tree(self)
        return self._fnode

    # shortcut
    def str_fnode(self, **kwargs):
        from .helper2 import MyPrettyPrinter
        return MyPrettyPrinter.str_fnode(self.sent, self.fnode, **kwargs)

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
# todo(+N): maybe could use "FNode", but probably to unify all these in the next version ...
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

# =====
# a simple sentence manager derived from dep tree (based on UD)

# "frame" node
class FNode:
    def __init__(self, l_type: str, d_label: str, widx: int):
        self.l_type = l_type  # layer type
        # --
        self.l_label = None  # dep-label or head
        self.d_label = d_label  # dep label
        self.widx = widx  # anchored at where?
        # --
        self.par = None
        self.chs = []
        self.head_chs = []
        self.label2chs = defaultdict(list)  # str->list: (deplab or head)
        # --
        self._caches = {}
        # --

    def __repr__(self):
        return f"{self.l_type}[ch={len(self.chs)}]"

    def add_ch(self, ch: 'FNode'):
        _sort_key = (lambda x: x.widx)  # always sort them left2right
        # --
        ch.par = self
        self.chs.append(ch)
        self.chs.sort(key=_sort_key)
        if ch.widx == self.widx or self.widx < 0:  # still self as head (always add root as head!)
            self.head_chs.append(ch)
            _l_label = 'head'
        else:
            _l_label = ch.d_label
        ch.l_label = _l_label
        self.label2chs[_l_label].append(ch)
        self.label2chs[_l_label].sort(key=_sort_key)
        # --
        self._caches.clear()

    @property
    def head_span(self):
        return self.get_span(def_inc=False)

    @property
    def full_span(self):
        return self.get_span(def_inc=True)

    # def_inc indicates by default include or not?
    def get_span(self, inc_set=(), exc_set=(), def_inc=True):
        _inc_set, _exc_set = set(inc_set), set(exc_set)
        _key = "|".join([",".join(sorted(_inc_set)), ",".join(sorted(_exc_set)), 'Y' if def_inc else 'N'])
        # --
        ret = self._caches.get(_key)
        if ret is None:
            all_spans = []
            in_chs = list(self.head_chs)  # note: copy!
            if self.widx >= 0:
                all_spans.append((self.widx, 1))  # at least head word!
                for z in self.chs:
                    if z.l_label in _exc_set:
                        continue  # exclude
                    elif z.l_label in _inc_set:
                        in_chs.append(z)
                    elif def_inc:
                        in_chs.append(z)
            else:  # special for root!
                in_chs.extend(self.chs)
            all_spans.extend([z.get_span(inc_set=inc_set, exc_set=exc_set, def_inc=def_inc) for z in in_chs])
            ret = _merge_spans(all_spans)  # note: repeat ones do not matter since return one span!
            # --
            self._caches[_key] = ret
        return ret

    # visit
    def visit(self, visitor, do_pre=True, do_post=False):
        if do_pre:
            visitor(self)
        for ch in self.chs:
            ch.visit(visitor, do_pre=do_pre, do_post=do_post)
        if do_post:
            visitor(self)
        # --

    # iter all
    def iter_all(self):
        yield self
        for ch in self.chs:
            yield from ch.iter_all()

    # ancestor path
    def get_ancestor_path(self, inc_self=False, top_down=True):
        path = []
        c = self if inc_self else self.par
        while c is not None:
            path.append(c)
            c = c.par
        if top_down:
            path.reverse()
        return path

    def find_common_ancestor(self, other: 'FNode'):
        path0 = self.get_ancestor_path(inc_self=True)
        path1 = other.get_ancestor_path(inc_self=True)
        common = []
        for a,b in zip(path0, path1):
            if a is b:  # note: exactly the same one!
                common.append(a)
        return common[-1]

# helper for reading
# --
# some helpers
def _sum_set(d, names):
    if '*' in names:
        ret = set(sum([list(z) for z in d.values()], []))
    else:
        ret = set()
        for n in names:
            ret.update(d[n])
    return ret
def _merge_spans(spans):
    a = min([z[0] for z in spans])
    b = max([z[0]+z[1] for z in spans])
    return (a, b-a)
# --
class FNodeReader:
    # --
    _readers = {}
    @staticmethod
    def get_reader(*args, **kwargs):
        _key = str(args + tuple([(k,v) for k,v in kwargs.items()]))
        if _key not in FNodeReader._readers:
            FNodeReader._readers[_key] = FNodeReader(*args, **kwargs)
        return FNodeReader._readers[_key]
    # --

    def __init__(self):
        self.udep_sets = {
            'root': {'root'},
            'loose': {'list', 'parataxis'},
            'conj': {'conj'},
            'core': {'nsubj', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp'},
            'noncore': {'obl', 'vocative', 'expl', 'dislocated', 'advcl', 'advmod', 'discourse', 'aux', 'cop', 'mark'},
            'nom': {'nmod', 'appos', 'nummod', 'acl', 'amod', 'det', 'clf', 'case'},
            'mwe': {'fixed', 'flat', 'compound', 'goeswith', 'orphan', 'reparandum'},  # simply count them as mwe
            'other': {'cc', 'punct', 'dep'},
        }
        self.udep_sets['pred'] = _sum_set(self.udep_sets, ['core', 'noncore'])
        self.udep_sets['all'] = _sum_set(self.udep_sets, ['*'])  # all names!
        # --
        self.layer_rules = [  # what "layers" to put
            'root', 'loose', 'conj', 'pred', 'nom', 'mwe',
        ]
        # --

    def read_tree(self, tree: DepTree):
        _sets = self.udep_sets
        heads, labels = tree.seq_head.vals, tree.seq_label.vals
        ch_lists = tree.chs_lists
        # first check labels
        labels = [z.split(":")[0].lower() for z in labels]
        if any(z not in _sets['all'] for z in labels):
            zwarn(f"Find non-UDv2 labels: {[z for z in labels if z not in _sets['all']]}")
            labels = [(z if z in _sets['all'] else 'dep') for z in labels]
        is_other = [(z in _sets['other']) for z in labels]
        # --
        def _read_node(_cur_idx, _cur_chs=None):
            if _cur_chs is None:
                _cur_chs = list(ch_lists[_cur_idx+1])  # +1 since ch_lists add ARTI-Root
            # look for higher layers
            _hit_layer = None
            _hit_chs = []
            for _layer in self.layer_rules:
                _layer_set = _sets[_layer]
                _hit_chs = [z for z in _cur_chs if labels[z] in _layer_set]
                if len(_hit_chs) > 0:
                    _hit_layer = _layer
                    if _layer == 'pred':  # also collect punct here!
                        _hit_chs = [z for z in _cur_chs if labels[z] in _layer_set or labels[z] == 'punct']
                    break
            # process them
            _left, _right = [z for z in _hit_chs if z < _cur_idx], [z for z in _hit_chs if z > _cur_idx]
            if _hit_layer is not None:
                _bounds = [min(_left) if len(_left)>0 else -1, max(_right) if len(_right)>0 else len(labels)]
            else:
                _hit_layer = 'leaf'
                _bounds = [_cur_idx, _cur_idx]
            # greedily gather "other"
            _other = [z for z in _cur_chs if is_other[z] and (z<_bounds[0] or z>_bounds[1])]
            # recursively read nodes
            _ret_node = FNode(_hit_layer, (labels[_cur_idx] if _cur_idx>=0 else 'root'), _cur_idx)
            _strips = _left + _right + _other
            if _hit_layer != 'leaf' and _cur_idx >= 0:  # no further down for arti-root
                _remain_chs = set(_cur_chs).difference(_strips)
                _head_node = _read_node(_cur_idx, sorted(_remain_chs))
                _ret_node.add_ch(_head_node)
            for z in _strips:
                _sub_node = _read_node(z)
                _ret_node.add_ch(_sub_node)
            return _ret_node
        # --
        ret = _read_node(-1, None)
        return ret

# collections of the "fnodes"
class DepSentStruct:
    def __init__(self, tree: DepTree):
        self.tree = tree
        self.root: FNode = FNodeReader.get_reader().read_tree(tree)
        # --
        # mappings from tokens to nodes
        self.widx2nodes = [[] for _ in range(len(tree))]  # bottom2up
        # --
        def _tmp_visit(_node):
            if _node.widx >= 0:
                self.widx2nodes[_node.widx].append(_node)
        # --
        self.root.visit(_tmp_visit, do_pre=False, do_post=True)
        # --

    # find the lowest node that contains the span
    # upward_set: go up certain d_labels, exclude_set: excluding certain tokens, up_leaf: try go up leaf
    def find_node(self, span, upward_set=None, exclude_set=None, up_leaf=True):
        # find all nodes
        all_nodes = []
        for widx in range(span[0], span[0]+span[1]):
            _node = self.widx2nodes[widx][0]  # lowest one (leaf)!
            if upward_set is not None:
                while _node.par is not None and _node.d_label in upward_set:
                    _node = _node.par  # go upward!
            if exclude_set is not None and _node.d_label in exclude_set:
                continue  # exclude this node!
            all_nodes.append(_node)
        if len(all_nodes) == 0:
            return None
        else:
            # find common ancestor
            ret = all_nodes[0]
            for other in all_nodes[1:]:
                ret = ret.find_common_ancestor(other)
            if up_leaf:  # try to go up than leaf!
                while ret.l_type == 'leaf' and ret.par is not None and ret.par.widx == ret.widx:
                    ret = ret.par
            return ret
