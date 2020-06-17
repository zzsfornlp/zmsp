#

# various decoding methods
# todo(note): conventions
#  the input scores utilize the lower-corner with pre-processing
#  the original scores are also provided
#  root scores are extra ones for the root
# for return: heads are offset-ed, but root is not

import numpy as np
import string

from msp.utils import Conf, zlog
from ..algo.nmst import mst_unproj, mst_proj

# =====
# part 1: main decoder

# =====
# chain baselines
def chain_l2r(scores, orig_scores, root_scores, is_proj: bool):
    slen = len(scores)
    return list(range(slen)), 0

def chain_r2l(scores, orig_scores, root_scores, is_proj: bool):
    slen = len(scores)
    return [x+2 for x in range(slen-1)]+[0], slen-1

def decode_rand(scores, orig_scores, root_scores, is_proj: bool):
    slen = len(scores)
    input_scores = np.random.rand(1, slen+1, slen+1).astype(np.float32)
    input_length = np.array([slen+1]).astype(np.int32)
    if is_proj:
        heads, _, _ = mst_proj(input_scores, input_length, False)
    else:
        heads, _, _ = mst_unproj(input_scores, input_length, False)
    ret_heads = heads[0][1:]
    ret_roots = [i for i,h in enumerate(ret_heads) if h==0]
    return ret_heads, ret_roots[0]  # simple the first root

# =====
# method 1: undirectional + selecting root

# undirected mst algorithm: Kruskal
def mst_kruskal(scores):
    cur_len = len(scores)  # scores should be [cur_len, cur_len], and only use the i0<i1 ones
    sorted_indexes = np.argsort(scores.flatten())
    node_sets = list(range(cur_len))
    edges = []
    for one_idx in reversed(sorted_indexes):
        i0, i1 = one_idx // cur_len, one_idx % cur_len
        if i0 < i1:
            s0, s1 = i0, i1
            while node_sets[s0] != s0:
                s0 = node_sets[s0]
            while node_sets[s1] != s1:
                s1 = node_sets[s1]
            if s0 != s1:
                edges.append((i0, i1))
                node_sets[s1] = s0
    assert len(edges) == cur_len-1
    # =====
    links = [set() for z in range(cur_len)]
    for i0, i1 in edges:
        links[i0].add(i1)
        links[i1].add(i0)
    return edges, links

# from undirected to directed
def _collect_heads(links, root):
    def _dfs_visit(a, heads):
        for b in links[a]:
            if heads[b]<0:  # not visited
                heads[b] = a+1  # heads always use the +1 offset
                _dfs_visit(b, heads)
    # using original index
    cur_heads = [-1] * len(links)
    cur_heads[root] = 0
    _dfs_visit(root, cur_heads)
    return cur_heads

# [slen, slen], [slen, slen], [slen]
def decode_m1(scores, orig_scores, root_scores, is_proj: bool):
    assert not is_proj
    # decode
    slen = len(scores)
    mst_edges, mst_links = mst_kruskal(scores)  # [slen]
    # decide on root (use directional info on original scores)
    if root_scores is None:
        root_scores = np.zeros(slen)
    #
    best_root = None
    best_heads = None
    best_score = -1.
    for cur_root in range(slen):
        cur_heads = _collect_heads(mst_links, cur_root)
        # m's change without h
        cur_score = root_scores[cur_root] + sum(orig_scores[h-1,m] for m,h in enumerate(cur_heads) if m!=cur_root)
        if cur_score > best_score:
            best_score = cur_score
            best_root = cur_root
            best_heads = cur_heads
    return best_heads, best_root

# =====
# method 2: directly using directional MST (can be non-proj or proj)

# [slen, slen], [slen, slen], [slen]
def decode_m2(scores, orig_scores, root_scores, is_proj: bool):
    # make scores into [m,h] and augment root lines
    slen = len(scores)
    pad_scores0 = np.concatenate([root_scores[np.newaxis, :], scores.T], 0)  # [slen+1, slen]
    pad_scores1 = np.concatenate([np.zeros([slen+1, 1]), pad_scores0], 1)  # [slen+1, slen+1]
    input_scores = pad_scores1[np.newaxis, :, :].astype(np.float32)
    input_length = np.array([slen+1]).astype(np.int32)
    if is_proj:
        heads, _, _ = mst_proj(input_scores, input_length, False)
    else:
        heads, _, _ = mst_unproj(input_scores, input_length, False)
    ret_heads = heads[0][1:]
    ret_roots = [i for i,h in enumerate(ret_heads) if h==0]
    return ret_heads, ret_roots[0]  # simple the first root

# =====
# method 3: easy first styled clustering

# [slen, slen], [slen, slen], [slen]
def decode_m3(scores, orig_scores, root_scores, is_proj: bool):
    assert is_proj

# =====
# method extra: sink function words

# =====
# =====

def get_decoder(method):
    return {
        "l2r": chain_l2r, "r2l": chain_r2l, "rand": decode_rand,
        "m1": decode_m1, "m2": decode_m2, "m3": decode_m3,
    }.get(method, None)

# =====
# part 2: fun decoder

# attach all fun to the right content word (unless boundary)
def fdec_right(fun_masks, **kwargs):
    slen = len(fun_masks)
    ret = [0] * slen
    prev_content_idx = -1
    prev_fun_list = []
    for widx, is_fun in enumerate(fun_masks):
        if is_fun:
            prev_fun_list.append(widx)
        else:
            for w in prev_fun_list:
                ret[w] = widx+1
            prev_fun_list.clear()
            prev_content_idx = widx
    # attach the rest
    for w in prev_fun_list:
        ret[w] = prev_content_idx+1
    return ret

# attach all fun to the left content word (unless boundary)
def fdec_left(fun_masks, **kwargs):
    slen = len(fun_masks)
    ret = [0] * slen
    # first deal with the first group
    widx = 0
    while widx<slen and fun_masks[widx]:
        widx += 1
    if widx>=slen:
        return ret  # all fun words
    #
    prev_content_idx = widx
    for w in range(widx):
        ret[w] = prev_content_idx+1
    #
    while widx<slen:
        if fun_masks[widx]:
            ret[widx] = prev_content_idx+1
        else:
            prev_content_idx = widx
        widx += 1
    return ret

# fun_scores: [h,m]
def fdec_max(fun_masks, fun_scores, **kwargs):
    fun_max_preds = np.argmax(fun_scores, 0) + 1
    return fun_masks.astype(np.int64) * fun_max_preds

# scores are all [h,m]
def fdec_cluster(fun_masks, fun_scores, orig_scores, **kwargs):
    pass

def get_fdecoder(method):
    return {
        "right": fdec_right, "left": fdec_left,
        "max": fdec_max, "cluster": fdec_cluster,
    }[method]

# =====
# reducer

class IRConf(Conf):
    def __init__(self):
        self.iterative = False
        self.weight_row = 1.
        self.weight_col = -1.
        self.weight_extra = 0.

class IterReducer:
    def __init__(self, conf: IRConf):
        self.conf = conf
        self.iterative = conf.iterative
        self.weight_row = conf.weight_row
        self.weight_col = conf.weight_col
        self.weight_extra = conf.weight_extra

    # ranking keys
    def get_rank_keys(self, scores, extra_scores):
        return scores.sum(1) * self.weight_row + scores.sum(0) * self.weight_col + extra_scores * self.weight_extra

    # iterative reduce: input N*N original scores
    def reduce(self, orig_scores, extra_scores=None):
        slen = len(orig_scores)
        # # what if right chain
        # right_chain = list(range(slen))
        # return right_chain, right_chain
        # #
        if extra_scores is None:
            extra_scores = np.zeros(slen)
        if self.iterative:
            reduce_idxes = []
            remaining_idxes = list(range(slen))
            while len(remaining_idxes)>0:
                cur_scores = orig_scores[remaining_idxes][:,remaining_idxes]
                cur_reduce_tidx = np.argmin(self.get_rank_keys(cur_scores, extra_scores[remaining_idxes]))
                cur_reduce_idx = remaining_idxes[cur_reduce_tidx]
                reduce_idxes.append(cur_reduce_idx)
                remaining_idxes.remove(cur_reduce_idx)  # must be there
            assert len(reduce_idxes) == slen
        else:
            # todo(note): directly rank all
            reduce_idxes = np.argsort(self.get_rank_keys(orig_scores, extra_scores))
        reduce_order = [None] * slen
        for one_order, one_idx in enumerate(reduce_idxes):
            reduce_order[one_idx] = one_order
        return reduce_order, reduce_idxes

# =====
# CKYParser

# nodes for the phrase tree: [a,b]
class TreeNode:
    def __init__(self, node_left: 'TreeNode'=None, node_right: 'TreeNode'=None, sent=None, leaf_idx: int=None):
        if leaf_idx is not None:
            assert sent is not None
            assert node_left is None and node_right is None
            self.is_leaf = True
            self.sent = sent
            self.range_a, self.range_b = leaf_idx, leaf_idx
            self.head_idx = leaf_idx  # only one node
        else:
            assert sent is None and leaf_idx is None
            self.is_leaf = False
            assert node_left.sent is node_right.sent
            self.sent = node_right.sent
            self.node_left, self.node_right = node_left, node_right
            assert node_left.range_b+1 == node_right.range_a
            self.range_a, self.range_b = node_left.range_a, node_right.range_b
            self.head_idx = None

    def set_head(self, head_idx):
        assert self.range_a<=head_idx and head_idx<=self.range_b
        self.head_idx = head_idx

    # add_head: plus head printings; fake_nt: fake X for non-terminal
    def get_repr(self, add_head, fake_nt):
        if self.is_leaf:
            cur_word = self.sent[self.range_a]
            cur_word = {"(": "-LRB-", ")": "-RRB-"}.get(cur_word, cur_word)  # escape
            if fake_nt:
                return f"(X {cur_word})"
            else:
                return cur_word
        else:
            ra, rb = self.node_left.get_repr(add_head, fake_nt), self.node_right.get_repr(add_head, fake_nt)
            fake_nt_str = "X " if fake_nt else ""
            if self.head_idx is None or not add_head:
                return f"({fake_nt_str}{ra} {rb})"
            else:
                head_str = self.sent[self.head_idx]
                return f"({fake_nt_str}[{head_str}] {ra} {rb})"

    def __repr__(self):
        return self.get_repr(True, False)

#
class CKYConf(Conf):
    def __init__(self):
        # phrase split score: lambda: a[sizeA, sizeB], b[sizeB, sizeA] -> scalar
        self.ps_method = "add_avg"
        self.ps_pp = False  # special two-layer processing with PUNCT as splitter
        # head finding method (similar to the Reducer)
        self.hf_use_left = False  # always make left as head
        self.hf_use_right = False  # always make right as head
        self.hf_use_fun_masks = False  # use extra help from masks (fun<non-fun)
        self.hf_use_pos_rule = False  # use specific rules on pos
        self.hf_use_ir = True  # use the same mechanism as in IterReducer, use hf_win_* for that purpose
        self.hf_use_vocab = False
        self.hf_vr_thresh = 100  # <=100 as reduce words
        # comparing self vs. other, using whole-phrase or head?
        self.hf_self_headed = True
        self.hf_other_headed = True
        # inside and outside weights
        self.hf_win_row = 1.
        self.hf_win_col = 0.
        self.hf_wout_row = 1.
        self.hf_wout_col = 0.
        self.hf_w_extra = 0.

#
class CKYParser:
    # extending some starting from string.punctuation
    # PUNCT_SET = set(r"""[]!"#$%&'()*+,./:;<=>?@[\\^_`{|}~„“”«»²–、。・（）「」『』：；！？〜，《》·]+""")
    # selecting the interesting ones
    PUNCT_SET = set(r""",.:;!?、，。：；！？""")

    def __init__(self, conf: CKYConf, r_vocab):
        self.conf = conf
        self.r_vocab = r_vocab
        #
        self.score_f = {
            "add_avg": lambda a,b: (a+b.T).mean(),
            "max_avg": lambda a,b: np.maximum(a, b.T).mean(),
        }[conf.ps_method]

    # input [N,N] original scores, and [N] extra-root-scores
    def parse(self, sent, uposes, fun_masks, orig_scores, extra_scores=None):
        slen = len(sent)
        # first get all leaf nodes
        all_leaf_nodes = [TreeNode(sent=sent, leaf_idx=z) for z in range(slen)]
        # cky
        if slen>1 and self.conf.ps_pp:
            # split by PUNCT
            is_punct = [all(z in CKYParser.PUNCT_SET for z in w) for w in sent]  # whole string is punct
            split_ranges = []
            prev_start = None
            for i, is_p in enumerate(is_punct):
                if is_p:
                    if prev_start is not None:
                        split_ranges.append((prev_start, i))
                        prev_start = None
                    split_ranges.append((i, i+1))
                else:
                    if prev_start is None:
                        prev_start = i
            if prev_start is not None:
                split_ranges.append((prev_start, slen))  # add final range
            # decode layer 1
            split_trees = [self._cky(all_leaf_nodes[a:b], orig_scores[a:b,a:b]) for a,b in split_ranges]
            # decode layer 2
            # todo(+N): current simply average here!
            l2_len = len(split_trees)
            l2_scores = np.zeros([l2_len, l2_len])
            for i in range(l2_len):
                for j in range(i+1, l2_len):
                    ia, ib = split_ranges[i]
                    ja, jb = split_ranges[j]
                    l2_scores[i,j] = orig_scores[ia:ib, ja:jb].mean()
                    l2_scores[j,i] = orig_scores[ja:jb, ia:ib].mean()
            best_tree = self._cky(split_trees, l2_scores)  # layer-2 decode
        else:
            best_tree = self._cky(all_leaf_nodes, orig_scores)  # directly decode
        # =====
        # then decide the head
        ret_heads, ret_root = self._decide_head(best_tree, orig_scores, extra_scores, uposes, fun_masks)
        #
        # zlog(best_tree, func="debug")
        return best_tree, (ret_heads, ret_root)

    # decode on the current list of leaf nodes (can be subtrees)
    def _cky(self, cur_leaf_nodes, cur_scores):
        slen = len(cur_leaf_nodes)
        assert slen>0
        if slen==1:
            return cur_leaf_nodes[0]  # directly return for singleton node
        # =====
        cky_best_scores = np.zeros([slen, slen])  # [a,b]
        cky_best_sp = np.zeros([slen, slen], dtype=np.int)
        for step in range(2, slen+1):  # do not need to calculate leaf nodes
            for a in range(slen+1-step):
                b = a + step - 1  # [a,b]
                # loop over split points
                cur_best_score = -1000.
                cur_best_sp = None
                for sp in range(a,b):  # [a,b] -> [a,sp], [sp+1,b]
                    cur_edge_score = self.score_f(cur_scores[a:sp+1, sp+1:b+1], cur_scores[sp+1:b+1, a:sp+1])
                    cur_score = cky_best_scores[a,sp] + cky_best_scores[sp+1,b] + cur_edge_score
                    if cur_score > cur_best_score:
                        cur_best_score = cur_score
                        cur_best_sp = sp
                assert cur_best_sp is not None
                cky_best_scores[a,b] = cur_best_score
                cky_best_sp[a,b] = cur_best_sp
        # traceback
        best_tree = self._traceback(cky_best_sp, cur_leaf_nodes, 0, slen-1)
        return best_tree

    # return TreeNode
    def _traceback(self, cky_best_sp, leaf_nodes, a, b):
        assert b>=a
        if a==b:
            return leaf_nodes[a]
        else:
            sp = cky_best_sp[a,b]
            node_left = self._traceback(cky_best_sp, leaf_nodes, a, sp)
            node_right = self._traceback(cky_best_sp, leaf_nodes, sp+1, b)
            return TreeNode(node_left=node_left, node_right=node_right)

    #
    def _pos_head_score(self, hidx_self, hidx_other, uposes):
        # todo(note): CONJ is from UDv1
        POS_RANKS = [["VERB"], ["NOUN", "PROPN"], ["ADJ"], ["ADV"],
                     ["ADP", "AUX", "CCONJ", "DET", "NUM", "PART", "PRON", "SCONJ", "CONJ"], ["INTJ", "SYM", "X"], ["PUNCT"]]
        POS2RANK = {}
        for i, ps in enumerate(POS_RANKS):
            for p in ps:
                POS2RANK[p] = i
        upos_self, upos_other = uposes[hidx_self], uposes[hidx_other]
        rank_self, rank_other = POS2RANK[upos_self], POS2RANK[upos_other]
        if rank_self < rank_other:
            return 1
        elif rank_self > rank_other:
            return -1
        else:
            # TODO(+N): maybe specific to en
            # if neighbouring, select the right one, else left
            if abs(hidx_self-hidx_other) == 0:
                return int(hidx_self>hidx_other)
            else:
                return int(hidx_self<hidx_other)

    # decide the heads (head finding)
    def _decide_head(self, top_tree: TreeNode, orig_scores, extra_scores, uposes, fun_masks):
        slen = len(orig_scores)
        if extra_scores is None:
            extra_scores = np.zeros(slen)
        ret_heads = [None] * slen
        ret_root = None
        # =====
        # get score
        def _score(node_self: TreeNode, node_other: TreeNode, node_parent: TreeNode):
            conf = self.conf
            # =====
            # determinist ones
            if conf.hf_use_left:
                return 100. if (node_self.head_idx<node_other.head_idx) else -100.
            elif conf.hf_use_right:
                return 100. if (node_self.head_idx>node_other.head_idx) else -100.
            elif conf.hf_use_pos_rule:
                return self._pos_head_score(node_self.head_idx, node_other.head_idx, uposes)
            else:
                pass
            # =====
            # filter ones
            if conf.hf_use_fun_masks:
                if fun_masks[node_self.head_idx] and not fun_masks[node_other.head_idx]:
                    return -100.
                elif not fun_masks[node_self.head_idx] and fun_masks[node_other.head_idx]:
                    return 100.
            if conf.hf_use_vocab:
                word_self, word_other = node_self.sent[node_self.head_idx], node_other.sent[node_other.head_idx]
                vthr = conf.hf_vr_thresh
                rank_self, rank_other = self.r_vocab.get(word_self.lower(), vthr+1), self.r_vocab.get(word_other.lower(), vthr+1)
                if rank_self<=vthr and rank_other>vthr:
                    return -100.
                elif rank_self>vthr and rank_other<=vthr:
                    return 100.
            # =====
            # fallback: score based ones
            if conf.hf_use_ir:
                self_head_idx = node_self.head_idx
                return orig_scores[self_head_idx].sum() * conf.hf_win_row \
                       + orig_scores[:,self_head_idx].sum() * conf.hf_win_col \
                       + extra_scores[self_head_idx] * conf.hf_w_extra
            else:
                x0_self, x1_self = (node_self.head_idx, node_self.head_idx+1) if conf.hf_self_headed \
                    else (node_self.range_a, node_self.range_b+1)
                x0_other, x1_other = (node_other.head_idx, node_other.head_idx+1) if conf.hf_other_headed \
                    else (node_other.range_a, node_other.range_b+1)
                x0_par, x1_par = node_parent.range_a, node_parent.range_b+1
                # inner score
                s_inner = (orig_scores[x0_self:x1_self, x0_other:x1_other]*conf.hf_win_row
                           + orig_scores[x0_other:x1_other, x0_self:x1_self].T*conf.hf_win_col).mean()
                # outer score
                s_outer_row = np.concatenate([orig_scores[x0_self:x1_self, :x0_par],
                                              orig_scores[x0_self:x1_self, x1_par:]], -1) * conf.hf_wout_row
                s_outer_col = np.concatenate([orig_scores[:x0_par, x0_self:x1_self],
                                              orig_scores[x1_par:, x0_self:x1_self]], 0).T * conf.hf_wout_col
                s_outer_sum = s_outer_row + s_outer_col
                s_outer = s_outer_sum.mean() if s_outer_sum.size>0 else 0.
                # extra score
                s_extra = extra_scores[x0_self:x1_self].mean() * conf.hf_w_extra
                return s_inner + s_outer + s_extra
        # =====
        # recursively decide
        def _rdec(cur_tree: TreeNode):
            if not cur_tree.is_leaf:
                _rdec(cur_tree.node_left)
                _rdec(cur_tree.node_right)
                score_left = _score(cur_tree.node_left, cur_tree.node_right, cur_tree)
                score_right = _score(cur_tree.node_right, cur_tree.node_left, cur_tree)
                head_left, head_right = cur_tree.node_left.head_idx, cur_tree.node_right.head_idx
                if score_left < score_right:
                    cur_tree.set_head(head_right)
                    ret_heads[head_left] = head_right + 1  # +1 offset
                else:
                    cur_tree.set_head(head_left)
                    ret_heads[head_right] = head_left + 1
        # =====
        _rdec(top_tree)
        ret_root = top_tree.head_idx
        ret_heads[ret_root] = 0
        return ret_heads, ret_root

# PYTHONPATH=../src/ python3 -m pdb ../src/tasks/cmd.py zdpar.stat2.decode3 input_file:_en_dev.ppos.pic already_pre_computed:1 mdec_method:cky hf_use_fun_masks:1 hf_use_ir:1 ps_pp:1
# b tasks/zdpar/stat2/helper_decode:388
