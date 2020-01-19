#

# various decoding methods
# todo(note): conventions
#  the input scores utilize the lower-corner with pre-processing
#  the original scores are also provided
#  root scores are extra ones for the root
# for return: heads are offset-ed, but root is not

import numpy as np

from msp.utils import Conf
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
    }[method]

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
# other helpers

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
