#

# mst algorithms implemented in python

import numpy as np
from mspx.utils import Constants

#
NEG_INF = Constants.REAL_PRAC_MIN

# =====
# unproj-mst algo: from NeuroNLP2 and seems originally come from mstparser
# scores: [bs, len-m, len-h, (N-label)], lengths = [bs]

# helper: find cycles, return (bool, set())
def _find_cycle(par, cur_length, cur_nodes):
    added = np.zeros(cur_length, np.bool)
    added[0] = True
    cycle = set()
    findcycle = False
    for i in range(1, cur_length):
        if findcycle:
            break
        if added[i] or not cur_nodes[i]:
            continue
        # init cycle
        tmp_cycle = set()
        tmp_cycle.add(i)
        added[i] = True
        findcycle = True
        l = i
        while par[l] not in tmp_cycle:
            l = par[l]
            if added[l]:
                findcycle = False
                break
            added[l] = True
            tmp_cycle.add(l)
        if findcycle:
            lorg = l
            cycle.add(lorg)
            l = par[lorg]
            while l != lorg:
                cycle.add(l)
                l = par[l]
            break
    return findcycle, cycle

# helper: the chuliuedmonds algorithm
def _chuLiuEdmonds(cur_score_matrix, cur_length, cur_oldI, cur_oldO, cur_nodes, cur_reps, cur_final_edges):
    # create greedy best graph (only considering current nodes)
    par = np.zeros(cur_length, dtype=np.int32)
    par[0] = -1
    for i in range(1, cur_length):
        # only interested at current nodes
        if cur_nodes[i]:
            max_score = NEG_INF
            max_par = -1
            # max_score = cur_score_matrix[i, 0]
            # max_par = 0
            for j in range(cur_length):
                if not cur_nodes[j] or j==i:
                    continue
                new_score = cur_score_matrix[i, j]
                if new_score > max_score:
                    max_score = new_score
                    max_par = j
            par[i] = max_par
    # find a cycle
    findcycle, cycle = _find_cycle(par, cur_length, cur_nodes)
    # no cycles, get all edges and return them.
    if not findcycle:
        cur_final_edges[0] = -1
        for i in range(1, cur_length):
            if not cur_nodes[i]:
                continue
            pr = cur_oldI[par[i], i]
            ch = cur_oldO[par[i], i]
            cur_final_edges[ch] = pr
        return
    # deal with the cycle
    cyc_len = len(cycle)
    cyc_weight = 0.0
    cyc_nodes = np.zeros(cyc_len, dtype=np.int32)
    for idx, cyc_node in enumerate(cycle):
        cyc_nodes[idx] = cyc_node
        cyc_weight += cur_score_matrix[cyc_node, par[cyc_node]]
    rep = cyc_nodes[0]
    for i in range(cur_length):
        if not cur_nodes[i] or i in cycle:
            continue
        max1 = NEG_INF
        wh1 = -1
        max2 = NEG_INF
        wh2 = -1
        for j in range(cyc_len):
            j1 = cyc_nodes[j]
            if cur_score_matrix[i, j1] > max1:
                max1 = cur_score_matrix[i, j1]
                wh1 = j1
            scr = cyc_weight + cur_score_matrix[j1, i] - cur_score_matrix[j1, par[j1]]
            if scr > max2:
                max2 = scr
                wh2 = j1
        cur_score_matrix[i, rep] = max1
        cur_oldI[rep, i] = cur_oldI[wh1, i]
        cur_oldO[rep, i] = cur_oldO[wh1, i]
        cur_score_matrix[rep, i] = max2
        cur_oldO[i, rep] = cur_oldO[i, wh2]
        cur_oldI[i, rep] = cur_oldI[i, wh2]
    rep_cons = []
    for i in range(cyc_len):
        rep_cons.append(set())
        cyc_node = cyc_nodes[i]
        for cc in cur_reps[cyc_node]:
            rep_cons[i].add(cc)
    for i in range(1, cyc_len):
        cyc_node = cyc_nodes[i]
        cur_nodes[cyc_node] = False
        for cc in cur_reps[cyc_node]:
            cur_reps[rep].add(cc)
    # recursively calling
    _chuLiuEdmonds(cur_score_matrix, cur_length, cur_oldI, cur_oldO, cur_nodes, cur_reps, cur_final_edges)
    # check each node in cycle, if one of its representatives is a key in the final_edges, it is the one.
    found = False
    wh = -1
    for i in range(cyc_len):
        for repc in rep_cons[i]:
            if repc in cur_final_edges:
                wh = cyc_nodes[i]
                found = True
                break
        if found:
            break
    l = par[wh]
    while l != wh:
        ch = cur_oldO[par[l], l]
        pr = cur_oldI[par[l], l]
        cur_final_edges[ch] = pr
        l = par[l]

# todo(warn): assume SYMBOLIC ROOT for both input/output
def mst_unproj(scores, lengths, labeled=True):
    # [bs, len, len] or [bs, len, len, N]
    if labeled:
        assert scores.ndim == 4, 'dimension of energies is not equal to 4'
    else:
        assert scores.ndim == 3, 'dimension of energies is not equal to 3'
    input_shape = scores.shape
    batch_size = input_shape[0]
    max_length = input_shape[1]
    # returned values
    ret_heads = np.zeros([batch_size, max_length], dtype=np.int32)
    ret_lables = np.zeros([batch_size, max_length], dtype=np.int32) if labeled else None
    ret_scores = np.zeros([batch_size, max_length], dtype=np.float32)
    #
    # loop for each instance in the batch
    for cur_i in range(batch_size):
        cur_scores = scores[cur_i]
        cur_length = lengths[cur_i]
        # chunked for real length
        cur_scores = cur_scores[:cur_length, :cur_length]
        # max over labels if labeled & chunk over length: [len-m, len-h]
        if labeled:
            cur_label_argmax = cur_scores.argmax(axis=-1)
            cur_scores = cur_scores.max(axis=-1)
        else:
            cur_label_argmax = None
        # initialize score matrix to original score matrix
        cur_score_matrix = np.array(cur_scores, copy=True)
        cur_oldI = np.zeros([cur_length, cur_length], dtype=np.int32)
        cur_oldO = np.zeros([cur_length, cur_length], dtype=np.int32)
        cur_nodes = np.zeros(cur_length, dtype=np.bool)
        cur_reps = [set() for _ in range(cur_length)]
        # initialize some values
        for s in range(cur_length):
            cur_score_matrix[s, s] = NEG_INF
            cur_nodes[s] = True
            cur_reps[s].add(s)
            for t in range(s + 1, cur_length):
                cur_oldI[s, t] = s
                cur_oldO[s, t] = t
                cur_oldI[t, s] = t
                cur_oldO[t, s] = s
        cur_final_edges = dict()
        # call the recursive f
        _chuLiuEdmonds(cur_score_matrix, cur_length, cur_oldI, cur_oldO, cur_nodes, cur_reps, cur_final_edges)
        #
        one_par = np.zeros(max_length, np.int32)
        one_scores = np.zeros(max_length, np.float32) + NEG_INF
        if labeled:
            one_type = np.zeros(max_length, np.int32)
        else:
            one_type = None
        for ch, pr in cur_final_edges.items():
            one_par[ch] = pr
            if labeled and ch != 0:
                one_type[ch] = cur_label_argmax[ch, pr]
            one_scores[ch] = cur_scores[ch, pr]
        one_par[0] = 0
        ret_heads[cur_i] = one_par
        ret_scores[cur_i] = one_scores
        if labeled:
            ret_lables[cur_i] = one_type
    return ret_heads, ret_lables, ret_scores

# =====
# todo(warn): the rest of the algorithms are not provided here (either in cmst or nn-based)
def mst_proj(scores, lengths, labeled=True):
    raise NotImplementedError("Compile/Use Cython/NN version for this one!")

def mst_greedy(scores, lengths, labeled=True):
    raise NotImplementedError("Compile/Use Cython/NN version for this one!")

#
def marginal_unproj(scores, lengths, labeled=True):
    raise NotImplementedError("Compile/Use Cython/NN version for this one!")

def marginal_proj(scores, lengths, labeled=True):
    raise NotImplementedError("Compile/Use Cython/NN version for this one!")

def marginal_greedy(scores, lengths, labeled=True):
    raise NotImplementedError("Compile/Use Cython/NN version for this one!")
