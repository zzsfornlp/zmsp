#

# The MST algorithms with cython
cimport cython
import numpy as np
cimport numpy as np
#
from libc.math cimport exp
from libc.math cimport log

#

cdef float NEG_INF = -12345678.         # same as msp.utils.Constants.REAL_PRAC_MIN
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t
ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

# todo(+N): there are many repeated codes!

# ===== MST graph decoding algorithms

# unporj mst algorithm

# par: real-idx -> parent, cur_length, cur_nodes: list of real-idx; return ctuple(bool, set)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _find_cycle(int[:] par, int cur_length, int[:] cur_nodes):
    cdef int[:] added = np.zeros(cur_length, np.int32)
    added[0] = 1
    cdef set cycle = set()
    cdef bint findcycle = False
    # loop for all the current nodes
    # ===== (inner cdefs)
    cdef int cur_idx = -1
    cdef set tmp_cycle = set()
    cdef int cur = -1
    cdef int n_start = -1
    # =====
    for i in range(len(cur_nodes)):
        cur_idx = cur_nodes[i]
        if added[cur_idx]:
            continue
        # current cycle
        tmp_cycle.clear()
        tmp_cycle.add(cur_idx)
        added[cur_idx] = 1
        findcycle = True
        cur = cur_idx
        while par[cur] not in tmp_cycle:
            cur = par[cur]
            if added[cur]:
                findcycle = False
                break
            added[cur] = 1
            tmp_cycle.add(cur)
        # get the cycle
        if findcycle:
            n_start = cur
            cycle.add(cur)
            cur = par[cur]
            while cur != n_start:
                cycle.add(cur)
                cur = par[cur]
            break
    #cdef (bint, set) ret = (findcycle, cycle)
    return findcycle, cycle

# the chuliuedmonds algorithm
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _chuLiuEdmonds(float[:, :] cur_score_matrix, int cur_length, int[:] cur_nodes, int[:] cur_final_edges):
    # create greedy best graph (only considering current nodes)
    cdef int[:] par = np.zeros(cur_length, dtype=np.int32)
    par[0] = -1
    cdef int cur_num_nodes = len(cur_nodes)
    # ===== (inner cdefs)
    cdef int cur_i = -1
    cdef float max_score = NEG_INF
    cdef int max_par = -1
    cdef int cur_j = -1
    cdef float new_score = NEG_INF
    # =====
    for i in range(cur_num_nodes):
        cur_i = cur_nodes[i]
        if cur_i == 0:
            continue
        # the max head
        max_score = NEG_INF
        max_par = -1
        for j in range(cur_num_nodes):
            cur_j = cur_nodes[j]
            if cur_i == cur_j:
                continue
            new_score = cur_score_matrix[cur_i, cur_j]
            if new_score > max_score:
                max_score = new_score
                max_par = cur_j
        par[cur_i] = max_par
    # find a cycle
    cy_ret = _find_cycle(par, cur_length, cur_nodes)
    cdef bint findcycle = cy_ret[0]
    cdef set cycle = cy_ret[1]
    # no cycles, get all edges and return them (will be modified at higher levels of recursive calling)
    # ===== (inner cdefs)
    cur_i = -1
    # =====
    if not findcycle:
        cur_final_edges[0] = -1
        for i in range(cur_num_nodes):
            cur_i = cur_nodes[i]
            cur_final_edges[cur_i] = par[cur_i]
        return
    # deal with the cycle
    cdef int cyc_len = len(cycle)
    cdef float cyc_weight = 0.0
    cdef int[:] cyc_nodes = np.zeros(cyc_len, dtype=np.int32)
    cdef int cyc_idx = 0
    # ===== (inner cdefs)
    cdef int cyc_node_i = -1
    # =====
    for cyc_node in cycle:
        cyc_node_i = cyc_node
        cyc_nodes[cyc_idx] = cyc_node_i
        cyc_weight += cur_score_matrix[cyc_node_i, par[cyc_node_i]]
        cyc_idx += 1
    # one representative node (store the max links here)
    cdef int rep = cyc_nodes[0]
    cdef int[:] max_out_link = np.zeros(cur_length, dtype=np.int32)
    cdef int[:] max_in_link = np.zeros(cur_length, dtype=np.int32)
    cdef list new_cur_nodes = []
    # ===== (inner cdefs)
    cur_i = -1
    cur_j = -1
    cdef float max1 = NEG_INF
    cdef int wh1 = -1
    cdef float max2 = NEG_INF
    cdef int wh2 = -1
    cdef float score_out = NEG_INF
    cdef float score_in = NEG_INF
    # =====
    for i in range(cur_num_nodes):
        cur_i = cur_nodes[i]
        if cur_i in cycle:
            continue
        # will appear in the new nodes
        new_cur_nodes.append(cur_i)
        # max output-link and input-link
        max1 = NEG_INF
        wh1 = -1
        max2 = NEG_INF
        wh2 = -1
        for j in range(cyc_len):
            cur_j = cyc_nodes[j]
            score_out = cur_score_matrix[cur_i, cur_j]
            if score_out > max1:
                max1 = score_out
                wh1 = cur_j
            score_in = cyc_weight + cur_score_matrix[cur_j, cur_i] - cur_score_matrix[cur_j, par[cur_j]]
            if score_in > max2:
                max2 = score_in
                wh2 = cur_j
        # change the score and the links
        cur_score_matrix[cur_i, rep] = max1
        max_out_link[cur_i] = wh1
        cur_score_matrix[rep, cur_i] = max2
        max_in_link[cur_i] = wh2
    # recursively calling
    new_cur_nodes.append(rep)
    cdef int[:] new_cur_nodes_arr = np.asarray(new_cur_nodes, dtype=ITYPE)
    _chuLiuEdmonds(cur_score_matrix, cur_length, new_cur_nodes_arr, cur_final_edges)
    # add the edges surviving in the cycle and fix the final_edges
    # ===== (inner cdefs)
    cur_i = -1
    cyc_node_i = -1
    cdef int cur_head = -1
    cdef int break_point = -1
    # =====
    cdef int new_cur_num_nodes = len(new_cur_nodes)
    for i in range(new_cur_num_nodes):
        cur_i = new_cur_nodes[i]
        if cur_i == 0:
            continue
        cur_head = cur_final_edges[cur_i]
        if cur_i == rep:
            # add and break the cycle
            for cyc_node in cyc_nodes:
                cyc_node_i = cyc_node
                cur_final_edges[cyc_node_i] = par[cyc_node_i]
            break_point = max_in_link[cur_head]
            cur_final_edges[break_point] = cur_head
        elif cur_head == rep:
            # fix the link
            cur_final_edges[cur_i] = max_out_link[cur_i]

# the algorithm
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cmst_unproj(np.ndarray scores, np.ndarray[ITYPE_t, ndim=1] lengths, bint labeled=True):
    # [bs, len, len] or [bs, len, len, N]
    cdef Py_ssize_t batch_size = scores.shape[0]
    cdef Py_ssize_t max_length = scores.shape[1]
    # returned values
    cdef np.ndarray[ITYPE_t, ndim=2] ret_heads = np.zeros([batch_size, max_length], dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=2] ret_labels = None
    if labeled:
        ret_labels = np.zeros([batch_size, max_length], dtype=ITYPE)
    cdef np.ndarray[DTYPE_t, ndim = 2] ret_scores = np.zeros([batch_size, max_length], dtype=DTYPE)
    #
    # loop for each instance in the batch
    # ===== (inner cdefs)
    cdef int cur_length = -1
    cdef np.ndarray[DTYPE_t, ndim = 2] cur_scores = None
    cdef np.ndarray[ITYPE_t, ndim = 2] cur_label_argmax = None
    cdef np.ndarray[DTYPE_t, ndim = 3] cur_scores_labeled = None
    #
    cdef np.ndarray[DTYPE_t, ndim = 2] cur_score_matrix = None
    cdef np.ndarray[ITYPE_t, ndim = 1] cur_nodes = None
    cdef np.ndarray[ITYPE_t, ndim = 1] cur_final_edges = None
    cdef np.ndarray[DTYPE_t, ndim = 1] one_scores = None
    cdef np.ndarray[ITYPE_t, ndim = 1] one_labels = None
    # =====
    for cur_i in range(batch_size):
        cur_length = lengths[cur_i]
        cur_scores = None
        cur_label_argmax = None
        if labeled:
            cur_scores_labeled = scores[cur_i][:cur_length, :cur_length]
            cur_label_argmax = np.zeros([cur_length, cur_length], dtype=ITYPE)
            np.argmax(cur_scores_labeled, axis=-1, out=cur_label_argmax)
            cur_scores = cur_scores_labeled.max(axis=-1)
        else:
            cur_scores = scores[cur_i][:cur_length, :cur_length]
        # initialize score matrix to original score matrix
        cur_score_matrix = np.array(cur_scores, copy=True)
        cur_nodes = np.arange(cur_length, dtype=ITYPE)
        for s in range(cur_length):
            cur_score_matrix[s, s] = NEG_INF
        cur_final_edges = np.zeros(cur_length, dtype=ITYPE)
        # call the recursive f
        _chuLiuEdmonds(cur_score_matrix, cur_length, cur_nodes, cur_final_edges)
        # get the results
        one_scores = np.zeros(max_length, dtype=DTYPE) + NEG_INF
        one_labels = None
        if labeled:
            one_labels = np.zeros(max_length, dtype=ITYPE)
        for ch, pr in enumerate(cur_final_edges):
            if labeled and ch!=0:
                one_labels[ch] = cur_label_argmax[ch, pr]
            one_scores[ch] = cur_scores[ch, pr]
        cur_final_edges[0] = 0
        ret_heads[cur_i][:cur_length] = cur_final_edges
        ret_scores[cur_i] = one_scores
        if labeled:
            ret_labels[cur_i] = one_labels
    return ret_heads, ret_labels, ret_scores

# =====
# other algorithm stubs, these are not supported here!

def cmst_greedy(scores, lengths, labeled=True):
    raise NotImplementedError("No supported cython version for this one!")

def cmarginal_unproj(scores, lengths, labeled=True):
    raise NotImplementedError("No supported cython version for this one!")

def cmarginal_greedy(scores, lengths, labeled=True):
    raise NotImplementedError("No supported cython version for this one!")

# =====
# projective mst & marginal algorithms (O(n^3) for first-order)

# todo(WARN): remember that the score-matrix is [m,h]

# -----
# inside-max algorithm for decoding
# todo(warn): can be errored if inputs have Nan which makes u==-2.
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _inside_max(float[:, :] cur_score_matrix, int cur_length, int[:] cur_final_edges):
    cdef float[:, :] chart_com = np.full([cur_length, cur_length], NEG_INF, dtype=DTYPE)
    cdef float[:, :] chart_incom = np.full([cur_length, cur_length], NEG_INF, dtype=DTYPE)
    cdef int[:, :] ptr_com = np.zeros([cur_length, cur_length], dtype=ITYPE)
    cdef int[:, :] ptr_incom = np.zeros([cur_length, cur_length], dtype=ITYPE)
    # ===== (inner cdefs)
    cdef int k=-1
    cdef int s=-1
    cdef int t=-1
    cdef int r=-1
    cdef float max_score = NEG_INF
    cdef u = -2
    cdef float tmp = 0.
    # =====
    # init
    for s in range(cur_length):
        chart_com[s,s] = 0.
        chart_incom[s,s] = 0.
    # loop
    for k in range(1, cur_length):
        # the distance k
        for s in range(cur_length):
            # span [s, t]
            t = s + k
            if t >= cur_length:
                break
            # maximum loop
            # 1. incomplete ones
            max_score = NEG_INF
            u = -2
            # I[s->t]/I[t<-s]: C[s->r], C[r+1<-t]
            for r in range(s, t):
                tmp = chart_com[s,r] + chart_com[t,r+1]
                if tmp > max_score:
                    max_score = tmp
                    u = r
            # 1.1 right -> left
            chart_incom[t,s] = max_score + cur_score_matrix[s,t]
            ptr_incom[t,s] = u
            # 1.2 left -> right
            chart_incom[s,t] = max_score + cur_score_matrix[t,s]
            ptr_incom[s,t] = u
            # 2. complete ones
            # 2.1 right -> left
            max_score = NEG_INF
            u = -2
            # C[s<-t]: C[s<-r], I[r<-t]
            for r in range(s,t):
                tmp = chart_incom[t,r] + chart_com[r,s]
                if tmp > max_score:
                    max_score = tmp
                    u = r
            chart_com[t,s] = max_score
            ptr_com[t,s] = u
            # 2.2 left -> right
            max_score = NEG_INF
            u = -2
            # C[s->t]: I[s->r], C[r->t]
            for r in range(s+1, t+1):
                tmp = chart_incom[s,r] + chart_com[r,t]
                if tmp > max_score:
                    max_score = tmp
                    u = r
            chart_com[s,t] = max_score
            ptr_com[s,t] = u
    # backtracking
    _inside_max_fill(cur_length, cur_final_edges, ptr_com, ptr_incom, 0, cur_length-1, True)

# fill results by backtracking
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _inside_max_fill(int cur_length, int[:] cur_final_edges, int[:, :] ptr_com, int[:, :] ptr_incom, int h, int m, bint comp):
    if h == m:
        return
    cdef int index = -1
    if not comp:
        index = ptr_incom[h,m]
        if h > m:
            _inside_max_fill(cur_length, cur_final_edges, ptr_com, ptr_incom, h, index+1, True)
            _inside_max_fill(cur_length, cur_final_edges, ptr_com, ptr_incom, m, index, True)
        else:
            _inside_max_fill(cur_length, cur_final_edges, ptr_com, ptr_incom, h, index, True)
            _inside_max_fill(cur_length, cur_final_edges, ptr_com, ptr_incom, m, index+1, True)
    else:
        index = ptr_com[h,m]
        assert cur_final_edges[index]<0, "Internal Error, assigning conflicts!"
        cur_final_edges[index] = h
        _inside_max_fill(cur_length, cur_final_edges, ptr_com, ptr_incom, h, index, False)
        _inside_max_fill(cur_length, cur_final_edges, ptr_com, ptr_incom, index, m, True)

# -----

# first-order mst-proj Eisner's algorithm
# -- preparation is similar(same) to the unproj one
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cmst_proj(np.ndarray scores, np.ndarray[ITYPE_t, ndim=1] lengths, bint labeled=True):
    # [bs, len, len] or [bs, len, len, N]
    cdef Py_ssize_t batch_size = scores.shape[0]
    cdef Py_ssize_t max_length = scores.shape[1]
    # returned values
    cdef np.ndarray[ITYPE_t, ndim=2] ret_heads = np.zeros([batch_size, max_length], dtype=ITYPE)
    cdef np.ndarray[ITYPE_t, ndim=2] ret_labels = None
    if labeled:
        ret_labels = np.zeros([batch_size, max_length], dtype=ITYPE)
    cdef np.ndarray[DTYPE_t, ndim = 2] ret_scores = np.zeros([batch_size, max_length], dtype=DTYPE)
    #
    # loop for each instance in the batch
    # ===== (inner cdefs)
    cdef int cur_length = -1
    cdef np.ndarray[DTYPE_t, ndim = 2] cur_scores = None
    cdef np.ndarray[ITYPE_t, ndim = 2] cur_label_argmax = None
    cdef np.ndarray[DTYPE_t, ndim = 3] cur_scores_labeled = None
    #
    cdef np.ndarray[ITYPE_t, ndim = 1] cur_final_edges = None
    cdef np.ndarray[DTYPE_t, ndim = 1] one_scores = None
    cdef np.ndarray[ITYPE_t, ndim = 1] one_labels = None
    # =====
    for cur_i in range(batch_size):
        cur_length = lengths[cur_i]
        cur_scores = None
        cur_label_argmax = None
        if labeled:
            cur_scores_labeled = scores[cur_i][:cur_length, :cur_length]
            cur_label_argmax = np.zeros([cur_length, cur_length], dtype=ITYPE)
            np.argmax(cur_scores_labeled, axis=-1, out=cur_label_argmax)
            cur_scores = cur_scores_labeled.max(axis=-1)
        else:
            cur_scores = scores[cur_i][:cur_length, :cur_length]
        # call the inside-max algorithm
        cur_final_edges = np.full(cur_length, -1, dtype=ITYPE)
        _inside_max(cur_scores, cur_length, cur_final_edges)
        # get the results
        one_scores = np.zeros(max_length, dtype=DTYPE) + NEG_INF
        one_labels = None
        if labeled:
            one_labels = np.zeros(max_length, dtype=ITYPE)
        for ch, pr in enumerate(cur_final_edges):
            if labeled and ch!=0:
                one_labels[ch] = cur_label_argmax[ch, pr]
            one_scores[ch] = cur_scores[ch, pr]
        cur_final_edges[0] = 0
        ret_heads[cur_i][:cur_length] = cur_final_edges
        ret_scores[cur_i] = one_scores
        if labeled:
            ret_labels[cur_i] = one_labels
    #
    return ret_heads, ret_labels, ret_scores

# =====
# projective inside-outside

#
cdef _logsumexp(float x, float y):
    if x > y:
        return x + log(1. + exp(y-x))
    else:
        return y + log(1. + exp(x-y))

# todo(warn): repeated codes: logsumexp version of _inside_max
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _inside_sum(float[:, :] cur_score_matrix, int cur_length, float[:, :] cur_beta_com, float[:, :] cur_beta_incom):
    # ===== (inner cdefs)
    cdef int k=-1
    cdef int s=-1
    cdef int t=-1
    cdef int r=-1
    cdef float sum_score=NEG_INF
    # =====
    # init
    for s in range(cur_length):
        cur_beta_com[s,s] = 0.
    # loop
    for k in range(1, cur_length):
        # the distance k
        for s in range(cur_length):
            # span [s, t]
            t = s + k
            if t >= cur_length:
                break
            # log-sum loop
            # 1. incomplete ones
            sum_score = NEG_INF
            # I[s->t]/I[s<-t]: C[s->r], C[r+1<-t]
            for r in range(s, t):
                sum_score = _logsumexp(sum_score, cur_beta_com[s,r]+ cur_beta_com[t,r+1])
            # 1.1 right -> left
            cur_beta_incom[t,s] = sum_score + cur_score_matrix[s,t]
            # 1.2 left -> right
            cur_beta_incom[s,t] = sum_score + cur_score_matrix[t,s]
            # 2. complete ones
            # 2.1 right -> left
            # C[s<-t]: C[s<-r], I[r<-t]
            sum_score = NEG_INF
            for r in range(s,t):
                sum_score = _logsumexp(sum_score, cur_beta_incom[t,r] + cur_beta_com[r,s])
            cur_beta_com[t,s] = sum_score
            # 2.2 left -> right
            # C[s->t]: I[s->r], C[r->t]
            sum_score = NEG_INF
            for r in range(s+1, t+1):
                sum_score = _logsumexp(sum_score, cur_beta_incom[s,r] + cur_beta_com[r,t])
            cur_beta_com[s,t] = sum_score
    # from 0 to the rest
    return cur_beta_com[0, cur_length-1]

# outside algorithm
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cdef _outside_sum(float[:, :] cur_score_matrix, int cur_length, float[:, :] cur_beta_com, float[:, :] cur_beta_incom, float[:, :] cur_alpha_com, float[:, :] cur_alpha_incom):
    # ===== (inner cdefs)
    cdef int k=-1
    cdef int s=-1
    cdef int t=-1
    cdef int x=-1
    cdef float sum_score_st=NEG_INF
    cdef float sum_score_ts=NEG_INF
    # =====
    # init
    cur_alpha_com[0, cur_length-1] = 0.
    cur_alpha_com[cur_length-1, 0] = NEG_INF
    # loop
    for k in range(cur_length-1, 0, -1):
        # the distance k
        for s in range(cur_length):
            # span [s, t]
            t = s + k
            if t >= cur_length:
                break
            # log-sum loop (reversed)
            # 1. complete spans
            sum_score_st = NEG_INF
            sum_score_ts = NEG_INF
            # mixed
            for x in range(0, s):
                # I[x->s] + C[s->t] -> C[x->t]
                sum_score_st = _logsumexp(sum_score_st, cur_beta_incom[x,s]+cur_alpha_com[x,t])
                # C[x->s-1] + C[s<-t] + Edge -> I[x->t]/I[x<-t]
                sum_score_ts = _logsumexp(sum_score_ts, cur_beta_com[x,s-1]+cur_alpha_incom[x,t]+cur_score_matrix[t,x])
                sum_score_ts = _logsumexp(sum_score_ts, cur_beta_com[x,s-1]+cur_alpha_incom[t,x]+cur_score_matrix[x,t])
            for x in range(t+1, cur_length):
                # C[s<-t] + I[t<-x] -> C[s<-x]
                sum_score_ts = _logsumexp(sum_score_ts, cur_beta_incom[x,t]+cur_alpha_com[x,s])
                # C[s->t] + C[t+1<-x] + Edge -> I[x->s]/I[s<-x]
                sum_score_st = _logsumexp(sum_score_st, cur_beta_com[x,t+1]+cur_alpha_incom[x,s]+cur_score_matrix[s,x])
                sum_score_st = _logsumexp(sum_score_st, cur_beta_com[x,t+1]+cur_alpha_incom[s,x]+cur_score_matrix[x,s])
            # 1.1 left -> right
            cur_alpha_com[s,t] = _logsumexp(cur_alpha_com[s,t], sum_score_st)
            # 1.2 right -> left
            cur_alpha_com[t,s] = _logsumexp(cur_alpha_com[t,s], sum_score_ts)
            #
            # 2. incomplete spans
            # 2.1 left -> right
            sum_score_st = NEG_INF
            for x in range(t, cur_length):
                sum_score_st = _logsumexp(sum_score_st, cur_beta_com[t,x]+cur_alpha_com[s,x])
            cur_alpha_incom[s,t] = _logsumexp(cur_alpha_incom[s,t], sum_score_st)
            # 2.2 right -> left
            sum_score_ts = NEG_INF
            for x in range(0, s+1):
                sum_score_ts = _logsumexp(sum_score_ts, cur_beta_com[s,x]+cur_alpha_com[t,x])
            cur_alpha_incom[t,s] = _logsumexp(cur_alpha_incom[t,s], sum_score_ts)
    # end

# todo(warn): for this one, only accept unlabeled inputs, since no need for labeled ones!
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cmarginal_proj(np.ndarray[DTYPE_t, ndim=3] scores, np.ndarray[ITYPE_t, ndim=1] lengths, bint labeled=False):
    assert not labeled, "No need to put labeled ones here, please logsum to unlabeled."
    #
    # [bs, len, len]
    cdef Py_ssize_t batch_size = scores.shape[0]
    cdef Py_ssize_t max_length = scores.shape[1]
    # returned values of marginals
    cdef float z = 0.
    cdef np.ndarray[DTYPE_t, ndim=3] marginals = np.full([batch_size, max_length, max_length], NEG_INF, dtype=DTYPE)
    #
    # loop for each instance in the batch
    # ===== (inner cdefs)
    cdef int cur_length = -1
    cdef np.ndarray[DTYPE_t, ndim = 2] cur_scores = None
    cdef np.ndarray[DTYPE_t, ndim = 2] cur_alpha_com = None
    cdef np.ndarray[DTYPE_t, ndim = 2] cur_alpha_incom = None
    cdef np.ndarray[DTYPE_t, ndim = 2] cur_beta_com = None
    cdef np.ndarray[DTYPE_t, ndim = 2] cur_beta_incom = None
    cdef int s = -1
    cdef int t = -1
    cdef float[:,:] cur_marginals = None
    # =====
    for cur_i in range(batch_size):
        cur_length = lengths[cur_i]
        cur_scores = scores[cur_i][:cur_length, :cur_length]
        # init as -Inf = log(0.)
        cur_alpha_com = np.full([cur_length, cur_length], NEG_INF, dtype=DTYPE)
        cur_alpha_incom = np.full([cur_length, cur_length], NEG_INF, dtype=DTYPE)
        cur_beta_com = np.full([cur_length, cur_length], NEG_INF, dtype=DTYPE)
        cur_beta_incom = np.full([cur_length, cur_length], NEG_INF, dtype=DTYPE)
        # inside outside
        z = _inside_sum(cur_scores, cur_length, cur_beta_com, cur_beta_incom)
        _outside_sum(cur_scores, cur_length, cur_beta_com, cur_beta_incom, cur_alpha_com, cur_alpha_incom)
        # get marginals
        cur_marginals = marginals[cur_i]
        for s in range(cur_length):
            for t in range(cur_length):
                if s == t:
                    continue
                cur_marginals[s,t] = cur_alpha_incom[t,s] + cur_beta_incom[t,s] - z
                cur_marginals[t,s] = cur_alpha_incom[s,t] + cur_beta_incom[s,t] - z
        # debug: print 2D matrix
        # for zz in [cur_scores, cur_beta_com, cur_beta_incom, cur_alpha_com, cur_alpha_incom]:
        #     print("-----")
        #     for z1 in range(cur_length):
        #         for z2 in range(cur_length):
        #             print(zz[z1,z2])
        #         print("\n")
    #
    #print(marginals)
    np.seterr(under='ignore')
    marginals = np.exp(marginals)
    return marginals
