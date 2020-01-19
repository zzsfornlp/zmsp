#

# sth about decoding

import numpy as np

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
    return edges

# by default use 0 as the default root
def decode_sent(scores, root=0):
    edges = mst_kruskal(scores)
    cur_len = len(scores)
    links = [set() for z in range(cur_len)]
    for i0, i1 in edges:
        links[i0].add(i1)
        links[i1].add(i0)
    # starting from the root & bfs
    heads = [-1] * cur_len
    visits = [root]
    ptr = 0
    while ptr < len(visits):
        cur = visits[ptr]
        for outgoing in links[cur]:
            if heads[outgoing]<0 and outgoing!=root:
                heads[outgoing] = cur
                visits.append(outgoing)
        ptr += 1
    return heads, links
