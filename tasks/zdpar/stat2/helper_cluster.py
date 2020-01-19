#

from typing import List
import numpy as np

#
class OneCluster:
    def __init__(self, nodes: List[int], head: int):
        self.nodes = nodes
        self.head = head

    def merge(self, other_cluster: 'OneCluster'):
        self.nodes.extend(other_cluster.nodes)
        # still keep self.head

# greedy nearby clustering with head
# bottom up decoding
class ClusterManager:
    def build_start_clusters(self, start_groups: List[List], start_heads: List[int]):
        return [OneCluster(a,b) for a,b in zip(start_groups, start_heads)]

    # slen is overall length, start_* is the current clusters, matrices are all in [h(slen),m(slen)] convention
    def cluster(self, slen: int, start_clusters: List[OneCluster], allowed_matrix, link_scores, head_scores):
        ret = [0] * slen
        procedures = []  # List[(h, m)]
        cur_clusters = start_clusters
        while True:
            # merge candidates between (i, i+1)
            merge_candidates = []
            merge_scores = []
            for midx in range(len(cur_clusters)-1):
                c1, c2 = cur_clusters[midx], cur_clusters[midx+1]
                c1_nodes, c1_head = c1.nodes, c1.head
                c2_nodes, c2_head = c2.nodes, c2.head
                if allowed_matrix[c1_head, c2_head] or allowed_matrix[c2_head, c1_head]:
                    merge_candidates.append(midx)
                    # here using the linking scores
                    one_score = None  # TODO(!)
                    merge_scores.append(one_score)
            # break if nothing left to merge
            if len(merge_candidates) == 0:
                break
            # greedily pick the best score
            best_idx = np.argmax(merge_scores).item()
            best_c1_idx = merge_candidates[best_idx]
            best_c2_idx = best_c1_idx + 1
            # TODO(!)
            # determine the direction
            # prepare for the next iter
        return ret, procedures
