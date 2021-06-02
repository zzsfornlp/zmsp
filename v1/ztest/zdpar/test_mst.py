#

from msp.utils import Timer
from tasks.zdpar.algo.mst import mst_unproj
from tasks.zdpar.algo.cmst import cmst_unproj, cmst_proj, cmarginal_proj
import numpy as np

# ===============
# directly from NeuralNLP2: https://github.com/XuezheMax/NeuroNLP2/blob/master/neuronlp2/tasks/parser.py
def decode_MST(energies, lengths, leading_symbolic=0, labeled=True):
    """
    decode best parsing tree with MST algorithm.
    :param energies: energies: numpy 4D tensor
        energies of each edge. the shape is [batch_size, num_labels, n_steps, n_steps],
        where the summy root is at index 0.
    :param masks: numpy 2D tensor
        masks in the shape [batch_size, n_steps].
    :param leading_symbolic: int
        number of symbolic dependency types leading in type alphabets)
    :return:
    """

    def find_cycle(par):
        added = np.zeros([length], np.bool)
        added[0] = True
        cycle = set()
        findcycle = False
        for i in range(1, length):
            if findcycle:
                break

            if added[i] or not curr_nodes[i]:
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

    def chuLiuEdmonds():
        par = np.zeros([length], dtype=np.int32)
        # create best graph
        par[0] = -1
        for i in range(1, length):
            # only interested at current nodes
            if curr_nodes[i]:
                max_score = score_matrix[0, i]
                par[i] = 0
                for j in range(1, length):
                    if j == i or not curr_nodes[j]:
                        continue

                    new_score = score_matrix[j, i]
                    if new_score > max_score:
                        max_score = new_score
                        par[i] = j

        # find a cycle
        findcycle, cycle = find_cycle(par)
        # no cycles, get all edges and return them.
        if not findcycle:
            final_edges[0] = -1
            for i in range(1, length):
                if not curr_nodes[i]:
                    continue

                pr = oldI[par[i], i]
                ch = oldO[par[i], i]
                final_edges[ch] = pr
            return

        cyc_len = len(cycle)
        cyc_weight = 0.0
        cyc_nodes = np.zeros([cyc_len], dtype=np.int32)
        id = 0
        for cyc_node in cycle:
            cyc_nodes[id] = cyc_node
            id += 1
            cyc_weight += score_matrix[par[cyc_node], cyc_node]

        rep = cyc_nodes[0]
        for i in range(length):
            if not curr_nodes[i] or i in cycle:
                continue

            max1 = float("-inf")
            wh1 = -1
            max2 = float("-inf")
            wh2 = -1

            for j in range(cyc_len):
                j1 = cyc_nodes[j]
                if score_matrix[j1, i] > max1:
                    max1 = score_matrix[j1, i]
                    wh1 = j1

                scr = cyc_weight + score_matrix[i, j1] - score_matrix[par[j1], j1]

                if scr > max2:
                    max2 = scr
                    wh2 = j1

            score_matrix[rep, i] = max1
            oldI[rep, i] = oldI[wh1, i]
            oldO[rep, i] = oldO[wh1, i]
            score_matrix[i, rep] = max2
            oldO[i, rep] = oldO[i, wh2]
            oldI[i, rep] = oldI[i, wh2]

        rep_cons = []
        for i in range(cyc_len):
            rep_cons.append(set())
            cyc_node = cyc_nodes[i]
            for cc in reps[cyc_node]:
                rep_cons[i].add(cc)

        for i in range(1, cyc_len):
            cyc_node = cyc_nodes[i]
            curr_nodes[cyc_node] = False
            for cc in reps[cyc_node]:
                reps[rep].add(cc)

        chuLiuEdmonds()

        # check each node in cycle, if one of its representatives is a key in the final_edges, it is the one.
        found = False
        wh = -1
        for i in range(cyc_len):
            for repc in rep_cons[i]:
                if repc in final_edges:
                    wh = cyc_nodes[i]
                    found = True
                    break
            if found:
                break

        l = par[wh]
        while l != wh:
            ch = oldO[par[l], l]
            pr = oldI[par[l], l]
            final_edges[ch] = pr
            l = par[l]

    if labeled:
        assert energies.ndim == 4, 'dimension of energies is not equal to 4'
    else:
        assert energies.ndim == 3, 'dimension of energies is not equal to 3'
    input_shape = energies.shape
    batch_size = input_shape[0]
    max_length = input_shape[2]

    pars = np.zeros([batch_size, max_length], dtype=np.int32)
    types = np.zeros([batch_size, max_length], dtype=np.int32) if labeled else None
    for i in range(batch_size):
        energy = energies[i]

        # calc the realy length of this instance
        length = lengths[i]

        # calc real energy matrix shape = [length, length, num_labels - #symbolic] (remove the label for symbolic types).
        if labeled:
            energy = energy[leading_symbolic:, :length, :length]
            # get best label for each edge.
            label_id_matrix = energy.argmax(axis=0) + leading_symbolic
            energy = energy.max(axis=0)
        else:
            energy = energy[:length, :length]
            label_id_matrix = None
        # get original score matrix
        orig_score_matrix = energy
        # initialize score matrix to original score matrix
        score_matrix = np.array(orig_score_matrix, copy=True)

        oldI = np.zeros([length, length], dtype=np.int32)
        oldO = np.zeros([length, length], dtype=np.int32)
        curr_nodes = np.zeros([length], dtype=np.bool)
        reps = []

        for s in range(length):
            orig_score_matrix[s, s] = float("-inf")
            score_matrix[s, s] = float("-inf")
            curr_nodes[s] = True
            reps.append(set())
            reps[s].add(s)
            for t in range(s + 1, length):
                oldI[s, t] = s
                oldO[s, t] = t

                oldI[t, s] = t
                oldO[t, s] = s

        final_edges = dict()
        chuLiuEdmonds()
        par = np.zeros([max_length], np.int32)
        if labeled:
            type = np.zeros([max_length], np.int32)
            type[0] = 0
        else:
            type = None

        for ch, pr in final_edges.items():
            par[ch] = pr
            if labeled and ch != 0:
                type[ch] = label_id_matrix[pr, ch]

        par[0] = 0
        pars[i] = par
        if labeled:
            types[i] = type
    return pars, types
# =======================

#
def rand_sample(size):
    x1 = np.random.random_sample(size)
    x2 = np.random.random_sample(size)
    y = (1000*x1 + 100*x2)
    return y.astype(np.float32)

#
def main():
    np.random.seed(123)
    R = 100
    BS = 32
    MAXL = 50
    LAB = 40
    all_scores = []
    all_orig_scores = []
    all_lengths = []
    all_one_labeled = []
    for _ in range(R):
        one_maxlen = np.random.randint(5, MAXL)
        one_labeled = bool(np.random.random_sample() < 0.5)
        if one_labeled:
            scores = rand_sample([BS, one_maxlen, one_maxlen, LAB])
            orig_scores = np.array(np.transpose(scores, [0, 3, 2, 1]), copy=True, dtype=np.float32)
        else:
            scores = rand_sample([BS, one_maxlen, one_maxlen])
            orig_scores = np.array(np.transpose(scores, [0, 2, 1]), copy=True, dtype=np.float32)
        lengths = np.random.randint(4, one_maxlen, BS)
        all_scores.append(scores)
        all_orig_scores.append(orig_scores)
        all_lengths.append(lengths)
        all_one_labeled.append(one_labeled)
    #
    all_orig_mst_results = []
    all_mst_results = []
    all_cmst_results = []
    all_cmst_results_proj = []
    # =====
    with Timer("", "CMARGINAL"):
        for scores, lengths, one_labeled in zip(all_scores, all_lengths, all_one_labeled):
            if one_labeled:
                # the number is not important
                unlabeled_scores = np.average(scores, axis=-1)
            else:
                unlabeled_scores = scores
            cur_marginals = cmarginal_proj(unlabeled_scores, lengths, labeled=False)
            # check marginals sum to 1. for each m
            cur_marginals_sum = np.sum(cur_marginals, axis=-1)      # [BS, m]
            for one_row, one_length in zip(cur_marginals_sum, lengths):
                # close_arr = np.isclose(one_row[1:one_length], 1.)
                close_arr = np.abs(one_row[1:one_length] - 1.) < 5e-2
                assert np.all(close_arr)
    # =====
    with Timer("", "CMST"):
        for scores, lengths, one_labeled in zip(all_scores, all_lengths, all_one_labeled):
            cmst_results = cmst_unproj(scores, lengths, labeled=one_labeled)
            all_cmst_results.append(cmst_results)
    with Timer("", "CMST_PROJ"):
        for scores, lengths, one_labeled in zip(all_scores, all_lengths, all_one_labeled):
            cmst_results_proj = cmst_proj(scores, lengths, labeled=one_labeled)
            all_cmst_results_proj.append(cmst_results_proj)
    with Timer("", "ORIG"):
        for orig_scores, lengths, one_labeled in zip(all_orig_scores, all_lengths, all_one_labeled):
            orig_mst_results = decode_MST(orig_scores, lengths, labeled=one_labeled)
            all_orig_mst_results.append(orig_mst_results)
    with Timer("", "MST"):
        for scores, lengths, one_labeled in zip(all_scores, all_lengths, all_one_labeled):
            mst_results = mst_unproj(scores, lengths, labeled=one_labeled)
            all_mst_results.append(mst_results)
    # check
    for idx in range(R):
        orig_mst_results, mst_results, cmst_results, one_labeled, one_maxlen = \
            all_orig_mst_results[idx], all_mst_results[idx], all_cmst_results[idx], all_one_labeled[idx], all_lengths[idx]
        cmst_results_proj = all_cmst_results_proj[idx]
        for i in range(2 if one_labeled else 1):
            assert np.all(orig_mst_results[i]==mst_results[i])
            assert np.all(cmst_results[i]==mst_results[i])
        # only check arcs for proj vs. unproj
        equal_rate = np.average(cmst_results_proj[0] == cmst_results[0])
        print("Equal rate at this time is %s." % equal_rate)

#
def main2():
    L = 5
    x = np.zeros([1,L,L], dtype=np.float32)
    cur_marginals = cmarginal_proj(x, np.asarray([L]), labeled=False)
    pass

#
if __name__ == '__main__':
    main2()
    main()

"""
Results of testing: about 5x speedup
## Start timer : CMARGINAL at 2.952. (Sun Jan  6 10:26:45 2019)
## End timer : CMARGINAL at 6.983, the period is 4.030 seconds. (Sun Jan  6 10:26:49 2019)
## Start timer : CMST at 6.983. (Sun Jan  6 10:26:49 2019)
## End timer : CMST at 7.639, the period is 0.656 seconds. (Sun Jan  6 10:26:50 2019)
## Start timer : CMST_PROJ at 7.639. (Sun Jan  6 10:26:50 2019)
## End timer : CMST_PROJ at 8.014, the period is 0.375 seconds. (Sun Jan  6 10:26:50 2019)
## Start timer : ORIG at 8.014. (Sun Jan  6 10:26:50 2019)
## End timer : ORIG at 11.450, the period is 3.437 seconds. (Sun Jan  6 10:26:54 2019)
## Start timer : MST at 11.450. (Sun Jan  6 10:26:54 2019)
## End timer : MST at 15.043, the period is 3.593 seconds. (Sun Jan  6 10:26:57 2019)
"""
