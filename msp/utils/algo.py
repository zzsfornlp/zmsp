# Algorithms Helper

import heapq

class AlgoHelper(object):
    # ===== Merge on sorted list with heap =====
    # helper: create local bindings for i0
    @staticmethod
    def _idxed_vals(vals, real_key, i0):
        for i1, v in enumerate(vals):
            yield (real_key(v), i0, i1)

    # based on heapq.merge
    # input: list of list of sorted values, output: values or idxes (i0, i1)
    @staticmethod
    def merge_sorted(values, k=None, idxed=False, key=None, reverse=False):
        if k is None:
            k = sum(len(v) for v in values)
        if idxed:
            if key is None:
                real_key = lambda x: x
            else:
                real_key = key
            new_values = []
            for i0, one_vals in enumerate(values):
                new_values.append(AlgoHelper._idxed_vals(one_vals, real_key, i0))
            values = new_values
            key = None
        rets = []
        for one in heapq.merge(*values, key=key, reverse=reverse):
            if len(rets) >= k:
                break
            rets.append(one)
        if idxed:
            rets = [(r[1], r[2]) for r in rets]
        return rets

    # like C++ STL functions, return index

    # the idx of first >v
    @staticmethod
    def upper_bound(seq, v):
        cur_left, count = 0, len(seq)
        while count > 0:
            step = count // 2
            it = cur_left + step
            if seq[it] <= v:
                cur_left = it+1
                count -= (step+1)
            else:
                count = step
        return cur_left

    # the idx of first >=v
    @staticmethod
    def lower_bound(seq, v):
        cur_left, count = 0, len(seq)
        while count > 0:
            step = count // 2
            it = cur_left + step
            if seq[it] < v:
                cur_left = it+1
                count -= (step+1)
            else:
                count = step
        return cur_left

    # return bool
    @staticmethod
    def binary_search(seq, v):
        first = AlgoHelper.lower_bound(seq, v)
        return first<len(seq) and (not v<seq[first])
