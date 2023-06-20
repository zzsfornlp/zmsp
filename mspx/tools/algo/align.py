#

# Algorithm Helper

__all__ = [
    "AlignHelper", "AlignedResult",
]

import numpy as np

# storing results of align
class AlignedResult:
    def __init__(self, merge_to_a1, merge_to_a2, a1_to_merge, a2_to_merge):
        self.merge_to_a1, self.merge_to_a2, self.a1_to_merge, self.a2_to_merge = \
            merge_to_a1, merge_to_a2, a1_to_merge, a2_to_merge
        # TODO(!): more convenient operations on this!

class AlignHelper:
    @staticmethod
    def edit_distance(s1, s2):
        # note: some parts borrowed from somewhere
        m, n = len(s1), len(s2)
        table = [[0] * (n + 1) for _ in range(m + 1)]  # [len1, len2]
        # --
        for i in range(m + 1):
            table[i][0] = i
        for j in range(n + 1):
            table[0][j] = j
        # --
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    table[i][j] = table[i - 1][j - 1]
                else:
                    table[i][j] = 1 + min(table[i - 1][j], table[i][j - 1], table[i - 1][j - 1])
        # --
        return table[-1][-1]

    # performing aligning by matching scores
    # -- input is 2d matrix of matching scores [len_a1, len_a2], should be >=0, 0 means absolute no match
    # -- the order is important for breaking ties for add_a1,match,add_a2: by default prefer add-a2 later
    # -- if "prefer_cont", then prefer continuous operations (not too many times of changing operations)
    # -- if "forbit_match0", then forbid matching when score<=0.
    @staticmethod
    def align_matches(match_score_arr, order_scores=None, prefer_cont=False, forbit_match0=True):
        DEFAULT_CODE = (0,1,2)  # a1/match/a2
        if order_scores is None:
            order_scores = DEFAULT_CODE
        assert np.all(match_score_arr>=0.)
        len1, len2 = match_score_arr.shape
        # recordings
        record_best_scores = np.zeros((1+len1, 1+len2), dtype=np.float32)  # best matching scores
        # pointers for back-tracing & also prefer order: by default add a1(x+=1); match; add a2(y+=1)
        record_best_codes = np.zeros((1+len1, 1+len2), dtype=np.int32)
        record_best_codes[0,:] = 2  # add a2 at y
        record_best_codes[:,0] = 0  # add a1 at x
        record_best_codes[0,0] = -1  # not anyone!
        # record number of change operations
        record_best_nc = np.zeros((1+len1, 1+len2), dtype=np.int32)
        record_best_nc[0,:] = 1  # only adding a2
        record_best_nc[:,0] = 1  # only adding a1
        record_best_nc[0,0] = 0  # no ops
        # --
        # loop: the looping order (ij or ji) does not matter
        _NEG = -10000
        for i in range(len1):
            ip1 = i + 1
            for j in range(len2):
                jp1 = j + 1
                s_a1 = record_best_scores[i,jp1]  # add a1 on x
                s_match = match_score_arr[i,j] + record_best_scores[i,j]  # match one
                if forbit_match0 and match_score_arr[i,j]<=0.:
                    s_match = _NEG
                s_a2 = record_best_scores[ip1,j]  # add a2 on y
                _last_codes = (record_best_codes[i,jp1], record_best_codes[i,j], record_best_codes[ip1,j])  # last time code
                _last_cost = (record_best_nc[i,jp1], record_best_nc[i,j], record_best_nc[ip1,j])  # last time cost
                _new_cost = tuple([a+int(b!=c) for a,b,c in zip(_last_cost, _last_codes, DEFAULT_CODE)])  # plus this time
                # --
                _sorting_cost = tuple([-z for z in _new_cost]) if prefer_cont else (0,0,0)
                ordered_selections = sorted(zip((s_a1, s_match, s_a2), _sorting_cost, order_scores, list(range(3))))
                sel_score, _, _, sel_idx = ordered_selections[-1]  # max score
                # --
                record_best_scores[ip1,jp1] = sel_score
                record_best_codes[ip1,jp1] = DEFAULT_CODE[sel_idx]
                record_best_nc[ip1,jp1] = _new_cost[sel_idx]
        # backtracking for whole seq and aligning point
        # results of idx matches
        merge_to_a1, merge_to_a2 = [], []  # merge_idx -> a?_idx or None
        a1_to_merge, a2_to_merge = [], []  # a?_idx -> merge_idx
        back_i, back_j = len1, len2
        cur_midx = -1
        while back_i+back_j>0:  # there are still remainings
            code = record_best_codes[back_i, back_j]
            if code == 0:  # add a1[back_i-1]
                back_i -= 1
                merge_to_a1.append(back_i)
                merge_to_a2.append(None)
                a1_to_merge.append(cur_midx)
            elif code == 1:  # add matched a1[back_i-1],a2[back_j-1]
                back_i -= 1
                back_j -= 1
                merge_to_a1.append(back_i)
                merge_to_a2.append(back_j)
                a1_to_merge.append(cur_midx)
                a2_to_merge.append(cur_midx)
            elif code == 2:  # add a2[back_j-1]
                back_j -= 1
                merge_to_a1.append(None)
                merge_to_a2.append(back_j)
                a2_to_merge.append(cur_midx)
            else:
                raise NotImplementedError()
            cur_midx -= 1
        # reverse things
        merge_to_a1.reverse()
        merge_to_a2.reverse()
        merge_len = len(merge_to_a1)
        a1_to_merge = [merge_len+z for z in reversed(a1_to_merge)]
        a2_to_merge = [merge_len+z for z in reversed(a2_to_merge)]
        # mergeto??: len(merged), a1tomerge: len(a1)
        ret = AlignedResult(merge_to_a1, merge_to_a2, a1_to_merge, a2_to_merge)
        return ret

    # =====
    # usually we want to match s2 to s1, thus hint is the start of s2 to s1
    @staticmethod
    def align_seqs(s1, s2, hint_s2_on_s1_offset: float=None, hint_scale=0.1,
                   match_f=(lambda x,y: float(x==y)), **align_kwargs):
        # first compare each pair to get match scores
        len1, len2 = len(s1), len(s2)
        match_score_arr = np.asarray([match_f(x,y) for x in s1 for y in s2]).reshape((len1, len2))
        if hint_s2_on_s1_offset is not None:
            assert hint_s2_on_s1_offset>=0 and hint_s2_on_s1_offset<len(s1), "Outside range of s1"
            posi_diff = np.arange(len1)[:, np.newaxis] - (np.arange(len2)+hint_s2_on_s1_offset)[np.newaxis, :]
            hint_rewards = hint_scale * np.exp(-np.abs(posi_diff))
            match_score_arr += (match_score_arr>0.).astype(np.float) * hint_rewards  # only add if >0
        # then get results
        return AlignHelper.align_matches(match_score_arr,  **align_kwargs)
