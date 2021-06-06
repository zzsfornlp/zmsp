#

# longest common subsequence

from typing import List
import numpy as np

class LCS:
    # performing aligning by matching scores
    # -- input is 2d matrix of matching scores [len_a1, len_a2], should be >=0, 0 means absolute no match
    # -- the order is important for breaking ties for add_a1,match,add_a2: by default prefer add-a1 and put add-a2 later
    @staticmethod
    def align_matches(match_score_arr: np.ndarray, order_scores=None):
        DEFAULT_CODE = (0, 1, 2)  # a1/match/a2
        if order_scores is None:
            order_scores = DEFAULT_CODE  # default score is add_a1=0, match=1, add_a2=2
        assert np.all(match_score_arr >= 0.), "Assume all scores >= 0.!"
        len1, len2 = match_score_arr.shape
        # recordings
        record_best_scores = np.zeros((1 + len1, 1 + len2), dtype=np.float32)  # best matching scores
        # pointers for back-tracing & also prefer order: by default add a1(x+=1); match; add a2(y+=1)
        record_best_codes = np.zeros((1 + len1, 1 + len2), dtype=np.int32)
        record_best_codes[0, :] = 2  # add a2 at y
        record_best_codes[:, 0] = 0  # add a1 at x
        record_best_codes[0, 0] = -1  # never used
        # loop: the looping order (ij or ji) does not matter
        for i in range(len1):
            ip1 = i + 1
            for j in range(len2):
                jp1 = j + 1
                s_match = match_score_arr[i, j] + record_best_scores[i, j]  # match one
                s_a1 = record_best_scores[i, jp1]  # add a1 on x
                s_a2 = record_best_scores[ip1, j]  # add a2 on y
                ordered_selections = sorted(zip((s_a1, s_match, s_a2), order_scores, DEFAULT_CODE))
                sel_score, _, sel_code = ordered_selections[-1]  # max score
                record_best_scores[ip1, jp1] = sel_score
                record_best_codes[ip1, jp1] = sel_code
        # backtracking for whole seq and aligning point
        # results of idx matches
        merge_to_a1, merge_to_a2 = [], []  # merge_idx -> a?_idx or None
        a1_to_merge, a2_to_merge = [], []  # a?_idx -> merge_idx
        back_i, back_j = len1, len2
        cur_midx = -1
        while back_i + back_j > 0:  # there are still remainings
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
        a1_to_merge = [merge_len + z for z in reversed(a1_to_merge)]
        a2_to_merge = [merge_len + z for z in reversed(a2_to_merge)]
        # idx_m -> idx_a1, ..., idx_a1 -> idx_merge, ...
        return merge_to_a1, merge_to_a2, a1_to_merge, a2_to_merge

    # =====
    # usually we want to match s2 to s1, thus hint is the start of s2 to s1
    @staticmethod
    def align_seqs(s1, s2, hint_s2_on_s1_offset: float=None, hint_scale=0.1, match_f=(lambda x,y: float(x==y))):
        # first compare each pair to get match scores
        len1, len2 = len(s1), len(s2)
        match_score_arr = np.asarray([match_f(x,y) for x in s1 for y in s2]).reshape((len1, len2))
        if hint_s2_on_s1_offset is not None:
            assert hint_s2_on_s1_offset>=0 and hint_s2_on_s1_offset<len(s1), "Outside range of s1"
            posi_diff = np.arange(len1)[:, np.newaxis] - (np.arange(len2)+hint_s2_on_s1_offset)[np.newaxis, :]
            hint_rewards = hint_scale * np.exp(-np.abs(posi_diff))
            match_score_arr += (match_score_arr>0.).astype(np.float) * hint_rewards  # only add if >0
        # then get results
        return LCS.align_matches(match_score_arr)

    # =====
    # special routine to merge a series of sequences (incrementally, left to right)
    # input: seq_list is List[List[item]], K is merging range in each step
    @staticmethod
    def merge_seqs(seq_list: List[List], K: int):
        cur_seq = []
        for s in seq_list:
            if len(cur_seq) < K:
                cur0, cur1 = [], cur_seq
            else:
                cur0, cur1 = cur_seq[:-K], cur_seq[-K:]  # use the most recent K for merging
            align_res = LCS.align_seqs(cur1, s)
            # get the merged one
            ma1, ma2 = align_res[:2]
            cur2 = [(s[b] if a is None else cur1[a]) for a,b in zip(ma1, ma2)]
            # finally concat
            cur_seq = cur0 + cur2
        return cur_seq
