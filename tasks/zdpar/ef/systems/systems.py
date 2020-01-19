#

# specific ef systems, currently only contain simple "reduce-free" ones
# todo(note): therefore, currently in all systems, wrong attachments will not include more "fated future cost"

from array import array
import numpy as np

from msp.utils import zlog
from msp.nn import BK
from .base_system import EfState, EfAction, Graph, ParseInstance

# -----
# helper to ensure no cycle

# used mainly for excluding cycles
# todo(note): consistent status: 1) cache_uppermost works for all, 2) cache_descendants works for all unattached ones

# -----
# todo(warn): the mask based one is deprecated!
# class NoCycleCache00:
#     def __init__(self, num_tok=None):
#         if num_tok:
#             self.cache_uppermost = BK.arange_idx(num_tok, device=BK.CPU_DEVICE)
#             # descendants (including self); actually mask of [h,m], todo(note): only valid for unattached ones
#             self.cache_descendants = BK.diagflat(BK.constants((num_tok,), 1, dtype=BK.uint8, device=BK.CPU_DEVICE))
#         else:
#             self.cache_uppermost = self.cache_descendants = None
#
#     def clone(self):
#         x = NoCycleCache00()
#         x.cache_uppermost = self.cache_uppermost.clone()
#         x.cache_descendants = self.cache_descendants.clone()
#         return x
#
#     def update(self, mod, head):
#         cache_uppermost, cache_descendants_masks = self.cache_uppermost, self.cache_descendants
#         cur_upm_node = cache_uppermost[head]
#         cur_descendants_mask = cache_descendants_masks[mod]
#         # update
#         cache_uppermost[cur_descendants_mask] = cur_upm_node
#         # todo(note): only care about the uppermost one since others have been masked or no need (by single-rule)
#         cache_descendants_masks[cur_upm_node, cur_descendants_mask] = 1

# the list based one
class NoCycleCache:
    def __init__(self, num_tok=None):
        if num_tok:
            self.cache_uppermost = np.arange(num_tok, dtype=np.int32)
            # descendants (including self); actually mask of [h,m], todo(note): only valid for unattached ones
            self.cache_descendants = [[i] for i in range(num_tok)]
        else:
            self.cache_uppermost = self.cache_descendants = None

    def clone(self):
        x = NoCycleCache()
        x.cache_uppermost = self.cache_uppermost.copy()
        x.cache_descendants = self.cache_descendants.copy()
        return x

    def update(self, mod, head):
        cache_uppermost, cache_descendants = self.cache_uppermost, self.cache_descendants
        cur_upm_node = cache_uppermost[head]
        cur_descendants = cache_descendants[mod]
        # update
        cache_uppermost[cur_descendants] = cur_upm_node
        # todo(note): only care about the uppermost one since others have been masked or no need (by single-rule)
        # copy on write!!
        cache_descendants[cur_upm_node] = cache_descendants[cur_upm_node] + cur_descendants

# =====
# specific system: free-style

class EfFreeState(EfState):
    def __init__(self, prev: 'EfFreeState'=None, action: EfAction=None, score=0., sg: Graph=None,
                 inst: ParseInstance=None, max_slen=-1, orig_bidx=-1):
        super().__init__(prev, action, score, sg, inst, max_slen, orig_bidx)
        if prev is None:
            self.nc_cache = NoCycleCache(self.max_slen)
        else:
            self.nc_cache = prev.nc_cache.clone()
            self.nc_cache.update(action.mod, action.head)

    def build_next(self, action: EfAction, score: float):
        return EfFreeState(self, action, score)

    # incrementally minus
    def update_cands_mask(self, prev_mask):
        # todo(note): for this mode, never add new cands but gradually eliminate illegal ones
        if self.prev is not None:
            # single head
            recent_mod, recent_head = self.action.mod, self.action.head
            prev_mask[recent_mod] = 0.
            # no cycle
            cache_uppermost, cache_descendants = self.nc_cache.cache_uppermost, self.nc_cache.cache_descendants
            cur_upm_node = cache_uppermost[recent_head]
            cur_descendants = cache_descendants[recent_mod]
            prev_mask[cur_upm_node, cur_descendants] = 0.  # these are enough
        return prev_mask

# =====
# specific system: top-down

class EfTdState(EfState):
    def __init__(self, prev: 'EfTdState'=None, action: EfAction=None, score=0., sg: Graph=None,
                 inst: ParseInstance=None, max_slen=-1, orig_bidx=-1):
        super().__init__(prev, action, score, sg, inst, max_slen, orig_bidx)
        #
        if prev is None:
            self.attached_cache = []  # nodes that are already attached
        else:
            self.attached_cache = prev.attached_cache + [action.mod]

    def build_next(self, action: EfAction, score: float):
        return EfTdState(self, action, score)

    # incrementally adding
    def update_cands_mask(self, prev_mask):
        if self.prev is None:
            prev_mask.zero_()  # clear all
            prev_mask[:, 0] = 1.  # only ROOT to others at the very first
        else:
            # allow one more head but disallow the new as mod
            recent_mod = self.action.mod
            prev_mask[:, recent_mod] = 1.
            prev_mask[self.attached_cache, recent_mod] = 0.
            prev_mask[recent_mod] = 0.
        # todo(note): no need to exclude cycles since for top-down, cycles go to ROOT and are already excluded
        return prev_mask

# =====
# specific system: directional: left-right/right-left

class EfDirState(EfState):
    def __init__(self, is_l2r, prev: 'EfDirState'=None, action: EfAction=None, score=0., sg: Graph=None,
                 inst: ParseInstance=None, max_slen=-1, orig_bidx=-1):
        super().__init__(prev, action, score, sg, inst, max_slen, orig_bidx)
        #
        self.is_l2r = is_l2r
        if is_l2r:
            self.dir_next_idx = self.length + 1
            self.dir_step = 1
        else:
            self.dir_next_idx = self.num_tok - 1 - self.length
            self.dir_step = -1
        #
        if prev is None:
            self.nc_cache = NoCycleCache(self.max_slen)
        else:
            self.nc_cache = prev.nc_cache.clone()
            self.nc_cache.update(action.mod, action.head)

    def build_next(self, action: EfAction, score: float):
        return EfDirState(self.is_l2r, self, action, score)

    # build new ones everytime
    def update_cands_mask(self, prev_mask):
        if self.prev is None:
            prev_mask.zero_()
        cur_next_idx = self.dir_next_idx
        prev_mask[cur_next_idx] = 1.
        # clear last step
        if cur_next_idx-self.dir_step < len(prev_mask):
            prev_mask[cur_next_idx-self.dir_step] = 0.  # Once-A-Bug: check for r2l mode
        # eliminate cycle
        cur_descendants = self.nc_cache.cache_descendants[cur_next_idx]
        if len(cur_descendants) > 0:
            prev_mask[cur_next_idx, cur_descendants] = 0.
        return prev_mask

    # todo(note): in this system, no paths can be merged since path determines structures
    def set_sig(self, labeled: bool):
        return None

# =====
# specific system: bottom-up(distance) -> actually near-to-far(near-first)
# -- distance according to the unattached nodes list
# todo(+N): currently does not fused with reduction, that is, not strictly bottom-up; still the same oracle?
#   but the problem is that when there are no oracles for some non-projective ones

class EfNfState(EfState):
    def __init__(self, dist, prev: 'EfNfState'=None, action: EfAction=None, score=0., sg: Graph=None,
                 inst: ParseInstance=None, max_slen=-1, orig_bidx=-1):
        super().__init__(prev, action, score, sg, inst, max_slen, orig_bidx)
        #
        self.dist = dist  # dist allowed for attaching
        # non-cycle cache & left/right un-attach neighbours
        max_len = self.num_tok
        if prev is None:
            self.nc_cache = NoCycleCache(self.max_slen)
            #
            self.left_link = list(range(-1, max_len-1))
            self.right_link = list(range(1, max_len+1))
            self.right_link[-1] = -1  # -1 means NULL
        else:
            self.nc_cache = prev.nc_cache.clone()
            self.nc_cache.update(action.mod, action.head)
            #
            self.left_link = prev.left_link.copy()
            self.right_link = prev.right_link.copy()
            mod = self.action.mod
            left, right = self.left_link[mod], self.right_link[mod]
            self.right_link[left] = right  # 0 cannot be mod
            if right>0:  # only if right is valid
                self.left_link[right] = left

    def build_next(self, action: EfAction, score: float):
        return EfNfState(self.dist, self, action, score)

    # incrementally adding: update for newly introduced "near nodes"
    def update_cands_mask(self, prev_mask):
        D = self.dist
        max_len = self.num_tok
        if self.prev is None:
            prev_mask.zero_()
            for i in range(1, max_len):
                prev_mask[i, max(0,i-D):min(max_len,i+D+1)] = 1.  # [i-D, i+D]
        else:
            recent_mod, recent_head = self.action.mod, self.action.head
            cache_uppermost, cache_descendants = self.nc_cache.cache_uppermost, self.nc_cache.cache_descendants
            # add new ones
            # -- first collect the unattached neighbours in range
            left_neighbours, right_neighbours = [recent_mod], [recent_mod]
            tmp_left_idx = tmp_right_idx = recent_mod
            for i in range(D):
                if tmp_left_idx >= 0:
                    tmp_left_idx = self.left_link[tmp_left_idx]
                left_neighbours.append(tmp_left_idx)
                if tmp_right_idx >= 0:
                    tmp_right_idx = self.right_link[tmp_right_idx]
                right_neighbours.append(tmp_right_idx)
            # -- make both left2right
            left_neighbours.reverse()
            # -- traverse the neighbours and add the new range (but remember to remove cycles)
            for i in range(D):
                # [..., left0, left1, ...] <=> [..., right1, right0, ...]
                left0, left1 = left_neighbours[i], left_neighbours[i+1]
                right1, right0 = right_neighbours[i], right_neighbours[i+1]
                # expand to new ones
                # add left span to right (discard -1 since 0 is always the sentinel)
                if left0>=0 and right0>=0:
                    prev_mask[right0, left0:left1] = 1.
                    prev_mask[right0, cache_descendants[right0]] = 0.
                # add right span to the left (remember to consider the rest of the sentence)
                if right1>=0 and left0>0:  # 0 cannot have parent
                    right0 = max_len if right0<0 else right0
                    prev_mask[left0, right1:right0+1] = 1.
                    prev_mask[left0, cache_descendants[left0]] = 0.
            # similar to the Free-mode: single-head and non-cycle constraint for general purpose
            prev_mask[recent_mod] = 0.
            cur_upm_node = cache_uppermost[recent_head]
            cur_descendants = cache_descendants[recent_mod]
            prev_mask[cur_upm_node, cur_descendants] = 0.  # these are enough
        return prev_mask

# ======
# system factory

class StateBuilder:
    def __init__(self, mode, nf_dist):
        self.mode = mode
        self.nf_dist = nf_dist
        self.build = {
            "free": EfFreeState,
            "t2d": EfTdState,
            "l2r": lambda **kwargs: EfDirState(True, **kwargs),
            "r2l": lambda **kwargs: EfDirState(False, **kwargs),
            "n2f": lambda **kwargs: EfNfState(nf_dist, **kwargs)
        }[mode]

    def __repr__(self):
        mode, nf_dist = self.mode, self.nf_dist
        return "<StateBuilder: system={mode}" + (f" dist={nf_dist}>" if mode=="n2f" else ">")
