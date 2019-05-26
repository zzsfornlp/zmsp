#

from .backends import BK
from msp.utils import zcheck, Helper

from collections import Iterable

# expressions and their identities, for manual batching
# !! only support dim[0], which is the batch size

# -- simple recordings
class ExprWrapperPool(object):
    pool = []

    @staticmethod
    def clear():
        ExprWrapperPool.pool = []

    @staticmethod
    def reg(one):
        ExprWrapperPool.pool.append(one)
        return len(ExprWrapperPool.pool)-1

    # todo(+cost): graph version to check validity
    @staticmethod
    def get(id):
        return ExprWrapperPool.pool[id]

# mostly not explicitly stored outside, but can be used temporally
class ExprWrapper(object):
    def __init__(self, val, bsize):
        # back value can be (Expr, list, dict,) or recursive ones on these three
        # -- but all end-exprs should have the same bsize & same structure
        self.bsize = bsize
        self.id = ExprWrapperPool.reg(val)

    @property
    def val(self):
        return ExprWrapperPool.get(self.id)

    # get raw value
    def get_val(self, item):
        return self.val[item]

    # return SlicedExpr for outside storing
    def split(self):
        ret = [SlicedExpr(self, idx) for idx in range(self.bsize)]
        return ret

# ===== Simply serve as a pointer
# stored outside
class SlicedExpr(object):
    def __init__(self, ew, slice_idx):
        self.ew = ew
        self.slice_idx = slice_idx

# ===== Select & Combine =====
class SliceManager(object):
    @staticmethod
    def _check_full(idxes, length):
        return len(idxes) == length and all(i==v for i,v in enumerate(idxes))

    # return: list of ew, list of list of slice-idx, tracking-idx(original)
    # todo(warn): lose original order of slices
    @staticmethod
    def collect_slices(slices):
        ews = []     # list of EW
        idxes = []     # list of list of (sidx, ori_idx)
        # scan slices and group
        tmp_id2idx = {}
        for ori_idx, s in enumerate(slices):
            one_ew, one_sidx = s.ew, s.slice_idx
            ew_id = one_ew.id
            #
            if ew_id not in tmp_id2idx:
                tmp_id2idx[ew_id] = len(ews)
                ews.append(one_ew)
                idxes.append([])
            #
            idx_in_vals = tmp_id2idx[ew_id]
            idxes[idx_in_vals].append((one_sidx, ori_idx))
        # sort and split
        slice_idxes, origin_idxes = [], []
        for one_idx, one_ew in zip(idxes, ews):
            one_idx.sort()      # low-idx to high-idx
            one_slice_idxes = [x[0] for x in one_idx]
            if SliceManager._check_full(one_slice_idxes, one_ew.bsize):
                slice_idxes.append(None)
            else:
                slice_idxes.append(one_slice_idxes)
            origin_idxes.append([x[1] for x in one_idx])
        return ews, slice_idxes, origin_idxes

    # return list of ew, list of combined indexes (retaining the original order)
    @staticmethod
    def _arrange_idxes(slices):
        values, bidxes = [], []
        # tmp
        tmp_bidx_bases = [0,]
        tmp_id2idx = {}
        for s in slices:
            one_ew, one_sidx = s.ew, s.slice_idx
            ew_id = one_ew.id
            if ew_id not in tmp_id2idx:
                tmp_id2idx[ew_id] = len(values)
                values.append(one_ew.val)
                tmp_bidx_bases.append(one_ew.bsize+tmp_bidx_bases[-1])
            #
            idx_in_vals = tmp_id2idx[ew_id]
            bidxes.append(tmp_bidx_bases[idx_in_vals]+one_sidx)
        # check for perfect match
        if SliceManager._check_full(bidxes, tmp_bidx_bases[-1]):
            bidxes = None
        return values, bidxes

    # combine all values according to the first value
    @staticmethod
    def _combine_recursive(values, bidxes):
        v0 = values[0]
        if isinstance(v0, dict):
            ret = {}
            for name in v0:
                next_values = [z[name] for z in values]
                ret[name] = SliceManager._combine_recursive(next_values, bidxes)
        elif isinstance(v0, list):
            ret = []
            for idx in range(len(v0)):
                next_values = [z[idx] for z in values]
                ret.append(SliceManager._combine_recursive(next_values, bidxes))
        else:
            zcheck(BK.is_expr(v0), "Illegal combine value type.")
            # todo(warn): first concat and then select, may use more memory
            ret = BK.concat(values, 0)
            if bidxes is not None:
                ret = BK.select(ret, bidxes, 0)
        return ret

    # combine according to keys
    # None->all, list->straight-through, set/dict->hierarchical
    @staticmethod
    def _combine_recursive_keys(values, bidxes, keys):
        if isinstance(keys, dict):
            ret = {}
            for k in keys:
                ret[k] = SliceManager._combine_recursive_keys([z[k] for z in values], bidxes, keys[k])
        elif isinstance(keys, set):
            ret = {}
            for k in keys:
                ret[k] = SliceManager._combine_recursive([z[k] for z in values], bidxes)
        else:
            # direct through
            if keys is None:
                keys = []
            elif not isinstance(keys, Iterable):
                keys = [keys]
            next_values = [Helper.apply_keys(z, keys) for z in values]
            ret = SliceManager._combine_recursive(next_values, bidxes)
        return ret

    # directly return python(val) or Expr for Scorer internal usage
    @staticmethod
    def combine_slices(slices, keys, skip_combine=False, skip_debug_check=False):
        # combine them
        if skip_combine:
            ew = slices[0].ew
            zcheck(len(slices)==ew.bsize, "At least get the same batch-size!")
            values, bidxes = [ew.val], None
            if skip_debug_check:
                _, flag = SliceManager._arrange_idxes(slices)
                zcheck(flag is None, "Failed skip-combine!")
        else:
            values, bidxes = SliceManager._arrange_idxes(slices)
        # prepare with the keys
        return SliceManager._combine_recursive_keys(values, bidxes, keys)
