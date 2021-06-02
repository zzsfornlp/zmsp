#

# prepare input features/factors of the seq, especially for padding
from msp.utils import Constants, zcheck
from collections import Iterable
import numpy as np

# prepare and truncate/pad the data along sentence-seq steps
# (batch, step, *) -> padded (step, batch, *)
# data: recursive list
# pad_lens, pad_vals: dim(data) sized list, pad_len<=0 means max
# dynamic_lens: use max_value if max_value<=pad_len
# mask_range: record mask for how many dims
class DataPadder(object):
    def __init__(self, dim, pad_lens=None, pad_vals=0, dynamic_lens=True, mask_range=0):
        self.dim = dim
        self.pad_lens = [-1]*dim if pad_lens is None else pad_lens
        self.pad_vals = [pad_vals]*dim if not isinstance(pad_vals, Iterable) else pad_vals
        self.dynamic_lens = [dynamic_lens]*dim if not isinstance(dynamic_lens, Iterable) else dynamic_lens
        self.mask_range = mask_range

    def _rec_size(self, d, cur_dim, sizes):
        diff = self.dim - cur_dim
        if diff<=0:
            return
        one_len = len(d)
        sizes[cur_dim] = max(one_len, sizes[cur_dim])
        for one_d in d:
            self._rec_size(one_d, cur_dim+1, sizes)

    def _rec_fill(self, d, cur_dim, sizes, strides, arr, arr_mask):
        diff = self.dim - cur_dim
        if diff<=0:
            return
        one_len = len(d)
        cur_pad = sizes[cur_dim]
        # fill
        if diff == 1:
            arr.extend(d[:cur_pad])
        else:
            for one_d in d[:cur_pad]:
                self._rec_fill(one_d, cur_dim+1, sizes, strides, arr, arr_mask)
        # pad
        miss = cur_pad-one_len
        if miss>0:
            arr.extend([self.pad_vals[cur_dim]] * (strides[cur_dim]*miss))
        # mask
        if cur_dim == self.mask_range-1:
            if miss>0:
                arr_mask.extend([1.]*one_len+[0.]*miss)
            else:
                arr_mask.extend([1.]*cur_pad)

    # return numpy-arr, mask
    def pad(self, data):
        # 1. first decide the sizes
        sizes = [0] * self.dim
        self._rec_size(data, 0, sizes)
        for idx in range(self.dim):
            pad_len = self.pad_lens[idx]
            if pad_len>0:
                if self.dynamic_lens[idx]:
                    sizes[idx] = min(sizes[idx], pad_len)
                else:
                    sizes[idx] = pad_len
        # 2. then iter the data and pad/trunc
        strides = [1]
        for one_s in reversed(sizes[1:]):
            strides.append(strides[-1]*one_s)
        strides.reverse()
        arr, arr_mask = [], []
        self._rec_fill(data, 0, sizes, strides, arr, arr_mask)
        ret = np.asarray(arr).reshape(sizes)
        if self.mask_range > 0:
            ret_mask = np.asarray(arr_mask, dtype=np.float32).reshape(sizes[:self.mask_range])
        else:
            ret_mask = None
        return ret, ret_mask
