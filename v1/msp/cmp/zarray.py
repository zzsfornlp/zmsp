#

# printing wrappers for array
# usually for printing and reporting, no need to consider efficiency!

import numpy as np

# todo(warn): should use pandas!

class ZArray:
    def __init__(self, arr: np.ndarray, names=None):
        self.arr = arr
        self.dims = arr.shape
        self.ndim = len(self.dims)
        self.idxes = [None] * self.ndim
        self.names = [None] * self.ndim
        # check #dim
        if names is None:
            names = [None] * self.ndim
        assert self.ndim == len(names)
        # check each dim
        for i in range(self.ndim):
            if names[i] is None:
                # dim-i, name-j
                self.names[i] = [f"zd{i}n{j}" for j in range(self.dims[i])]
            else:
                self.names[i] = names[i]
            #
            self.idxes[i] = {k:ii for ii,k in enumerate(self.names[i])}
            assert len(self.names[i]) == self.dims[i]
            assert len(self.idxes[i]) == self.dims[i]

    # can perform reduction at outside
    def get_arr(self):
        return self.arr

    def get_names(self, axis, idxes):
        if isinstance(idxes, (list, tuple)):
            return [self.names[axis][z] for z in idxes]
        else:
            return self.names[axis][idxes]

    def get_idxes(self, axis, names):
        if isinstance(names, (list, tuple)):
            return [self.idxes[axis][z] for z in names]
        else:
            return self.idxes[axis][names]

    def get_names_axis(self, axes=None):
        if axes is None:
            return self.names
        elif isinstance(axes, (list, tuple)):
            return [self.names[z] for z in axes]
        else:
            return self.names[axes]

    # =====
    # row major
    # latex-mode: fsep=" & " lsep="\\\\\n"
    @staticmethod
    def str_table(matrix_arr, names_pair, max_blank=False, fsep=" ", lsep="\n"):
        matrix_shape = matrix_arr.shape
        assert len(matrix_shape) == 2 and len(names_pair) == 2 and (len(names_pair[0]), len(names_pair[1])) == matrix_shape
        #
        print_matrix = [[""] + [str(z) for z in names_pair[1]]]
        for row in range(len(matrix_arr)):
            print_matrix.append([str(names_pair[0][row])] + [str(z) for z in matrix_arr[row]])
        if max_blank:
            # get max-len for each column
            print_matrix = np.array(print_matrix)
            num_col = print_matrix.shape[1]
            for col in range(num_col):
                col_arr = print_matrix[:, col]
                max_len = max(len(z) for z in col_arr)
                # todo(+1): aligned to the left!
                col_format = "%%-%ds" % max_len
                print_matrix[:, col] = [col_format % z for z in col_arr]
        #
        ss = lsep.join([fsep.join(row_arr) for row_arr in print_matrix])
        return ss

    @staticmethod
    # input_pairs: list of (key, value)
    # todo(warn): padding should indicate the dtype, np.* for numbers, None for object
    def create_zarr(input_pairs, padding, key_sorters=None):
        if len(input_pairs) == 0:
            return ZArray(np.array([]))     # empty
        n_dim = len(input_pairs[0])
        key_dicts = [{} for _ in range(n_dim)]      # name -> {'idx', 'count', 'name', }
        if key_sorters is None:
            key_sorters = [None] * n_dim
        # default is sort by appearance
        assert len(key_sorters) == n_dim
        key_sorters = ['idx' if z is None else z for z in key_sorters]
        # collect all the keys for each dimension
        for key, value in input_pairs:
            assert len(key) == n_dim
            for i in range(n_dim):
                one_k = key[i]
                one_d = key_dicts[i]
                if one_k not in one_d:
                    one_d[one_k] = {"idx": len(one_d), "count": 1, "name": one_k}
                else:
                    one_d[one_k]["count"] += 1
        # get real orders
        ordered_names = [sorted(d.keys(), key=lambda z: d[z][s]) for d,s in zip(key_dicts, key_sorters)]
        ordered_dicts = [{n:i for i,n in enumerate(z)} for z in ordered_names]
        # expand tensor (decide the type by padding)
        arr_shape = [len(z) for z in ordered_names]
        arr = np.full(arr_shape, padding)
        for key, value in input_pairs:
            the_idx = tuple([ordered_dicts[i][key[i]] for i in range(n_dim)])
            arr[the_idx] = value
        return ZArray(arr, ordered_names)
