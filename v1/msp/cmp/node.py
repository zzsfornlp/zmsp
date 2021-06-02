#

class ZObject(object):
    def __init__(self, m=None):
        if m is not None:
            self.update(m)

    def update(self, m):
        for k, v in m.items():
            setattr(self, k, v)

# one recording node
class RecNode(object):
    def __init__(self, parent=None, myk=None):
        self.count = 0
        # cur level
        self.parent = parent
        self.myk = myk
        # next level
        self.nums = {}
        self.nexts = {}
        self.idxes = []

    def keys(self):
        return self.nums.keys()

    # add the end of None
    def add_seq(self, seq, val=1, idx=None):
        #
        def _update_node(one_node):
            one_node.count += val
            if idx is not None:
                one_node.idxes.append(idx)
        #
        if not isinstance(seq, (list, tuple)):
            seq = [seq]
        # recursive adding
        cur_node = self
        _update_node(cur_node)
        while len(seq) > 0:
            seq0, seq = seq[0], seq[1:]
            if isinstance(seq0, (list, tuple)):
                # todo(warn): adding list of properties!
                assert len(seq) == 0, "Only support ending properties!"
                for one_prop in seq0:
                    if one_prop not in cur_node.nums:
                        cur_node.nums[one_prop] = 0
                        cur_node.nexts[one_prop] = RecNode(self, one_prop)
                    cur_node.nums[one_prop] += val
                    _update_node(cur_node.nexts[one_prop])
                cur_node = None
                break
            else:
                if seq0 not in cur_node.nums:
                    cur_node.nums[seq0] = 0
                    cur_node.nexts[seq0] = RecNode(self, seq0)
                cur_node.nums[seq0] += val
                cur_node = cur_node.nexts[seq0]
                _update_node(cur_node)

    #
    def my_info(self):
        return self.parent.k_info(self.myk)

    # next levels
    #
    def get_perc(self, k, smooth=0., class_num=None):
        class_all = len(self.nums) if class_num is None else class_num
        count_all = self.count + smooth*class_all
        count_k = self.nums.get(k, 0.) + smooth
        #
        if count_all == 0.:
            return 0.
        else:
            return count_k / count_all

    def get_num(self, k):
        return self.nums.get(k, 0)

    def get_next(self, k):
        return self.nexts.get(k, None)

    # queries (next levels?)
    def k_info(self, k):
        ret = {"key": k, "all": self.count, "num": self.get_num(k), "perc": self.get_perc(k),
                "str": self.k_str(k), "next": self.get_next(k)}
        return ret

    def k_str(self, k):
        return "%s:%d/%.4f" % (k, self.get_num(k), self.get_perc(k))

    def ks_info(self, sort_by="key", max_num=-1):
        if isinstance(sort_by, str):
            # sort_f = lambda x: getattr(x, sort_by)
            sort_f = lambda x: x[sort_by]
        else:
            sort_f = sort_by
        #
        all_infos = [self.k_info(k) for k in self.nums]
        all_infos.sort(key=sort_f)
        if max_num >= 0:
            return all_infos[:max]
        else:
            return all_infos

    # also can add accu-perc
    def ks_str(self, sort_by="key", max_num=-1, sep=" ", accu_perc=True):
        the_infos = self.ks_info(sort_by, max_num)
        #
        thems = ["(%d/%d)" % (self.count, len(self.nums))]
        accu_v = 0.
        for one_info in the_infos:
            z = one_info["str"]
            if accu_perc:
                accu_v += one_info["perc"]
                z += "(%.2f)" % accu_v
            thems.append(z)
        return sep.join(thems)
