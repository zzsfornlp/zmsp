#
from msp.utils import Constants, zcheck

# =====
# inner graph representation
# todo(+N): this part can be re-written in cython or c++

# this class only maintain the properties of basic graph info, other things are not here!
class LinearState(object):
    # _STATUS = "NONE"(unknown), "EXPAND"(survived), "MERGED", "END"
    STATUS_NONE, STATUS_EXPAND, STATUS_MERGED, STATUS_END = 0, 1, 2, 3

    # ==== constructions
    def __init__(self, prev: 'LinearState'=None, action=None, score=0., sg: 'LinearGraph'=None, padded=False):
        # 1. basic info
        self.sg = sg            # related SearchGraph
        self.prev = prev        # parent
        self.length = 0         # how many steps from start
        # graph info
        self.id = None
        self.nexts = None           # valid next states
        self.merge_list = None      # merge others
        self.merger = None          # merged by other

        # 2. status
        self.padded = padded
        self.status = LinearState.STATUS_NONE

        # 3. values
        self.action = action
        self.score = score
        self.score_accu = 0.
        # =====
        # depend on prev state
        if prev is not None:
            self.sg = prev.sg
            self.length = prev.length + 1
            self.score_accu = prev.score_accu + self.score
        # no need to preserve graph info if not recording it
        if padded:
            self.sg = None
        if self.sg is not None:
            self.nexts = []             # valid next states
            self.merge_list = []        # merge others
            self.id = self.sg.reg(self)
            if prev is not None:
                prev.status = LinearState.STATUS_EXPAND
                prev.nexts.append(self)

    def get_pid(self):
        if self.prev is None:
            return -1
        else:
            return self.prev.id

    def __len__(self):
        return self.length

    def __repr__(self):
        return f"State[{self.id}<-{self.get_pid()}] len={self.length}, act={self.action}, sc={self.score_accu:.3f}({self.score:.3f})"

    # graph related
    def is_end(self):
        return self.status == LinearState.STATUS_END

    def is_start(self):
        return self.prev is None

    def mark_end(self):
        self.status = LinearState.STATUS_END
        if self.sg is not None:
            self.sg.add_end(self)

    # get the history path towards root
    def get_path(self, forward_order=True, depth=Constants.INT_PRAC_MAX):
        cur = self
        ret = []
        # todo(warn): excluding the usually artificial root-node
        while cur.prev is not None and len(ret) < depth:
            ret.append(cur)
            cur = cur.prev
        if forward_order:
            ret.reverse()
        return ret

    # merge: modify both states
    def merge_by(self, s: 'LinearState'):
        zcheck(s.merger is None, "Err: multiple level of merges!")
        self.status = LinearState.STATUS_MERGED
        self.merger = s
        s.add_merge(self)

    def add_merge(self, s: 'LinearState'):
        if self.merge_list:
            self.merge_list.append(s)

# the linear search graph (lattice)
class LinearGraph(object):
    def __init__(self, info=None):
        # information about the settings like srcs
        self.info = info
        # unique ids for states (must be >=1)
        self.counts = 0
        # special nodes
        self.root = None
        self.ends = []

    # register a new state, return its id in this graph
    def reg(self, s: LinearState):
        self.counts += 1
        # todo(+N): record more info about the state?
        return self.counts

    def set_root(self, s: LinearState):
        zcheck(s.sg is self, "SGErr: State does not belong here")
        zcheck(self.root is None, "SGErr: Can only have one root")
        zcheck(s.is_start(), "SGErr: Only start node can be root")
        self.root = s

    def add_end(self, s: LinearState):
        self.ends.append(s)
