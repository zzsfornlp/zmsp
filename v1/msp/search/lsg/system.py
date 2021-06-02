#

# the core system components (the strctures)

from typing import List
from msp.utils import Constants, zcheck

# basic state information, only maintains the properties of basic graph info, other things are in derived classes.
class State:
    # _STATUS = "NONE"(unknown), "EXPAND"(survived), "MERGED", "END"
    STATUS_NONE, STATUS_EXPAND, STATUS_MERGED, STATUS_END = 0, 1, 2, 3

    # ==== constructions
    def __init__(self, prev: 'State'=None, action: 'Action'=None, score: float=0., sg: 'Graph'=None):
        # 1. basic info
        self.sg: 'Graph' = sg            # related SearchGraph
        self.prev: 'State' = prev        # parent
        self.length: int = 0         # how many steps from start
        # graph info
        self.id: int = None
        self.nexts: List = None           # valid next states
        self.merge_list: List = None      # merge others
        self.merger: State = None          # merged by other

        # 2. status
        self.status: int = State.STATUS_NONE
        self.restart: bool = False

        # 3. values
        self.action: Action = action
        # model scores
        self.score: float = score
        self.score_accu: float = 0.
        # actual cost (unknown currently, to be filled by Oracler)
        self.cost: float = None
        self.cost_accu: float = None
        # signature
        self.sig = None  # None actually means no sig, but remember None != None for sig
        # =====
        # depend on prev state
        if prev is not None:
            self.sg = prev.sg
            self.length = prev.length + 1
            self.score_accu = prev.score_accu + self.score
            # todo(note): set the end points for the action
            self.action.set_from_to(prev, self)
        else:
            self.restart = True  # root is also a restart
            assert action is None, "Err: starting State does not accept Action."
        # no need to preserve graph info if not recording it
        if self.sg is not None:
            self.nexts = []             # valid next states
            self.merge_list = []        # merge others
            self.id = self.sg.reg(self)
            if prev is not None:
                prev.status = State.STATUS_EXPAND
                prev.nexts.append(self)

    def get_pid(self):
        if self.prev is None:
            return -1
        else:
            return self.prev.id

    def __len__(self):
        return self.length

    def __repr__(self):
        return f"State[{self.id}<-{self.get_pid()}] len={self.length}, act={self.action}, " \
            f"sc={self.score_accu:.3f}({self.score:.3f})"

    # graph related
    def is_end(self):
        return self.status == State.STATUS_END

    def is_start(self):
        return self.prev is None

    def is_restart(self):
        return self.restart

    def mark_restart(self):
        self.restart = True

    def mark_end(self):
        self.status = State.STATUS_END
        if self.sg is not None:
            self.sg.add_end(self)

    # get the history path towards root
    def get_path(self, forward_order=True, depth=Constants.INT_PRAC_MAX, include_restart=False):
        cur = self
        ret = []
        count = 0
        # get them until START or meet max depth
        while cur is not None and count < depth:
            if not include_restart and cur.restart:
                break
            ret.append(cur)
            cur = cur.prev
            count += 1
        # prepare the direction in the list
        if forward_order:
            ret.reverse()
        return ret

    # merge: modify both states
    def merge_by(self, s: 'State'):
        zcheck(s.merger is None, "Err: multiple level of merges!")
        self.status = State.STATUS_MERGED
        self.merger = s
        s.add_merge(self)

    def add_merge(self, s: 'State'):
        if self.merge_list:
            self.merge_list.append(s)

    # =====
    def get_loss_aug_score(self, margin):
        return self.score_accu+margin*self.cost_accu

# one action from one state to another
class Action:
    def __init__(self):
        self.state_from: State = None
        self.state_to: State = None

    def set_from_to(self, fstate: State, tstate: State):
        self.state_from = fstate
        self.state_to = tstate

# one expansion from a state, can lead to real candidate states if further selected by the selector
class Candidates:
    pass

# the linear search graph (lattice)
class Graph(object):
    def __init__(self, info=None):
        # information about the settings like srcs
        self.info = info
        # unique ids for states (must be >=1)
        self.counts = 0
        # special nodes
        self.root = None
        self.ends = []

    # register a new state, return its id in this graph
    def reg(self, s: State):
        self.counts += 1
        # todo(+N): record more info about the state?
        return self.counts

    def set_root(self, s: State):
        zcheck(s.sg is self, "SGErr: State does not belong here")
        zcheck(self.root is None, "SGErr: Can only have one root")
        zcheck(s.is_start(), "SGErr: Only start node can be root")
        self.root = s

    def add_end(self, s: State):
        self.ends.append(s)

#
# other component classes for the searching procedure (decoding or training)

# =====

class Oracler:
    pass

class Coster:
    def set_costs(self, states: List[State]):
        raise NotImplementedError()

class Signaturer:
    def set_sigs(self, states: List[State]):
        raise NotImplementedError()

# =====
