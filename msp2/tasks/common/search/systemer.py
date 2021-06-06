#

# the transition/searching system

__all__ = [
    "SearchStateStatus", "SearchState", "SearchAction", "SearchGraph",
    "Coster", "Signaturer",
]

from enum import Enum
from typing import List
from msp2.utils import Constants

# state status
class SearchStateStatus(Enum):
    NONE = 1
    EXPAND = 2
    MERGED = 3
    END = 4

# basic state information, only maintains the properties of basic graph info, other things are in derived classes.
class SearchState:
    def __init__(self, prev: 'SearchState'=None, action: 'SearchAction'=None, score: float=0., sg: 'SearchGraph'=None):
        # 1. basic info
        self.sg: SearchGraph = sg  # related SearchGraph
        self.prev: SearchState = prev  # parent
        self.length: int = 0  # how many steps from start
        self.id: int = None  # id in sg
        # 2. status
        self.status: SearchStateStatus = SearchStateStatus.NONE  # as a start
        # 3. values
        self.action: SearchAction = action
        # model scores
        self.score: float = score
        self.score_accu: float = 0.
        # --
        # depend on prev state
        if prev is not None:
            self.sg = prev.sg  # same sg as prev
            self.length = prev.length + 1  # length+1
            self.score_accu = prev.score_accu + self.score  # accumulate score
        else:
            assert action is None, "Err: starting State does not accept Action."
        # extra graph info if needed
        if self.sg is not None:
            self.id = self.sg.reg(self)
            if prev is not None:
                prev.status = SearchStateStatus.EXPAND

    @property
    def pid(self):
        return -1 if self.prev is None else self.prev.id

    def __len__(self):
        return self.length

    def __repr__(self):
        return f"State[{self.id}<-{self.pid}] len={self.length}, act={self.action}, " \
            f"sc={self.score_accu:.3f}({self.score:.3f})"

    # status related
    def is_end(self):
        return self.status == SearchStateStatus.END

    def mark_end(self):
        self.status = SearchStateStatus.END
        if self.sg is not None:
            self.sg.add_end(self)

    def is_start(self):
        return self.prev is None

    # get the history path towards root
    def get_path(self, forward_order=True, depth=Constants.INT_PRAC_MAX):
        cur = self
        ret = []
        count = 0
        # get them until START or meet max depth
        while cur is not None and count < depth:
            ret.append(cur)
            cur = cur.prev
            count += 1
        # prepare the direction in the list
        if forward_order:
            ret.reverse()
        return ret

# one action: prev + action -> next
class SearchAction:
    pass

# the linear search graph (lattice) (can be optional in some modes)
class SearchGraph:
    def __init__(self, info=None):
        # information about the settings like srcs
        self.info = info
        # unique ids for states (must be >=1)
        self.counts = 0
        # special nodes
        self.root = None  # only one root (start-point)
        self.ends = []

    # register a new state, return its id in this graph
    def reg(self, s: SearchState):
        self.counts += 1
        # todo(+N): record more info about the state?
        return self.counts

    def set_root(self, s: SearchState):
        assert s.sg is self, "SGErr: State does not belong here"
        assert self.root is None, "SGErr: Can only have one root"
        assert s.is_start(), "SGErr: Only start node can be root"
        self.root = s

    def add_end(self, s: SearchState):
        self.ends.append(s)

# others
class Coster:
    def set_costs(self, states: List[SearchState]):
        raise NotImplementedError()

class Signaturer:
    def set_sigs(self, states: List[SearchState]):
        raise NotImplementedError()
