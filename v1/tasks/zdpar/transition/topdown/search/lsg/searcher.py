#

# the (batched) Linear Searchers
# the three layers: State, Agenda, Searcher

from typing import List
from .graph import LinearState, LinearGraph

# =====
# main components

# agenda for one instance
class BfsLinearAgenda:
    def __init__(self, inst, init_beam: List[LinearState]=None, init_gbeam: List[LinearState]=None):
        self.inst = inst
        self.beam = [] if init_beam is None else init_beam          # plain search beam
        self.gbeam = [] if init_gbeam is None else init_gbeam       # gold-informed ones
        # for loss collection
        self.local_golds = []               # gold (oracle) states for local loss
        # for results collection
        self.ends = []
        self.last_beam = []

    @staticmethod
    def init_agenda(state_type, inst, require_sg: bool):
        sg = None
        if require_sg:
            sg = LinearGraph()
        init_state = state_type(prev=None, sg=sg, inst=inst)
        a = BfsLinearAgenda(inst, init_beam=[init_state])
        return a

    # TODO(+N): what to do with ended states?
    def is_end(self):
        return all(x.is_end() for x in self.beam) and all(x.is_end() for x in self.gbeam)

# =====
# batched searcher
class BfsLinearSearcher:
    # =====
    def expand(self, ags: List[BfsLinearAgenda]):
        raise NotImplementedError()

    def select(self, ags: List[BfsLinearAgenda], candidates):
        raise NotImplementedError()

    def end(self, ags: List[BfsLinearAgenda], selections):
        raise NotImplementedError()
    # =====

    def refresh(self, *args, **kwargs):
        raise NotImplementedError()

    def go(self, ags: List[BfsLinearAgenda]):
        while True:
            # finish if all ended
            if all(a.is_end() for a in ags):
                break
            # step 1: expand (list of list of candidates)
            candidates = self.expand(ags)
            # step 2: score (batched all) & select
            selections = self.select(ags, candidates)
            # step 3: next (inplaced modifications)
            self.end(ags, selections)
        # results are stored in ags
        # pass

# b tasks/zdpar/transition/topdown/search/lsg/searcher:63
