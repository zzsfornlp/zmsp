#

# the (batched) BFS-Style Linear Searchers

from typing import List, Tuple, Dict, Callable
from .searcher import SearcherAgenda, SearcherCache, SearcherComponent, Searcher
from .systemer import *

# agenda for one instance
class BfsAgenda(SearcherAgenda):
    def __init__(self, inst: object, init_states: List[SearchState]):
        self.inst = inst
        self.currents = init_states  # current states
        self.ends = []  # ending states
        # history steps
        self.histories = [init_states]

    @staticmethod
    def init(state_creator: Callable, inst: object, sg: SearchGraph = None):
        init_state = state_creator(prev=None, sg=sg, inst=inst)
        a = BfsAgenda(inst, init_states=[init_state])
        return a

    def update(self, new_states: List[SearchState]):
        self.histories.append(new_states)
        self.currents = new_states

    def add_end(self, state: SearchState):
        self.ends.append(state)

    def is_all_end(self):
        return all(x.is_end() for x in self.currents)

# =====
# (batched) main components

# cache (mainly storing batched tensors)
class BfsCache(SearcherCache):
    pass

# expanding local candidates (with possible scorer calculations)
# this one is mostly system-specific!
class BfsExpander(SearcherComponent):
    def expand(self, ags: List[BfsAgenda], cache: BfsCache) -> List[List[SearchState]]:
        raise NotImplementedError()

# global arranger on all local selected ones
class BfsGlobalArranger(SearcherComponent):
    def __init__(self, beam_size: int, coster: Coster, signaturer: Signaturer):
        self.beam_size = beam_size
        # components to provide state-info
        self.coster = coster
        self.signaturer = signaturer
        self.margin = None

    def __repr__(self):
        return f"<{self.__class__.__name__}, K={self.beam_size}, coster={self.coster}, signaturer={self.signaturer}>"

    def refresh(self, margin=0., **kwargs):
        self.margin = margin

    # return whether survived
    def _add_or_merged(self, sig_map: Dict, cand: SearchState) -> bool:
        cur_sig = cand.sig
        if cur_sig is not None:
            cur_merger_state = sig_map.get(cur_sig, None)
            if cur_merger_state is None:
                sig_map[cur_sig] = cand
                return True
            else:
                cand.merge_by(cur_merger_state)
                return False
        return True

    # rank and select for the beam
    def arrange(self, ags: List[BfsAgenda], local_selections: List[List[SearchState]]) -> List[List[SearchState]]:
        coster, signaturer = self.coster, self.signaturer
        beam_size = self.beam_size
        cur_margin = self.margin
        # --
        batch_len = len(ags)
        global_selections: List[List[SearchState]] = [None] * batch_len
        assert batch_len == len(local_selections)
        # for each instance
        for batch_idx in range(batch_len):
            cands = local_selections[batch_idx]
            # sort by score+cost*margin: set cost if needed (mostly in training)
            if coster is not None:
                coster.set_costs(cands)
                cands.sort(key=lambda x: x.get_loss_aug_score(self.margin), reverse=True)
            else:  # directly by score
                cands.sort(key=lambda x: x.score_accu, reverse=True)
            # if signature, then we may merge things
            if signaturer is not None:
                signaturer.set_sigs(cands)
                # loop them all
                id_map, sig_map = {}, {}
                final_cands = []
                for one_cand in cands:
                    if self._add_or_merged(sig_map, one_cand):
                        final_cands.append(one_cand)
                        # todo(note): id map is for excluding repeated states between plain and oracle
                        # id_map[(plain_cand.prev, plain_cand.action)] = plain_cand
                        if len(final_cands) >= beam_size:
                            break
            else:  # otherwise, simply cut off by beam size
                final_cands = cands[:beam_size]
            global_selections[batch_idx] = final_cands
        return global_selections

# the preparation for the next step
class BfsEnder(SearcherComponent):
    # modify the agenda inplace
    def end(self, ags: List[BfsAgenda], final_selections: List[List[SearchState]]):
        assert len(ags) == len(final_selections)
        for ag, one_selections in zip(ags, final_selections):
            new_states = []
            for s in one_selections:
                if self.is_end(s):  # check end
                    s.mark_end()
                    ag.add_end(s)
                else:
                    new_states.append(s)
            ag.update(new_states)

    # todo(note): to be implemented for specific system
    def is_end(self, state: SearchState):
        raise NotImplementedError()

    # at the final end of the searching, by default do nothing
    def wrapup(self, ags: List[BfsAgenda]):
        pass

# =====
# the (batched) searcher itself

class BfsSearcher(Searcher):
    def __init__(self):
        # components
        self.expander: BfsExpander = None
        self.global_arranger: BfsGlobalArranger = None
        self.ender: BfsEnder = None

    def go(self, ags: List[BfsAgenda], cache: BfsCache):
        while True:
            # finish if all ended
            if all(a.is_all_end() for a in ags):
                break
            # step 1: expand for local candidates
            local_selections = self.expander.expand(ags, cache)
            # step 2: select globally (skip if not necessary)
            if self.global_arranger is None:
                final_selections = local_selections
            else:
                final_selections = self.global_arranger.arrange(ags, local_selections)
            # step 3: next step (inplaced modifications)
            self.ender.end(ags, final_selections)
        # final ending
        self.ender.wrapup(ags)
        # results are stored in ags
        # pass
