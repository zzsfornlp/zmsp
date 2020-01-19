#

# the (batched) BFS-Style Linear Searchers
# the three layers: State, Agenda, Searcher

from typing import List, Tuple, Dict
from .system import State, Graph, Oracler, Coster, Signaturer
from .search import Agenda, Searcher, Component

# agenda for one instance
class BfsAgenda(Agenda):
    def __init__(self, inst, init_beam: List[State]=None, init_gbeam: List[State]=None):
        self.inst = inst
        self.beam = [] if init_beam is None else init_beam  # plain search beam
        self.gbeam = [] if init_gbeam is None else init_gbeam  # gold-informed ones
        # for loss collection
        self.local_golds = []  # gold (oracle) states for local loss
        self.special_points = []  # special snapshots of the plain/gold beams (for special update mode in training)
        # for results collection
        self.ends = [], []  # endings for plain/gold ones

    @staticmethod
    def init_agenda(state_creator, inst, require_sg: bool):
        sg = None
        if require_sg:
            sg = Graph()
        init_state = state_creator(prev=None, sg=sg, inst=inst)
        a = BfsAgenda(inst, init_beam=[init_state])
        return a

    def is_end(self):
        return all(x.is_end() for x in self.beam) and all(x.is_end() for x in self.gbeam)

# a snapshot (special point) of the current status of beam
class BfsSPoint:
    def __init__(self, plain_finals, oracle_finals, violation=0.):
        self.plain_finals = plain_finals
        self.oracle_finals = oracle_finals
        self.violation = violation

# =====
# scoring helpers

# todo(WARN): it seems that in previous code here, the margin get doubled?
#  But might not be True if cost is different than the simple ones, thus temporally leave it here!

def state_plain_ranker_getter(margin: float=0.):
    return lambda x: x.get_loss_aug_score(margin)

def state_oracle_ranker(s: State):
    return (-s.cost_accu, s.score_accu)

# =====
# (batched) main components

# expanding candidates (with possible scorer calculations)
# this one is mostly system-specific!
class BfsExpander(Component):
    def expand(self, ags: List[BfsAgenda]):
        raise NotImplementedError()

# select with multiple modes (plain + oracle)
# some usually-used cases: 1) plain(topk/sample)-search for decoding
class BfsLocalSelector(Component):
    def __init__(self, plain_mode, oracle_mode, plain_k_arc, plain_k_label, oracle_k_arc, oracle_k_label, oracler: Oracler):
        self.plain_mode = plain_mode
        self.oracle_mode = oracle_mode
        self.plain_k_arc, self.plain_k_label, self.oracle_k_arc, self.oracle_k_label = \
            plain_k_arc, plain_k_label, oracle_k_arc, oracle_k_label
        self.oracler = oracler

    def __repr__(self):
        return f"<{self.__class__.__name__}: plain:mode={self.plain_mode},arcK={self.plain_k_arc},labelK={self.plain_k_label}, " \
            f"oracle:mode={self.oracle_mode},arcK={self.oracle_k_arc},labelK={self.oracle_k_label}, oracler={self.oracler}>"

    # =====
    # these two only return flattened results

    def select_plain(self, ags: List[BfsAgenda], candidates, mode, k_arc, k_label) -> List[List]:
        raise NotImplementedError()

    def select_oracle(self, ags: List[BfsAgenda], candidates, mode, k_arc, k_label) -> List[List]:
        raise NotImplementedError()

    # =====
    # finally assembling final results

    # return List[(plain-list, oracle-list)]
    def select(self, ags: List[BfsAgenda], candidates) -> List[Tuple[List, List]]:
        plain_results = self.select_plain(ags, candidates, self.plain_mode, self.plain_k_arc, self.plain_k_label)
        oracle_results = self.select_oracle(ags, candidates, self.oracle_mode, self.oracle_k_arc, self.oracle_k_label)
        # assemble for final results
        sidx = 0
        final_results = []
        for ag in ags:
            one_cands_plain, one_cands_oracle = [], []  # plain-beam, gold-beam
            for plain_state in ag.beam:
                cur_plain_nexts, cur_oracle_nexts = plain_results[sidx], oracle_results[sidx]
                one_cands_plain.extend(cur_plain_nexts)  # plain+plain => plain
                one_cands_oracle.extend(cur_oracle_nexts)  # plain+oracle => oracle
                sidx += 1
            for oracle_state in ag.gbeam:
                cur_plain_nexts, cur_oracle_nexts = plain_results[sidx], oracle_results[sidx]
                # WHAT.extend(cur_plain_nexts)  # oracle+plain => ? todo(note): discard!
                one_cands_oracle.extend(cur_oracle_nexts)  # oracle+oracle => oracle
                sidx += 1
            final_results.append((one_cands_plain, one_cands_oracle))
        assert len(plain_results) == sidx and len(oracle_results) == sidx
        return final_results

class BfsGlobalArranger(Component):
    def __init__(self, plain_beam_size: int, gold_beam_size: int, coster: Coster, signaturer: Signaturer):
        self.plain_beam_size = plain_beam_size
        self.gold_beam_size = gold_beam_size
        # components to provide state-info
        self.coster = coster
        self.signaturer = signaturer
        self.margin = 0.
        #
        self.plain_ranker = state_plain_ranker_getter(self.margin)
        self.oracle_ranker = state_oracle_ranker

    def __repr__(self):
        return f"<{self.__class__.__name__}: plainK={self.plain_beam_size}, oracleK={self.gold_beam_size}, " \
            f"coster={self.coster}, signaturer={self.signaturer}>"

    def refresh(self, margin=0., **kwargs):
        self.margin = margin
        self.plain_ranker = state_plain_ranker_getter(self.margin)

    # return whether survived
    def _add_or_merged(self, sig_map: Dict, cand: State) -> bool:
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
    def arrange(self, ags: List[BfsAgenda], local_selections: List) -> List[Tuple[List, List]]:
        coster, signaturer = self.coster, self.signaturer
        plain_beam_size, gold_beam_size = self.plain_beam_size, self.gold_beam_size
        #
        cur_margin = self.margin
        batch_len = len(ags)
        global_selections: List[Tuple[List, List]] = [None] * batch_len
        assert batch_len == len(local_selections)
        # for each instance
        for batch_idx in range(batch_len):
            pcands, gcands = local_selections[batch_idx]
            # plain sort by score+cost*margin, oracle sort by (-cost, score)
            # set cost if needed (mostly in training)
            if coster is not None:
                coster.set_costs(pcands)
                pcands.sort(key=self.plain_ranker, reverse=True)
                coster.set_costs(gcands)
                gcands.sort(key=self.oracle_ranker, reverse=True)  # ignore margin for oracle ones
            else:
                pcands.sort(key=self.plain_ranker, reverse=True)
                assert len(gcands) <= 1, "Err: No Coster to rank the oracle states!"
            # set signature if needed
            if signaturer is not None:
                signaturer.set_sigs(pcands)
                signaturer.set_sigs(gcands)
            # loop them all
            id_map, sig_map = {}, {}
            plain_finals, oracle_finals = [], []
            for plain_cand in pcands:
                if self._add_or_merged(sig_map, plain_cand):
                    plain_finals.append(plain_cand)
                    # todo(note): id map is for excluding repeated states between plain and oracle
                    id_map[(plain_cand.prev, plain_cand.action)] = plain_cand
                    if len(plain_finals) >= plain_beam_size:
                        break
            for oracle_cand in gcands:
                cur_id = (oracle_cand.prev, oracle_cand.action)
                if cur_id not in id_map:
                    if self._add_or_merged(sig_map, oracle_cand):
                        oracle_finals.append(oracle_cand)
                        if len(oracle_finals) >= gold_beam_size:
                            break
            global_selections[batch_idx] = (plain_finals, oracle_finals)
        return global_selections

# the preparation for the next step
class BfsEnder(Component):
    def __init__(self, ending_mode):
        self.ending_mode = ending_mode
        self._modify_f = {"plain": self._modify_plain, "eu": self._modify_eu,
                          "maxv": self._modify_maxv, "bso": self._modify_bso}[ending_mode]
        self.margin = 0.
        #
        self.plain_ranker = state_plain_ranker_getter(self.margin)
        self.oracle_ranker = state_oracle_ranker

    def __repr__(self):
        return f"<{self.__class__.__name__}: ending_mode={self.ending_mode}>"

    def refresh(self, margin=0., **kwargs):
        self.margin = margin
        self.plain_ranker = state_plain_ranker_getter(self.margin)

    def end(self, ags: List[BfsAgenda], final_selections: List):
        for ag, one_selections in zip(ags, final_selections):
            new_beam, new_gbeam = [], []
            if len(one_selections[0]) > 0:
                # with special modifications
                plain_finals, oracle_finals = self._modify_f(ag, one_selections)
                # pick out ended ones for both beams
                for which_end, which_beam, which_finals in zip([0,1], [new_beam, new_gbeam], [plain_finals, oracle_finals]):
                    for s in which_finals:
                        if self.is_end(s):
                            s.mark_end()
                            ag.ends[which_end].append(s)
                        else:
                            which_beam.append(s)
            ag.beam = new_beam
            ag.gbeam = new_gbeam

    def wrapup(self, ags: List[BfsAgenda]):
        # whether adding the endings states
        ending_mode = self.ending_mode
        if ending_mode == "plain" or ending_mode == "bso":
            for ag in ags:
                ag.special_points.append(BfsSPoint(ag.ends[0], ag.ends[1]))
        elif ending_mode == "eu":
            for ag in ags:
                if len(ag.special_points) == 0:
                    ag.special_points.append(BfsSPoint(ag.ends[0], ag.ends[1]))
        elif ending_mode == "maxv":
            pass
        else:
            raise NotImplementedError()

    # =====
    # special modes for modifying the beam

    def _modify_plain(self, ag: BfsAgenda, final_selections):
        plain_finals, oracle_finals = final_selections
        return plain_finals, oracle_finals

    def _modify_eu(self, ag: BfsAgenda, final_selections):
        # check whether gold drop out
        plain_finals, oracle_finals = final_selections
        if len(oracle_finals)>0:
            best_oracle = max(oracle_finals, key=self.oracle_ranker)
            # if oracle dropped out of the beam
            if all(best_oracle.cost_accu<x.cost_accu for x in plain_finals):
                ag.special_points.append(BfsSPoint(plain_finals, oracle_finals))
                return [], []
        return plain_finals, oracle_finals

    def _modify_bso(self, ag: BfsAgenda, final_selections):
        # check whether gold drop out
        plain_finals, oracle_finals = final_selections
        if len(oracle_finals)>0:
            best_oracle = max(oracle_finals, key=self.oracle_ranker)
            # if oracle dropped out of the beam
            if all(best_oracle.cost_accu < x.cost_accu for x in plain_finals):
                ag.special_points.append(BfsSPoint(plain_finals, oracle_finals))
                best_oracle.mark_restart()
                return [best_oracle], []  # add it back!!
        return plain_finals, oracle_finals

    def _modify_maxv(self, ag: BfsAgenda, final_selections):
        # check violation
        plain_finals, oracle_finals = final_selections
        best_plain = max(plain_finals, key=self.plain_ranker)
        best_oracle = max(plain_finals+oracle_finals, key=self.oracle_ranker)
        cur_violation = self.plain_ranker(best_plain) - self.plain_ranker(best_oracle)  # score diff
        # keep the max-violation one
        if len(ag.special_points) == 0:
            ag.special_points.append(BfsSPoint(plain_finals, oracle_finals, cur_violation))
        elif cur_violation > ag.special_points[0].violation:
            ag.special_points[0] = (BfsSPoint(plain_finals, oracle_finals, cur_violation))
        return plain_finals, oracle_finals
    # =====

    # todo(note): to be implemented for specific system
    def is_end(self, state: State):
        raise NotImplementedError()

# =====
# the (batched) searcher itself

class BfsSearcher(Searcher):
    def __init__(self):
        # components
        self.expander: BfsExpander = None
        self.local_selector: BfsLocalSelector = None
        self.global_arranger: BfsGlobalArranger = None
        self.ender: BfsEnder = None

    def go(self, ags: List[BfsAgenda]):
        while True:
            # finish if all ended
            if all(a.is_end() for a in ags):
                break
            # step 1: expand (list of list of candidates)
            candidates = self.expander.expand(ags)
            # step 2: score (batched all) & select locally
            local_selections = self.local_selector.select(ags, candidates)
            # step 3: select globally (skip if not necessary)
            if self.global_arranger is None:
                final_selections = local_selections
            else:
                final_selections = self.global_arranger.arrange(ags, local_selections)
            # step 4: next step (inplaced modifications)
            self.ender.end(ags, final_selections)
        # final ending
        self.ender.wrapup(ags)
        # results are stored in ags
        # pass

# =====
# creator

# todo(+N): currently specific for specific systems, can the common things be grouped together?

class BfsSearcherFactory:
    # =====
    # general creator: put all components together and get a Searcher
    pass
