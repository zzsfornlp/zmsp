#

# base State

from typing import List
from array import array

from msp.nn import BK
from msp.search.lsg import State, Action, Graph, Oracler, Coster, Signaturer

from ...common.data import ParseInstance

# =====
# basic units

class EfState(State):
    def __init__(self, prev: 'EfState'=None, action: 'EfAction'=None, score=0., sg: Graph=None,
                 inst: ParseInstance=None, max_slen=-1, orig_bidx=-1):
        super().__init__(prev, action, score, sg)
        # todo(+N): can it be more efficient to maintain (not-copy) the structure info?
        # record the basic info
        if prev is None:
            self.inst: ParseInstance = inst
            self.num_tok = len(inst) + 1  # num of tokens plus artificial root
            self.num_rest = self.num_tok - 1  # num of arcs remained to attach
            self.list_arc = [-1] * self.num_tok  # attached heads
            self.list_label = [-1] * self.num_tok  # attached labels
            self.idxes_chs = [[] for _ in range(self.num_tok)]  # child token idx list (by adding order)
            self.labels_chs = [[] for _ in range(self.num_tok)]  # should be the same size and corresponds to "ch_nodes"
        else:
            self.inst = prev.inst
            self.num_tok = prev.num_tok
            self.num_rest = prev.num_rest - 1  # currently always Attach actions
            self.list_arc = prev.list_arc.copy()
            self.list_label = prev.list_label.copy()
            # self.idxes_chs = [x.copy() for x in prev.idxes_chs]
            # self.labels_chs = [x.copy() for x in prev.labels_chs]
            # todo(warn): only shallow copy here, later copy on write!
            self.idxes_chs = prev.idxes_chs.copy()
            self.labels_chs = prev.labels_chs.copy()
            # only attach actions
            cur_head, cur_mod, cur_label = action.head, action.mod, action.label
            assert self.list_arc[cur_mod]<0
            self.list_arc[cur_mod] = cur_head
            self.list_label[cur_mod] = cur_label
            # self.idxes_chs[cur_head].append(cur_mod)
            # self.labels_chs[cur_head].append(cur_label)
            # copy on write
            self.idxes_chs[cur_head] = self.idxes_chs[cur_head] + [cur_mod]
            self.labels_chs[cur_head] = self.labels_chs[cur_head] + [cur_label]
        # =====
        # other calculate as needed values
        # useful for batching
        self.running_bidx: int = None  # running batch-idx (position in running batches)
        # original batch-idx (instances) todo(warn): to be set at the start
        if prev is not None:
            self.orig_bidx = prev.orig_bidx
        else:
            self.orig_bidx = orig_bidx
        # max length in the current batch; todo(+N): super non-elegant
        if prev is not None:
            self.max_slen = prev.max_slen
        else:
            self.max_slen = max_slen
        # related with cost/oracle (whether current arc/label is correct)
        self.wrong_al = None

    # =====
    # helpers

    @staticmethod
    def set_running_bidxes(flattened_states: List['EfState']):
        for idx, s in enumerate(flattened_states):
            s.running_bidx = idx

    def set_orig_bidx(self, bidx: int):
        self.orig_bidx = bidx

    # =====
    # specific procedures (default as EfFree System)

    # extend from self to another state
    def build_next(self, action: 'EfAction', score: float):
        raise NotImplementedError()

    # todo(note): prev_mask is prev's cands, and the illegal ones are already excluded at the very start
    def update_cands_mask(self, prev_mask):
        raise NotImplementedError()

    def update_oracle_mask(self, prev_arc_mask, prev_label):
        raise NotImplementedError()

    # self.cost/self.cost_accu
    # todo(+N): systems other than the top-down one are not arc-decomp and have future-loss because of cycle constraint!
    #  Therefore, this cost is less than the actual (plus-future) cost
    #  but maybe the (partial) oracle is still the correct oracle?
    def set_cost(self, weight_arc: float, weight_label: float):
        if self.cost_accu is not None:
            return self.cost_accu
        # for this system, simply consider only the last action will be fine
        if self.prev is None:
            self.cost_accu = self.cost = 0.
        else:
            prev_cost_accu = self.prev.set_cost(weight_arc, weight_label)
            this_head, this_mod, this_label = self.action._key
            gold_heads, gold_labels = self.inst.heads.vals, self.inst.labels.idxes
            wrong_arc, wrong_label = int(gold_heads[this_mod]!=this_head), int(gold_labels[this_mod]!=this_label)
            # todo(note): wrong_arc always means wrong_label!
            wrong_label = min(wrong_arc+wrong_label, 1)
            self.wrong_al = (wrong_arc, wrong_label)
            self.cost = weight_arc * wrong_arc + weight_label * wrong_label
            self.cost_accu = prev_cost_accu + self.cost
        return self.cost_accu

    # todo(+2): are there more efficient ways?
    # self.sig
    def set_sig(self, labeled: bool):
        if self.sig is not None:
            return self.sig
        atype = "b" if (self.num_tok<127) else "h"  # todo(note): hope most sentences will be shorter than 127
        sig_list = (self.list_arc + self.list_label) if labeled else self.list_arc
        sig = array(atype, sig_list).tobytes()
        self.sig = sig
        return sig

    # todo(WARN): currently in this system, costs are already added in Selector, but it still ignores further loss!
    def get_loss_aug_score(self, margin):
        # todo(+N): return self.score_accu + margin*self.future_loss
        return self.score_accu

# only Attach action
class EfAction(Action):
    def __init__(self, head, mod, label):
        super().__init__()
        # info
        self.head = head
        self.mod = mod
        self.label = label
        #
        self._key = (head, mod, label)

    def __repr__(self):
        return f"{self.head}->{self.mod}[{self.label}]"

    def __hash__(self):
        return hash(self._key)

    def __eq__(self, other):
        return self._key == other._key

# =====
#

# todo(warn): not used here!
# # group of candidates from one State, wait for Selector
# class EfCandidates(Candidates):
#     def __init__(self, state: EfState, cands_mask: np.ndarray):
#         self.state = state
#         self.cands_mask = cands_mask

# =====
# other components

class EfOracler(Oracler):
    def set_oracle_masks(self, states: List[EfState], prev_oracle_mask_ct, prev_oracle_label_ct):
        for idx, s in enumerate(states):
            s.update_oracle_mask(prev_oracle_mask_ct[idx], prev_oracle_label_ct[idx])

    # most of the time, only need to init once and the oracle masks need not changes
    # todo(note): this is assigned at CPU
    @staticmethod
    def init_oracle_mask(inst: ParseInstance, prev_arc_mask, prev_label):
        gold_heads = inst.heads.vals[1:]
        gold_labels = inst.labels.idxes[1:]
        gold_idxes = [i + 1 for i in range(len(gold_heads))]
        prev_arc_mask[gold_idxes, gold_heads] = 1.
        prev_label[gold_idxes, gold_heads] = BK.input_idx(gold_labels, BK.CPU_DEVICE)
        return prev_arc_mask, prev_label

    def __repr__(self):
        return "<EfOracler>"

class EfCoster(Coster):
    def __init__(self, weight_arc: float, weight_label: float):
        self.weight_arc = weight_arc
        self.weight_label = weight_label

    def set_costs(self, states: List[EfState]):
        weight_arc, weight_label = self.weight_arc, self.weight_label
        for s in states:
            s.set_cost(weight_arc, weight_label)

    def __repr__(self):
        return f"<EfCoster: weight_arc={self.weight_arc}, weight_label={self.weight_label}>"

class EfSignaturer(Signaturer):
    def __init__(self, labeled: bool):
        self.labeled = labeled

    def set_sigs(self, states: List[EfState]):
        labeled = self.labeled
        for s in states:
            s.set_sig(labeled)

    def __repr__(self):
        return f"<EfSignaturer: labeled={self.labeled}>"
