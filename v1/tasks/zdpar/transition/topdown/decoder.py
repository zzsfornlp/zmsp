#

from typing import List
import numpy as np
from array import array
from copy import copy

from msp.utils import Conf, zcheck, Random, zlog, Constants, GLOBAL_RECORDER
from msp.nn import BK, ExprWrapper, SliceManager, SlicedExpr
from msp.nn.layers import BasicNode, Affine, BiAffineScorer, Embedding, RnnNode, MultiHeadAttention, AttConf, AttDistHelper
from msp.zext.process_train import ScheduledValue
from msp.zext.seq_helper import DataPadder

from ...common.data import ParseInstance
from .search.lsg import LinearState, LinearGraph, BfsLinearAgenda, BfsLinearSearcher

# manage single search instance
class TdState(LinearState):
    # todo(warn): global variable, not elegant and there are too many dependencies on this one!!
    is_bfs: bool = None

    def __init__(self, prev: 'TdState'=None, action=None, score=0., sg: LinearGraph=None, padded=False, inst: ParseInstance=None):
        super().__init__(prev, action, score, sg, padded)
        # todo(+N): can it be more efficient to maintain (not-copy) the structure info?
        # record the state for the top-down transition system
        if prev is None:
            self.inst = inst
            self.depth = 0                            # real depth
            self.depth_eff = 0                        # effective depth = depth if down else depth+1
            self.num_tok = len(inst) + 1              # num of tokens plus artificial root
            self.num_rest = self.num_tok - 1          # num of arcs to attach
            self.list_arc = [-1] * self.num_tok       # attached heads
            self.list_label = [-1] * self.num_tok     # attached labels
            self.reduced = [False] * self.num_tok     # whether this node has been reduced
            assert action is None
            self.state_par = None                    # parent state node
            self.state_last_cur = None               # previous cur state (cur = last_cur + one-more-children)
            self.idx_cur = 0                         # cur token idx
            # todo(warn): 0 serves as input ROOT label, and output reduce label, can never co-exist!
            self.label_cur = 0                       # cur label between self and parent
            self.idx_par = 0                         # parent token idx (0's parent is still 0)
            self.idxes_chs = []                      # child token idx list (by adding order)
            self.labels_chs = []                     # should be the same size and corresponds to "ch_nodes"
            #
            self.idx_ch_leftmost, self.idx_ch_rightmost = 0, 0
            # todo(+N): should make it another Class
            # BFS info
            assert TdState.is_bfs is not None
            self.is_bfs = TdState.is_bfs
            self.bfs_root = self
            if self.is_bfs:
                self.bfs_buffer = [0, []]  # next-idx, next-info list
        else:
            self.inst = prev.inst
            self.num_tok = prev.num_tok
            self.num_rest = prev.num_rest
            self.list_arc = prev.list_arc.copy()
            self.list_label = prev.list_label.copy()
            self.reduced = prev.reduced.copy()
            # BFS
            self.is_bfs = prev.is_bfs
            self.bfs_root = prev.bfs_root
            #
            # action = (head, child or head means up, attach_label or None)
            act_head, act_attach, act_label = action
            assert act_head == prev.idx_cur
            if act_head == act_attach:
                if self.is_bfs:
                    # go next: reduce
                    bfs_buffer_idx, bfs_buffer_list = prev.bfs_buffer
                    new_bfs_buffer_list = bfs_buffer_list.copy()
                    if bfs_buffer_idx < len(new_bfs_buffer_list):
                        next_par_state, next_cidx, next_label = new_bfs_buffer_list[bfs_buffer_idx]
                        new_bfs_buffer_list[bfs_buffer_idx] = None
                        self.bfs_buffer = [bfs_buffer_idx+1, new_bfs_buffer_list]
                    else:
                        # go back to root again!!
                        next_par_state, next_cidx, next_label = self.bfs_root, 0, 0
                        self.bfs_buffer = [0, []]
                    self.state_par = next_par_state  # parent state
                    self.state_last_cur = None  # no prev parallel states for this one
                    self.idx_cur = next_cidx
                    self.label_cur = next_label
                    self.idx_par = next_par_state.idx_cur
                    self.idxes_chs = []
                    self.labels_chs = []
                    self.reduced[act_head] = True
                    self.idx_ch_leftmost = self.idx_ch_rightmost = next_cidx
                    self.depth = self.depth_eff = -1  # todo(warn): not used!!
                else:
                    # go up: reduce
                    last_cur_state = prev.state_par
                    self.state_par = last_cur_state.state_par
                    self.state_last_cur = last_cur_state
                    self.idx_cur = last_cur_state.idx_cur
                    self.label_cur = last_cur_state.label_cur
                    self.idx_par = last_cur_state.idx_par
                    self.idxes_chs = last_cur_state.idxes_chs.copy()
                    self.idxes_chs.append(act_head)
                    self.labels_chs = last_cur_state.labels_chs.copy()
                    self.labels_chs.append(self.list_label[act_head])
                    self.reduced[act_head] = True  # reduce
                    # leftmost and rightmost children or self
                    self.idx_ch_leftmost, self.idx_ch_rightmost = \
                        min(act_head, last_cur_state.idx_ch_leftmost), max(act_head, last_cur_state.idx_ch_rightmost)
                    #
                    self.depth = prev.depth - 1
                    self.depth_eff = prev.depth
            else:
                # attach
                self.num_rest -= 1
                self.list_arc[act_attach] = act_head
                self.list_label[act_attach] = act_label
                #
                if self.is_bfs:
                    # attach, but not go down, just record in buffer
                    self.state_par = prev.state_par
                    self.state_last_cur = prev
                    self.idx_cur = prev.idx_cur
                    self.label_cur = prev.label_cur
                    self.idx_par = prev.idx_par
                    #
                    self.idxes_chs = prev.idxes_chs.copy()
                    self.idxes_chs.append(act_attach)
                    self.labels_chs = prev.labels_chs.copy()
                    self.labels_chs.append(act_label)
                    # leftmost and rightmost children or self
                    self.idx_ch_leftmost, self.idx_ch_rightmost = \
                        min(act_attach, prev.idx_ch_leftmost), max(act_attach, prev.idx_ch_rightmost)
                    self.depth = self.depth_eff = -1  # todo(warn): not used!!
                    # fill buffer for the children node
                    bfs_buffer_idx, bfs_buffer_list = prev.bfs_buffer
                    new_bfs_buffer_list = bfs_buffer_list.copy()  # todo(warn): copy to make it correct for beam-search
                    new_bfs_buffer_list.append((self, act_attach, act_label))
                    self.bfs_buffer = [bfs_buffer_idx, new_bfs_buffer_list]
                else:
                    # go down: attach
                    self.state_par = prev
                    self.state_last_cur = None
                    self.idx_cur = act_attach
                    self.label_cur = act_label
                    self.idx_par = act_head
                    self.idxes_chs = []
                    self.labels_chs = []
                    # leftmost and rightmost children or self
                    self.idx_ch_leftmost, self.idx_ch_rightmost = act_attach, act_attach
                    #
                    self.depth = self.depth_eff = prev.depth + 1
        self.not_root = 0. if (self.state_par is None) else 1.          # still have parents, not the root node
        # =====
        # other calculate as needed values
        #
        # Scorer: possible rnn hidden layers slices
        self.spine_rnn_cache = None
        self.trailing_rnn_cache = None
        # Scores: score slices
        self.arc_score_slice = None
        self.label_score_slice = None
        # for Oracle: togo child list
        self.oracle_ch_cache = None         # pair of idx in the list (next to read children idx): (ch-idx, des-idx)
        # for loss: loss_cur+loss_future
        self.oracle_loss_cache = None       # (loss-arc, loss-corr_arc-label, delta-arc, delta-label)
        # for signature and merging
        self.sig_cache = None               # (arc-array, label-array, sig-bytes)

# =====
# manage the oracle related stuffs
# TODO(+N): different oracles for bfs decoding!
class TdOracleManager:
    def __init__(self, oracle_strategy: str, oracle_projective: bool, free_dist_alpha: float, labeling_order: str):
        self.oracle_strategy = oracle_strategy
        self.free_dist_alpha = free_dist_alpha
        self.is_l2r, self.is_i2o, self.is_free, self.is_label, self.is_n2f = \
            [oracle_strategy==z for z in ["l2r", "i2o", "free", "label", "n2f"]]
        zcheck(any([self.is_l2r, self.is_i2o, self.is_free, self.is_label, self.is_n2f]),
               f"Err: UNK oracle-strategy {oracle_strategy}")
        zcheck(not oracle_projective, "Err: not implemented for projective oracle!")
        #
        self.label_order = {"core": TdOracleManager.LABELING_RANKINGS_CORE,
                            "freq": TdOracleManager.LABELING_RANKINGS_FREQ}[labeling_order]

    # todo(+N): to be improved maybe
    # by core relation -> non-core
    LABELING_RANKINGS_CORE = {v:i for i,v in enumerate([
        # core
        "nsubj", "csubj", "obj", "iobj", "ccomp", "xcomp",
        # nominal
        "nmod", "appos", "nummod", "amod", "acl",
        # non core
        "obl", "vocative", "expl", "dislocated", "advcl", "advmod", "discourse",
        # others
        "fixed", "flat", "compound", "list", "parataxis", "orphan", "goeswith", "reparandum",
        # functions
        "conj", "cc", "aux", "cop", "mark", "det", "clf", "case", "punct", "root", "dep",
    ])}
    # by english frequency
    LABELING_RANKINGS_FREQ = {v:i for i,v in enumerate(['punct', 'case', 'nmod', 'amod', 'det', 'obl', 'nsubj', 'root', 'advmod', 'conj', 'obj', 'cc', 'mark', 'aux', 'acl', 'nummod', 'flat', 'cop', 'advcl', 'xcomp', 'appos', 'compound', 'expl', 'ccomp', 'fixed', 'iobj', 'parataxis', 'dep', 'csubj', 'orphan', 'discourse', 'clf', 'goeswith', 'vocative', 'list', 'dislocated', 'reparandum'])}

    # =====
    # used for init stream
    def init_inst(self, inst: ParseInstance):
        # inst.get_children_mask_arr(False)
        inst.set_children_info(self.oracle_strategy, label_ranking_dict=self.label_order, free_dist_alpha=self.free_dist_alpha)
        return inst

    # used for refreshing before each running (currently mainly for shuffling free-mode children)
    def refresh_insts(self, insts: List[ParseInstance]):
        if self.is_free:
            for inst in insts:
                inst.shuffle_children_free()
        elif self.is_n2f:
            for inst in insts:
                inst.shuffle_children_n2f()

    # =====
    # get oracles for the input states, this procedure is same for all the modes
    # todo(+N): currently only return one oracle (determined or random)
    # todo(+N): simple oracle, but is this one optimal for TD?
    def get_oracles(self, states: List[TdState]):
        ret_nodes = []
        ret_labels = []
        for s in states:
            idx_cur = s.idx_cur
            inst = s.inst
            list_arc = s.list_arc
            state_last_cur = s.state_last_cur
            cur_ch_list = inst.children_list[idx_cur]
            cur_de_list = inst.descendant_list[idx_cur]
            len_ch_list = len(cur_ch_list)
            len_de_list = len(cur_de_list)
            # the start; note that ch_idx is the indirect index
            if state_last_cur is None:
                ch_idx = 0
                de_idx = 0
            else:
                prev_ch_idx, prev_de_idx = state_last_cur.oracle_ch_cache
                # todo(note): will not repeat since the attached one will be filtered out
                ch_idx = prev_ch_idx
                de_idx = prev_de_idx
            # find attachable one
            while ch_idx<len_ch_list and list_arc[cur_ch_list[ch_idx]]>=0:
                ch_idx += 1
            while de_idx<len_de_list and list_arc[cur_de_list[de_idx]]>=0:
                de_idx += 1
            # assign
            s.oracle_ch_cache = (ch_idx, de_idx)
            if ch_idx<len_ch_list:
                # 1) still have children to attach
                new_ch = cur_ch_list[ch_idx]
                ret_nodes.append(new_ch)
                ret_labels.append(inst.labels.idxes[new_ch])
            # elif de_idx<len_de_list:
            #     # 2) descendant, this attach decision itself is wrong, but will not lead to more errors
            elif not TdState.is_bfs and idx_cur==0 and de_idx<len_de_list:
                # todo(+N): is this more reasonable?
                # 2) let ROOT collect all unattached nodes, but still in breadth first styled as arranged in de_list
                new_ch = cur_de_list[de_idx]
                ret_nodes.append(new_ch)
                ret_labels.append(inst.labels.idxes[new_ch])
                # raise RuntimeError("Remove this when start to test advanced training strategies!")
            else:
                # 3) all descendant attached, reduce!
                ret_nodes.append(idx_cur)
                # todo(warn): 0 also as the reduce label!
                ret_labels.append(0)
        return ret_nodes, ret_labels

    # todo(warn): fill all possible oracles
    def fill_oracles(self, states: List[TdState], cands_mask_arr):
        oracle_mask_arr = np.full_like(cands_mask_arr, 0.)  # arc masking
        oracle_label_arr = np.full_like(cands_mask_arr, -1, dtype=np.int64)
        if self.is_free:
            # get all unattached children, otherwise reduce
            for sidx, s in enumerate(states):
                idx_cur = s.idx_cur
                inst = s.inst
                list_arc = s.list_arc
                cur_ch_list = inst.children_list[idx_cur]
                valid_ch_list = [z for z in cur_ch_list if list_arc[z]<0]
                if len(valid_ch_list) > 0:
                    # attach
                    oracle_mask_arr[sidx][valid_ch_list] = 1.
                elif idx_cur == 0:
                    # todo(+N): again, let the ROOT collect all unattached ones
                    unfinished_ones = [z+1 for z, a in enumerate(list_arc[1:]) if a<0]
                    oracle_mask_arr[sidx][unfinished_ones] = 1.
                else:
                    # reduce
                    oracle_mask_arr[sidx][idx_cur] = 1.
                # fill labels for the chidren
                tofill_labels = inst.labels.idxes
                oracle_label_arr[sidx][:len(tofill_labels)] = tofill_labels
        else:
            # specific order, fill the oracle one
            ret_nodes, _ = self.get_oracles(states)
            oracle_mask_arr[np.arange(len(states)), ret_nodes] = 1.
        oracle_mask_arr *= cands_mask_arr
        return oracle_mask_arr, oracle_label_arr

    # =====
    # todo(warn): this loss calculation only corresponds to free-mode!
    # for global learning or credit assignment, get the current+future loss for a state
    def set_losses(self, state: TdState):
        prev = state.prev
        if prev is None:
            state.oracle_loss_cache = (0, 0, 0, 0)
        else:
            if prev.oracle_loss_cache is None:
                self.set_losses(prev)
            loss_arc, loss_carc_label, _, _ = prev.oracle_loss_cache
            act_head, act_attach, act_label = state.action
            inst = state.inst
            list_arc = state.list_arc
            delta_arc, delta_label = 0, 0
            if act_head == act_attach:      # reduce operation which can lead to missing losses
                # todo(warn): only currently unattached children!!
                # todo(+N): might be not efficient for looping all children
                # missing_loss = max(0, len(inst.children_list[act_head]) - len(prev.idxes_chs))
                missing_loss = len([z for z in inst.children_list[act_head] if list_arc[z]<0])
                delta_arc, delta_label = missing_loss, 0
            else:
                real_head = inst.heads.vals[act_attach]
                if real_head != act_head:
                    if state.reduced[real_head]:
                        # not blamed here
                        delta_arc, delta_label = 0, 0
                    else:
                        # new attach error, but no blaming for label loss
                        delta_arc, delta_label = 1, 0
                elif inst.labels.idxes[act_attach] != act_label:
                    delta_arc, delta_label = 0, 1
            loss_arc += delta_arc
            loss_carc_label += delta_label
            state.oracle_loss_cache = (loss_arc, loss_carc_label, delta_arc, delta_label)

# =====
# calculate signatures
class TdSigManager:
    def __init__(self, sig_type:str="plain"):
        self.counter = 0
        self.positive_bias = 2      # bias to make all the indices positive to fit in unsigned value
        self.sig_plain, self.sig_none = [sig_type==z for z in ("plain", "none")]
        zcheck(self.sig_plain or self.sig_none, f"UNK sig type: {sig_type}")

    def get_sig(self, state: TdState):
        if self.sig_plain:
            return self.get_sig_plain(state)
        elif self.sig_none:
            # no signature, everyone is different!
            self.counter += 1
            return self.counter
        else:
            raise NotImplementedError()

    # ignore the sigs of all reduced children, only care about unattached nodes and concerned nodes in the spine
    def get_sig_plain(self, state: TdState):
        positive_bias = self.positive_bias
        prev = state.prev
        if prev is None:
            num_tok = state.num_tok
            atype = "B" if (num_tok+positive_bias<256) else "H"
            # todo(note): plus bias here!
            start_arr = array(atype, [-1+positive_bias] * num_tok)
            state.sig_cache = (start_arr, start_arr, b'')
        else:
            if prev.sig_cache is None:
                self.get_sig_plain(prev)
            prev_arc_array, prev_label_array, _ = prev.sig_cache
            new_arc_array, new_label_array = copy(prev_arc_array), copy(prev_label_array)
            # tell by the action
            act_head, act_attach, act_label = state.action
            if act_head == act_attach:
                # todo(note): ignore further level of children here, set 0!
                for c in state.idxes_chs:
                    new_arc_array[c] = 0
                    new_label_array[c] = 0
            else:
                # todo(note): plus bias here!
                new_arc_array[act_attach] = act_head + positive_bias
                new_label_array[act_attach] = act_label + positive_bias
            state.sig_cache = (new_arc_array, new_label_array, new_arc_array.tobytes()+new_label_array.tobytes())
        return state.sig_cache[-1]

# =====
# components

# -----
# scorer

class TdScorerConf(Conf):
    def __init__(self):
        self._input_dim = -1    # enc's last dimension
        self._num_label = -1    # number of labels
        #
        self.output_local_norm = True       # whether norm at output
        self.apply_mask_in_scores = True    # masking out the non-candidates in the scoring phase
        self.dim_label = 30
        # todo(warn): not used in Scorer, but in Selector!
        self.score_reduce_label = False   # whether include label scores for reduce-op (label=0)
        self.score_reduce_fix_zero = False  # fix reduce arc score to 0
        # space transferring
        self.arc_space = 512
        self.lab_space = 128
        # pre-computation for the head side: (cur/chsib_set/par -> spineRNN -> TrailingRNN)
        # todo(+N): pre-transformation for each type of nodes? but may be too much params.
        self.head_pre_size = 512    # size of repr after pre-computation
        self.head_pre_useff = True  # ff for the first layer? otherwise simply sum
        self.use_label_feat = True  # whether using label feature for cur and children nodes
        self.use_chsib = True   # use children node's siblings
        self.chsib_num = 5      # how many (recent) ch's siblings to consider, 0 means all: [-num:]
        self.chsib_f = "att"    # how to represent the ch set: att=attention, sum=sum, ff=feadfoward (always 0 vector if no ch)
        self.chsib_att = AttConf().init_from_kwargs(d_kqv=256, att_dropout=0.1, head_count=2)
        self.use_par = True    # use parent of current head node (grand parent)
        self.use_spine_rnn = False       # use spine rnn for the spine-line of ancestor nodes
        self.spine_rnn_type = "lstm2"
        self.use_trailing_rnn = False    # trailing rnn's size should be the same as the head's output size
        self.trailing_rnn_type = "lstm2"
        # -----
        # final biaffine scoring
        self.ff_hid_size = 0
        self.ff_hid_layer = 0
        self.use_biaffine = True
        self.use_ff = True
        self.use_ff2 = False
        self.biaffine_div = 1.
        # distance clip?
        self.arc_dist_clip = -1
        self.arc_use_neg = False

    def get_dist_aconf(self):
        return AttConf().init_from_kwargs(clip_dist=self.arc_dist_clip, use_neg_dist=self.arc_use_neg)

# another helper class for children set calculation
class TdChSibReprer(BasicNode):
    def __init__(self, pc: BK.ParamCollection, sconf: TdScorerConf):
        super().__init__(pc, None, None)
        # concat enc and label if use-label
        self.dim = (sconf._input_dim+sconf.dim_label) if sconf.use_label_feat else sconf._input_dim
        self.is_att, self.is_sum, self.is_ff = [sconf.chsib_f==z for z in ["att", "sum", "ff"]]
        self.ff_reshape = [-1, self.dim*sconf.chsib_num]        # only for ff
        if self.is_att:
            self.fnode = self.add_sub_node("fn", MultiHeadAttention(pc, self.dim, self.dim, self.dim, sconf.chsib_att))
        elif self.is_sum:
            self.fnode = None
        elif self.is_ff:
            zcheck(sconf.chsib_num>0, "Err: Cannot ff with 0 child")
            self.fnode = self.add_sub_node("fn", Affine(pc, self.dim*sconf.chsib_num, self.dim, act="elu"))
        else:
            raise NotImplementedError(f"UNK chsib method: {sconf.chsib_f}")

    def get_output_dims(self, *input_dims):
        return (self.dim, )

    def zeros(self, batch):
        return BK.zeros((batch, self.dim))

    def __call__(self, chs_input_state_t, chs_input_mem_t, chs_mask_t, chs_valid_t):
        if self.is_att:
            # [*, max-ch, size], ~, [*, 1, size], [*, max-ch] -> [*, size]
            ret = self.fnode(chs_input_mem_t, chs_input_mem_t, chs_input_state_t.unsqueeze(-2), chs_mask_t).squeeze(-2)
        # ignore head for the rest
        elif self.is_sum:
            # ignore head
            ret = (chs_input_mem_t*chs_mask_t.unsqueeze(-1)).sum(-2)
        elif self.is_ff:
            reshaped_input_state_t = (chs_input_mem_t*chs_mask_t.unsqueeze(-1)).view(self.ff_reshape)
            ret = self.fnode(reshaped_input_state_t)
        else:
            ret = None
        # out-most mask
        # [*, D_DL] * [*, 1]
        return ret * (chs_valid_t.unsqueeze(-1))

# helper class for head pre-computation
# (group1:cur/chsib_set/par + group2:spineRNN + group3:TrailingRNN)
class TdHeadReprer(BasicNode):
    def __init__(self, pc: BK.ParamCollection, sconf: TdScorerConf):
        super().__init__(pc, None, None)
        self.head_pre_size = sconf.head_pre_size
        self.use_label_feat = sconf.use_label_feat
        # padders for child nodes
        self.input_enc_dim = sconf._input_dim
        self.chs_start_posi = -sconf.chsib_num
        self.ch_idx_padder = DataPadder(2, pad_vals=0, mask_range=2)        # [*, num-ch]
        self.ch_label_padder = DataPadder(2, pad_vals=0)
        # =====
        # todo(note): here 0 for the ROOT
        self.label_embeddings = self.add_sub_node("label", Embedding(pc, sconf._num_label, sconf.dim_label, fix_row0=False))
        # =====
        # todo(note): now adopting flatten groupings for basic, spine-rnn and trailing-rnn
        # group 1: [cur_node, chsib, par_node] -> head_pre_size
        self.use_chsib = sconf.use_chsib
        self.use_par = sconf.use_par
        # cur node
        hr_input_sizes = [sconf._input_dim+sconf.dim_label] if self.use_label_feat else [sconf._input_dim]
        if sconf.use_chsib:
            self.chsib_reprer = self.add_sub_node("chsib", TdChSibReprer(pc, sconf))
            hr_input_sizes.append(self.chsib_reprer.get_output_dims()[0])
        if sconf.use_par:
            # no par label here!
            hr_input_sizes.append(sconf._input_dim)
        # group 2: Spine RNN
        self.use_spine_rnn = sconf.use_spine_rnn
        if self.use_spine_rnn:
            self.spine_rnn = self.add_sub_node("spine", RnnNode.get_rnn_node(sconf.spine_rnn_type, pc, hr_input_sizes[0], self.head_pre_size))
            hr_input_sizes.append(self.head_pre_size)
        # group 3: Trailing RNN
        self.use_trailing_rnn = sconf.use_trailing_rnn
        if self.use_trailing_rnn:
            self.trailing_rnn = self.add_sub_node("trailing", RnnNode.get_rnn_node(sconf.trailing_rnn_type, pc, hr_input_sizes[0],self.head_pre_size))
            hr_input_sizes.append(self.head_pre_size)
        # finally sum
        if sconf.head_pre_useff:
            self.final_ff = self.add_sub_node("f1", Affine(pc, hr_input_sizes, self.head_pre_size, act="elu"))
        else:
            # todo(+2)
            self.final_ff = None

    # calculating head representations, state_bidx_bases is the flattened_idx_base
    def __call__(self, states: List[TdState], state_bidx_bases_expr: BK.Expr, flattened_enc_expr: BK.Expr):
        local_bsize = len(states)
        head_repr_inputs = []
        # (group 1) [cur_node, chsib, par_node]
        # -- cur_node
        cur_idxes = [s.idx_cur for s in states]
        cur_rel_pos_t = BK.input_idx(cur_idxes)
        cur_idxes_real_t = cur_rel_pos_t + state_bidx_bases_expr
        cur_expr_node = BK.select(flattened_enc_expr, cur_idxes_real_t)     # [*, enc_size]
        if self.use_label_feat:
            cur_labels = [s.label_cur for s in states]
            cur_labels_real_t = BK.input_idx(cur_labels)
            cur_expr_label = self.label_embeddings(cur_labels_real_t)           # [*, dim_label]
            cur_enc_t = BK.concat([cur_expr_node, cur_expr_label], -1)
        else:
            cur_enc_t = cur_expr_node
        head_repr_inputs.append(cur_enc_t)          # [*, D+DL]
        # -- chsib
        if self.use_chsib:
            # slicing does not check range
            chs_idxes = [s.idxes_chs[self.chs_start_posi:] for s in states]
            chs_valid = [(0. if len(z)==0 else 1.) for z in chs_idxes]
            # padding
            padded_chs_idxes, padded_chs_mask = self.ch_idx_padder.pad(chs_idxes)       # [*, max-ch], [*, max-ch]
            # todo(warn): if all no children
            if padded_chs_idxes.shape[1] == 0:
                chs_repr = self.chsib_reprer.zeros(local_bsize)
            else:
                # select chs reprs
                padded_chs_idxes_t = BK.input_idx(padded_chs_idxes) + state_bidx_bases_expr.unsqueeze(-1)
                output_shape = [-1, BK.get_shape(padded_chs_idxes_t, -1), self.input_enc_dim]
                chs_enc_t = BK.select(flattened_enc_expr, padded_chs_idxes_t.view(-1)).view(output_shape)  # [*,max-ch,D]
                # other inputs
                chs_mask_t = BK.input_real(padded_chs_mask)  # [*, max-ch]
                chs_valid_t = BK.input_real(chs_valid)  # [*,]
                # labels
                if self.use_label_feat:
                    chs_labels = [s.labels_chs[self.chs_start_posi:] for s in states]
                    padded_chs_labels, _ = self.ch_label_padder.pad(chs_labels)         # [*, max-ch]
                    chs_labels_t = self.label_embeddings(padded_chs_labels)             # [*, max-ch, DL]
                    chs_input_mem_t = BK.concat([chs_enc_t, chs_labels_t], -1)          # [*, max-ch, D+DL]
                else:
                    chs_input_mem_t = chs_enc_t
                # calculate representations
                chs_input_state_t = cur_enc_t                                           # [*, D+DL]
                chs_repr = self.chsib_reprer(chs_input_state_t, chs_input_mem_t, chs_mask_t, chs_valid_t)  # [*, D+DL]
            #
            head_repr_inputs.append(chs_repr)       # [*, D+DL]
        # -- par_node
        if self.use_par:
            # here, no use of parent's label and mask with zero-vector for roots' parents
            par_idxes = [s.idx_par for s in states]
            par_idxes_real = BK.input_idx(par_idxes) + state_bidx_bases_expr
            par_masks = [s.not_root for s in states]
            # [*, enc_size]
            par_expr = BK.select(flattened_enc_expr, par_idxes_real) * BK.input_real(par_masks).unsqueeze(-1)
            head_repr_inputs.append(par_expr)
        # -----
        # (group 2): spine-only RNN
        if self.use_spine_rnn:
            # create zero-init states for RNN
            empty_slice = ExprWrapper.make_unit(self.spine_rnn.zero_init_hidden(1))
            # collect previous RNN states from parent states and combine
            # prev_hid_slices = [(empty_slice if s.state_par is None else s.state_par.spine_rnn_cache) for s in states]
            prev_hid_slices = [(s.state_par.spine_rnn_cache if s.not_root else empty_slice) for s in states]
            prev_hid_combined = SliceManager.combine_slices(prev_hid_slices, None)
            # RNN step
            layer2_hid_state = self.spine_rnn(cur_enc_t, prev_hid_combined, None)
            # store
            new_slices = ExprWrapper(layer2_hid_state, local_bsize).split()
            for bidx, s in enumerate(states):
                s.spine_rnn_cache = new_slices[bidx]
            # todo(note): inner side of the RNN-units
            head_repr_inputs.append(layer2_hid_state[0])
        # (group 3): all-history-trajectory RNN (the processing is similar to layer2)
        # create zero-init states for RNN
        if self.use_trailing_rnn:
            # create zero-init states for RNN
            empty_slice = ExprWrapper.make_unit(self.trailing_rnn.zero_init_hidden(1))
            # collect previous RNN states from parent states and combine
            prev_hid_slices = [(empty_slice if s.prev is None else s.prev.trailing_rnn_cache) for s in states]
            prev_hid_combined = SliceManager.combine_slices(prev_hid_slices, None)
            # RNN step
            layer3_hid_state = self.trailing_rnn(cur_enc_t, prev_hid_combined, None)
            # store
            new_slices = ExprWrapper(layer3_hid_state, local_bsize).split()
            for bidx, s in enumerate(states):
                s.trailing_rnn_cache = new_slices[bidx]
            # todo(note): inner side of the RNN-units
            head_repr_inputs.append(layer3_hid_state[0])
        # ==== finally
        final_repr = self.final_ff(head_repr_inputs)  # [*, dim]
        return final_repr, cur_rel_pos_t

# -----
# the main scorer
# three steps for scoring: enc-prepare, arc-score, label-score
class TdScorer(BasicNode):
    def __init__(self, pc: BK.ParamCollection, sconf: TdScorerConf):
        super().__init__(pc, None, None)
        # options
        input_dim = sconf._input_dim
        arc_space = sconf.arc_space
        lab_space = sconf.lab_space
        ff_hid_size = sconf.ff_hid_size
        ff_hid_layer = sconf.ff_hid_layer
        use_biaffine = sconf.use_biaffine
        use_ff = sconf.use_ff
        use_ff2 = sconf.use_ff2
        #
        head_pre_size = sconf.head_pre_size
        self.cands_padder = DataPadder(2, pad_vals=0, mask_range=2)         # [*, num-cands]
        #
        self.input_dim = input_dim
        self.num_label = sconf._num_label
        self.output_local_norm = sconf.output_local_norm
        self.apply_mask_in_scores = sconf.apply_mask_in_scores
        self.score_reduce_label = sconf.score_reduce_label
        self.score_reduce_fix_zero = sconf.score_reduce_fix_zero
        # todo(warn): the biaffine direction is the opposite in graph one, but does not matter
        biaffine_div = sconf.biaffine_div
        # attach/arc
        self.arc_m = self.add_sub_node("am", Affine(pc, input_dim, arc_space, act="elu"))
        self.arc_h = self.add_sub_node("ah", Affine(pc, head_pre_size, arc_space, act="elu"))
        self.arc_scorer = self.add_sub_node("as", BiAffineScorer(pc, arc_space, arc_space, 1, ff_hid_size, ff_hid_layer=ff_hid_layer, use_biaffine=use_biaffine, use_ff=use_ff, use_ff2=use_ff2, biaffine_div=biaffine_div))
        # only add distance for arc
        if sconf.arc_dist_clip > 0:
            self.dist_helper = self.add_sub_node("dh", AttDistHelper(pc, sconf.get_dist_aconf(), arc_space))
        else:
            self.dist_helper = None
        # labeling
        self.lab_m = self.add_sub_node("lm", Affine(pc, input_dim, lab_space, act="elu"))
        self.lab_h = self.add_sub_node("lh", Affine(pc, head_pre_size, lab_space, act="elu"))
        self.lab_scorer = self.add_sub_node("ls", BiAffineScorer(pc, lab_space, lab_space, self.num_label, ff_hid_size, ff_hid_layer=ff_hid_layer, use_biaffine=use_biaffine, use_ff=use_ff, use_ff2=use_ff2, biaffine_div=biaffine_div))
        # head preparation
        self.head_preper = self.add_sub_node("h", TdHeadReprer(pc, sconf))
        #
        # alpha for local norm
        self.local_norm_alpha = 1.0

    # step 0: pre-compute for the mod candidate nodes
    # transform to specific space (for labeling, using stacking-like architecture for the head side)
    # [*, len, input_dim] -> *[*, len, space_dim]
    def transform_space(self, enc_expr):
        # flatten for later usage for head nodes
        flattened_enc_expr = enc_expr.view([-1, self.input_dim])
        flattened_enc_stride = BK.get_shape(enc_expr, -2)        # sentence length
        # calculate for m nodes
        am_expr = self.arc_m(enc_expr)
        lm_expr = self.lab_m(enc_expr)
        am_pack = self.arc_scorer.precompute_input0(am_expr)
        lm_pack = self.lab_scorer.precompute_input0(lm_expr)
        scoring_expr_pack = (enc_expr, flattened_enc_expr, flattened_enc_stride, am_pack, lm_pack)
        return scoring_expr_pack

    # local normalization (with prob-raising alpha), return log-prob
    def local_normalize(self, scores, alpha):
        if alpha != 1.:
            scores = BK.log_softmax(scores, -1) * alpha
        return BK.log_softmax(scores, -1)

    def set_local_norm_alpha(self, alpha):
        if alpha != self.local_norm_alpha:
            self.local_norm_alpha = alpha
            zlog(f"Setting scorer's alpha to {self.local_norm_alpha}")

    # step 1: compute for arc
    # INPUT: ENC_PACK, List[*], [*], [*, slen] -> OUTPUT: score[*, slen], ARC_PACK
    # todo(note): tradeoff use masks instead of idxes to lessen CPU-burden, nearly half GPU computation is masked out
    def score_arc(self, cache_pack, states: List[TdState], state_bidxes_expr: BK.Expr, cand_masks_expr: BK.Expr):
        bsize = len(states)
        enc_expr, flattened_enc_expr, flattened_enc_stride, am_pack, lm_pack = cache_pack
        # flattened idx bases
        state_bidx_bases_expr = state_bidxes_expr * flattened_enc_stride
        # prepare head
        head_expr, rel_pos_t = self.head_preper(states, state_bidx_bases_expr, flattened_enc_expr)     # [*, head_pre_size], [*]
        ah_expr = self.arc_h(head_expr).unsqueeze(-2)                                       # [*, 1, arc_space]
        lh_expr = self.lab_h(head_expr).unsqueeze(-2)                                       # [*, 1, lab_space]
        # prepare cands
        # [*, slen, *space] (todo(note): select at batch-idx instead of flattened-idx)
        state_am_pack = [(BK.select(z, state_bidxes_expr) if z is not None else None) for z in am_pack]
        # arc distance
        if self.dist_helper is not None:
            hm_dist = rel_pos_t.unsqueeze(-1) - BK.unsqueeze(BK.arange_idx(0, flattened_enc_stride), 0)
            ah_rel1, _ = self.dist_helper.obatin_from_distance(hm_dist)    # [*, L, arc_space]
        else:
            ah_rel1 = None
        # [*, slen]
        # todo(note): effective for local-norm mode, no effects for global models since later we will exclude things
        score_candidate_mask = cand_masks_expr if self.apply_mask_in_scores else None
        # full_arc_score = self.arc_scorer.postcompute_input1(
        #     state_am_pack, ah_expr, mask0=score_candidate_mask, rel1_t=ah_rel1).squeeze(-1)
        full_arc_score = self.arc_scorer.postcompute_input1(state_am_pack, ah_expr, rel1_t=ah_rel1).squeeze(-1)
        if self.score_reduce_fix_zero:
            full_arc_score[BK.arange_idx(bsize), rel_pos_t] = 0.
        # todo(warn): mask as the final step
        if score_candidate_mask is not None:
            full_arc_score += Constants.REAL_PRAC_MIN*(1.-score_candidate_mask)
        if self.output_local_norm:
            full_arc_score = self.local_normalize(full_arc_score, self.local_norm_alpha)
        return full_arc_score, (state_bidx_bases_expr, state_bidxes_expr, cand_masks_expr, lh_expr)

    # step 2: compute for label (full or partial)
    # labeling_cand_idxes=None means scoring for all cands, otherwise partially
    # INPUT: ~, [*, len_cand]; OUTPUT: [*, ?, num-label]
    # todo(+N): if self.score_reduce_fix_zero and self.score_reduce_label?
    def score_label(self, cache_pack, cache_arc_pack, labeling_cand_idxes_expr: BK.Expr=None):
        enc_expr, flattened_enc_expr, flattened_enc_stride, _, lm_pack = cache_pack
        state_bidx_bases_expr, state_bidxes_expr, cand_masks_expr, lh_expr = cache_arc_pack
        #
        if labeling_cand_idxes_expr is None:
            # [*, slen, label] (full score, also select at batch-idx)
            state_lm_pack = [(BK.select(z, state_bidxes_expr) if z is not None else None) for z in lm_pack]
            score_candidate_mask = cand_masks_expr if self.apply_mask_in_scores else None
            full_label_score = self.lab_scorer.postcompute_input1(state_lm_pack, lh_expr, mask0=score_candidate_mask)
            if self.output_local_norm:
                full_label_score = self.local_normalize(full_label_score, self.local_norm_alpha)
            return full_label_score
        else:
            # [*, ?, label] (partial score)
            flattened_labeling_cand_idxes_expr = labeling_cand_idxes_expr + state_bidx_bases_expr.unsqueeze(-1)
            output_shape = BK.get_shape(flattened_labeling_cand_idxes_expr) + [-1]
            flat_idx = flattened_labeling_cand_idxes_expr.view(-1)   # flatten first-two dims
            selected_lm_pack = [(BK.select(z.view(-1, BK.get_shape(z, -1)), flat_idx).view(output_shape)
                                 if z is not None else None) for z in lm_pack]  # [*, ?, *space]
            selected_label_score = self.lab_scorer.postcompute_input1(selected_lm_pack, lh_expr)
            if self.output_local_norm:
                selected_label_score = self.local_normalize(selected_label_score, self.local_norm_alpha)
            return selected_label_score
# -----

# ---
# expander, get the candidates of each state for the next step
class TdExpander:
    def expand(self, s: TdState):
        raise NotImplementedError()

    @staticmethod
    def get_expander(strategy, projective):
        if projective:
            # todo(+N)
            raise NotImplementedError("Err: Projective methods have not been implemented yet.")
        else:
            _MAP = {"i2o": TdExpanderI2oNP, "l2r": TdExpanderL2rNP, "free": TdExpanderFreeNP,
                    "n2f": TdExpanderN2fNP, "n2f2": TdExpanderN2f2NP}
            return _MAP[strategy]()

# Non-projective inside to outside
class TdExpanderI2oNP(TdExpander):
    def expand(self, s: TdState):
        num_tok = s.num_tok
        idx_cur = s.idx_cur
        list_arc = s.list_arc
        idxes_chs = s.idxes_chs
        last_ch_node = idx_cur if (len(idxes_chs)==0) else idxes_chs[-1]
        if last_ch_node>idx_cur:
            # right branch
            ret = [i for i in range(last_ch_node+1, num_tok) if list_arc[i]<0]
        else:
            # left branch, but can also jump to right branch
            ret = [i for i in range(1, last_ch_node) if list_arc[i]<0] + [i for i in range(idx_cur+1, num_tok) if list_arc[i]<0]
        if idx_cur>0 or TdState.is_bfs:
            ret.append(idx_cur)
        return ret

# Non-projective left to right
class TdExpanderL2rNP(TdExpander):
    def expand(self, s: TdState):
        num_tok = s.num_tok
        idx_cur = s.idx_cur
        list_arc = s.list_arc
        idxes_chs = s.idxes_chs
        last_ch_node = 0 if (len(idxes_chs) == 0) else idxes_chs[-1]
        ret = [i for i in range(last_ch_node+1, num_tok) if list_arc[i]<0]
        if idx_cur>0 or TdState.is_bfs:
            ret.append(idx_cur)
        return ret

# Non-projective free
class TdExpanderFreeNP(TdExpander):
    def expand(self, s: TdState):
        num_tok = s.num_tok
        idx_cur = s.idx_cur
        list_arc = s.list_arc
        ret = [i for i in range(1, num_tok) if list_arc[i]<0]
        if idx_cur>0 or TdState.is_bfs:
            # including a reduce operation (todo(note): since cur_arc[cur_node] must be >=0)
            ret.append(idx_cur)
        return ret

# Non-projective n2f: strict mode
class TdExpanderN2fNP(TdExpander):
    def expand(self, s: TdState):
        num_tok = s.num_tok
        idx_cur = s.idx_cur
        list_arc = s.list_arc
        idxes_chs = s.idxes_chs
        last_ch_node = idx_cur if (len(idxes_chs) == 0) else idxes_chs[-1]
        last_ch_node_distance = abs(idx_cur-last_ch_node)
        ret = [i for i in range(1, num_tok) if (list_arc[i]<0 and abs(idx_cur-i)>=last_ch_node_distance)]
        if idx_cur>0 or TdState.is_bfs:
            ret.append(idx_cur)
        return ret

# Non-projective n2f2: non-strict mode
class TdExpanderN2f2NP(TdExpander):
    def expand(self, s: TdState):
        num_tok = s.num_tok
        idx_cur = s.idx_cur
        list_arc = s.list_arc
        ret = [i for i in range(1, s.idx_ch_leftmost) if list_arc[i]<0] + \
              [i for i in range(s.idx_ch_rightmost+1, num_tok) if list_arc[i]<0]
        if idx_cur>0 or TdState.is_bfs:
            ret.append(idx_cur)
        return ret

# ---
# local selector (scoring + local selector)
class TdLocalSelector:
    def __init__(self, scorer: TdScorer, local_arc_beam_size: int, local_label_beam_size: int, oracle_manager=None):
        self.scorer: TdScorer = scorer
        self.num_label = self.scorer.num_label
        self.score_reduce_label = self.scorer.score_reduce_label
        # for mode with topk
        self.local_arc_beam_size = local_arc_beam_size
        self.local_label_beam_size = min(local_label_beam_size, self.scorer.num_label)
        # for mode with oracle
        self.oracle_manager: TdOracleManager = oracle_manager
        # for ss-mode
        self.rand_stream = Random.stream(Random.random_sample)

    # return selected states
    # todo(warn): in mixed mode, run _score_and_choose in the batch, and ignore useless ones
    def select(self, ags: List[BfsLinearAgenda], candidates, cache_pack):
        raise NotImplementedError()

    # common scoring and selecting procedure (three modes, can select more for mixing)
    # flatten out all the score expressions
    # todo(warn): beam_size is only useful for topk, sampling always obtain 1 selection
    # todo(warn): no score modification (+margin) in this local step!
    # todo(warn): first arc-topk then label, will this lead to search-error?
    def _score_and_choose(self, candidates, cache_pack, if_topk, if_tsample, if_sample, if_oracle,
                          selection_mask_arr=None, log_prob_sum=False):
        # bsize * (TdState, orig-batch-idxes, candidate-masks)
        flattened_states, state_bidxes, cands_mask_arr = candidates
        state_bidxes_expr = BK.input_idx(state_bidxes)
        cand_masks_expr = BK.input_real(cands_mask_arr)
        # =====
        # First: Common arc score: [*, slen]
        full_arc_score, cache_arc_pack = self.scorer.score_arc(cache_pack, flattened_states, state_bidxes_expr, cand_masks_expr)
        bsize, slen = BK.get_shape(full_arc_score)
        ew_full_arc_score = ExprWrapper(full_arc_score.view(-1), bsize*slen)        # [bsize*slen]
        nlabel = self.num_label
        # todo(note): add masking (not inplaced) here for topk selection, after possible local-norm,
        #  the one before local-norm is controlled by the flag in scorer
        selection_mask_expr = cand_masks_expr if (selection_mask_arr is None) else BK.input_real(selection_mask_arr)
        if log_prob_sum:
            assert selection_mask_arr is not None
            oracle_log_prob_sum = (BK.exp(full_arc_score) * selection_mask_expr).sum(-1).log()  # [bsize,]
        else:
            oracle_log_prob_sum = None
        full_arc_score = full_arc_score + Constants.REAL_PRAC_MIN*(1.-selection_mask_expr)
        # =====
        # Rest: different selecting method: (arc-K, label-K, scores_arc, idxes_arc, ew_label, scores_label, idxes_label)
        select_rets = [None] * 4
        # -----
        # 1/2. topk/topk-sample selection
        if if_topk or if_tsample:
            arc_topk = min(slen-1, self.local_arc_beam_size)        # todo(note): cannot select 0 (root) as child
            # [*, arc_topk]
            # todo(warn): be careful that there can be log0=-inf, which might lead to NAN?
            topk_arc_scores, topk_arc_idxes = BK.topk(full_arc_score, arc_topk, dim=-1, sorted=False)
            # -----
            # 1. topk
            if if_topk:
                # label score: [*, arc_topk, nlabel]
                topk_label_score = self.scorer.score_label(cache_pack, cache_arc_pack, topk_arc_idxes)
                ew_topk_label_score = ExprWrapper(topk_label_score.view(-1), bsize*arc_topk*nlabel)  # [bs*arcK*nl]
                label_topk = self.local_label_beam_size
                # topk^2 [*, arc_topk, label_topk]
                topk2_label_scores, topk2_label_idxes = BK.topk(topk_label_score, label_topk, dim=-1, sorted=False)
                select_rets[0] = (arc_topk, label_topk, topk_arc_scores, topk_arc_idxes,
                                  ew_topk_label_score, topk2_label_scores, topk2_label_idxes)
            # 2. topk_sample (use gumble)
            if if_tsample:
                # sampled arc
                tsample_arc_scores, tsample_arc_local_idxes = BK.category_sample(topk_arc_scores)
                tsample_arc_idxes = BK.gather(topk_arc_idxes, tsample_arc_local_idxes, -1)
                # label: [*, 1, nlabel]
                tsample_label_score = self.scorer.score_label(cache_pack, cache_arc_pack, tsample_arc_idxes)
                ew_tsample_label_score = ExprWrapper(tsample_label_score.view(-1), bsize*nlabel)  # [bs*1*nl]
                # todo(+2): directly sample here for simplicity: [*, 1, 1]
                tsample2_label_scores, tsample2_label_idxes = BK.category_sample(tsample_label_score, -1)
                select_rets[1] = (1, 1, tsample_arc_scores, tsample_arc_idxes,
                                  ew_tsample_label_score, tsample2_label_scores, tsample2_label_idxes)
        # -----
        # 3. sampling (use gumble)
        # todo(+N): current only support 1 sampling instance
        if if_sample:
            # sampled arc [*, 1]
            sample_arc_scores, sample_arc_idxes = BK.category_sample(full_arc_score)
            # scoring labels [*, 1, nlabel]
            sample_label_score = self.scorer.score_label(cache_pack, cache_arc_pack, sample_arc_idxes)
            ew_sample_label_score = ExprWrapper(sample_label_score.view(-1), bsize*nlabel)
            # sampled labels [*, 1, 1]
            sample2_label_scores, sample2_label_idxes = BK.category_sample(sample_label_score, -1)
            select_rets[2] = (1, 1, sample_arc_scores, sample_arc_idxes,
                              ew_sample_label_score, sample2_label_scores, sample2_label_idxes)
        # -----
        # 4. oracle (need external help)
        # todo(+N): current only support 1 oracle instance
        if if_oracle:
            oracle_arc_idxes, oracle_label_idxes = self.oracle_manager.get_oracles(flattened_states)
            # [*, 1], [*, 1, 1]
            oracle_arc_idxes, oracle_label_idxes = BK.input_idx(oracle_arc_idxes).unsqueeze(-1), \
                                                   BK.input_idx(oracle_label_idxes).unsqueeze(-1).unsqueeze(-1)
            # arc scores [*, 1]
            oracle_arc_scores = BK.gather(full_arc_score, oracle_arc_idxes)
            # label all scores [*, 1, nlabel]
            oracle_label_score = self.scorer.score_label(cache_pack, cache_arc_pack, oracle_arc_idxes)
            ew_oracle_label_score = ExprWrapper(oracle_label_score.view(-1), bsize*nlabel)
            # label scores [*, 1, 1]
            oracle2_label_scores = BK.gather(oracle_label_score, oracle_label_idxes)
            select_rets[3] = (1, 1, oracle_arc_scores, oracle_arc_idxes,
                              ew_oracle_label_score, oracle2_label_scores, oracle_label_idxes)
        # return them all
        return (bsize, slen, nlabel, ew_full_arc_score), select_rets, oracle_log_prob_sum

    # similar to the previous one but with special mode!
    def _score_and_sample_oracle(self, candidates, cache_pack, sel_mask_arr, sel_label_arr, log_prob_sum):
        # bsize * (TdState, orig-batch-idxes, candidate-masks)
        flattened_states, state_bidxes, cands_mask_arr = candidates
        state_bidxes_expr = BK.input_idx(state_bidxes)
        cand_masks_expr = BK.input_real(cands_mask_arr)
        # =====
        # First: Common arc score: [*, slen]
        full_arc_score, cache_arc_pack = self.scorer.score_arc(cache_pack, flattened_states, state_bidxes_expr, cand_masks_expr)
        bsize, slen = BK.get_shape(full_arc_score)
        ew_full_arc_score = ExprWrapper(full_arc_score.view(-1), bsize * slen)  # [bsize*slen]
        nlabel = self.num_label
        #
        selection_mask_expr = BK.input_real(sel_mask_arr)
        sel_label_expr = BK.input_idx(sel_label_arr)  # [bsize, LEN]
        if log_prob_sum:
            assert sel_mask_arr is not None
            oracle_log_prob_sum = (BK.exp(full_arc_score) * selection_mask_expr).sum(-1).log()  # [bsize,]
        else:
            oracle_log_prob_sum = None
        full_arc_score = full_arc_score + Constants.REAL_PRAC_MIN * (1. - selection_mask_expr)
        # step 1, sample arc
        # sampled arc [*, 1]
        sample_arc_scores, sample_arc_idxes = BK.category_sample(full_arc_score)
        # scoring labels [*, 1, nlabel]
        sample_label_score = self.scorer.score_label(cache_pack, cache_arc_pack, sample_arc_idxes)
        ew_sample_label_score = ExprWrapper(sample_label_score.view(-1), bsize * nlabel)
        # get oracle labels [*, 1, 1]
        oracle_label_idxes = BK.gather(sel_label_expr, sample_arc_idxes).unsqueeze(-1)
        oracle2_label_scores = BK.gather(sample_label_score, oracle_label_idxes)
        select_rets = [(1, 1, sample_arc_scores, sample_arc_idxes,
                              ew_sample_label_score, oracle2_label_scores, oracle_label_idxes)]
        # return them all
        return (bsize, slen, nlabel, ew_full_arc_score), select_rets, oracle_log_prob_sum

    # =====
    # generate and fill-in the new candidate states from the local selections
    # -- return list[(beam_expands, gbeam_expands)]

    # common procedure for one state
    def _add_states(self, cands: List, state: TdState, nlabel, arc_k, label_k, cands_mask_arr_s, arc_scores_arr_s, arc_idxes_arr_s, label_scores_arr_s, label_idxes_arr_s, ew_arc, ew_label, ew_arc_idx_base, ew_label_idx_base, label_all_scores_t):
        aidx = 0
        state_cur_idx = state.idx_cur
        assert len(arc_scores_arr_s)==arc_k and len(arc_idxes_arr_s)==arc_k, "Err: Wrong arc-k"
        for one_arc_score, one_arc_idx in zip(arc_scores_arr_s, arc_idxes_arr_s):
            one_arc_score, one_arc_idx = float(one_arc_score), int(one_arc_idx)
            # only allow legal cands, also let reduce operations have labels (but not outputed)!
            # todo(+N): this means that reduce on ROOT is filtered out!
            if cands_mask_arr_s[one_arc_idx]>0.:
                arc_score_slice = SlicedExpr(ew_arc, ew_arc_idx_base + one_arc_idx)
                cur_ew_label_idx_base = ew_label_idx_base + aidx * nlabel
                if one_arc_idx == state_cur_idx:
                    if self.score_reduce_label:
                        # todo(warn): reduce operation, only accept padding label! # todo(+N): is this one expensive?
                        reduce_label_score = (label_all_scores_t[cur_ew_label_idx_base]).item()      # L-scores[base+0]
                        reduce_label_slice = SlicedExpr(ew_label, cur_ew_label_idx_base)
                    else:
                        reduce_label_score = 0.
                        reduce_label_slice = None
                    new_state = TdState(prev=state, action=(one_arc_idx, one_arc_idx, 0), score=(one_arc_score+reduce_label_score))
                    new_state.label_score_slice = reduce_label_slice
                    new_state.arc_score_slice = arc_score_slice
                    cands.append(new_state)
                else:  # loop for all the labels
                    assert len(label_scores_arr_s[aidx])==label_k and len(label_idxes_arr_s[aidx])==label_k, "Err: wrong label-K"
                    for one_label_score, one_label_idx in zip(label_scores_arr_s[aidx], label_idxes_arr_s[aidx]):
                        one_label_score, one_label_idx = float(one_label_score), int(one_label_idx)
                        new_state = TdState(prev=state, action=(state_cur_idx, one_arc_idx, one_label_idx),
                                            score=(one_arc_score+one_label_score))
                        new_state.arc_score_slice = arc_score_slice
                        new_state.label_score_slice = SlicedExpr(ew_label, cur_ew_label_idx_base + one_label_idx)
                        cands.append(new_state)
            aidx += 1

    # mode 1: only on plain beam (for topk/sample/oracle)
    def _new_states_plain(self, ags: List[BfsLinearAgenda], cands_mask_arr: np.ndarray, general_pack, plain_pack, local_gold_collect):
        bsize, slen, nlabel, ew_full_arc_score = general_pack
        arc_k, label_k, arc_scores, arc_idxes, ew_label_scores, label_scores, label_idxes = plain_pack
        stride_ew_label = arc_k*nlabel
        # -----
        # ew_arc: [bsize*slen], arc_scores: [bsize, arc_k],
        # ew_label: [bsize*arc_k*nlabel], label_scores: [bsize, ark_k, label_k]
        arc_scores_arr, arc_idxes_arr, label_scores_arr, label_idxes_arr = \
            BK.get_value(arc_scores), BK.get_value(arc_idxes), BK.get_value(label_scores), BK.get_value(label_idxes)
        label_alls_t = BK.get_cpu_tensor(ew_label_scores.val)
        rets = []
        bidx = 0
        for ag in ags:
            pcands = []
            # plain beam
            for state in ag.beam:
                self._add_states(pcands, state, nlabel, arc_k, label_k, cands_mask_arr[bidx], arc_scores_arr[bidx],
                                 arc_idxes_arr[bidx], label_scores_arr[bidx], label_idxes_arr[bidx],
                                 ew_full_arc_score, ew_label_scores, bidx*slen, bidx*stride_ew_label, label_alls_t)
                bidx += 1
            # gold beam
            assert len(ag.gbeam) == 0, "Wrong mode"
            # bidx += len(ag.gbeam)
            rets.append((pcands, []))
            # todo(warn): stat for debugging-like purpose
            # if self.oracle_manager is not None:
            #     for cur_cand_sample in pcands:
            #         self._stat_ss(cur_cand_sample, None)
            if local_gold_collect:
                ag.local_golds.extend(pcands)
        assert bidx == bsize, "Unmatched dim0 size!"
        return rets

    # mode 2: scheduled sampling (for sample-oracle)
    def _new_states_ss(self, ags: List[BfsLinearAgenda], cands_mask_arr: np.ndarray, general_pack, plain_pack, gold_pack, rate_plain, ss_strict_oracle: bool, ss_include_correct_rate: float):
        bsize, slen, nlabel, ew_full_arc_score = general_pack
        arc_k1, label_k1, arc_scores1, arc_idxes1, ew_label_scores1, label_scores1, label_idxes1 = plain_pack
        arc_k2, label_k2, arc_scores2, arc_idxes2, ew_label_scores2, label_scores2, label_idxes2 = gold_pack
        stride_ew_label1 = arc_k1 * nlabel
        stride_ew_label2 = arc_k2 * nlabel
        # -----
        # ew_arc: [bsize*slen], arc_scores: [bsize, arc_k],
        # ew_label: [bsize*arc_k*nlabel], label_scores: [bsize, ark_k, label_k]
        arc_scores_arr1, arc_idxes_arr1, label_scores_arr1, label_idxes_arr1 = \
            BK.get_value(arc_scores1), BK.get_value(arc_idxes1), BK.get_value(label_scores1), BK.get_value(label_idxes1)
        arc_scores_arr2, arc_idxes_arr2, label_scores_arr2, label_idxes_arr2 = \
            BK.get_value(arc_scores2), BK.get_value(arc_idxes2), BK.get_value(label_scores2), BK.get_value(label_idxes2)
        label_alls_t1 = BK.get_cpu_tensor(ew_label_scores1.val)
        label_alls_t2 = BK.get_cpu_tensor(ew_label_scores2.val)
        rets = []
        bidx = 0
        for ag in ags:
            pcands = []
            # plain beam
            for state in ag.beam:
                cands1 = []
                cands2 = []
                self._add_states(cands1, state, nlabel, arc_k1, label_k1, cands_mask_arr[bidx], arc_scores_arr1[bidx],
                                 arc_idxes_arr1[bidx], label_scores_arr1[bidx], label_idxes_arr1[bidx],
                                 ew_full_arc_score, ew_label_scores1, bidx*slen, bidx*stride_ew_label1, label_alls_t1)
                self._add_states(cands2, state, nlabel, arc_k2, label_k2, cands_mask_arr[bidx], arc_scores_arr2[bidx],
                                 arc_idxes_arr2[bidx], label_scores_arr2[bidx], label_idxes_arr2[bidx],
                                 ew_full_arc_score, ew_label_scores2, bidx*slen, bidx*stride_ew_label2, label_alls_t2)
                # =====
                assert len(cands1)==1 and len(cands2)==1, "Too many cands for scheduled sampling mode!"
                if ss_strict_oracle:
                    # for local losses
                    ag.local_golds.extend(cands2)
                    # next expanding: sampling with rate
                    if next(self.rand_stream) < rate_plain:
                        pcands.extend(cands1)
                    else:
                        pcands.extend(cands2)
                else:
                    cur_cand_sample, cur_cand_oracle = cands1[0], cands2[0]
                    self.oracle_manager.set_losses(cur_cand_sample)
                    _, _, cur_delta_arc, cur_delta_label = cur_cand_sample.oracle_loss_cache
                    if (cur_delta_arc+cur_delta_label)==0:
                        # if sample a good state, always follow it, but whether include it?
                        if next(self.rand_stream) < ss_include_correct_rate:
                            ag.local_golds.append(cur_cand_sample)
                        pcands.append(cur_cand_sample)
                    else:
                        ag.local_golds.append(cur_cand_oracle)
                        pcands.append(cur_cand_sample if (next(self.rand_stream) < rate_plain) else cur_cand_oracle)
                    # todo(warn): stat for debugging-like purpose
                    # self._stat_ss(cur_cand_sample, cur_cand_oracle)
                # =====
                bidx += 1
            # gold beam
            assert len(ag.gbeam) == 0, "Wrong mode"
            # bidx += len(ag.gbeam)
            rets.append((pcands, []))
        assert bidx == bsize, "Unmatched dim0 size!"
        return rets

    def _stat_ss(self, cur_cand_sample, cur_cand_oracle):
        GLOBAL_RECORDER.record_kv("ss_step", 1)
        self.oracle_manager.set_losses(cur_cand_sample)
        cur_loss_arc, cur_loss_label, cur_delta_arc, cur_delta_label = [str(z) for z in cur_cand_sample.oracle_loss_cache]
        GLOBAL_RECORDER.record_kv("ss_lossda_" + cur_delta_arc, 1)
        GLOBAL_RECORDER.record_kv("ss_lossdl_" + cur_delta_label, 1)
        # GLOBAL_RECORDER.record_kv("ss_lossla_"+cur_loss_arc, 1)
        # GLOBAL_RECORDER.record_kv("ss_lossll_"+cur_loss_label, 1)
        if cur_cand_sample.length >= 2 * cur_cand_sample.num_tok - 2:
            GLOBAL_RECORDER.record_kv("ss_end", 1)
            GLOBAL_RECORDER.record_kv("ss_endla_" + cur_loss_arc, 1)
            GLOBAL_RECORDER.record_kv("ss_endll_" + cur_loss_label, 1)

    # mode 3: mixed plain+gold (for global learning with topk/sample + oracle)
    def _new_states_both(self, ags: List[BfsLinearAgenda], cands_mask_arr: np.ndarray, general_pack, plain_pack, gold_pack):
        bsize, slen, nlabel, ew_full_arc_score = general_pack
        arc_k1, label_k1, arc_scores1, arc_idxes1, ew_label_scores1, label_scores1, label_idxes1 = plain_pack
        arc_k2, label_k2, arc_scores2, arc_idxes2, ew_label_scores2, label_scores2, label_idxes2 = gold_pack
        stride_ew_label1 = arc_k1 * nlabel
        stride_ew_label2 = arc_k2 * nlabel
        # -----
        # ew_arc: [bsize*slen], arc_scores: [bsize, arc_k],
        # ew_label: [bsize*arc_k*nlabel], label_scores: [bsize, ark_k, label_k]
        arc_scores_arr1, arc_idxes_arr1, label_scores_arr1, label_idxes_arr1 = \
            BK.get_value(arc_scores1), BK.get_value(arc_idxes1), BK.get_value(label_scores1), BK.get_value(label_idxes1)
        arc_scores_arr2, arc_idxes_arr2, label_scores_arr2, label_idxes_arr2 = \
            BK.get_value(arc_scores2), BK.get_value(arc_idxes2), BK.get_value(label_scores2), BK.get_value(label_idxes2)
        label_alls_t1 = BK.get_cpu_tensor(ew_label_scores1.val)
        label_alls_t2 = BK.get_cpu_tensor(ew_label_scores2.val)
        rets = []
        bidx = 0
        for ag in ags:
            pcands = []
            gcands = []
            # plain beam
            for state in ag.beam:
                # plain expanding
                self._add_states(pcands, state, nlabel, arc_k1, label_k1, cands_mask_arr[bidx], arc_scores_arr1[bidx],
                                 arc_idxes_arr1[bidx], label_scores_arr1[bidx], label_idxes_arr1[bidx],
                                 ew_full_arc_score, ew_label_scores1, bidx*slen, bidx*stride_ew_label1, label_alls_t1)
                # todo(note): plain+oracle also goes to gbeam: let the GlobalSelector decide.
                self._add_states(gcands, state, nlabel, arc_k2, label_k2, cands_mask_arr[bidx], arc_scores_arr2[bidx],
                                 arc_idxes_arr2[bidx], label_scores_arr2[bidx], label_idxes_arr2[bidx],
                                 ew_full_arc_score, ew_label_scores2, bidx*slen, bidx*stride_ew_label2, label_alls_t2)
                bidx += 1
            # gold beam
            for state in ag.gbeam:
                # todo(note): only add gold+oracle
                self._add_states(gcands, state, nlabel, arc_k2, label_k2, cands_mask_arr[bidx], arc_scores_arr2[bidx],
                                 arc_idxes_arr2[bidx], label_scores_arr2[bidx], label_idxes_arr2[bidx],
                                 ew_full_arc_score, ew_label_scores2, bidx*slen, bidx*stride_ew_label2, label_alls_t2)
                bidx += 1
            rets.append((pcands, gcands))
        assert bidx == bsize, "Unmatched dim0 size!"
        return rets

# pure topk, for decoding
class TdLocalSelectorTopk(TdLocalSelector):
    def __init__(self, scorer: TdScorer, local_arc_beam_size: int, local_label_beam_size: int, oracle_manager=None, force_oracle: bool=False):
        super().__init__(scorer, local_arc_beam_size, local_label_beam_size, oracle_manager)
        self.force_oracle = force_oracle

    def select(self, ags: List[BfsLinearAgenda], candidates, cache_pack):
        flattened_states, _, cands_mask_arr = candidates
        sel_arr, _ = self.oracle_manager.fill_oracles(flattened_states, cands_mask_arr) if self.force_oracle else (None, None)
        general_pack, more_packs, _ = self._score_and_choose(candidates, cache_pack, True, False, False, False, sel_arr)
        return self._new_states_plain(ags, cands_mask_arr, general_pack, more_packs[0], False)

# pure oracle, for basic static learning
class TdLocalSelectorOracle(TdLocalSelector):
    def __init__(self, scorer: TdScorer, local_arc_beam_size: int, local_label_beam_size: int, oracle_manager, log_sum_prob):
        super().__init__(scorer, local_arc_beam_size, local_label_beam_size, oracle_manager)
        self.log_sum_prob = log_sum_prob

    def select(self, ags: List[BfsLinearAgenda], candidates, cache_pack):
        lps = self.log_sum_prob
        flattened_states, _, cands_mask_arr = candidates
        sel_arr, _ = self.oracle_manager.fill_oracles(flattened_states, cands_mask_arr) if lps else (None, None)
        general_pack, more_packs, oracle_log_prob_sum = \
            self._score_and_choose(candidates, cache_pack, False, False, False, True, sel_arr, lps)
        rets = self._new_states_plain(ags, cands_mask_arr, general_pack, more_packs[3], True)
        # todo(+2): repeated codes
        # todo(warn): replace scores with log-prob-sum, states returned are exactly gold-states in this mode
        if lps:
            bsize, = BK.get_shape(oracle_log_prob_sum)
            lps_full_arc_score = ExprWrapper(oracle_log_prob_sum, bsize)
            bidx = 0
            for pcands, _ in rets:
                for p in pcands:
                    p.arc_score_slice = SlicedExpr(lps_full_arc_score, bidx)
                    bidx += 1
            assert bidx == bsize
        return rets

# pure sampler
class TdLocalSelectorSample(TdLocalSelector):
    def __init__(self, scorer: TdScorer, local_arc_beam_size: int, local_label_beam_size: int, tsample: bool, oracle_manager=None):
        super().__init__(scorer, local_arc_beam_size, local_label_beam_size, oracle_manager)
        self.tsample = tsample

    def select(self, ags: List[BfsLinearAgenda], candidates, cache_pack):
        cands_mask_arr = candidates[-1]
        if self.tsample:
            general_pack, more_packs, _ = self._score_and_choose(candidates, cache_pack, False, True, False, False)
            return self._new_states_plain(ags, cands_mask_arr, general_pack, more_packs[1], False)
        else:
            general_pack, more_packs, _ = self._score_and_choose(candidates, cache_pack, False, False, True, False)
            return self._new_states_plain(ags, cands_mask_arr, general_pack, more_packs[2], False)

# oracle sampler
class TdLocalSelectorOracleSample(TdLocalSelector):
    def __init__(self, scorer: TdScorer, local_arc_beam_size: int, local_label_beam_size: int, oracle_manager, log_sum_prob):
        super().__init__(scorer, local_arc_beam_size, local_label_beam_size, oracle_manager)
        self.log_sum_prob = log_sum_prob

    def select(self, ags: List[BfsLinearAgenda], candidates, cache_pack):
        flattened_states, _, cands_mask_arr = candidates
        sel_arr, sel_labs = self.oracle_manager.fill_oracles(flattened_states, cands_mask_arr)
        lps = self.log_sum_prob
        general_pack, more_packs, oracle_log_prob_sum = self._score_and_sample_oracle(candidates, cache_pack, sel_arr, sel_labs, lps)
        rets = self._new_states_plain(ags, cands_mask_arr, general_pack, more_packs[0], False)
        # todo(warn): replace scores with log-prob-sum
        if lps:
            bsize, = BK.get_shape(oracle_log_prob_sum)
            lps_full_arc_score = ExprWrapper(oracle_log_prob_sum, bsize)
            bidx = 0
            for pcands, _ in rets:
                for p in pcands:
                    p.arc_score_slice = SlicedExpr(lps_full_arc_score, bidx)
                    bidx += 1
            assert bidx == bsize
        return rets

# sample (with or without oracle) with 1 instance, for (scheduled) sampling
class TdLocalSelectorSS(TdLocalSelector):
    def __init__(self, scorer: TdScorer, local_arc_beam_size: int, local_label_beam_size: int, oracle_manager, ss_rate_plain: ScheduledValue, tsample: bool, ss_strict_oracle: bool, ss_include_correct_rate: float):
        super().__init__(scorer, local_arc_beam_size, local_label_beam_size, oracle_manager)
        self.tsample = tsample
        self.ss_rate_plain = ss_rate_plain
        self.ss_strict_oracle = ss_strict_oracle
        self.ss_include_correct_rate = ss_include_correct_rate

    def select(self, ags: List[BfsLinearAgenda], candidates, cache_pack):
        rate_plain = self.ss_rate_plain.value
        sso = self.ss_strict_oracle
        sic = self.ss_include_correct_rate
        cands_mask_arr = candidates[-1]
        if rate_plain<=0.:  # pure oracle mode
            general_pack, more_packs, _ = self._score_and_choose(candidates, cache_pack, False, False, False, True)
            rs = self._new_states_plain(ags, cands_mask_arr, general_pack, more_packs[3], True)
        elif self.tsample:
            general_pack, more_packs, _ = self._score_and_choose(candidates, cache_pack, False, True, False, True)
            rs = self._new_states_ss(ags, cands_mask_arr, general_pack, more_packs[1], more_packs[3], rate_plain, sso, sic)
        else:
            general_pack, more_packs, _ = self._score_and_choose(candidates, cache_pack, False, False, True, True)
            rs = self._new_states_ss(ags, cands_mask_arr, general_pack, more_packs[2], more_packs[3], rate_plain, sso, sic)
        return rs

# TODO(+N)
class TdLocalSelectorMixed(TdLocalSelector):
    pass

# ---
# global selector/arranger (ranking the beam)
class TdGlobalArranger:
    def __init__(self, global_beam_size: int):
        self.global_beam_size = global_beam_size

    def arrange(self, ags: List[BfsLinearAgenda], local_cands: List):
        raise NotImplementedError()

    @staticmethod
    def create(global_beam_size: int, merge_sig_type: str, attach_num_beam_size: int):
        if merge_sig_type == "none" and attach_num_beam_size>=global_beam_size:
            ret = TdGlobalArrangerPlain(global_beam_size)
        else:
            ret = TdGlobalArrangerMerge(global_beam_size, merge_sig_type, attach_num_beam_size)
        zlog(f"Create GlobalArranger: {ret}")
        return ret

# plain mode (no mixing)
class TdGlobalArrangerPlain(TdGlobalArranger):
    def __init__(self, global_beam_size: int):
        super().__init__(global_beam_size)

    def arrange(self, ags: List[BfsLinearAgenda], local_cands: List):
        batch_len = len(ags)
        selections = [None] * batch_len
        assert batch_len == len(local_cands)
        for batch_idx in range(batch_len):
            pcands = local_cands[batch_idx][0]      # # ignore gcands
            pcands.sort(key=lambda x: x.score_accu, reverse=True)
            selections[batch_idx] = (pcands[:self.global_beam_size], None)
        return selections

# slightly advanced mode: with second-beam and merging
class TdGlobalArrangerMerge(TdGlobalArranger):
    def __init__(self, global_beam_size: int, merge_sig_type: str, attach_num_beam_size: int):
        super().__init__(global_beam_size)
        #
        self.sig_manager = TdSigManager(merge_sig_type)
        self.attach_num_beam_size = attach_num_beam_size

    def arrange(self, ags: List[BfsLinearAgenda], local_cands: List):
        batch_len = len(ags)
        selections = [None] * batch_len
        assert batch_len == len(local_cands)
        global_beam_size, sig_manager, attach_num_beam_size = \
            self.global_beam_size, self.sig_manager, self.attach_num_beam_size
        # go
        for batch_idx in range(batch_len):
            pcands = local_cands[batch_idx][0]      # ignore gcands
            pcands.sort(key=lambda x: x.score_accu, reverse=True)
            # select with merging
            scands = []
            beam_sig, beam_attach_num = {}, {}      # bytes->State, int->int
            for one_cand in pcands:
                # secondary attach_num beam
                one_remain_num = one_cand.num_rest
                already_count = beam_attach_num.get(one_remain_num, 0)
                if already_count >= attach_num_beam_size:
                    continue
                beam_attach_num[one_remain_num] = already_count + 1
                # sig beam
                # todo(+N): more secondary beams?
                one_sig = sig_manager.get_sig(one_cand)
                merger_state = beam_sig.get(one_sig, None)
                if merger_state is None:
                    beam_sig[one_sig] = one_cand
                else:
                    one_cand.merge_by(merger_state)     # merge by the higher-scored one
                    continue
                # adding
                scands.append(one_cand)
                # finally main beam
                if len(scands) >= global_beam_size:
                    break
            selections[batch_idx] = (scands, None)
        return selections

# TODO(+N) mixing mode / allow-merge (for advanced decoding and training)

# -----
# final step, end once loop and prepare for the next
class TdEnder:
    def end(self, ags: List[BfsLinearAgenda], selections):
        raise NotImplementedError()

# plain mode
class TdEnderPlain(TdEnder):
    def end(self, ags: List[BfsLinearAgenda], selections):
        for ag, one_selections in zip(ags, selections):
            new_beam = []
            for s in one_selections[0]:
                ending_length = (2*s.num_tok-1) if TdState.is_bfs else (2*s.num_tok-2)
                # ending_length = (2*s.num_tok-2)
                # todo(note): finish all 2*n-2 steps, can be different than num_rest>0 for global models
                if s.length >= ending_length:
                    s.mark_end()
                    ag.ends.append(s)
                else:
                    new_beam.append(s)
            # replace and record last one
            if len(ag.beam) > 0:
                ag.last_beam = ag.beam
            ag.beam = new_beam

# =====
# the overall search helper or actually the searcher
class TdSearcher(BfsLinearSearcher):
    def __int__(self):
        # components
        self.expander: TdExpander = None
        self.local_selector: TdLocalSelector = None
        self.global_arranger: TdGlobalArranger = None
        self.ender: TdEnder = None
        # cache for the scorer
        self.cache_pack = None
        self.max_length = -1
        # TODO(+N): where to add margin?
        self.margin = 0.

    # setup for each search
    def refresh(self, cache_pack):
        self.cache_pack = cache_pack
        self.max_length = cache_pack[2]         # todo(+1): not elegant
        # todo(+N): need to refresh other things?

    # expand candidates for each state to get scores
    def expand(self, ags: List[BfsLinearAgenda]):
        expander = self.expander
        this_bsize = sum(len(z.beam) + len(z.gbeam) for z in ags)          # current flattened bsize
        #
        state_bidxes = np.zeros(this_bsize, dtype=np.int32)
        cands_mask_arr = np.zeros([this_bsize, self.max_length], dtype=np.float32)
        flattened_states = []
        #
        bidx = 0
        for sidx, ag in enumerate(ags):
            # add all states
            for one_beam in [ag.beam, ag.gbeam]:
                flattened_states.extend(one_beam)
                for state in one_beam:
                    one_cands = expander.expand(state)
                    state_bidxes[bidx] = sidx
                    cands_mask_arr[bidx][one_cands] = 1.
                    bidx += 1
        return (flattened_states, state_bidxes, cands_mask_arr)

    # score and select
    def select(self, ags: List[BfsLinearAgenda], candidates):
        # step 1: scoring and local select
        flattened_states, state_bidxes, cands_mask_arr = candidates
        # List[(pcands, gcands)]
        local_selections = self.local_selector.select(ags, candidates, self.cache_pack)
        # step 2: global select with the local candidates
        # List[List(cands)]
        if self.global_arranger is None:
            # for single-instance ones
            global_selections = local_selections
        else:
            global_selections = self.global_arranger.arrange(ags, local_selections)
        return global_selections

    # preparing for next loop or ending
    def end(self, ags: List[BfsLinearAgenda], selections):
        self.ender.end(ags, selections)

    # =====
    # put all components together and get a Searcher

    # 1. pure beam searcher for decoding
    @staticmethod
    def create_beam_searcher(scorer: TdScorer, oracle_manager, iconf, force_oracle: bool):
        searcher = TdSearcher()
        searcher.expander = TdExpander.get_expander(iconf.expand_strategy, iconf.expand_projective)
        searcher.local_selector = TdLocalSelectorTopk(scorer, iconf.local_arc_beam_size, iconf.local_label_beam_size, oracle_manager, force_oracle)
        # todo(+N): merge?
        searcher.global_arranger = TdGlobalArranger.create(iconf.global_beam_size, iconf.merge_sig_type, iconf.attach_num_beam_size)
        searcher.ender = TdEnderPlain()
        return searcher

    # 2. pure oracle-follower for teacher forcing like training
    @staticmethod
    def create_oracle_follower(scorer: TdScorer, oracle_manager, iconf, log_prob_sum: bool):
        searcher = TdSearcher()
        searcher.expander = TdExpander.get_expander(iconf.expand_strategy, iconf.expand_projective)
        # todo(warn): single instance!
        searcher.local_selector = TdLocalSelectorOracle(scorer, iconf.local_arc_beam_size, iconf.local_label_beam_size, oracle_manager, log_prob_sum)
        searcher.global_arranger = None
        searcher.ender = TdEnderPlain()
        return searcher

    # 3. mixing oracle/sample randomly for scheduled sampling training
    @staticmethod
    def create_scheduled_sampler(scorer: TdScorer, oracle_manager, iconf, ss_rate_sample: ScheduledValue, topk_sample: bool, ss_strict_oracle: bool, ss_include_correct_rate: float):
        searcher = TdSearcher()
        searcher.expander = TdExpander.get_expander(iconf.expand_strategy, iconf.expand_projective)
        # todo(warn): single instance!
        searcher.local_selector = TdLocalSelectorSS(scorer, iconf.local_arc_beam_size, iconf.local_label_beam_size, oracle_manager, ss_rate_sample, topk_sample, ss_strict_oracle, ss_include_correct_rate)
        searcher.global_arranger = None
        searcher.ender = TdEnderPlain()
        return searcher

    # 4. pure sampler for RL training
    # use oracle_manager only for recording things
    @staticmethod
    def create_rl_sampler(scorer: TdScorer, oracle_manager, iconf, topk_sample: bool):
        searcher = TdSearcher()
        searcher.expander = TdExpander.get_expander(iconf.expand_strategy, iconf.expand_projective)
        # todo(warn): single instance!
        searcher.local_selector = TdLocalSelectorSample(scorer, iconf.local_arc_beam_size, iconf.local_label_beam_size, topk_sample, oracle_manager)
        searcher.global_arranger = None
        searcher.ender = TdEnderPlain()
        return searcher

    # 4.5 pure sampler for oracle-force/sampler training
    # use oracle_manager only for recording things
    @staticmethod
    def create_of_sampler(scorer: TdScorer, oracle_manager, iconf, log_prob_sum: bool):
        searcher = TdSearcher()
        searcher.expander = TdExpander.get_expander(iconf.expand_strategy, iconf.expand_projective)
        # todo(warn): single instance!
        searcher.local_selector = TdLocalSelectorOracleSample(
            scorer, iconf.local_arc_beam_size, iconf.local_label_beam_size, oracle_manager, log_prob_sum)
        searcher.global_arranger = None
        searcher.ender = TdEnderPlain()
        return searcher

    # 5. multi-sample or beam-search + oracle for beam-as-all RL-styled min-risk training
    # TODO(+N)
    @staticmethod
    def create_min_risker():
        pass

    # 6. TODO(+N)

# b tasks/zdpar/transition/topdown/decoder:623
