#

from typing import List
from msp.utils import Conf, Constants
from msp.nn import BK
from msp.nn.layers import BasicNode
from msp.zext.seq_helper import DataPadder
from msp.zext.ie import HLabelVocab, HLabelIdx
from ..labeler import HLabelNode, HLabelNodeConf
from ...common.data import Sentence

# =====
# the overall node extractor (plus labeling)
# this module will be idx based for output
class NodeExtractorConfBase(Conf):
    def __init__(self):
        self.lab_conf = HLabelNodeConf()

# the extractor base class
class NodeExtractorBase(BasicNode):
    def __init__(self, pc, conf: NodeExtractorConfBase, vocab: HLabelVocab, extract_type: str):
        super().__init__(pc, None, None)
        self.conf = conf
        self.vocab = vocab
        self.hl: HLabelNode = self.add_sub_node("hl", HLabelNode(pc, conf.lab_conf, vocab))
        self.hl_output_size = self.hl.prediction_sizes[self.hl.eff_max_layer-1]  # num of output labels
        #
        self.extract_type = extract_type
        self.items_getter = {"evt": self.get_events, "ef": lambda sent: sent.entity_fillers}[extract_type]
        self.constrain_evt_types = None
        # 2d pad
        self.padder_mask = DataPadder(2, pad_vals=0.)
        self.padder_idxes = DataPadder(2, pad_vals=0)  # todo(warn): 0 for full-nil
        self.padder_items = DataPadder(2, pad_vals=None)
        # 3d pad
        self.padder_mask_3d = DataPadder(3, pad_vals=0.)
        self.padder_items_3d = DataPadder(3, pad_vals=None)

    # =====
    # idx tranforms
    def hlidx2idx(self, hlidx: HLabelIdx) -> int:
        return hlidx.get_idx(self.hl.eff_max_layer-1)

    def idx2hlidx(self, idx: int) -> HLabelIdx:
        return self.vocab.get_hlidx(idx, self.hl.eff_max_layer)

    # events possibly filtered by constrain_evt_types
    def get_events(self, sent):
        ret = sent.events
        constrain_evt_types = self.constrain_evt_types
        if constrain_evt_types is None:
            return ret
        else:
            return [z for z in ret if z.type in constrain_evt_types]

    def set_constrain_evt_types(self, constrain_evt_types):
        self.constrain_evt_types = constrain_evt_types

    # =====
    # main procedure

    def loss(self, insts: List, input_lexi, input_expr, input_mask, margin=0.):
        raise NotImplementedError()

    def predict(self, insts: List, input_lexi, input_expr, input_mask):
        raise NotImplementedError()

    def lookup(self, insts: List, input_lexi, input_expr, input_mask):
        raise NotImplementedError()

    # =====
    # basic input specifying for this module
    # todo(+N): currently entangling things with the least elegant way?

    # batch inputs for head mode
    def batch_inputs_h(self, insts: List[Sentence]):
        key, items_getter = self.extract_type, self.items_getter
        nil_idx = 0
        # get gold/input data and batch
        all_masks, all_idxes, all_items, all_valid = [], [], [], []
        all_idxes2, all_items2 = [], []  # secondary types
        for sent in insts:
            preps = sent.preps.get(key)
            # not cached, rebuild them
            if preps is None:
                length = sent.length
                items = items_getter(sent)
                # token-idx -> ...
                prep_masks, prep_idxes, prep_items = [0.]*length, [nil_idx]*length, [None]*length
                prep_idxes2, prep_items2 = [nil_idx]*length, [None]*length
                if items is None:
                    # todo(note): there are samples that do not have entity annotations (KBP15)
                    #  final 0/1 indicates valid or not
                    prep_valid = 0.
                else:
                    prep_valid = 1.
                    for one_item in items:
                        this_hwidx = one_item.mention.hard_span.head_wid
                        this_hlidx = one_item.type_idx
                        # todo(+N): ignore except the first two types (already ranked by type-freq)
                        if prep_idxes[this_hwidx] == 0:
                            prep_masks[this_hwidx] = 1.
                            prep_idxes[this_hwidx] = self.hlidx2idx(this_hlidx)  # change to int here!
                            prep_items[this_hwidx] = one_item
                        elif prep_idxes2[this_hwidx] == 0:
                            prep_idxes2[this_hwidx] = self.hlidx2idx(this_hlidx)  # change to int here!
                            prep_items2[this_hwidx] = one_item
                sent.preps[key] = (prep_masks, prep_idxes, prep_items, prep_valid, prep_idxes2, prep_items2)
            else:
                prep_masks, prep_idxes, prep_items, prep_valid, prep_idxes2, prep_items2 = preps
            # =====
            all_masks.append(prep_masks)
            all_idxes.append(prep_idxes)
            all_items.append(prep_items)
            all_valid.append(prep_valid)
            all_idxes2.append(prep_idxes2)
            all_items2.append(prep_items2)
        # pad and batch
        mention_masks = BK.input_real(self.padder_mask.pad(all_masks)[0])  # [*, slen]
        mention_idxes = BK.input_idx(self.padder_idxes.pad(all_idxes)[0])  # [*, slen]
        mention_items_arr, _ = self.padder_items.pad(all_items)  # [*, slen]
        mention_valid = BK.input_real(all_valid)  # [*]
        mention_idxes2 = BK.input_idx(self.padder_idxes.pad(all_idxes2)[0])  # [*, slen]
        mention_items2_arr, _ = self.padder_items.pad(all_items2)  # [*, slen]
        return mention_masks, mention_idxes, mention_items_arr, mention_valid, mention_idxes2, mention_items2_arr

    # batch inputs for gene0 mode (separate for each label)
    def batch_inputs_g0(self, insts: List[Sentence]):
        # similar to "batch_inputs_h", but further extend for each label
        key, items_getter = self.extract_type, self.items_getter
        # nil_idx = 0
        # get gold/input data and batch
        output_size = self.hl_output_size
        all_masks, all_items, all_valid = [], [], []
        for sent in insts:
            preps = sent.preps.get(key)
            # not cached, rebuild them
            if preps is None:
                length = sent.length
                items = items_getter(sent)
                # token-idx -> [slen, out-size]
                prep_masks = [[0. for _i1 in range(output_size)] for _i0 in range(length)]
                prep_items = [[None for _i1 in range(output_size)] for _i0 in range(length)]
                if items is None:
                    # todo(note): there are samples that do not have entity annotations (KBP15)
                    #  final 0/1 indicates valid or not
                    prep_valid = 0.
                else:
                    prep_valid = 1.
                    for one_item in items:
                        this_hwidx = one_item.mention.hard_span.head_wid
                        this_hlidx = one_item.type_idx
                        this_tidx = self.hlidx2idx(this_hlidx)  # change to int here!
                        # todo(+N): simply ignore repeated ones with same type and trigger
                        if prep_masks[this_hwidx][this_tidx] == 0.:
                            prep_masks[this_hwidx][this_tidx] = 1.
                            prep_items[this_hwidx][this_tidx] = one_item
                sent.preps[key] = (prep_masks, prep_items, prep_valid)
            else:
                prep_masks, prep_items, prep_valid = preps
            # =====
            all_masks.append(prep_masks)
            all_items.append(prep_items)
            all_valid.append(prep_valid)
        # pad and batch
        mention_masks = BK.input_real(self.padder_mask_3d.pad(all_masks)[0])  # [*, slen, L]
        mention_idxes = None
        mention_items_arr, _ = self.padder_items_3d.pad(all_items)  # [*, slen, L]
        mention_valid = BK.input_real(all_valid)  # [*]
        return mention_masks, mention_idxes, mention_items_arr, mention_valid

    # batch inputs for gene1 mode (seq-gene mode)
    # todo(note): the return is different than previous, here directly idx-based
    def batch_inputs_g1(self, insts: List[Sentence]):
        train_reverse_evetns = self.conf.train_reverse_evetns  # todo(note): this option is from derived class
        _tmp_f = lambda x: list(reversed(x)) if train_reverse_evetns else lambda x: x
        key, items_getter = self.extract_type, self.items_getter
        # nil_idx = 0  # nil means eos
        # get gold/input data and batch
        all_widxes, all_lidxes, all_vmasks, all_items, all_valid = [], [], [], [], []
        for sent in insts:
            preps = sent.preps.get(key)
            # not cached, rebuild them
            if preps is None:
                items = items_getter(sent)
                # todo(note): directly add, assume they are already sorted in a good way (widx+lidx); 0(nil) as eos
                if items is None:
                    prep_valid = 0.
                    # prep_widxes, prep_lidxes, prep_vmasks, prep_items = [0], [0], [1.], [None]
                    prep_widxes, prep_lidxes, prep_vmasks, prep_items = [], [], [], []
                else:
                    prep_valid = 1.
                    prep_widxes = _tmp_f([z.mention.hard_span.head_wid for z in items]) + [0]
                    prep_lidxes = _tmp_f([self.hlidx2idx(z.type_idx) for z in items]) + [0]
                    prep_vmasks = [1.] * (len(items)+1)
                    prep_items = _tmp_f(items.copy()) + [None]
                sent.preps[key] = (prep_widxes, prep_lidxes, prep_vmasks, prep_items, prep_valid)
            else:
                prep_widxes, prep_lidxes, prep_vmasks, prep_items, prep_valid = preps
            # =====
            all_widxes.append(prep_widxes)
            all_lidxes.append(prep_lidxes)
            all_vmasks.append(prep_vmasks)
            all_items.append(prep_items)
            all_valid.append(prep_valid)
        # pad and batch
        mention_widxes = BK.input_idx(self.padder_idxes.pad(all_widxes)[0])  # [*, ?]
        mention_lidxes = BK.input_idx(self.padder_idxes.pad(all_lidxes)[0])  # [*, ?]
        mention_vmasks = BK.input_real(self.padder_mask.pad(all_vmasks)[0])  # [*, ?]
        mention_items_arr, _ = self.padder_items.pad(all_items)  # [*, ?]
        mention_valid = BK.input_real(all_valid)  # [*]
        return mention_widxes, mention_lidxes, mention_vmasks, mention_items_arr, mention_valid
