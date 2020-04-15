#

# the overall model

from typing import List
import numpy as np
from copy import deepcopy

from msp.utils import Random
from msp.nn.layers import BasicNode, Affine
from msp.nn import BK
from msp.zext.seq_helper import DataPadder
from msp.zext.ie import HLabelIdx, HLabelVocab
from msp.zext.process_train import SVConf, ScheduledValue

from ..common.data import DocInstance, Sentence, Mention, HardSpan, EntityFiller, Event
from ..common.vocab import IEVocabPackage
from ..common.model import MyIEModel, MyIEModelConf, BaseInferenceConf, BaseTrainingConf

from .extract import NodeExtractorBase, NodeExtractorConfHead, NodeExtractorHead, \
    NodeExtractorConfGene0, NodeExtractorGene0, NodeExtractorConfGene1, NodeExtractorGene1
from .linkage import LinkerConf, Linker

# =====
# confs

class InferenceConf(BaseInferenceConf):
    def __init__(self):
        super().__init__()
        # whether lookup for each component?
        self.lookup_ef = 0.
        self.lookup_evt = 0.
        # whether pred if not lookup
        self.pred_ef = 0.
        self.pred_arg = 0.
        # -----
        self.expand_evt_compound = False  # expand VERB+compound+?, but only improves a little, do not consider now.

class TrainingConf(BaseTrainingConf):
    def __init__(self):
        super().__init__()
        # lookup in training?
        self.lookup_ef = 0.
        self.lookup_evt = 0.
        # training weights for the components
        # self.lambda_ef = 0.
        # self.lambda_evt = 1.
        # self.lambda_arg = 0.
        self.lambda_ef = SVConf().init_from_kwargs(val=0.0)
        self.lambda_evt = SVConf().init_from_kwargs(val=1.0)
        self.lambda_arg = SVConf().init_from_kwargs(val=0.0)

class MySimpleIEModelConf(MyIEModelConf):
    def __init__(self):
        super().__init__(InferenceConf(), TrainingConf())
        # extractor components confs
        self.c_ef = NodeExtractorConfHead()
        self.evt_model_type = "h"  # h, g0, g1
        self.c_evt_h = NodeExtractorConfHead()
        self.c_evt_g0 = NodeExtractorConfGene0()
        self.c_evt_g1 = NodeExtractorConfGene1()
        self.c_arg = LinkerConf()

# =====
# model

class MySimpleIEModel(MyIEModel):
    def __init__(self, conf: MySimpleIEModelConf, vpack: IEVocabPackage):
        super().__init__(conf, vpack)
        # components
        self.ef_extractor: NodeExtractorBase = self.decoders[0]
        self.evt_extractor: NodeExtractorBase = self.decoders[1]
        self.arg_linker: Linker = self.decoders[2]
        #
        self.hl_ef: HLabelVocab = self.vpack.get_voc("hl_ef")
        self.hl_evt: HLabelVocab = self.vpack.get_voc("hl_evt")
        self.hl_arg: HLabelVocab = self.vpack.get_voc("hl_arg")
        # lambdas for training
        self.lambda_ef = ScheduledValue("lambda_ef", conf.tconf.lambda_ef)
        self.lambda_evt = ScheduledValue("lambda_evt", conf.tconf.lambda_evt)
        self.lambda_arg = ScheduledValue("lambda_arg", conf.tconf.lambda_arg)
        self.add_scheduled_values(self.lambda_ef)
        self.add_scheduled_values(self.lambda_evt)
        self.add_scheduled_values(self.lambda_arg)

    # todo(note): this one should be careful, since it will be called in base's __init__
    def build_decoders(self) -> List[BasicNode]:
        # assign dims and get the node
        # entity and filler
        self.conf.c_ef._input_dim = self.enc_ef_output_dim
        self.conf.c_ef._lexi_dim = self.lexi_output_dim
        ef_extractor = NodeExtractorHead(self.pc, self.conf.c_ef, self.vpack.get_voc("hl_ef"), "ef")
        # event
        evt_model_type = self.conf.evt_model_type
        if evt_model_type == "h":
            self.conf.c_evt_h._input_dim = self.enc_evt_output_dim
            self.conf.c_evt_h._lexi_dim = self.lexi_output_dim
            evt_extractor = NodeExtractorHead(self.pc, self.conf.c_evt_h, self.vpack.get_voc("hl_evt"), "evt")
        elif evt_model_type == "g0":
            self.conf.c_evt_g0._input_dim = self.enc_evt_output_dim
            self.conf.c_evt_g0._lexi_dim = self.lexi_output_dim
            evt_extractor = NodeExtractorGene0(self.pc, self.conf.c_evt_g0, self.vpack.get_voc("hl_evt"), "evt")
        elif evt_model_type == "g1":
            self.conf.c_evt_g1._input_dim = self.enc_evt_output_dim
            self.conf.c_evt_g1._lexi_dim = self.lexi_output_dim
            evt_extractor = NodeExtractorGene1(self.pc, self.conf.c_evt_g1, self.vpack.get_voc("hl_evt"), "evt")
        else:
            raise NotImplementedError(f"UNK evt_model_type {evt_model_type}")
        # arg_linker
        self.conf.c_arg._input_dim = self.enc_evt_output_dim
        assert self.enc_ef_output_dim == self.enc_evt_output_dim
        arg_linker = Linker(self.pc, self.conf.c_arg, self.vpack.get_voc("hl_arg"))
        return [ef_extractor, evt_extractor, arg_linker]

    # =====
    # helpers (for inference)
    def ef_creator(self, partial_id: str, m: Mention, hlidx: HLabelIdx, logprob: float):
        # todo(warn): here does not distinguish entity or filler
        return EntityFiller("ef-"+partial_id, m, str(hlidx), None, True, type_idx=hlidx, score=logprob)

    def evt_creator(self, partial_id: str, m: Mention, hlidx: HLabelIdx, logprob: float):
        return Event("evt-"+partial_id, m, str(hlidx), type_idx=hlidx, score=logprob)

    # decoding
    def inference_on_batch(self, insts: List[DocInstance], **kwargs):
        self.refresh_batch(False)
        test_constrain_evt_types = self.test_constrain_evt_types
        ndoc, nsent = len(insts), 0
        iconf = self.conf.iconf
        with BK.no_grad_env():
            # splitting into buckets
            all_packs = self.bter.run(insts, training=False)
            for one_pack in all_packs:
                # =====
                # predict
                sent_insts, lexi_repr, enc_repr_ef, enc_repr_evt, mask_arr = one_pack
                nsent += len(sent_insts)
                mask_expr = BK.input_real(mask_arr)
                # entity and filler
                if iconf.lookup_ef:
                    ef_items, ef_widxes, ef_valid_mask, ef_lab_idxes, ef_lab_embeds = \
                        self._lookup_mentions(sent_insts, lexi_repr, enc_repr_ef, mask_expr, self.ef_extractor, ret_copy=True)
                elif iconf.pred_ef:
                    ef_items, ef_widxes, ef_valid_mask, ef_lab_idxes, ef_lab_embeds = \
                        self._inference_mentions(sent_insts, lexi_repr, enc_repr_ef, mask_expr, self.ef_extractor, self.ef_creator)
                else:
                    ef_items = [[] for _ in range(len(sent_insts))]
                    ef_valid_mask = BK.zeros((len(sent_insts), 0))
                    ef_widxes = ef_lab_idxes = ef_lab_embeds = None
                # event
                if iconf.lookup_evt:
                    evt_items, evt_widxes, evt_valid_mask, evt_lab_idxes, evt_lab_embeds = \
                        self._lookup_mentions(sent_insts, lexi_repr, enc_repr_evt, mask_expr, self.evt_extractor, ret_copy=True)
                else:
                    evt_items, evt_widxes, evt_valid_mask, evt_lab_idxes, evt_lab_embeds = \
                        self._inference_mentions(sent_insts, lexi_repr, enc_repr_evt, mask_expr, self.evt_extractor, self.evt_creator)
                # arg
                if iconf.pred_arg:
                    # todo(note): for this step of decoding, we only consider inner-sentence pairs
                    # todo(note): inplaced
                    self._inference_args(ef_items, ef_widxes, ef_valid_mask, ef_lab_idxes, enc_repr_ef,
                                         evt_items, evt_widxes, evt_valid_mask, evt_lab_idxes, enc_repr_evt)
                # =====
                # assign
                for one_sent_inst, one_ef_items, one_ef_valid, one_evt_items, one_evt_valid in \
                        zip(sent_insts, ef_items, BK.get_value(ef_valid_mask), evt_items, BK.get_value(evt_valid_mask)):
                    # entity and filler
                    one_ef_items = [z for z,va in zip(one_ef_items, one_ef_valid) if (va and z is not None)]
                    one_sent_inst.pred_entity_fillers = one_ef_items
                    # event
                    one_evt_items = [z for z,va in zip(one_evt_items, one_evt_valid) if (va and z is not None)]
                    if test_constrain_evt_types is not None:
                        one_evt_items = [z for z in one_evt_items if z.type in test_constrain_evt_types]
                    # =====
                    # todo(note): special rule (actually a simple rule based extender)
                    if iconf.expand_evt_compound:
                        for one_evt in one_evt_items:
                            one_hard_span = one_evt.mention.hard_span
                            sid, hwid, _ = one_hard_span.position(True)
                            assert one_hard_span.length == 1  # currently no way to predict more
                            if hwid+1 < one_sent_inst.length:
                                if one_sent_inst.uposes.vals[hwid]=="VERB" and one_sent_inst.ud_heads.vals[hwid+1]==hwid \
                                        and one_sent_inst.ud_labels.vals[hwid+1]=="compound":
                                    one_hard_span.length += 1
                    # =====
                    one_sent_inst.pred_events = one_evt_items
        return {"doc": ndoc, "sent": nsent}

    # training
    def fb_on_batch(self, annotated_insts: List[DocInstance], training=True, loss_factor=1., **kwargs):
        self.refresh_batch(training)
        self.evt_extractor.set_constrain_evt_types(self.train_constrain_evt_types)  # also ignore irrelevant types for training
        ndoc, nsent = len(annotated_insts), 0
        margin = self.margin.value
        lambda_ef, lambda_evt, lambda_arg = self.lambda_ef.value, self.lambda_evt.value, self.lambda_arg.value
        lookup_ef, lookup_evt = self.conf.tconf.lookup_ef, self.conf.tconf.lookup_evt
        #
        has_loss_ef = has_loss_evt = has_loss_arg = lambda_arg > 0.
        has_loss_ef = has_loss_ef or (lambda_ef > 0.)
        has_loss_evt = has_loss_evt or (lambda_evt > 0.)
        # splitting into buckets
        all_packs = self.bter.run(annotated_insts, training=training)
        all_ef_losses = []
        all_evt_losses = []
        all_arg_losses = []
        for one_pack in all_packs:
            # =====
            # predict
            sent_insts, lexi_repr, enc_repr_ef, enc_repr_evt, mask_arr = one_pack
            nsent += len(sent_insts)
            mask_expr = BK.input_real(mask_arr)
            # entity and filler
            if lookup_ef:
                ef_items, ef_widxes, ef_valid_mask, ef_lab_idxes, ef_lab_embeds = \
                    self._lookup_mentions(sent_insts, lexi_repr, enc_repr_ef, mask_expr, self.ef_extractor)
            elif has_loss_ef:
                ef_losses, ef_items, ef_widxes, ef_valid_mask, ef_lab_idxes, ef_lab_embeds = \
                    self._fb_mentions(sent_insts, lexi_repr, enc_repr_ef, mask_expr, self.ef_extractor, margin)
                all_ef_losses.append(ef_losses)
            # event
            if lookup_evt:
                evt_items, evt_widxes, evt_valid_mask, evt_lab_idxes, evt_lab_embeds = \
                    self._lookup_mentions(sent_insts, lexi_repr, enc_repr_evt, mask_expr, self.evt_extractor)
            elif has_loss_evt:
                evt_losses, evt_items, evt_widxes, evt_valid_mask, evt_lab_idxes, evt_lab_embeds = \
                    self._fb_mentions(sent_insts, lexi_repr, enc_repr_evt, mask_expr, self.evt_extractor, margin)
                all_evt_losses.append(evt_losses)
            # arg
            if has_loss_arg:
                # todo(note): for training, we only consider inner-sentence pairs,
                #  since most of the training data is like this
                arg_losses = self._fb_args(ef_items, ef_widxes, ef_valid_mask, ef_lab_idxes, enc_repr_ef,
                                           evt_items, evt_widxes, evt_valid_mask, evt_lab_idxes, enc_repr_evt, margin)
                all_arg_losses.append(arg_losses)
        # =====
        # final loss sum and backward
        info = {"doc": ndoc, "sent": nsent, "fb": 1}
        if len(all_packs) == 0:
            return info
        self.collect_loss_and_backward(["ef", "evt", "arg"], [all_ef_losses, all_evt_losses, all_arg_losses],
                                       [lambda_ef, lambda_evt, lambda_arg], info, training, loss_factor)
        return info

    # =====
    # shared procedures

    # todo(note): for mention detection, still performing at sentence level
    #  (but the input enc features can come from doc-level)

    # ======
    # mentions

    def _inference_mentions(self, insts: List[Sentence], lexi_repr, enc_repr, mask_expr, extractor: NodeExtractorBase, item_creator):
        sel_logprobs, sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds = \
            extractor.predict(insts, lexi_repr, enc_repr, mask_expr)
        # handling outputs here: prepare new items
        head_idxes_arr = BK.get_value(sel_idxes)  # [*, max-count]
        lab_idxes_arr = BK.get_value(sel_lab_idxes)  # [*, max-count]
        logprobs_arr = BK.get_value(sel_logprobs)  # [*, max-count]
        valid_arr = BK.get_value(sel_valid_mask)  # [*, max-count]
        all_items = []
        bsize, mc = valid_arr.shape
        for one_idxes, one_valids, one_lab_idxes, one_logprobs, one_sent in \
                zip(head_idxes_arr, valid_arr, lab_idxes_arr, logprobs_arr, insts):
            sid = one_sent.sid
            partial_id0 = f"{one_sent.doc.doc_id}-s{one_sent.sid}-i"
            for this_i in range(mc):
                this_valid = float(one_valids[this_i])
                if this_valid == 0:  # must be compact
                    assert np.all(one_valids[this_i:]==0.)
                    all_items.extend([None] * (mc-this_i))
                    break
                # todo(note): we need to assign various info at the outside
                this_mention = Mention(HardSpan(sid, int(one_idxes[this_i]), None, None))
                # todo(note): where to filter None?
                this_hlidx = extractor.idx2hlidx(one_lab_idxes[this_i])
                all_items.append(item_creator(partial_id0+str(this_i), this_mention, this_hlidx, float(one_logprobs[this_i])))
        # only return the items and the ones useful for later steps: List(sent)[List(items)], *[*, max-count]
        ret_items = np.asarray(all_items, dtype=object).reshape((bsize, mc))
        return ret_items, sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds

    def _fb_mentions(self, insts: List[Sentence], lexi_repr, enc_repr, mask_expr, extractor: NodeExtractorBase, margin):
        # loss and predictions
        return extractor.loss(insts, lexi_repr, enc_repr, mask_expr, margin=margin)

    def _lookup_mentions(self, insts: List[Sentence], lexi_repr, enc_repr, mask_expr, extractor: NodeExtractorBase,
                         ret_copy: bool = False):
        ret_items, sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds = \
            extractor.lookup(insts, lexi_repr, enc_repr, mask_expr)
        # todo(+N): here, the links will be dropped between ef and evt
        if ret_copy:
            ret_items = deepcopy(ret_items)
        return ret_items, sel_idxes, sel_valid_mask, sel_lab_idxes, sel_lab_embeds

    # =====
    # args

    def _inference_args(self, ef_items, ef_widxes, ef_valid_mask, ef_lab_idxes, enc_repr_ef,
                        evt_items, evt_widxes, evt_valid_mask, evt_lab_idxes, enc_repr_evt):
        arg_linker = self.arg_linker
        repr_ef = BK.gather_first_dims(enc_repr_ef, ef_widxes, -2)  # [*, len-ef, D]
        repr_evt = BK.gather_first_dims(enc_repr_evt, evt_widxes, -2)  # [*, len-evt, D]
        role_logprobs, role_predictions = arg_linker.predict(repr_ef, repr_evt, ef_lab_idxes, evt_lab_idxes,
                                                             ef_valid_mask, evt_valid_mask)
        # add them inplaced
        roles_arr = BK.get_value(role_predictions)  # [*, len-ef, len-evt]
        logprobs_arr = BK.get_value(role_logprobs)
        for bidx, one_roles_arr in enumerate(roles_arr):
            one_ef_items, one_evt_items = ef_items[bidx], evt_items[bidx]
            # =====
            # todo(note): delete origin links!
            for z in one_ef_items:
                if z is not None:
                    z.links.clear()
            for z in one_evt_items:
                if z is not None:
                    z.links.clear()
            # =====
            one_logprobs = logprobs_arr[bidx]
            for ef_idx, one_ef in enumerate(one_ef_items):
                if one_ef is None:
                    continue
                for evt_idx, one_evt in enumerate(one_evt_items):
                    if one_evt is None:
                        continue
                    one_role_idx = int(one_roles_arr[ef_idx, evt_idx])
                    if one_role_idx > 0:  # link
                        this_hlidx = arg_linker.idx2hlidx(one_role_idx)
                        one_evt.add_arg(one_ef, role=str(this_hlidx), role_idx=this_hlidx,
                                        score=float(one_logprobs[ef_idx, evt_idx]))

    def _fb_args(self, ef_items, ef_widxes, ef_valid_mask, ef_lab_idxes, enc_repr_ef,
                 evt_items, evt_widxes, evt_valid_mask, evt_lab_idxes, enc_repr_evt, margin):
        # get the gold idxes
        arg_linker = self.arg_linker
        bsize, len_ef = ef_items.shape
        bsize2, len_evt = evt_items.shape
        assert bsize == bsize2
        gold_idxes = np.zeros([bsize, len_ef, len_evt], dtype=np.long)
        for one_gold_idxes, one_ef_items, one_evt_items in zip(gold_idxes, ef_items, evt_items):
            # todo(note): check each pair
            for ef_idx, one_ef in enumerate(one_ef_items):
                if one_ef is None:
                    continue
                role_map = {id(z.evt): z.role_idx for z in one_ef.links}  # todo(note): since we get the original linked ones
                for evt_idx, one_evt in enumerate(one_evt_items):
                    pairwise_role_hlidx = role_map.get(id(one_evt))
                    if pairwise_role_hlidx is not None:
                        pairwise_role_idx = arg_linker.hlidx2idx(pairwise_role_hlidx)
                        assert pairwise_role_idx > 0
                        one_gold_idxes[ef_idx, evt_idx] = pairwise_role_idx
        # get loss
        repr_ef = BK.gather_first_dims(enc_repr_ef, ef_widxes, -2)  # [*, len-ef, D]
        repr_evt = BK.gather_first_dims(enc_repr_evt, evt_widxes, -2)  # [*, len-evt, D]
        if np.prod(gold_idxes.shape) == 0:
            # no instances!
            return [[BK.zeros([]), BK.zeros([])]]
        else:
            gold_idxes_t = BK.input_idx(gold_idxes)
            return arg_linker.loss(repr_ef, repr_evt, ef_lab_idxes, evt_lab_idxes, ef_valid_mask, evt_valid_mask,
                                   gold_idxes_t, margin)

    def _lookup_args(self, **kwargs):
        raise NotImplementedError("Not need for this here!")

# b tasks/zie/models2/model:142
