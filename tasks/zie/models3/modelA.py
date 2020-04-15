#

# especially for arg extraction with given events

from typing import List
import numpy as np
from collections import defaultdict
from copy import copy as shallow_copy
from copy import deepcopy as deep_copy

from msp.utils import Random, zlog, Conf
from msp.nn.layers import BasicNode, Affine, NoDropRop
from msp.nn import BK
from msp.zext.seq_helper import DataPadder
from msp.zext.ie import HLabelIdx, HLabelVocab
from msp.zext.process_train import SVConf, ScheduledValue

from ..common.data import DocInstance, Sentence, Mention, HardSpan, EntityFiller, Event
from ..common.vocab import IEVocabPackage
from ..common.model import MyIEModel, MyIEModelConf, BaseInferenceConf, BaseTrainingConf

from .model_enc import M3EncConf, M3Encoder, MultiSentItem
from .model_dec import TaskSpecAdp, PrepHelper, ArgLinkerConf, ArgLinker
from .model_expand import ArgSpanExpanderConf, ArgSpanExpander

# =====
# conf

class M3AInferenceConf(BaseInferenceConf):
    def __init__(self):
        super().__init__()
        #
        self.batch_size = 10  # single doc?
        self.decode_verbose = False
        #
        self.lookup_ef = False  # use input ef as cands rather than predict
        self.pred_arg = True
        self.pred_span = True
        #
        self.exclude_nolink_ef = True  # only keep the ones as args

class M3ATrainingConf(BaseTrainingConf):
    def __init__(self):
        super().__init__()
        # lambdas
        # arg cand
        self.lambda_cand = SVConf().init_from_kwargs(val=0.5)
        # arg link
        self.lambda_arg = SVConf().init_from_kwargs(val=1.0)
        # arg span expander
        self.lambda_span = SVConf().init_from_kwargs(val=0.5)

class M3AIEModelConf(MyIEModelConf):
    def __init__(self):
        super().__init__(M3AInferenceConf(), M3ATrainingConf())
        # components
        # encoding, todo(note): here replace bter conf!!
        self.bt_conf = M3EncConf()
        # decoding
        self.c_cand = CandidateExtractorConf()
        self.c_arg = ArgLinkerConf()
        self.c_span = ArgSpanExpanderConf()
        # others
        self.mix_pred_ef_rate = 1.  # inside mix rate & default outside rate
        self.mix_pred_ef_rate_outside = -1.  # active when >0

    def do_validate(self):
        if self.mix_pred_ef_rate_outside<0:
            self.mix_pred_ef_rate_outside = self.mix_pred_ef_rate

# =====
# model
class M3AIEModel(MyIEModel):
    def __init__(self, conf: M3AIEModelConf, vpack: IEVocabPackage):
        super().__init__(conf, vpack)
        # components
        self.cand_extractor: CandidateExtractor = self.decoders[0]
        self.arg_linker: ArgLinker = self.decoders[1]
        self.span_expander: ArgSpanExpander = self.decoders[2]
        # vocab
        self.hl_arg: HLabelVocab = self.vpack.get_voc("hl_arg")
        # lambdas for training
        self.lambda_cand = ScheduledValue("lambda_cand", conf.tconf.lambda_cand)
        self.lambda_arg = ScheduledValue("lambda_arg", conf.tconf.lambda_arg)
        self.lambda_span = ScheduledValue("lambda_span", conf.tconf.lambda_span)
        self.add_scheduled_values(self.lambda_cand)
        self.add_scheduled_values(self.lambda_arg)
        self.add_scheduled_values(self.lambda_span)
        # others
        self.random_sample_stream = Random.stream(Random.random_sample)

    def get_inst_preper(self, training, **kwargs):
        conf = self.conf
        def _preper(inst):
            self.bter.prepare_inst(inst)
            # todo(note): not here for arg
            return inst
        return _preper

    def build_encoder(self) -> BasicNode:
        return M3Encoder(self.pc, self.conf.bt_conf, self.conf.tconf, self.vpack)

    def build_decoders(self) -> List[BasicNode]:
        input_enc_dims = self.bter.speical_output_dims()  # todo(note): from M3Enc
        cand_extractor = CandidateExtractor(self.pc, self.conf.c_cand, input_enc_dims)
        arg_linker = ArgLinker(self.pc, self.conf.c_arg, self.vpack.get_voc("hl_arg"), input_enc_dims,
                               borrowed_evt_hlnode=None, borrowed_evt_adp=None)  # todo(note): not using these modes!
        arg_span_expander = ArgSpanExpander(self.pc, self.conf.c_span, input_enc_dims)
        return [cand_extractor, arg_linker, arg_span_expander]

    # =====
    # special inst transforming
    def _insts2msitems(self, insts):
        # -----
        # a new sent with new containers but old contents for ef
        def _copy_sent(sent, cur_event):
            ret = shallow_copy(sent)
            ret.events = []
            ret.pred_events = []
            if cur_event is not None:
                # only one event for center
                ret.events.append(cur_event)  # for events, use the original one
                copied_event = deep_copy(cur_event)  # for pred, use the copied one
                copied_event.links.clear()  # and clear links(args) for prediction
                ret.pred_events.append(copied_event)
            ret.entity_fillers = shallow_copy(ret.entity_fillers)  # for training
            ret.pred_entity_fillers = []  # to predict
            ret.orig_sent = sent  # used in prediction to append back to the original instances
            return ret
        # -----
        ret = []
        for inst in insts:
            center_cands = []
            if isinstance(inst, DocInstance):
                center_cands.extend(inst.sents)
            else:
                assert isinstance(inst, MultiSentItem)
                center_cands.append(inst.sents[inst.center_idx])  # only include the center one if already ms
            # get the new specially copied ones
            for one_center_sent in center_cands:
                one_ms_item = one_center_sent.preps["ms"]
                one_center_idx = one_ms_item.center_idx
                # todo(note): new instance for each event (use input events!!)
                for one_event in one_center_sent.events:
                    one_ms = shallow_copy(one_ms_item)
                    one_ms.sents = [_copy_sent(s, (one_event if (i==one_center_idx) else None))
                                    for i,s in enumerate(one_ms.sents)]  # special copy sents
                    ret.append(one_ms)
        return ret

    # lookup efs as cands
    def _lookup_efs(self, ms_items):
        for ms_item in ms_items:
            for s in ms_item.sents:
                s.pred_entity_fillers.clear()
                for one_ef in s.entity_fillers:
                    copied_ef = deep_copy(one_ef)
                    copied_ef.links.clear()  # clean links
                    s.pred_entity_fillers.append(copied_ef)

    # put back predictions
    # todo(note): there can be repeated efs for different events, but let it be there, will be merged outside if needed
    def _putback_preds(self, ms_items):
        constrain_types = self.test_constrain_evt_types
        exclude_nolink_ef = self.conf.iconf.exclude_nolink_ef
        for ms_item in ms_items:
            for s in ms_item.sents:
                orig_sent = s.orig_sent
                if constrain_types is not None:  # filter here
                    orig_sent.pred_events.extend([z for z in s.pred_events if z.type in constrain_types])
                else:
                    orig_sent.pred_events.extend(s.pred_events)
                if exclude_nolink_ef:
                    orig_sent.pred_entity_fillers.extend([z for z in s.pred_entity_fillers if len(z.links)>0])
                else:
                    orig_sent.pred_entity_fillers.extend(s.pred_entity_fillers)

    # testing
    def inference_on_batch(self, insts: List[DocInstance], **kwargs):
        self.refresh_batch(False)
        # -----
        if len(insts) == 0:
            return {}
        # -----
        ndoc, nsent = len(insts), 0
        iconf = self.conf.iconf
        # =====
        # get tmp ms_items for each event
        input_ms_items = self._insts2msitems(insts)
        # -----
        if len(input_ms_items) == 0:
            return {}
        # -----
        with BK.no_grad_env():
            # splitting into buckets
            all_packs = self.bter.run(input_ms_items, training=False)
            for one_pack in all_packs:
                ms_items, bert_expr, basic_expr = one_pack
                nsent += len(ms_items)
                # cands
                if iconf.lookup_ef:
                    self._lookup_efs(ms_items)
                else:
                    self.cand_extractor.predict(ms_items, bert_expr, basic_expr)
                # args
                if iconf.pred_arg:
                    self.arg_linker.predict(ms_items, bert_expr, basic_expr)
                # span
                if iconf.pred_span:
                    self.span_expander.predict(ms_items, bert_expr)
        # put back all predictions
        self._putback_preds(input_ms_items)
        # collect all stats
        num_ef, num_evt, num_arg = 0, 0, 0
        for one_doc in insts:
            for one_sent in one_doc.sents:
                num_ef += len(one_sent.pred_entity_fillers)
                num_evt += len(one_sent.pred_events)
                num_arg += sum(len(z.links) for z in one_sent.pred_events)
        info = {"doc": ndoc, "sent": nsent, "num_ef": num_ef, "num_evt": num_evt, "num_arg": num_arg}
        if iconf.decode_verbose:
            zlog(f"Decode one mini-batch: {info}")
        return info

    # training
    def fb_on_batch(self, annotated_insts: List[DocInstance], training=True, loss_factor=1., **kwargs):
        self.refresh_batch(training)
        assert self.train_constrain_evt_types is None, "Not implemented for training constrain_types"
        ndoc, nsent = len(annotated_insts), 0
        lambda_cand, lambda_arg, lambda_span = self.lambda_cand.value, self.lambda_arg.value, self.lambda_span.value
        # simply multi-task training, no explict interactions between them
        # -----
        # only include the ones with enough annotations
        valid_insts = []
        for one_inst in annotated_insts:
            if all(z.events is not None and z.entity_fillers is not None for z in one_inst.sents):
                valid_insts.append(one_inst)
        input_ms_items = self._insts2msitems(valid_insts)
        if len(input_ms_items)==0:
            return {}
        # -----
        all_packs = self.bter.run(input_ms_items, training=training)
        all_cand_losses, all_arg_losses, all_span_losses = [], [], []
        # =====
        mix_pred_ef_rate = self.conf.mix_pred_ef_rate
        mix_pred_ef_rate_outside = self.conf.mix_pred_ef_rate_outside
        mix_pred_ef_count = 0
        # cur_margin = self.margin.value  # todo(+N): currently not used!
        for one_pack in all_packs:
            ms_items, bert_expr, basic_expr = one_pack
            nsent += len(ms_items)
            if lambda_cand>0.:
                # todo(note): no need to clean up pred ones since sentences and containers are copied
                cand_losses = self.cand_extractor.loss(ms_items, bert_expr, basic_expr)
                all_cand_losses.append(cand_losses)
                # predict as candidates
                with BK.no_grad_env():
                    self.cand_extractor.predict(ms_items, bert_expr, basic_expr)
                    # mix into gold ones; no need to cleanup since these are copies by _insts2msitems
                    for one_msent in ms_items:
                        center_idx = one_msent.center_idx
                        for one_sidx, one_sent in enumerate(one_msent.sents):
                            hit_posi = set()
                            for one_ef in one_sent.entity_fillers:
                                posi = one_ef.mention.hard_span.position()
                                hit_posi.add(posi)
                            # add predicted ones
                            cur_mix_rate = mix_pred_ef_rate if (center_idx==one_sidx) else mix_pred_ef_rate_outside
                            for one_ef in one_sent.pred_entity_fillers:
                                posi = one_ef.mention.hard_span.position()
                                if posi not in hit_posi and next(self.random_sample_stream) <= cur_mix_rate:
                                    hit_posi.add(posi)
                                    # one_ef.is_mix = True
                                    # todo(note): these are not TRUE efs, but only mixing preds as neg examples for training
                                    one_sent.entity_fillers.append(one_ef)
                                    mix_pred_ef_count += 1
            if lambda_arg>0.:
                # todo(note): since currently we are predicting all candidates for one event
                arg_losses = self.arg_linker.loss(ms_items, bert_expr, basic_expr, dynamic_prepare=True)
                all_arg_losses.append(arg_losses)
            if lambda_span>0.:
                span_losses = self.span_expander.loss(ms_items, bert_expr)
                all_span_losses.append(span_losses)
        # =====
        # final loss sum and backward
        info = {"doc": ndoc, "sent": nsent, "fb": 1, "mix_pef": mix_pred_ef_count}
        if len(all_packs) > 0:
            self.collect_loss_and_backward(["cand", "arg", "span"], [all_cand_losses, all_arg_losses, all_span_losses],
                                           [lambda_cand, lambda_arg, lambda_span], info, training, loss_factor)
        return info

# =====
# special components
# todo(note): always assume one event per ms_item; mostly adopted from those from "model_dec.py"

class CandidateExtractorConf(Conf):
    def __init__(self):
        self.hidden_dim = 512
        # training
        self.train_neg_rate = 0.5  # inside neg rate & default outside rate
        self.train_neg_rate_outside = -1.  # active if >=0.
        self.nil_penalty = 100.  # penalty for nil(idx=0)
        # testing
        self.pred_sent_ratio = 0.5  # at most how many cands to the ratio of sent_length
        self.pred_sent_ratio_sep = True  # separate ratio for each sent? or mix them together and sort?

    def do_validate(self):
        if self.train_neg_rate_outside<0:
            self.train_neg_rate_outside = self.train_neg_rate

class CandidateExtractor(BasicNode):
    def __init__(self, pc, conf: CandidateExtractorConf, input_enc_dims):
        super().__init__(pc, None, None)
        self.conf = conf
        # scorer
        self.adp = self.add_sub_node('adp', TaskSpecAdp(pc, input_enc_dims, [], conf.hidden_dim))
        adp_hidden_size = self.adp.get_output_dims()[0]
        self.predictor = self.add_sub_node('pred', Affine(pc, adp_hidden_size, 2, init_rop=NoDropRop()))  # 0 as nil
        # others
        self.id_counter = defaultdict(int)  # docid->ef-count (make sure unique ef-id)
        self.valid_hlidx = HLabelIdx(["unk"], [1])

    def loss(self, ms_items: List, bert_expr, basic_expr, margin=0.):
        conf = self.conf
        bsize = len(ms_items)
        # build targets (include all sents)
        # todo(note): use "x.entity_fillers" for getting gold args
        offsets_t, masks_t, _, items_arr, labels_t = PrepHelper.prep_targets(
            ms_items, lambda x: x.entity_fillers, True, True, conf.train_neg_rate, conf.train_neg_rate_outside, True)
        labels_t.clamp_(max=1)  # either 0 or 1
        # -----
        # return 0 if all no targets
        if BK.get_shape(offsets_t, -1) == 0:
            zzz = BK.zeros([])
            return [[zzz, zzz, zzz]]
        # -----
        arange_t = BK.arange_idx(bsize).unsqueeze(-1)  # [bsize, 1]
        sel_bert_t = bert_expr[arange_t, offsets_t]  # [bsize, ?, Fold, D]
        sel_basic_t = None if basic_expr is None else basic_expr[arange_t, offsets_t]  # [bsize, ?, D']
        hiddens = self.adp(sel_bert_t, sel_basic_t, [])  # [bsize, ?, D"]
        # build loss
        logits = self.predictor(hiddens)  # [bsize, ?, Out]
        log_probs = BK.log_softmax(logits, -1)
        picked_log_probs = - BK.gather_one_lastdim(log_probs, labels_t).squeeze(-1)  # [bsize, ?]
        masked_losses = picked_log_probs * masks_t
        # loss_sum, loss_count, gold_count
        return [[masked_losses.sum(), masks_t.sum(), (labels_t > 0).float().sum()]]

    def predict(self, ms_items: List, bert_expr, basic_expr):
        conf = self.conf
        bsize = len(ms_items)
        # build targets (include all sents)
        offsets_t, masks_t, _, _, _ = PrepHelper.prep_targets(ms_items, lambda x: [], True, True, 1., 1., False)
        arange_t = BK.arange_idx(bsize).unsqueeze(-1)  # [bsize, 1]
        sel_bert_t = bert_expr[arange_t, offsets_t]  # [bsize, ?, Fold, D]
        sel_basic_t = None if basic_expr is None else basic_expr[arange_t, offsets_t]  # [bsize, ?, D']
        hiddens = self.adp(sel_bert_t, sel_basic_t, [])  # [bsize, ?, D"]
        logits = self.predictor(hiddens)  # [bsize, ?, Out]
        # -----
        log_probs = BK.log_softmax(logits, -1)
        log_probs[:,:,0] -= conf.nil_penalty  # encourage more predictions
        topk_log_probs, topk_log_labels = log_probs.max(dim=-1)  # [bsize, ?, k]
        # decoding
        head_offsets_arr = BK.get_value(offsets_t)  # [bs, ?]
        masks_arr = BK.get_value(masks_t)
        topk_log_probs_arr, topk_log_labels_arr = BK.get_value(topk_log_probs), BK.get_value(topk_log_labels)  # [bsize, ?, k]
        for one_ms_item, one_offsets_arr, one_masks_arr, one_logprobs_arr, one_labels_arr \
                in zip(ms_items, head_offsets_arr, masks_arr, topk_log_probs_arr, topk_log_labels_arr):
            # build tidx2sidx
            one_sents = one_ms_item.sents
            one_offsets = one_ms_item.offsets
            tidx2sidx = []
            for idx in range(1, len(one_offsets)):
                tidx2sidx.extend([idx-1]*(one_offsets[idx]-one_offsets[idx-1]))
            # get all candidates
            all_candidates = [[] for _ in one_sents]
            for cur_offset, cur_valid, cur_logprob, cur_label in zip(one_offsets_arr, one_masks_arr, one_logprobs_arr, one_labels_arr):
                if not cur_valid or cur_label<=0:
                    continue
                # which sent
                cur_offset = int(cur_offset)
                cur_sidx = tidx2sidx[cur_offset]
                cur_sent = one_sents[cur_sidx]
                minus_offset = one_ms_item.offsets[cur_sidx] - 1  # again consider the ROOT
                cur_mention = Mention(HardSpan(cur_sent.sid, cur_offset-minus_offset, None, None))
                all_candidates[cur_sidx].append((cur_sent, cur_mention, cur_label, cur_logprob))
            # keep certain ratio for each sent separately?
            final_candidates = []
            if conf.pred_sent_ratio_sep:
                for one_sent, one_sent_candidates in zip(one_sents, all_candidates):
                    cur_keep_num = max(int(conf.pred_sent_ratio * (one_sent.length-1)), 1)
                    one_sent_candidates.sort(key=lambda x: x[-1], reverse=True)
                    final_candidates.extend(one_sent_candidates[:cur_keep_num])
            else:
                all_size = 0
                for one_sent, one_sent_candidates in zip(one_sents, all_candidates):
                    all_size += one_sent.length - 1
                    final_candidates.extend(one_sent_candidates)
                final_candidates.sort(key=lambda x: x[-1], reverse=True)
                final_keep_num = max(int(conf.pred_sent_ratio * all_size), len(one_sents))
                final_candidates = final_candidates[:final_keep_num]
            # add them all
            for cur_sent, cur_mention, cur_label, cur_logprob in final_candidates:
                cur_logprob = float(cur_logprob)
                doc_id = cur_sent.doc.doc_id
                self.id_counter[doc_id] += 1
                new_id = f"ef-{doc_id}-{self.id_counter[doc_id]}"
                hlidx = self.valid_hlidx
                new_ef = EntityFiller(new_id, cur_mention, str(hlidx), None, True, type_idx=hlidx, score=cur_logprob)
                cur_sent.pred_entity_fillers.append(new_ef)

# b tasks/zie/models3/modelA.py:265
