#

# another one: simplify things and remover unnecessary parts

from typing import List
import numpy as np

from msp.utils import Random, zlog
from msp.nn.layers import BasicNode, Affine
from msp.nn import BK
from msp.zext.seq_helper import DataPadder
from msp.zext.ie import HLabelIdx, HLabelVocab
from msp.zext.process_train import SVConf, ScheduledValue

from ..common.data import DocInstance, Sentence, Mention, HardSpan, EntityFiller, Event
from ..common.vocab import IEVocabPackage
from ..common.model import MyIEModel, MyIEModelConf, BaseInferenceConf, BaseTrainingConf
from ..common.helper_span import SpanExpanderDep, SpanExpanderExternal

from .model_enc import M3EncConf, M3Encoder
from .model_dec import MentionExtractorConf, MentionExtractor, ArgLinkerConf, ArgLinker
from .model_expand import ArgSpanExpanderConf, ArgSpanExpander

# =====
# conf

#
class M3InferenceConf(BaseInferenceConf):
    def __init__(self):
        super().__init__()
        #
        self.batch_size = 1  # single doc calculation
        # whether lookup for each component?
        self.lookup_ef = 0.
        self.lookup_evt = 0.
        # whether pred if not lookup
        self.pred_ef = 1.
        self.pred_evt = 1.
        self.pred_arg = 1.
        self.pred_span = 0.
        #
        self.decode_verbose = False
        # expand span by what?
        self.expand_span_method = "dep"
        self.expand_span_ext_file = ""

#
class M3TrainingConf(BaseTrainingConf):
    def __init__(self):
        super().__init__()
        # lambdas for training
        # mention extract
        self.lambda_mention_ef = SVConf().init_from_kwargs(val=1.0)
        self.lambda_mention_evt = SVConf().init_from_kwargs(val=1.0)
        # ef/evt mention affinity score
        self.lambda_affi_ef = SVConf().init_from_kwargs(val=0.)
        self.lambda_affi_evt = SVConf().init_from_kwargs(val=0.)
        # arg link
        self.lambda_arg = SVConf().init_from_kwargs(val=1.0)
        # arg span expander
        self.lambda_span = SVConf().init_from_kwargs(val=0.)

#
class M3IEModelConf(MyIEModelConf):
    def __init__(self):
        super().__init__(M3InferenceConf(), M3TrainingConf())
        # components
        # encoding
        # todo(note): here replace bter conf!!
        self.bt_conf = M3EncConf()
        # decoding
        self.c_ef = MentionExtractorConf()
        self.c_evt = MentionExtractorConf().init_from_kwargs(dec_topk_label=2)  # allow 2 types for event
        self.c_arg = ArgLinkerConf()
        self.c_span = ArgSpanExpanderConf()
        # special mode: include pred efs in training for args
        # todo(note): this is not an elegant mode since mixing preds into golds
        self.mix_pred_ef = False
        self.mix_pred_ef_rate = 0.25
        # simple pos based ef getter
        self.pos_ef_getter = False
        # self.pos_ef_getter_list = ["PRON", "PROPN", "NOUN"]
        self.pos_ef_getter_list = ["PRON", "PROPN", "NOUN", "ADJ", "VERB", "NUM", "SYM"]

# =====
class M3IEModel(MyIEModel):
    def __init__(self, conf: M3IEModelConf, vpack: IEVocabPackage):
        super().__init__(conf, vpack)
        # components
        self.ef_extractor: MentionExtractor = self.decoders[0]
        self.evt_extractor: MentionExtractor = self.decoders[1]
        self.arg_linker: ArgLinker = self.decoders[2]
        self.span_expander: ArgSpanExpander = self.decoders[3]
        # vocab
        self.hl_ef: HLabelVocab = self.vpack.get_voc("hl_ef")
        self.hl_evt: HLabelVocab = self.vpack.get_voc("hl_evt")
        self.hl_arg: HLabelVocab = self.vpack.get_voc("hl_arg")
        # lambdas for training
        self.lambda_mention_ef = ScheduledValue("lambda_mention_ef", conf.tconf.lambda_mention_ef)
        self.lambda_mention_evt = ScheduledValue("lambda_mention_evt", conf.tconf.lambda_mention_evt)
        self.lambda_affi_ef = ScheduledValue("lambda_affi_ef", conf.tconf.lambda_affi_ef)
        self.lambda_affi_evt = ScheduledValue("lambda_affi_evt", conf.tconf.lambda_affi_evt)
        self.lambda_arg = ScheduledValue("lambda_arg", conf.tconf.lambda_arg)
        self.lambda_span = ScheduledValue("lambda_span", conf.tconf.lambda_span)
        self.add_scheduled_values(self.lambda_mention_ef)
        self.add_scheduled_values(self.lambda_mention_evt)
        self.add_scheduled_values(self.lambda_affi_ef)
        self.add_scheduled_values(self.lambda_affi_evt)
        self.add_scheduled_values(self.lambda_arg)
        self.add_scheduled_values(self.lambda_span)
        #
        self.random_sample_stream = Random.stream(Random.random_sample)
        self.pos_ef_getter_set = set(conf.pos_ef_getter_list)
        #
        if conf.iconf.expand_span_method == "dep":
            self.static_span_expander = SpanExpanderDep()
        elif conf.iconf.expand_span_method == "ext":
            self.static_span_expander = SpanExpanderExternal(conf.iconf.expand_span_ext_file)
        else:
            zlog("No static span expander!")
            self.static_span_expander = None

    def get_inst_preper(self, training, **kwargs):
        conf = self.conf
        def _training_preper(inst):
            if conf.pos_ef_getter:
                self.fake_efs(inst, lambda x: x.entity_fillers, False)
            self.bter.prepare_inst(inst)
            self.arg_linker.prepare_inst(inst)
            return inst
        def _testing_preper(inst):
            if conf.pos_ef_getter:
                self.fake_efs(inst, lambda x: x.pred_entity_fillers, True)
            self.bter.prepare_inst(inst)
            return inst
        return _training_preper if training else _testing_preper

    # make arg cands (fake ef) by filtering pos
    def fake_efs(self, inst, ef_getter, clear):
        # todo(note): fake ef get idx=1 (not 0 since it might be elimininated)
        fake_hlidx = self.hl_ef.layered_hlidx[-1][1]
        fake_count = 0
        for sent in inst.sents:
            target_entity_fillers = ef_getter(sent)
            if clear:
                target_entity_fillers.clear()
            covered_wids = set()
            for one_ef in target_entity_fillers:
                if one_ef.mention is not None:
                    covered_wids.add(one_ef.mention.hard_span.head_wid)
            added_efs = []
            for wid, upos in enumerate(sent.uposes.vals[1:], 1):
                if wid not in covered_wids and upos in self.pos_ef_getter_set:
                    this_mention = Mention(HardSpan(sent.sid, wid, None, None))
                    this_ef = EntityFiller("pos_ef-"+str(fake_count), this_mention, str(fake_hlidx), None, True, type_idx=fake_hlidx)
                    added_efs.append(this_ef)
                    fake_count += 1
            target_entity_fillers.extend(added_efs)

    def build_encoder(self) -> BasicNode:
        return M3Encoder(self.pc, self.conf.bt_conf, self.conf.tconf, self.vpack)

    def build_decoders(self) -> List[BasicNode]:
        input_enc_dims = self.bter.speical_output_dims()  # todo(note): from M3Enc
        ef_extractor = MentionExtractor(self.pc, self.conf.c_ef, self.vpack.get_voc("hl_ef"), "ef", input_enc_dims)
        evt_extractor = MentionExtractor(self.pc, self.conf.c_evt, self.vpack.get_voc("hl_evt"), "evt", input_enc_dims)
        arg_linker = ArgLinker(self.pc, self.conf.c_arg, self.vpack.get_voc("hl_arg"), input_enc_dims,
                               borrowed_evt_hlnode=evt_extractor.predictor, borrowed_evt_adp=evt_extractor.adp)
        arg_span_expander = ArgSpanExpander(self.pc, self.conf.c_span, input_enc_dims)
        return [ef_extractor, evt_extractor, arg_linker, arg_span_expander]

    # testing
    def inference_on_batch(self, insts: List[DocInstance], **kwargs):
        self.refresh_batch(False)
        test_constrain_evt_types = self.test_constrain_evt_types
        # -----
        if len(insts) == 0:
            return {}
        # -----
        ndoc, nsent = len(insts), 0
        iconf = self.conf.iconf
        with BK.no_grad_env():
            # splitting into buckets
            all_packs = self.bter.run(insts, training=False)
            for one_pack in all_packs:
                ms_items, bert_expr, basic_expr = one_pack
                nsent += len(ms_items)
                # ef
                if iconf.lookup_ef:
                    self.ef_extractor.lookup(ms_items)
                elif iconf.pred_ef:
                    self.ef_extractor.predict(ms_items, bert_expr, basic_expr)
                # evt
                if iconf.lookup_evt:
                    self.evt_extractor.lookup(ms_items, constrain_types=test_constrain_evt_types)
                elif iconf.pred_evt:
                    self.evt_extractor.predict(ms_items, bert_expr, basic_expr, constrain_types=test_constrain_evt_types)
            # deal with arg after pred all!!
            if iconf.pred_arg:
                for one_pack in all_packs:
                    ms_items, bert_expr, basic_expr = one_pack
                    self.arg_linker.predict(ms_items, bert_expr, basic_expr)
                    if iconf.pred_span:
                        self.span_expander.predict(ms_items, bert_expr)
        # collect all stats
        num_ef, num_evt, num_arg = 0, 0, 0
        for one_doc in insts:
            for one_sent in one_doc.sents:
                num_ef += len(one_sent.pred_entity_fillers)
                num_evt += len(one_sent.pred_events)
                num_arg += sum(len(z.links) for z in one_sent.pred_events)
                if self.static_span_expander is not None:
                    assert not iconf.pred_span, "Not compatible of these two modes!"
                    for one_ef in one_sent.pred_entity_fillers:
                        # todo(note): expand phrase by rule
                        one_hard_span = one_ef.mention.hard_span
                        head_wid = one_hard_span.head_wid
                        one_hard_span.wid, one_hard_span.length = self.static_span_expander.expand_span(head_wid, one_sent)
        info = {"doc": ndoc, "sent": nsent, "num_ef": num_ef, "num_evt": num_evt, "num_arg": num_arg}
        if iconf.decode_verbose:
            zlog(f"Decode one mini-batch: {info}")
        return info

    # training
    def fb_on_batch(self, annotated_insts: List[DocInstance], training=True, loss_factor=1., **kwargs):
        self.refresh_batch(training)
        assert self.train_constrain_evt_types is None, "Not implemented for training constrain_types"
        # -----
        if len(annotated_insts) == 0:
            return {}
        # -----
        ndoc, nsent = len(annotated_insts), 0
        lambda_mention_ef, lambda_mention_evt, lambda_arg, lambda_span = \
            self.lambda_mention_ef.value, self.lambda_mention_evt.value, self.lambda_arg.value, self.lambda_span.value
        # simply multi-task training, no explict interactions between them
        all_packs = self.bter.run(annotated_insts, training=training)
        all_ef_mention_losses = []
        all_evt_mention_losses = []
        all_arg_losses = []
        all_span_losses = []
        #
        mix_pred_ef_rate = self.conf.mix_pred_ef_rate
        mix_pred_ef = self.conf.mix_pred_ef
        mix_pred_ef_count = 0
        # =====
        cur_margin = self.margin.value
        for one_pack in all_packs:
            ms_items, bert_expr, basic_expr = one_pack
            nsent += len(ms_items)
            if lambda_mention_ef>0.:
                if mix_pred_ef:
                    # clear previous added fake ones
                    for one_msent in ms_items:
                        center_sent = one_msent.sents[one_msent.center_idx]
                        center_sent.entity_fillers = [z for z in center_sent.entity_fillers if not hasattr(z, "is_mix")]
                    # -----
                ef_losses = self.ef_extractor.loss(ms_items, bert_expr, basic_expr)
                all_ef_mention_losses.append(ef_losses)
            if lambda_mention_evt>0.:
                evt_losses = self.evt_extractor.loss(ms_items, bert_expr, basic_expr, margin=cur_margin)
                all_evt_mention_losses.append(evt_losses)
            if lambda_span>0.:
                span_losses = self.span_expander.loss(ms_items, bert_expr)
                all_span_losses.append(span_losses)
            # predict efs as candidates for args
            if mix_pred_ef:
                with BK.no_grad_env():
                    self.ef_extractor.predict(ms_items, bert_expr, basic_expr)
                    # mix into gold ones
                    for one_msent in ms_items:
                        center_sent = one_msent.sents[one_msent.center_idx]
                        # since we might cache insts, we do not consider previous mixed ones
                        hit_posi = set()
                        center_sent.entity_fillers = [z for z in center_sent.entity_fillers if not hasattr(z, "is_mix")]
                        for one_ef in center_sent.entity_fillers:
                            posi = one_ef.mention.hard_span.position()
                            hit_posi.add(posi)
                        # add predicted ones
                        for one_ef in center_sent.pred_entity_fillers:
                            posi = one_ef.mention.hard_span.position()
                            if posi not in hit_posi and next(self.random_sample_stream)<=mix_pred_ef_rate:
                                hit_posi.add(posi)
                                one_ef.is_mix = True
                                # todo(note): these are not TRUE efs, but only mixing preds as neg examples for training
                                center_sent.entity_fillers.append(one_ef)
                                mix_pred_ef_count += 1
        # =====
        # in some mode, we may want to collect predicted efs
        for one_pack in all_packs:
            ms_items, bert_expr, basic_expr = one_pack
            if lambda_arg>0.:
                arg_losses = self.arg_linker.loss(ms_items, bert_expr, basic_expr, dynamic_prepare=mix_pred_ef)
                all_arg_losses.append(arg_losses)
        # =====
        # final loss sum and backward
        info = {"doc": ndoc, "sent": nsent, "fb": 1, "mix_pef": mix_pred_ef_count}
        if len(all_packs) == 0:
            return info
        self.collect_loss_and_backward(["ef", "evt", "arg", "span"],
                                       [all_ef_mention_losses, all_evt_mention_losses, all_arg_losses, all_span_losses],
                                       [lambda_mention_ef, lambda_mention_evt, lambda_arg, lambda_span],
                                       info, training, loss_factor)
        # =====
        # # for debug
        # zlog([d.dataset for d in annotated_insts])
        # zlog(info)
        # =====
        return info

# b tasks/zie/models3/model:135
# b tasks/zie/models3/model:262, M3IEModel._debug_time==24
