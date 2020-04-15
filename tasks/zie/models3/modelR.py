#

# especially for realis prediction given event
# (maybe also with type prediction for aux loss)

from typing import List
from copy import copy as shallow_copy
from msp.utils import Random, zlog, Conf
from msp.nn import BK
from msp.nn.layers import BasicNode, Affine, NoDropRop
from ..common.data import DocInstance, EVENT_REALIS_LIST
from ..common.vocab import IEVocabPackage
from ..common.model import MyIEModel, MyIEModelConf, BaseInferenceConf, BaseTrainingConf
from .model_enc import M3EncConf, M3Encoder, MultiSentItem
from .model_dec import TaskSpecAdp, HLabelVocab, PrepHelper

# =====
# conf

class M3RInferenceConf(BaseInferenceConf):
    def __init__(self):
        super().__init__()
        #
        self.batch_size = 10  # single doc?
        self.decode_verbose = False

class M3RTrainingConf(BaseTrainingConf):
    def __init__(self):
        super().__init__()

class M3RIEModelConf(MyIEModelConf):
    def __init__(self):
        super().__init__(M3RInferenceConf(), M3RTrainingConf())
        # components
        # encoding, todo(note): here replace bter conf!!
        self.bt_conf = M3EncConf()
        # decoding
        self.c_pred = RealisTypePredictorConf()

class M3RIEModel(MyIEModel):
    def __init__(self, conf: M3RIEModelConf, vpack: IEVocabPackage):
        super().__init__(conf, vpack)
        # components
        self.predictor: RealisTypePredictor = self.decoders[0]
        # vocab
        self.hl_evt: HLabelVocab = self.vpack.get_voc("hl_evt")

    # prepare for encoder
    def get_inst_preper(self, training, **kwargs):
        def _preper(inst):
            self.bter.prepare_inst(inst)
            return inst
        return _preper

    def build_encoder(self) -> BasicNode:
        return M3Encoder(self.pc, self.conf.bt_conf, self.conf.tconf, self.vpack)

    def build_decoders(self) -> List[BasicNode]:
        input_enc_dims = self.bter.speical_output_dims()  # todo(note): from M3Enc
        predictor = RealisTypePredictor(self.pc, self.conf.c_pred, self.vpack.get_voc("hl_evt"), input_enc_dims)
        return [predictor]

    # =====
    # testing
    def inference_on_batch(self, insts: List[DocInstance], **kwargs):
        self.refresh_batch(False)
        # -----
        if len(insts) == 0:
            return {}
        # -----
        # todo(note): first do shallow copy!
        for one_doc in insts:
            for one_sent in one_doc.sents:
                one_sent.pred_entity_fillers = [z for z in one_sent.entity_fillers]
                one_sent.pred_events = [shallow_copy(z) for z in one_sent.events]
        # -----
        ndoc, nsent = len(insts), 0
        iconf = self.conf.iconf
        with BK.no_grad_env():
            # splitting into buckets
            all_packs = self.bter.run(insts, training=False)
            for one_pack in all_packs:
                ms_items, bert_expr, basic_expr = one_pack
                nsent += len(ms_items)
                self.predictor.predict(ms_items, bert_expr, basic_expr)
        info = {"doc": ndoc, "sent": nsent, "num_evt": sum(len(z.pred_events) for z in insts)}
        if iconf.decode_verbose:
            zlog(f"Decode one mini-batch: {info}")
        return info

    # training
    def fb_on_batch(self, annotated_insts: List[DocInstance], training=True, loss_factor=1., **kwargs):
        self.refresh_batch(training)
        # -----
        if len(annotated_insts) == 0:
            return {}
        # -----
        ndoc, nsent = len(annotated_insts), 0
        all_packs = self.bter.run(annotated_insts, training=training)
        all_losses = []
        for one_pack in all_packs:
            ms_items, bert_expr, basic_expr = one_pack
            nsent += len(ms_items)
            pred_loss = self.predictor.loss(ms_items, bert_expr, basic_expr)
            all_losses.append(pred_loss)
        # -----
        info = {"doc": ndoc, "sent": nsent, "fb": 1}
        if len(all_packs) == 0:
            return info
        self.collect_loss_and_backward(["evt"], [all_losses], [1.], info, training, loss_factor)
        return info

# =====
# simple classifier for predicting realis (maybe with aux task for type)

class RealisTypePredictorConf(Conf):
    def __init__(self):
        self.hidden_dim = 512
        # lambdas
        self.lambda_realis = 1.
        self.lambda_type = 0.5
        # testing
        self.pred_realis = True
        self.pred_type = False  # whether predict type?

class RealisTypePredictor(BasicNode):
    def __init__(self, pc, conf: RealisTypePredictorConf, vocab: HLabelVocab, input_enc_dims):
        super().__init__(pc, None, None)
        self.conf = conf
        self.vocab = vocab
        assert vocab.nil_as_zero
        VOCAB_LAYER = -1  # todo(note): simply use final largest layer
        self.lidx2hlidx = vocab.layered_hlidx[VOCAB_LAYER]  # int-idx -> HLabelIdx
        # scorer
        self.adp = self.add_sub_node('adp', TaskSpecAdp(pc, input_enc_dims, [], conf.hidden_dim))
        adp_hidden_size = self.adp.get_output_dims()[0]
        # fixed types for realis
        self.realis_predictor = self.add_sub_node('pr', Affine(pc, adp_hidden_size, len(EVENT_REALIS_LIST), init_rop=NoDropRop()))
        # type predictor as a possible aux task
        self.type_predictor = self.add_sub_node('pt', Affine(pc, adp_hidden_size, len(self.lidx2hlidx), init_rop=NoDropRop()))

    # =====

    def loss(self, ms_items: List, bert_expr, basic_expr):
        conf = self.conf
        bsize = len(ms_items)
        # use gold targets: only use positive samples!!
        offsets_t, masks_t, _, items_arr, labels_t = PrepHelper.prep_targets(
            ms_items, lambda x: x.events, True, False, 0., 0., True)  # [bs, ?]
        realis_flist = [(-1 if (z is None or z.realis_idx is None) else z.realis_idx) for z in items_arr.flatten()]
        realis_t = BK.input_idx(realis_flist).view(items_arr.shape)  # [bs, ?]
        realis_mask = (realis_t >= 0).float()
        realis_t.clamp_(min=0)  # make sure all idxes are legal
        # -----
        # return 0 if all no targets
        if BK.get_shape(offsets_t, -1) == 0:
            zzz = BK.zeros([])
            return [[zzz, zzz, zzz], [zzz, zzz, zzz]]  # realis, types
        # -----
        arange_t = BK.arange_idx(bsize).unsqueeze(-1)  # [bsize, 1]
        sel_bert_t = bert_expr[arange_t, offsets_t]  # [bsize, ?, Fold, D]
        sel_basic_t = None if basic_expr is None else basic_expr[arange_t, offsets_t]  # [bsize, ?, D']
        hiddens = self.adp(sel_bert_t, sel_basic_t, [])  # [bsize, ?, D"]
        # build losses
        loss_item_realis = self._get_one_loss(self.realis_predictor, hiddens, realis_t, realis_mask, conf.lambda_realis)
        loss_item_type = self._get_one_loss(self.type_predictor, hiddens, labels_t, masks_t, conf.lambda_type)
        return [loss_item_realis, loss_item_type]

    def _get_one_loss(self, predictor, hidden_t, labels_t, masks_t, lambda_loss):
        logits = predictor(hidden_t)  # [bsize, ?, Out]
        log_probs = BK.log_softmax(logits, -1)
        picked_neg_log_probs = - BK.gather_one_lastdim(log_probs, labels_t).squeeze(-1)  # [bsize, ?]
        masked_losses = picked_neg_log_probs * masks_t
        # loss_sum, loss_count, gold_count(only for type)
        return [masked_losses.sum()*lambda_loss, masks_t.sum(), (labels_t>0).float().sum()]

    def predict(self, ms_items: List, bert_expr, basic_expr):
        conf = self.conf
        bsize = len(ms_items)
        # todo(note): use the pred_events which are shallow copied from inputs
        offsets_t, masks_t, _, items_arr, _ = PrepHelper.prep_targets(
            ms_items, lambda x: x.pred_events, True, False, 0., 0., True)  # [bs, ?]
        # -----
        if BK.get_shape(offsets_t, -1) == 0:
            return  # no input
        # -----
        # similar ones
        arange_t = BK.arange_idx(bsize).unsqueeze(-1)  # [bsize, 1]
        sel_bert_t = bert_expr[arange_t, offsets_t]  # [bsize, ?, Fold, D]
        sel_basic_t = None if basic_expr is None else basic_expr[arange_t, offsets_t]  # [bsize, ?, D']
        hiddens = self.adp(sel_bert_t, sel_basic_t, [])  # [bsize, ?, D"]
        # predict: only top-1!
        if conf.pred_realis:
            self._pred_and_put_res(self.realis_predictor, hiddens, items_arr, self._put_realis)
        if conf.pred_type:
            self._pred_and_put_res(self.type_predictor, hiddens, items_arr, self._put_type)

    # =====
    # specific setting method
    def _put_realis(self, evt, score, label_idx):
        evt.set_realis(realis_idx=int(label_idx), realis_score=float(score))

    def _put_type(self, evt, score, label_idx):
        if label_idx == 0:
            return
        this_hlidx = self.lidx2hlidx[label_idx]
        # directly rewrite things!
        evt.score = float(score)
        evt.type = str(this_hlidx)
        evt.type_idx = this_hlidx
    # =====

    # predict and put results inplace!
    def _pred_and_put_res(self, predictor, hidden_t, evt_arr, put_f):
        logits = predictor(hidden_t)  # [bsize, ?, Out]
        log_probs = BK.log_softmax(logits, -1)
        max_log_probs, max_label_idxes = log_probs.max(-1)  # [bs, ?], simply argmax prediction
        max_log_probs_arr, max_label_idxes_arr = BK.get_value(max_log_probs), BK.get_value(max_label_idxes)
        for evt_row, lprob_row, lidx_row in zip(evt_arr, max_log_probs_arr, max_label_idxes_arr):
            for one_evt, one_lprob, one_lidx in zip(evt_row, lprob_row, lidx_row):
                if one_evt is not None:
                    put_f(one_evt, one_lprob, one_lidx)  # callback for inplace setting
