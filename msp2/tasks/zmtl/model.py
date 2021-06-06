#

# the new model with the mtl modules

from typing import List, Dict
from collections import Counter
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.nn.modules import BaseModelConf, BaseModel
from msp2.data.vocab import VocabPackage
from msp2.data.inst import yield_sents, Sent, DataPadder, Doc, Frame
from msp2.data.stream import BatchHelper
from msp2.utils import ConfEntryChoices, zlog
from .modules import *

# =====

class ZmtlModelConf(BaseModelConf):
    def __init__(self):
        super().__init__()
        # components
        # -- input + enc
        self.enc_conf = ConfEntryChoices({"plain": ZEncoderPlainConf(), "bert": ZEncoderBertConf()}, "bert")
        # -- dec
        self.srl_conf = ConfEntryChoices({"srl": ZDecoderSRLConf(), "srl2": ZDecoderSRL2Conf(), "none": None}, "srl2")  # srl
        self.pos_conf = ConfEntryChoices({"upos": ZDecoderUPOSConf(), "none": None}, "none")  # upos
        self.dep_conf = ConfEntryChoices({"udep": ZDecoderUDEPConf(), "none": None}, "none")  # udep
        # --
        self.frame_based = False  # whether inst should be frame or sent?
        # decode
        self.decode_sent_thresh_diff = 20  # sent diff thresh in decoding
        self.decode_sent_thresh_batch = 8  # num sent one time
        # =====
        # aug model forward!
        self.aug_model_confs = []  # "." means self!!
        self.aug_model_paths = []  # should match confs
        self.aug_times = 0  # maybe repeat for example, in MC-dropout
        self.aug_training_flag = False  # when aug forward, training or not?
        # =====
        # score model forward!
        self.score_times = 1  # repeat for scoring, for estimating model confidence.
        self.score_training_flag = False  # when score forward, training or not?
        # =====
        # special backward
        self.pcgrad = ConfEntryChoices({"yes": PCGradHelperConf(), "none": None}, "none")
        self.opt_sparse = ConfEntryChoices({"yes": OptimSparseHelperConf(), "none": None}, "none")

@node_reg(ZmtlModelConf)
class ZmtlModel(BaseModel):
    def __init__(self, conf: ZmtlModelConf, vpack: VocabPackage, no_pre_build_optims=False):
        super().__init__(conf)
        conf: ZmtlModelConf = self.conf
        self.vpack = vpack
        # =====
        # components
        # -- encoder
        self.enc: ZEncoder = conf.enc_conf.make_node(vpack)
        # -- decoders
        self.srl = conf.srl_conf.make_node("srl", vpack.get_voc('evt'), vpack.get_voc('arg'), self.enc) if conf.srl_conf is not None else None
        self.pos = conf.pos_conf.make_node("pos", vpack.get_voc('upos'), self.enc) if conf.pos_conf is not None else None
        self.dep = conf.dep_conf.make_node("dep", vpack.get_voc('deplab'), self.enc) if conf.dep_conf is not None else None
        # -- mediator
        self.med = ZMediator(self.enc, [self.pos, self.dep, self.srl])
        # -- pcgrad
        self.pcgrad = conf.pcgrad.make_node() if conf.pcgrad is not None else None
        # -- sparsity
        self.opt_sparse = conf.opt_sparse.make_node() if conf.opt_sparse is not None else None
        # =====
        # --
        if not no_pre_build_optims:
            zzz = self.optims  # finally build optim!
        # --
        # load aug models
        assert conf.aug_training_flag or conf.aug_times<=1, "Meaningless to forward multiple times in testing mode!"
        self.aug_models = self._load_aug_models(conf.aug_model_confs, conf.aug_model_paths)
        # for score models
        assert conf.score_training_flag or conf.score_times<=1, "Meaningless to forward multiple times in testing mode!"
        # --

    # load aug models
    def _load_aug_models(self, aug_confs: List[str], aug_paths: List[str]):
        assert len(aug_confs) == len(aug_paths)
        # --
        ret_models = []
        for one_conf, one_path in zip(aug_confs, aug_paths):
            if one_conf == ".":
                assert one_path == "."
                zlog("Add self as aug model!")
                one_m = self  # note: simply add self!!
            else:
                model_conf = ZmtlModelConf()
                model_conf.update_from_args([one_conf], quite=True, check=False, add_global_key=False, validate=True)
                zlog(f"Load model conf from {one_conf}!")
                one_m = ZmtlModel(model_conf, self.vpack, no_pre_build_optims=True)  # note: reuse vpack, require same vocab!!
                one_m.load(one_path, strict=True)
                zlog(f"Load model from {one_path}!")
            ret_models.append(one_m)
        return ret_models
    # --

    # --
    # helper
    def _yield_insts(self, insts):
        if self.conf.frame_based:
            for inst in insts:
                if isinstance(inst, Frame):  # already one frame
                    yield inst
                elif isinstance(inst, Doc):  # spread if only that is a Doc
                    for sent in inst.sents:
                        yield from sent.events
                else:
                    yield from inst.events
        else:
            for inst in insts:
                if isinstance(inst, Doc):  # spread if only that is a Doc
                    yield from inst.sents
                else:
                    yield inst
        # --

    def get_emb(self):
        if isinstance(self.enc, ZEncoderPlain):
            return self.enc.emb
        else:
            return None

    def update(self, lrate: float, grad_factor: float):
        super().update(lrate, grad_factor)
        # --
        if self.opt_sparse is not None:
            self.opt_sparse.do_update(self.enc.get_layered_params(), lrate)
        # --

    # =====
    def enc_forward(self, enc_cached_input, out_scores: Dict, refresh_training: bool):
        if refresh_training is not None:
            self.refresh_batch(refresh_training)
        med = self.med
        self.enc.forward(None, med, cached_input=enc_cached_input)
        ZMediator.append_scores(med.main_scores, out_scores)
        med.restart()
        # --

    # force_lidx means we only train for the 'force_lidx'
    def loss_on_batch(self, insts: List, loss_factor=1., training=True, force_lidx=None, **kwargs):
        conf: ZmtlModelConf = self.conf
        self.refresh_batch(training)
        # --
        # import torch
        # torch.autograd.set_detect_anomaly(True)
        # --
        actual_insts = list(self._yield_insts(insts))
        med = self.med
        enc_cached_input = self.enc.prepare_inputs(actual_insts)
        # ==
        # if needed, forward other models (can be self)
        aug_scores = {}
        with BK.no_grad_env():
            if conf.aug_times >= 1:
                # forward all at once!!
                _mm_input = enc_cached_input if (conf.aug_times == 1) else self.enc.prepare_inputs(actual_insts*conf.aug_times)
                for mm in self.aug_models:  # add them all to aug_scores!!
                    mm.enc_forward(_mm_input, aug_scores, conf.aug_training_flag)
        # ==
        self.refresh_batch(training)
        med.force_lidx = force_lidx  # note: special assign
        # enc
        self.enc.forward(None, med, cached_input=enc_cached_input)
        # dec
        med.aug_scores = aug_scores  # note: assign here!!
        all_losses = med.do_losses()
        # --
        # final loss and backward
        info = {"inst0": len(insts), "inst": len(actual_insts), "fb": 1, "fb0": 0}
        final_loss, loss_info = self.collect_loss(all_losses, ret_dict=(self.pcgrad is not None))
        info.update(loss_info)
        if training:
            if self.pcgrad is not None:
                # self.pcgrad.do_backward(self.parameters(), final_loss, loss_factor)
                # note: we only specially treat enc's, for others, grads will always be accumulated!
                self.pcgrad.do_backward(self.enc.parameters(), final_loss, loss_factor)
            else:  # as usual
                # assert final_loss.requires_grad
                if BK.get_value(final_loss).item() > 0:  # note: loss should be >0 usually!!
                    BK.backward(final_loss, loss_factor)
                else:  # no need to backwrad if no loss
                    info["fb0"] = 1
        med.restart()  # clean!
        med.force_lidx = None  # clear!
        return info

    def predict_on_batch(self, insts: List, **kwargs):
        conf: ZmtlModelConf = self.conf
        self.refresh_batch(False)
        # --
        actual_insts = list(self._yield_insts(insts))
        med = self.med
        # --
        info_counter = Counter()
        if len(actual_insts) > 0:  # avoid empty inputs!
            with BK.no_grad_env():
                # batch run inside if input is doc
                inst_buckets = BatchHelper.group_buckets(
                    actual_insts, thresh_diff=conf.decode_sent_thresh_diff, thresh_all=conf.decode_sent_thresh_batch,
                    size_f=lambda x: 1, sort_key=lambda x: len(x.sent))
                for one_insts in inst_buckets:
                    # --
                    self.refresh_batch(False)  # need to refresh each time!!
                    # --
                    # enc
                    rets = self.enc.forward(one_insts, med)
                    info_counter += Counter(rets[-1])
                    # dec
                    one_info = med.do_preds()
                    info_counter += one_info
        # --
        info = {"inst0": len(insts), "inst": len(actual_insts)}
        info.update(info_counter)
        med.restart()  # clean
        return info

    def score_on_batch(self, insts: List, **kwargs):
        conf: ZmtlModelConf = self.conf
        # --
        with BK.no_grad_env():
            self.refresh_batch(conf.score_training_flag)
            actual_insts = list(self._yield_insts(insts))
            # forward enc
            med = self.med
            enc_cached_input = self.enc.prepare_inputs(actual_insts * conf.score_times)  # multiple times
            self.enc.forward(None, med, cached_input=enc_cached_input)
            # do score with dec
            # note: do we need to split here?
            info_counter = med.do_scores(orig_insts=actual_insts)
            # --
            info = {"inst0": len(insts), "inst": len(actual_insts), "forw": 1}
            info.update(info_counter)
            # --
            med.restart()
            return info

# --
# b msp2/tasks/zmtl/model:76
