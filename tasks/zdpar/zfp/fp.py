#

# the parser

from typing import List
import numpy as np

from msp.utils import Conf, Random, zlog, JsonRW, zfatal, zwarn
from msp.data import VocabPackage, MultiHelper
from msp.model import Model
from msp.nn import BK
from msp.nn import refresh as nn_refresh
from msp.nn.layers import BasicNode, Affine, RefreshOptions, NoDropRop
#
from msp.zext.seq_helper import DataPadder
from msp.zext.process_train import RConf, SVConf, ScheduledValue, OptimConf

from ..common.data import ParseInstance
from .enc import FpEncConf, FpEncoder
from .dec import FpDecConf, FpDecoder
from .masklm import MaskLMNodeConf, MaskLMNode

# =====
# confs

# decoding conf
class FpInferenceConf(Conf):
    def __init__(self):
        # overall
        self.batch_size = 32
        self.infer_single_length = 80  # single-inst batch if >= this length

# training conf
class FpTrainingConf(RConf):
    def __init__(self):
        super().__init__()
        # about files
        self.no_build_dict = False
        self.load_model = False
        self.load_process = False
        # batch arranger
        self.batch_size = 32
        self.train_min_length = 0
        self.train_skip_length = 100
        self.shuffle_train = True
        # optimizer and lrate factor for enc&dec&sl(mid)
        self.enc_optim = OptimConf()
        self.enc_lrf = SVConf().init_from_kwargs(val=1., which_idx="eidx", mode="none", min_val=0., max_val=1.)
        self.dec_optim = OptimConf()
        self.dec_lrf = SVConf().init_from_kwargs(val=1., which_idx="eidx", mode="none", min_val=0., max_val=1.)
        self.mid_optim = OptimConf()
        self.mid_lrf = SVConf().init_from_kwargs(val=1., which_idx="eidx", mode="none", min_val=0., max_val=1.)
        # margin
        self.margin = SVConf().init_from_kwargs(val=0.0)
        # ===
        # overwrite default ones
        self.patience = 3
        self.anneal_times = 5
        self.max_epochs = 50

# base conf
class FpParserConf(Conf):
    def __init__(self, iconf: FpInferenceConf=None, tconf: FpTrainingConf=None):
        self.iconf = FpInferenceConf() if iconf is None else iconf
        self.tconf = FpTrainingConf() if tconf is None else tconf
        # Model
        self.encoder_conf = FpEncConf()
        self.decoder_conf = FpDecConf()
        self.masklm_conf = MaskLMNodeConf()
        # lambdas for training
        self.lambda_parse = SVConf().init_from_kwargs(val=1.0)
        self.lambda_masklm = SVConf().init_from_kwargs(val=0.0)

# =====
# model

class FpParser(Model):
    def __init__(self, conf: FpParserConf, vpack: VocabPackage):
        self.conf = conf
        self.vpack = vpack
        tconf = conf.tconf
        # ===== Vocab =====
        self.label_vocab = vpack.get_voc("label")
        # ===== Model =====
        self.pc = BK.ParamCollection(True)
        # bottom-part: input + encoder
        self.enc = FpEncoder(self.pc, conf.encoder_conf, vpack)
        self.enc_output_dim = self.enc.get_output_dims()[0]
        self.enc_lrf_sv = ScheduledValue("enc_lrf", tconf.enc_lrf)
        self.pc.optimizer_set(tconf.enc_optim.optim, self.enc_lrf_sv, tconf.enc_optim,
                              params=self.enc.get_parameters(), check_repeat=True, check_full=True)
        # middle-part: structured layer at the middle (build later for convenient re-loading)
        self.slayer = self.build_slayer()
        self.mid_lrf_sv = ScheduledValue("mid_lrf", tconf.mid_lrf)
        if self.slayer is not None:
            self.pc.optimizer_set(tconf.mid_optim.optim, self.mid_lrf_sv, tconf.mid_optim,
                                  params=self.slayer.get_parameters(), check_repeat=True, check_full=True)
        # upper-part: decoder
        self.dec = self.build_decoder()
        self.dec_lrf_sv = ScheduledValue("dec_lrf", tconf.dec_lrf)
        self.pc.optimizer_set(tconf.dec_optim.optim, self.dec_lrf_sv, tconf.dec_optim,
                              params=self.dec.get_parameters(), check_repeat=True, check_full=True)
        # extra aux loss
        conf.masklm_conf._input_dim = self.enc_output_dim
        self.masklm = MaskLMNode(self.pc, conf.masklm_conf, vpack)
        self.pc.optimizer_set(tconf.dec_optim.optim, self.dec_lrf_sv, tconf.dec_optim,
                              params=self.masklm.get_parameters(), check_repeat=True, check_full=True)
        # ===== For training =====
        # schedule values
        self.margin = ScheduledValue("margin", tconf.margin)
        self.lambda_parse = ScheduledValue("lambda_parse", conf.lambda_parse)
        self.lambda_masklm = ScheduledValue("lambda_masklm", conf.lambda_masklm)
        self._scheduled_values = [self.margin, self.enc_lrf_sv, self.mid_lrf_sv, self.dec_lrf_sv,
                                  self.lambda_parse, self.lambda_masklm]
        # for refreshing dropouts
        self.previous_refresh_training = True

    # to be implemented
    def build_decoder(self):
        conf = self.conf
        # todo(note): might need to change if using slayer
        conf.decoder_conf._input_dim = self.enc_output_dim
        conf.decoder_conf._num_label = self.label_vocab.trg_len(True)  # todo(note): use the original idx
        dec = FpDecoder(self.pc, conf.decoder_conf, self.label_vocab)
        return dec

    def build_slayer(self):
        return None  # None means no slayer

    # called before each mini-batch
    def refresh_batch(self, training: bool):
        # refresh graph
        # todo(warn): make sure to remember clear this one
        nn_refresh()
        # refresh nodes
        if not training:
            if not self.previous_refresh_training:
                # todo(+1): currently no need to refresh testing mode multiple times
                return
            self.previous_refresh_training = False
            embed_rop = other_rop = RefreshOptions(training=False)  # default no dropout
        else:
            embed_rop, other_rop = self.enc.prepare_training_rop()
            # todo(warn): once-bug, don't forget this one!!
            self.previous_refresh_training = True
        # manually refresh
        self.enc.special_refresh(embed_rop, other_rop)
        for node in [self.dec, self.slayer, self.masklm]:
            if node is not None:
                node.refresh(other_rop)

    def update(self, lrate, grad_factor):
        self.pc.optimizer_update(lrate, grad_factor)

    def add_scheduled_values(self, v):
        self._scheduled_values.append(v)

    def get_scheduled_values(self):
        return self._scheduled_values

    # == load and save models
    # todo(warn): no need to load confs here
    def load(self, path, strict=True):
        self.pc.load(path, strict)
        # self.conf = JsonRW.load_from_file(path+".json")
        zlog(f"Load {self.__class__.__name__} model from {path}.", func="io")

    def save(self, path):
        self.pc.save(path)
        JsonRW.to_file(self.conf, path + ".json")
        zlog(f"Save {self.__class__.__name__} model to {path}.", func="io")

    def aug_words_and_embs(self, aug_vocab, aug_wv):
        return self.enc.aug_words_and_embs(aug_vocab, aug_wv)

    # common routines
    def collect_loss_and_backward(self, loss_names, loss_ts, loss_lambdas, info, training, loss_factor):
        final_losses = []
        for one_name, one_losses, one_lambda in zip(loss_names, loss_ts, loss_lambdas):
            if one_lambda>0. and len(one_losses)>0:
                num_sub_losses = len(one_losses[0])
                coll_sub_losses = []
                for i in range(num_sub_losses):
                    # todo(note): (loss_sum, loss_count, gold_count[opt])
                    this_loss_sum = BK.stack([z[i][0] for z in one_losses]).sum()
                    this_loss_count = BK.stack([z[i][1] for z in one_losses]).sum()
                    info[f"loss_sum_{one_name}{i}"] = this_loss_sum.item()
                    info[f"loss_count_{one_name}{i}"] = this_loss_count.item()
                    # optional extra count
                    if len(one_losses[0][i]) >= 3:  # has gold count
                        info[f"loss_count_extra_{one_name}{i}"] = BK.stack([z[i][2] for z in one_losses]).sum().item()
                    # todo(note): any case that loss-count can be 0?
                    coll_sub_losses.append(this_loss_sum / (this_loss_count + 1e-5))
                # sub losses are already multiplied by sub-lambdas
                weighted_sub_loss = BK.stack(coll_sub_losses).sum() * one_lambda
                final_losses.append(weighted_sub_loss)
        if len(final_losses)>0:
            final_loss = BK.stack(final_losses).sum()
            if training and final_loss.requires_grad:
                BK.backward(final_loss, loss_factor)

    # =====
    # training
    def fb_on_batch(self, annotated_insts: List[ParseInstance], training=True, loss_factor=1., **kwargs):
        self.refresh_batch(training)
        cur_lambda_parse, cur_lambda_masklm = self.lambda_parse.value, self.lambda_masklm.value
        # prepare for masklm todo(note): always prepare for input-mask-dropout-like
        input_word_mask_repl_arr, output_pred_mask_repl_arr, ouput_pred_idx_arr = self.masklm.prepare(annotated_insts, training)
        # encode
        enc_expr, mask_expr = self.enc.run(annotated_insts, training, input_word_mask_repl=input_word_mask_repl_arr)
        # get loss
        if cur_lambda_parse > 0.:
            parsing_loss = [self.dec.loss(annotated_insts, enc_expr, mask_expr)]
        else:
            parsing_loss = []
        if cur_lambda_masklm > 0.:
            masklm_loss = [self.masklm.loss(enc_expr, output_pred_mask_repl_arr, ouput_pred_idx_arr)]
        else:
            masklm_loss = []
        # -----
        info = {"fb": 1, "sent": len(annotated_insts), "tok": sum(map(len, annotated_insts))}
        self.collect_loss_and_backward(["parse", "masklm"], [parsing_loss, masklm_loss],
                                       [cur_lambda_parse, cur_lambda_masklm], info, training, loss_factor)
        return info

    # decoding
    def inference_on_batch(self, insts: List[ParseInstance], **kwargs):
        with BK.no_grad_env():
            self.refresh_batch(False)
            # encode
            enc_expr, mask_expr = self.enc.run(insts, False, input_word_mask_repl=None)  # no masklm in testing
            # decode
            self.dec.predict(insts, enc_expr, mask_expr)
            # =====
            # test for masklm
            input_word_mask_repl_arr, output_pred_mask_repl_arr, ouput_pred_idx_arr = self.masklm.prepare(insts, False)
            enc_expr2, mask_expr2 = self.enc.run(insts, False, input_word_mask_repl=input_word_mask_repl_arr)
            masklm_loss = self.masklm.loss(enc_expr2, output_pred_mask_repl_arr, ouput_pred_idx_arr)
            masklm_loss_val, masklm_loss_count, masklm_corr_count = [z.item() for z in masklm_loss[0]]
            # =====
        info = {"sent": len(insts), "tok": sum(map(len, insts)),
                "masklm_loss_val": masklm_loss_val, "masklm_loss_count": masklm_loss_count, "masklm_corr_count": masklm_corr_count}
        return info

# b tasks/zdpar/zfp/fp:223
