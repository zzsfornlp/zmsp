#

# the multitask model

from typing import List, Dict
import numpy as np
from msp.utils import zlog, zwarn, Random, Helper, Conf
from msp.data import VocabPackage
from msp.nn import BK
from msp.nn.layers import BasicNode, PosiEmbedding2, RefreshOptions, Dropout
from msp.nn.modules import EncConf, MyEncoder
from msp.nn.modules.berter2 import BertFeaturesWeightLayer
from msp.zext.process_train import SVConf, ScheduledValue
from .base import BaseModelConf, BaseModel, BaseModuleConf, BaseModule, LossHelper
from .helper import RPrepConf, RPrepNode, EntropyHelper
from .mods.embedder import EmbedderNodeConf, EmbedderNode, Inputter, AugWord2Node
from .mods.vrec import VRecEncoderConf, VRecEncoder, VRecConf, VRecNode
from .mods.masklm import MaskLMNodeConf, MaskLMNode
from .mods.plainlm import PlainLMNodeConf, PlainLMNode
from .mods.orderpr import OrderPredNodeConf, OrderPredNode
from .mods.dpar import DparG1DecoderConf, DparG1Decoder
from .mods.seqlab import SeqLabNodeConf, SeqLabNode
from .mods.seqcrf import SeqCrfNodeConf, SeqCrfNode
from ..data.insts import GeneralSentence, NpArrField

# =====
# the overall Model

class MtlMlmModelConf(BaseModelConf):
    def __init__(self):
        super().__init__()
        # components
        self.emb_conf = EmbedderNodeConf()
        self.mlm_conf = MaskLMNodeConf()
        self.plm_conf = PlainLMNodeConf()
        self.orp_conf = OrderPredNodeConf()
        self.orp_loss_special = False  # special loss for orp
        # non-pretraining parts
        self.dpar_conf = DparG1DecoderConf()
        self.upos_conf = SeqLabNodeConf()
        self.ner_conf = SeqCrfNodeConf()
        self.do_ner = False  # an extra flag
        self.ner_use_crf = True  # whether use crf for ner?
        # where pairwise repr comes from
        self.default_attn_count = 128  # by default, setting this to what?
        self.prepr_choice = "attn_max"  # rdist, rdist_abs, attn_sum, attn_avg, attn_max, attn_last
        # which encoder to use
        self.enc_choice = "vrec"  # by default, use the new one
        self.venc_conf = VRecEncoderConf()
        self.oenc_conf = EncConf().init_from_kwargs(enc_hidden=512, enc_rnn_layer=0)
        # another small encoder
        self.rprep_conf = RPrepConf()
        # agreement module
        self.lambda_agree = SVConf().init_from_kwargs(val=0., which_idx="eidx", mode="none")
        self.agree_loss_f = "js"
        # separate generator seed especially for testing mask
        self.testing_rand_gen_seed = 0
        self.testing_get_attns = False
        # todo(+N): should make this into another conf
        # for training and testing
        self.no_build_dict = False
        # -- length thresh
        self.train_max_length = 80
        self.train_min_length = 5
        self.test_single_length = 80
        # -- batch size
        self.train_shuffle = True
        self.train_batch_size = 80
        self.train_inst_copy = 1  # how many times to copy the training insts (for different masks)
        self.test_batch_size = 32
        # -- maxibatch for BatchArranger
        self.train_maxibatch_size = 20  # cache for this number of batches in BatchArranger
        self.test_maxibatch_size = -1  # -1 means cache all
        # -- batch_size_f
        self.train_batch_on_len = False  # whether use sent length as budgets rather then sent count
        self.test_batch_on_len = False
        # -----
        self.train_preload_model = False
        self.train_preload_process = False
        # interactive testing
        self.test_interactive = False  # special mode
        # load pre-training model?
        self.load_pretrain_model_name = ""
        # -----
        # special mode with extra word2
        # todo(+N): ugly patch!
        self.aug_word2 = False
        self.aug_word2_dim = 300
        self.aug_word2_pretrain = ""
        self.aug_word2_save_dir = ""
        self.aug_word2_aug_encoder = True  # special model's special mode, feature-based original encoder and stack another one!!
        self.aug_detach_ratio = 0.5
        self.aug_detach_dropout = 0.33
        self.aug_detach_numlayer = 6  # use mixture of how many last layers?

    def do_validate(self):
        if self.orp_loss_special:
            assert self.orp_conf.disturb_mode == "local_shuffle2"

class MtlMlmModel(BaseModel):
    def __init__(self, conf: MtlMlmModelConf, vpack: VocabPackage):
        super().__init__(conf)
        # for easier checking
        self.word_vocab = vpack.get_voc("word")
        # components
        self.embedder = self.add_node("emb", EmbedderNode(self.pc, conf.emb_conf, vpack))
        self.inputter = Inputter(self.embedder, vpack)  # not a node
        self.emb_out_dim = self.embedder.get_output_dims()[0]
        self.enc_attn_count = conf.default_attn_count
        if conf.enc_choice == "vrec":
            self.encoder = self.add_component("enc", VRecEncoder(self.pc, self.emb_out_dim, conf.venc_conf))
            self.enc_attn_count = self.encoder.attn_count
        elif conf.enc_choice == "original":
            conf.oenc_conf._input_dim = self.emb_out_dim
            self.encoder = self.add_node("enc", MyEncoder(self.pc, conf.oenc_conf))
        else:
            raise NotImplementedError()
        zlog(f"Finished building model's encoder {self.encoder}, all size is {self.encoder.count_allsize_parameters()}")
        self.enc_out_dim = self.encoder.get_output_dims()[0]
        # --
        conf.rprep_conf._rprep_vr_conf.matt_conf.head_count = self.enc_attn_count  # make head-count agree
        self.rpreper = self.add_node("rprep", RPrepNode(self.pc, self.enc_out_dim, conf.rprep_conf))
        # --
        self.lambda_agree = self.add_scheduled_value(ScheduledValue(f"agr:lambda", conf.lambda_agree))
        self.agree_loss_f = EntropyHelper.get_method(conf.agree_loss_f)
        # --
        self.masklm = self.add_component("mlm", MaskLMNode(self.pc, self.enc_out_dim, conf.mlm_conf, self.inputter))
        self.plainlm = self.add_component("plm", PlainLMNode(self.pc, self.enc_out_dim, conf.plm_conf, self.inputter))
        # todo(note): here we use attn as dim_pair, do not use pair if not using vrec!!
        self.orderpr = self.add_component("orp", OrderPredNode(
            self.pc, self.enc_out_dim, self.enc_attn_count, conf.orp_conf, self.inputter))
        # =====
        # pre-training pre-load point!!
        if conf.load_pretrain_model_name:
            zlog(f"At preload_pretrain point: Loading from {conf.load_pretrain_model_name}")
            self.pc.load(conf.load_pretrain_model_name, strict=False)
        # =====
        self.dpar = self.add_component("dpar", DparG1Decoder(
            self.pc, self.enc_out_dim, self.enc_attn_count, conf.dpar_conf, self.inputter))
        self.upos = self.add_component("upos", SeqLabNode(
            self.pc, "pos", self.enc_out_dim, self.conf.upos_conf, self.inputter))
        if conf.do_ner:
            if conf.ner_use_crf:
                self.ner = self.add_component("ner", SeqCrfNode(
                    self.pc, "ner", self.enc_out_dim, self.conf.ner_conf, self.inputter))
            else:
                self.ner = self.add_component("ner", SeqLabNode(
                    self.pc, "ner", self.enc_out_dim, self.conf.ner_conf, self.inputter))
        else:
            self.ner = None
        # for pairwise reprs (no trainable params here!)
        self.rel_dist_embed = self.add_node("oremb", PosiEmbedding2(self.pc, n_dim=self.enc_attn_count, max_val=100))
        self._prepr_f_attn_sum = lambda cache, rdist: BK.stack(cache.list_attn, 0).sum(0) if (len(cache.list_attn))>0 else None
        self._prepr_f_attn_avg = lambda cache, rdist: BK.stack(cache.list_attn, 0).mean(0) if (len(cache.list_attn))>0 else None
        self._prepr_f_attn_max = lambda cache, rdist: BK.stack(cache.list_attn, 0).max(0)[0] if (len(cache.list_attn))>0 else None
        self._prepr_f_attn_last = lambda cache, rdist: cache.list_attn[-1] if (len(cache.list_attn))>0 else None
        self._prepr_f_rdist = lambda cache, rdist: self._get_rel_dist_embed(rdist, False)
        self._prepr_f_rdist_abs = lambda cache, rdist: self._get_rel_dist_embed(rdist, True)
        self.prepr_f = getattr(self, "_prepr_f_"+conf.prepr_choice)  # shortcut
        # --
        self.testing_rand_gen = Random.create_sep_generator(conf.testing_rand_gen_seed)  # especial gen for testing
        # =====
        if conf.orp_loss_special:
            self.orderpr.add_node_special(self.masklm)
        # =====
        # extra one!!
        self.aug_word2 = self.aug_encoder = self.aug_mixturer = None
        if conf.aug_word2:
            self.aug_word2 = self.add_node("aug2", AugWord2Node(self.pc, conf.emb_conf, vpack,
                                                                "word2", conf.aug_word2_dim, self.emb_out_dim))
            if conf.aug_word2_aug_encoder:
                assert conf.enc_choice == "vrec"
                self.aug_detach_drop = self.add_node("dd", Dropout(self.pc, (self.enc_out_dim,), fix_rate=conf.aug_detach_dropout))
                self.aug_encoder = self.add_component("Aenc", VRecEncoder(self.pc, self.emb_out_dim, conf.venc_conf))
                self.aug_mixturer = self.add_node("Amix", BertFeaturesWeightLayer(self.pc, conf.aug_detach_numlayer))

    # helper: embed and encode
    def _emb_and_enc(self, cur_input_map: Dict, collect_loss: bool, insts=None):
        conf = self.conf
        # -----
        # special mode
        if conf.aug_word2 and conf.aug_word2_aug_encoder:
            _rop = RefreshOptions(training=False)  # special feature-mode!!
            self.embedder.refresh(_rop)
            self.encoder.refresh(_rop)
        # -----
        emb_t, mask_t = self.embedder(cur_input_map)
        rel_dist = cur_input_map.get("rel_dist", None)
        if rel_dist is not None:
            rel_dist = BK.input_idx(rel_dist)
        if conf.enc_choice == "vrec":
            enc_t, cache, enc_loss = self.encoder(emb_t, src_mask=mask_t, rel_dist=rel_dist, collect_loss=collect_loss)
        elif conf.enc_choice == "original":  # todo(note): change back to arr for back compatibility
            assert rel_dist is None, "Original encoder does not support rel_dist"
            enc_t = self.encoder(emb_t, BK.get_value(mask_t))
            cache, enc_loss = None, None
        else:
            raise NotImplementedError()
        # another encoder based on attn
        final_enc_t = self.rpreper(emb_t, enc_t, cache)  # [*, slen, D] => final encoder output
        if conf.aug_word2:
            emb2_t = self.aug_word2(insts)
            if conf.aug_word2_aug_encoder:
                # simply add them all together, detach orig-enc as features
                stack_hidden_t = BK.stack(cache.list_hidden[-conf.aug_detach_numlayer:], -2).detach()
                features = self.aug_mixturer(stack_hidden_t)
                aug_input = (emb2_t + conf.aug_detach_ratio*self.aug_detach_drop(features))
                final_enc_t, cache, enc_loss = self.aug_encoder(aug_input, src_mask=mask_t,
                                                                rel_dist=rel_dist, collect_loss=collect_loss)
            else:
                final_enc_t = (final_enc_t + emb2_t)  # otherwise, simply adding
        return emb_t, mask_t, final_enc_t, cache, enc_loss

    # ---
    # helper on agreement loss
    def _get_agr_loss(self, loss_prefix, cache1, cache2=None, copy_num=1):
        # =====
        def _ts_pair_f(_attn_info_pair):
            _ainfo1, _ainfo2 = _attn_info_pair
            _rets = []
            for _t1, _d1, _t2, _d2 in zip(_ainfo1[3], _ainfo1[4], _ainfo2[3], _ainfo2[4]):
                assert _d1 == _d2, "disagree on which dim to collect agr_loss!"
                _rets.append((_t1, _t2, _d1))
            return _rets
        # --
        def _ts_self_f(_list_attn_info):
            _rets = []
            for _t, _d in zip(_list_attn_info[3], _list_attn_info[4]):
                # extra dim at idx 0; todo(note): must repeat insts at outmost idx: repeated = insts * copy_num
                _t1 = _t.view([copy_num, -1] + BK.get_shape(_t)[1:])  # [copy, bs, ...]
                # roll it by 1
                _t2 = BK.concat([_t1[-1].unsqueeze(0), _t1[:-1]], dim=0)
                _rets.append((_t1, _t2, _d))
            return _rets
        # --
        _arg_loss_f = lambda x: self.agree_loss_f(*x)
        # =====
        cur_lambda_agr = self.lambda_agree.value
        if cur_lambda_agr > 0.:
            if cache2 is None:  # roll self
                assert copy_num > 1
                rets = VRecEncoder.get_losses_from_attn_list(
                    cache1.list_attn_info, _ts_self_f, _arg_loss_f, loss_prefix, cur_lambda_agr)
            else:
                rets = VRecEncoder.get_losses_from_attn_list(
                    list(zip(cache1.list_attn_info, cache2.list_attn_info)), _ts_pair_f, _arg_loss_f, loss_prefix, cur_lambda_agr)
        else:
            rets = []
        return rets

    def _get_rel_dist_embed(self, rel_dist, use_abs: bool):
        if use_abs:
            rel_dist = BK.input_idx(rel_dist).abs()
        ret = self.rel_dist_embed(rel_dist)  # [bs, len, len, H]
        return ret

    def _get_rel_dist(self, len_q: int, len_k: int = None):
        if len_k is None:
            len_k = len_q
        dist_x = BK.arange_idx(0, len_k).unsqueeze(0)  # [1, len_k]
        dist_y = BK.arange_idx(0, len_q).unsqueeze(1)  # [len_q, 1]
        distance = dist_x - dist_y  # [len_q, len_k]
        return distance

    # ---
    def fb_on_batch(self, insts: List[GeneralSentence], training=True, loss_factor=1.,
                    rand_gen=None, assign_attns=False, **kwargs):
        # =====
        # import torch
        # torch.autograd.set_detect_anomaly(True)
        # with torch.autograd.detect_anomaly():
        # =====
        conf = self.conf
        self.refresh_batch(training)
        if len(insts) == 0:
            return {"fb": 0, "sent": 0, "tok": 0}
        # -----
        # copying instances for training: expand at dim0
        cur_copy = conf.train_inst_copy if training else 1
        copied_insts = insts * cur_copy
        all_losses = []
        # -----
        # original input
        input_map = self.inputter(copied_insts)
        # for the pretraining modules
        has_loss_mlm, has_loss_orp = (self.masklm.loss_lambda.value > 0.), (self.orderpr.loss_lambda.value > 0.)
        if (not has_loss_orp) and has_loss_mlm:  # only for mlm
            masked_input_map, input_erase_mask_arr = self.masklm.mask_input(input_map, rand_gen=rand_gen)
            emb_t, mask_t, enc_t, cache, enc_loss = self._emb_and_enc(masked_input_map, collect_loss=True)
            all_losses.append(enc_loss)
            # mlm loss; todo(note): currently only using one layer
            mlm_loss = self.masklm.loss(enc_t, input_erase_mask_arr, input_map)
            all_losses.append(mlm_loss)
            # assign values
            if assign_attns:  # may repeat and only keep that last one, but does not matter!
                self._assign_attns_item(copied_insts, "mask", input_erase_mask_arr=input_erase_mask_arr, cache=cache)
            # agreement loss
            if cur_copy > 1:
                all_losses.extend(self._get_agr_loss("agr_mlm", cache, copy_num=cur_copy))
        if has_loss_orp:
            disturbed_input_map = self.orderpr.disturb_input(input_map, rand_gen=rand_gen)
            if has_loss_mlm:  # further mask some
                disturb_keep_arr = disturbed_input_map.get("disturb_keep", None)
                assert disturb_keep_arr is not None, "No keep region for mlm!"
                # todo(note): in this mode we assume add_root, so here exclude arti-root by [:,1:]
                masked_input_map, input_erase_mask_arr = \
                    self.masklm.mask_input(input_map, rand_gen=rand_gen, extra_mask_arr=disturb_keep_arr[:,1:])
                disturbed_input_map.update(masked_input_map)  # update
            emb_t, mask_t, enc_t, cache, enc_loss = self._emb_and_enc(disturbed_input_map, collect_loss=True)
            all_losses.append(enc_loss)
            # orp loss
            if conf.orp_loss_special:
                orp_loss = self.orderpr.loss_special(enc_t, mask_t, disturbed_input_map.get("disturb_keep", None),
                                                     disturbed_input_map, self.masklm)
            else:
                orp_input_attn = self.prepr_f(cache, disturbed_input_map.get("rel_dist"))
                orp_loss = self.orderpr.loss(enc_t, orp_input_attn, mask_t, disturbed_input_map.get("disturb_keep", None))
            all_losses.append(orp_loss)
            # mlm loss
            if has_loss_mlm:
                mlm_loss = self.masklm.loss(enc_t, input_erase_mask_arr, input_map)
                all_losses.append(mlm_loss)
            # assign values
            if assign_attns:  # may repeat and only keep that last one, but does not matter!
                self._assign_attns_item(copied_insts, "dist", abs_posi_arr=disturbed_input_map.get("posi"), cache=cache)
            # agreement loss
            if cur_copy > 1:
                all_losses.extend(self._get_agr_loss("agr_orp", cache, copy_num=cur_copy))
        if self.plainlm.loss_lambda.value > 0.:
            if conf.enc_choice == "vrec":  # special case for blm
                emb_t, mask_t = self.embedder(input_map)
                rel_dist = input_map.get("rel_dist", None)
                if rel_dist is not None:
                    rel_dist = BK.input_idx(rel_dist)
                # two directions
                true_rel_dist = self._get_rel_dist(BK.get_shape(mask_t, -1))  # q-k: [len_q, len_k]
                enc_t1, cache1, enc_loss1 = self.encoder(emb_t, src_mask=mask_t, qk_mask=(true_rel_dist<=0).float(),
                                                         rel_dist=rel_dist, collect_loss=True)
                enc_t2, cache2, enc_loss2 = self.encoder(emb_t, src_mask=mask_t, qk_mask=(true_rel_dist>=0).float(),
                                                         rel_dist=rel_dist, collect_loss=True)
                assert not self.rpreper.active, "TODO: Not supported for this mode"
                all_losses.extend([enc_loss1, enc_loss2])
                # plm loss with explict two inputs
                plm_loss = self.plainlm.loss([enc_t1, enc_t2], input_map)
                all_losses.append(plm_loss)
            else:
                # here use original input
                emb_t, mask_t, enc_t, cache, enc_loss = self._emb_and_enc(input_map, collect_loss=True)
                all_losses.append(enc_loss)
                # plm loss
                plm_loss = self.plainlm.loss(enc_t, input_map)
                all_losses.append(plm_loss)
            # agreement loss
            assert self.lambda_agree.value==0., "Not implemented for this mode"
        # =====
        # task loss
        dpar_loss_lambda, upos_loss_lambda, ner_loss_lambda = \
            [0. if z is None else z.loss_lambda.value for z in [self.dpar, self.upos, self.ner]]
        if any(z>0. for z in [dpar_loss_lambda, upos_loss_lambda, ner_loss_lambda]):
            # here use original input
            emb_t, mask_t, enc_t, cache, enc_loss = self._emb_and_enc(input_map, collect_loss=True, insts=insts)
            all_losses.append(enc_loss)
            # parsing loss
            if dpar_loss_lambda > 0.:
                dpar_input_attn = self.prepr_f(cache, self._get_rel_dist(BK.get_shape(mask_t, -1)))
                dpar_loss = self.dpar.loss(copied_insts, enc_t, dpar_input_attn, mask_t)
                all_losses.append(dpar_loss)
            # pos loss
            if upos_loss_lambda > 0.:
                upos_loss = self.upos.loss(copied_insts, enc_t, mask_t)
                all_losses.append(upos_loss)
            # ner loss
            if ner_loss_lambda > 0.:
                ner_loss = self.ner.loss(copied_insts, enc_t, mask_t)
                all_losses.append(ner_loss)
        # -----
        info = self.collect_loss_and_backward(all_losses, training, loss_factor)
        info.update({"fb": 1, "sent": len(insts), "tok": sum(len(z) for z in insts)})
        return info

    def inference_on_batch(self, insts: List[GeneralSentence], **kwargs):
        conf = self.conf
        self.refresh_batch(False)
        with BK.no_grad_env():
            # special mode
            # use: CUDA_VISIBLE_DEVICES=3 PYTHONPATH=../../src/ python3 -m pdb ../../src/tasks/cmd.py zmlm.main.test ${RUN_DIR}/_conf device:0 dict_dir:${RUN_DIR}/ model_load_name:${RUN_DIR}/zmodel.best test:./_en.debug test_interactive:1
            if conf.test_interactive:
                iinput_sent = input(">> (Interactive testing) Input sent sep by blanks: ")
                iinput_tokens = iinput_sent.split()
                if len(iinput_sent) > 0:
                    iinput_inst = GeneralSentence.create(iinput_tokens)
                    iinput_inst.word_seq.set_idxes([self.word_vocab.get_else_unk(w) for w in iinput_inst.word_seq.vals])
                    iinput_inst.char_seq.build_idxes(self.inputter.vpack.get_voc("char"))
                    iinput_map = self.inputter([iinput_inst])
                    iinput_erase_mask = np.asarray([[z=="Z" for z in iinput_tokens]]).astype(dtype=np.float32)
                    iinput_masked_map = self.inputter.mask_input(iinput_map, iinput_erase_mask, set("pos"))
                    emb_t, mask_t, enc_t, cache, enc_loss = self._emb_and_enc(iinput_masked_map, collect_loss=False, insts=[iinput_inst])
                    mlm_loss = self.masklm.loss(enc_t, iinput_erase_mask, iinput_map)
                    dpar_input_attn = self.prepr_f(cache, self._get_rel_dist(BK.get_shape(mask_t, -1)))
                    self.dpar.predict([iinput_inst], enc_t, dpar_input_attn, mask_t)
                    self.upos.predict([iinput_inst], enc_t, mask_t)
                    # print them
                    import pandas as pd
                    cur_fields = {
                        "idxes": list(range(1, len(iinput_inst)+1)),
                        "word": iinput_inst.word_seq.vals, "pos": iinput_inst.pred_pos_seq.vals,
                        "head": iinput_inst.pred_dep_tree.heads[1:], "dlab": iinput_inst.pred_dep_tree.labels[1:]}
                    zlog(f"Result:\n{pd.DataFrame(cur_fields).to_string()}")
                return {}  # simply return here for interactive mode
            # -----
            # test for MLM simply as in training (use special separate rand_gen to keep the masks the same for testing)
            # todo(+2): do we need to keep testing/validing during training the same? Currently not!
            info = self.fb_on_batch(insts, training=False, rand_gen=self.testing_rand_gen, assign_attns=conf.testing_get_attns)
            # -----
            if len(insts) == 0:
                return info
            # decode for dpar
            input_map = self.inputter(insts)
            emb_t, mask_t, enc_t, cache, _ = self._emb_and_enc(input_map, collect_loss=False, insts=insts)
            dpar_input_attn = self.prepr_f(cache, self._get_rel_dist(BK.get_shape(mask_t, -1)))
            self.dpar.predict(insts, enc_t, dpar_input_attn, mask_t)
            self.upos.predict(insts, enc_t, mask_t)
            if self.ner is not None:
                self.ner.predict(insts, enc_t, mask_t)
            # -----
            if conf.testing_get_attns:
                if conf.enc_choice == "vrec":
                    self._assign_attns_item(insts, "orig", cache=cache)
                elif conf.enc_choice in ["original"]:
                    pass
                else:
                    raise NotImplementedError()
            return info

    def _assign_attns_item(self, insts, prefix, input_erase_mask_arr=None, abs_posi_arr=None, cache=None):
        if cache is not None:
            attn_names, attn_list = [], []
            for one_sidx, one_attn in enumerate(cache.list_attn):
                attn_names.append(f"{prefix}_att{one_sidx}")
                attn_list.append(one_attn)
            if cache.accu_attn is not None:
                attn_names.append(f"{prefix}_att_accu")
                attn_list.append(cache.accu_attn)
            for one_name, one_attn in zip(attn_names, attn_list):
                # (step_idx, ) -> [bs, len_q, len_k, head]
                one_attn_arr = BK.get_value(one_attn)
                for bidx, inst in enumerate(insts):
                    save_arr = one_attn_arr[bidx]
                    inst.add_item(one_name, NpArrField(save_arr, float_decimal=4), assert_non_exist=False)
        if abs_posi_arr is not None:
            for bidx, inst in enumerate(insts):
                inst.add_item(f"{prefix}_abs_posi",
                              NpArrField(abs_posi_arr[bidx], float_decimal=0), assert_non_exist=False)
        if input_erase_mask_arr is not None:
            for bidx, inst in enumerate(insts):
                inst.add_item(f"{prefix}_erase_mask",
                              NpArrField(input_erase_mask_arr[bidx], float_decimal=4), assert_non_exist=False)
        # -----

# b tasks/zmlm/model/mtl:56
