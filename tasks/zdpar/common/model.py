#

# basic modules handling inputs and encodings for the parsers

from typing import List
import numpy as np

from msp.utils import Conf, Random, zlog, JsonRW, zfatal, zwarn
from msp.data import VocabPackage, MultiHelper
from msp.model import Model
from msp.nn import BK
from msp.nn import refresh as nn_refresh
from msp.nn.layers import BasicNode, Affine, RefreshOptions, NoDropRop
from msp.nn.modules import EmbedConf, MyEmbedder, EncConf, MyEncoder
#
from msp.zext.seq_helper import DataPadder
from msp.zext.process_train import RConf, SVConf, ScheduledValue, OptimConf

from .data import ParseInstance

# =====
# the input modeling part

# -----
# for joint pos prediction

class JPosConf(Conf):
    def __init__(self):
        self._input_dim = -1  # to be filled
        self.jpos_multitask = False  # the overall switch for multitask
        self.jpos_lambda = 0.  # lambda(switch) for training: loss_parsing + lambda*loss_pos
        self.jpos_decode = True  # switch for decoding
        self.jpos_enc = EncConf().init_from_kwargs(enc_rnn_type="lstm2", enc_hidden=1024, enc_rnn_layer=0)
        self.jpos_stacking = True  # stacking as inputs (using adding here)

class JPosModule(BasicNode):
    def __init__(self, pc: BK.ParamCollection, jconf: JPosConf, pos_vocab):
        super().__init__(pc, None, None)
        self.jpos_stacking = jconf.jpos_stacking
        self.jpos_multitask = jconf.jpos_multitask
        self.jpos_lambda = jconf.jpos_lambda
        self.jpos_decode = jconf.jpos_decode
        # encoder0
        jconf.jpos_enc._input_dim = jconf._input_dim
        self.enc = self.add_sub_node("enc0", MyEncoder(self.pc, jconf.jpos_enc))
        self.enc_output_dim = self.enc.get_output_dims()[0]
        # output
        # todo(warn): here, include some other things for convenience
        num_labels = len(pos_vocab)
        self.pred = self.add_sub_node("pred", Affine(self.pc, self.enc_output_dim, num_labels, init_rop=NoDropRop()))
        # further stacking (if not, then simply multi-task learning)
        if jconf.jpos_stacking:
            self.pos_weights = self.add_param("w", (num_labels, self.enc_output_dim))  # [n, dim] to be added
        else:
            self.pos_weights = None

    def get_output_dims(self, *input_dims):
        return (self.enc_output_dim, )

    # return (out_expr, jpos_pack)
    def __call__(self, input_repr, mask_arr, require_loss, require_pred, gold_pos_arr=None):
        enc0_expr = self.enc(input_repr, mask_arr)  # [*, len, d]
        #
        enc1_expr = enc0_expr
        pos_probs, pos_losses_expr, pos_preds_expr = None, None, None
        if self.jpos_multitask:
            # get probabilities
            pos_logits = self.pred(enc0_expr)  # [*, len, nl]
            pos_probs = BK.softmax(pos_logits, dim=-1)
            # stacking for input -> output
            if self.jpos_stacking:
                enc1_expr = enc0_expr + BK.matmul(pos_probs, self.pos_weights)
            # simple cross entropy loss
            if require_loss and self.jpos_lambda>0.:
                gold_probs = BK.gather_one_lastdim(pos_probs, gold_pos_arr).squeeze(-1)  # [*, len]
                # todo(warn): multiplying the factor here, but not maksing here (masking in the final steps)
                pos_losses_expr = (-self.jpos_lambda) * gold_probs.log()
            # simple argmax for prediction
            if require_pred and self.jpos_decode:
                pos_preds_expr = pos_probs.max(dim=-1)[1]
        return enc1_expr, (pos_probs, pos_losses_expr, pos_preds_expr)

# -----
# the real bottom part

class BTConf(Conf):
    def __init__(self):
        # embedding and encoding layer
        self.emb_conf = EmbedConf().init_from_kwargs(dim_word=300, dim_char=30, dim_extras='[50]', extra_names='["pos"]')
        self.enc_conf = EncConf().init_from_kwargs(enc_rnn_type="lstm2", enc_hidden=1024, enc_rnn_layer=3)
        # joint pos encoder (layer0)
        self.jpos_conf = JPosConf()
        # =====
        # other options
        # inputs
        self.char_max_length = 45
        # dropouts
        self.drop_embed = 0.33
        self.dropmd_embed = 0.
        self.drop_hidden = 0.33
        self.gdrop_rnn = 0.33            # gdrop (always fixed for recurrent connections)
        self.idrop_rnn = 0.33            # idrop for rnn
        self.fix_drop = True            # fix drop for one run for each dropout
        self.singleton_unk = 0.5        # replace singleton words with UNK when training
        self.singleton_thr = 2          # only replace singleton if freq(val) <= this (also decay with 1/freq)

    def do_validate(self):
        if self.jpos_conf.jpos_multitask:
            # assert self.jpos_conf.jpos_lambda > 0.  # for training
            assert 'pos' not in self.emb_conf.extra_names, "No usage for pos prediction?"

# bottom part of the parsers, also include some shared procedures
class ParserBT(BasicNode):
    def __init__(self, pc: BK.ParamCollection, bconf: BTConf, vpack: VocabPackage):
        super().__init__(pc, None, None)
        self.bconf = bconf
        # ===== Vocab =====
        self.word_vocab = vpack.get_voc("word")
        self.char_vocab = vpack.get_voc("char")
        self.pos_vocab = vpack.get_voc("pos")
        # ===== Model =====
        # embedding
        self.emb = self.add_sub_node("emb", MyEmbedder(self.pc, bconf.emb_conf, vpack))
        emb_output_dim = self.emb.get_output_dims()[0]
        # encoder0 for jpos
        # todo(note): will do nothing if not use_jpos
        bconf.jpos_conf._input_dim = emb_output_dim
        self.jpos_enc = self.add_sub_node("enc0", JPosModule(self.pc, bconf.jpos_conf, self.pos_vocab))
        enc0_output_dim = self.jpos_enc.get_output_dims()[0]
        # encoder
        # todo(0): feed compute-on-the-fly hp
        bconf.enc_conf._input_dim = enc0_output_dim
        self.enc = self.add_sub_node("enc", MyEncoder(self.pc, bconf.enc_conf))
        self.enc_output_dim = self.enc.get_output_dims()[0]
        # ===== Input Specification =====
        # inputs (word, char, pos) and vocabulary
        self.need_word = self.emb.has_word
        self.need_char = self.emb.has_char
        # todo(warn): currently only allow extra fields for POS
        self.need_pos = False
        if len(self.emb.extra_names) > 0:
            assert len(self.emb.extra_names) == 1 and self.emb.extra_names[0] == "pos"
            self.need_pos = True
        # todo(warn): currently only allow one aux field
        self.need_aux = False
        if len(self.emb.dim_auxes) > 0:
            assert len(self.emb.dim_auxes) == 1
            self.need_aux = True
        #
        self.word_padder = DataPadder(2, pad_vals=self.word_vocab.pad, mask_range=2)
        self.char_padder = DataPadder(3, pad_lens=(0, 0, bconf.char_max_length), pad_vals=self.char_vocab.pad)
        self.pos_padder = DataPadder(2, pad_vals=self.pos_vocab.pad)
        #
        self.random_sample_stream = Random.stream(Random.random_sample)

    def get_output_dims(self, *input_dims):
        return (self.enc_output_dim, )

    #
    def refresh(self, rop=None):
        zfatal("Should call special_refresh instead!")

    # whether enabling joint-pos multitask
    def jpos_multitask_enabled(self):
        return self.jpos_enc.jpos_multitask

    # ====
    # special routines

    def special_refresh(self, embed_rop, other_rop):
        self.emb.refresh(embed_rop)
        self.enc.refresh(other_rop)
        self.jpos_enc.refresh(other_rop)

    def prepare_training_rop(self):
        mconf = self.bconf
        embed_rop = RefreshOptions(hdrop=mconf.drop_embed, dropmd=mconf.dropmd_embed, fix_drop=mconf.fix_drop)
        other_rop = RefreshOptions(hdrop=mconf.drop_hidden, idrop=mconf.idrop_rnn, gdrop=mconf.gdrop_rnn,
                                   fix_drop=mconf.fix_drop)
        return embed_rop, other_rop

    # =====
    # run
    def _prepare_input(self, insts, training):
        word_arr, char_arr, extra_arrs, aux_arrs = None, None, [], []
        # ===== specially prepare for the words
        wv = self.word_vocab
        W_UNK = wv.unk
        UNK_REP_RATE = self.bconf.singleton_unk
        UNK_REP_THR = self.bconf.singleton_thr
        word_act_idxes = []
        if training and UNK_REP_RATE>0.:    # replace unfreq/singleton words with UNK
            for one_inst in insts:
                one_act_idxes = []
                for one_idx in one_inst.words.idxes:
                    one_freq = wv.idx2val(one_idx)
                    if one_freq is not None and one_freq >= 1 and one_freq <= UNK_REP_THR:
                        if next(self.random_sample_stream) < (UNK_REP_RATE/one_freq):
                            one_idx = W_UNK
                    one_act_idxes.append(one_idx)
                word_act_idxes.append(one_act_idxes)
        else:
            word_act_idxes = [z.words.idxes for z in insts]
        # todo(warn): still need the masks
        word_arr, mask_arr = self.word_padder.pad(word_act_idxes)
        # =====
        if not self.need_word:
            word_arr = None
        if self.need_char:
            chars = [z.chars.idxes for z in insts]
            char_arr, _ = self.char_padder.pad(chars)
        if self.need_pos or self.jpos_multitask_enabled():
            poses = [z.poses.idxes for z in insts]
            pos_arr, _ = self.pos_padder.pad(poses)
            if self.need_pos:
                extra_arrs.append(pos_arr)
        else:
            pos_arr = None
        if self.need_aux:
            aux_arr_list = [z.extra_features["aux_repr"] for z in insts]
            # pad
            padded_seq_len = int(mask_arr.shape[1])
            final_aux_arr_list = []
            for cur_arr in aux_arr_list:
                cur_len = len(cur_arr)
                if cur_len > padded_seq_len:
                    final_aux_arr_list.append(cur_arr[:padded_seq_len])
                else:
                    final_aux_arr_list.append(np.pad(cur_arr, ((0,padded_seq_len-cur_len),(0,0)), 'constant'))
            aux_arrs.append(np.stack(final_aux_arr_list, 0))
        #
        input_repr = self.emb(word_arr, char_arr, extra_arrs, aux_arrs)
        # [BS, Len, Dim], [BS, Len]
        return input_repr, mask_arr, pos_arr

    # todo(warn): for rnn, need to transpose masks, thus need np.array
    # return input_repr, enc_repr, mask_arr
    def run(self, insts, training):
        # ===== calculate
        # [BS, Len, Di], [BS, Len], [BS, len]
        input_repr, mask_arr, gold_pos_arr = self._prepare_input(insts, training)
        # enc0 for joint-pos multitask
        input_repr0, jpos_pack = self.jpos_enc(
            input_repr, mask_arr, require_loss=training, require_pred=(not training), gold_pos_arr=gold_pos_arr)
        # [BS, Len, De]
        enc_repr = self.enc(input_repr0, mask_arr)
        return input_repr, enc_repr, jpos_pack, mask_arr

    # special routine
    def aug_words_and_embs(self, aug_vocab, aug_wv):
        orig_vocab = self.word_vocab
        if self.emb.has_word:
            orig_arr = self.emb.word_embed.E.detach().cpu().numpy()
            # todo(+2): find same-spelling words in the original vocab if not-hit in the extra_embed?
            # todo(warn): here aug_vocab should be find in aug_wv
            aug_arr = aug_vocab.filter_embed(aug_wv, assert_all_hit=True)
            new_vocab, new_arr = MultiHelper.aug_vocab_and_arr(orig_vocab, orig_arr, aug_vocab, aug_arr, aug_override=True)
            # assign
            self.word_vocab = new_vocab
            self.emb.word_embed.replace_weights(new_arr)
        else:
            zwarn("No need to aug vocab since delexicalized model!!")
            new_vocab = orig_vocab
        return new_vocab

# =====
# base parser

# decoding conf
class BaseInferenceConf(Conf):
    def __init__(self):
        # overall
        self.batch_size = 32
        self.infer_single_length = 100  # single-inst batch if >= this length

# training conf
class BaseTrainingConf(RConf):
    def __init__(self):
        super().__init__()
        # about files
        self.no_build_dict = False
        self.load_model = False
        self.load_process = False
        # batch arranger
        self.batch_size = 32
        self.train_min_length = 0
        self.train_skip_length = 120
        self.shuffle_train = True
        # optimizer and lrate factor for enc&dec&sl(mid)
        self.enc_optim = OptimConf()
        self.enc_lrf = SVConf().init_from_kwargs(val=1., which_idx="eidx", mode="none", min_val=0., max_val=1.)
        self.dec_optim = OptimConf()
        self.dec_lrf = SVConf().init_from_kwargs(val=1., which_idx="eidx", mode="none", min_val=0., max_val=1.)
        # todo(note): by default freeze this one!
        self.dec2_optim = OptimConf()
        self.dec2_lrf = SVConf().init_from_kwargs(val=0., which_idx="eidx", mode="none", min_val=0., max_val=1.)
        self.mid_optim = OptimConf()
        self.mid_lrf = SVConf().init_from_kwargs(val=1., which_idx="eidx", mode="none", min_val=0., max_val=1.)
        # margin
        self.margin = SVConf().init_from_kwargs(val=0.0)
        # scheduled sampling (0. as always oracle)
        self.sched_sampling = SVConf().init_from_kwargs(val=0.0)
        # other regularization
        self.reg_scores_lambda = 0.  # score norm regularization
        # ===
        # overwrite default ones
        self.patience = 8
        self.anneal_times = 10
        self.max_epochs = 200

# base conf
class BaseParserConf(Conf):
    def __init__(self, iconf: BaseInferenceConf, tconf: BaseTrainingConf):
        self.iconf = iconf
        self.tconf = tconf
        # Model
        self.new_name_conv = False  # set this to false to be compatible with previous models (param names)!
        self.bt_conf = BTConf()
        # others (in inherited classes)

# base parser
class BaseParser(Model):
    def __init__(self, conf: BaseParserConf, vpack: VocabPackage):
        self.conf = conf
        self.vpack = vpack
        tconf = conf.tconf
        # ===== Vocab =====
        self.label_vocab = vpack.get_voc("label")
        # ===== Model =====
        self.pc = BK.ParamCollection(conf.new_name_conv)
        # bottom-part: input + encoder
        self.bter = ParserBT(self.pc, conf.bt_conf, vpack)
        self.enc_output_dim = self.bter.get_output_dims()[0]
        self.enc_lrf_sv = ScheduledValue("enc_lrf", tconf.enc_lrf)
        self.pc.optimizer_set(tconf.enc_optim.optim, self.enc_lrf_sv, tconf.enc_optim,
                              params=self.bter.get_parameters(), check_repeat=True, check_full=True)
        # upper-part: decoder
        # todo(+2): very ugly here!
        self.scorer = self.build_decoder()
        self.dec_lrf_sv = ScheduledValue("dec_lrf", tconf.dec_lrf)
        self.dec2_lrf_sv = ScheduledValue("dec2_lrf", tconf.dec2_lrf)
        try:
            params, params2 = self.scorer.get_split_params()
            self.pc.optimizer_set(tconf.dec_optim.optim, self.dec_lrf_sv, tconf.dec_optim,
                                  params=params, check_repeat=True, check_full=False)
            self.pc.optimizer_set(tconf.dec2_optim.optim, self.dec2_lrf_sv, tconf.dec2_optim,
                                  params=params2, check_repeat=True, check_full=True)
        except:
            self.pc.optimizer_set(tconf.dec_optim.optim, self.dec_lrf_sv, tconf.dec_optim,
                                  params=self.scorer.get_parameters(), check_repeat=True, check_full=True)
        # middle-part: structured layer at the middle (build later for convenient re-loading)
        self.slayer = None
        self.mid_lrf_sv = ScheduledValue("mid_lrf", tconf.mid_lrf)
        # ===== For training =====
        # schedule values
        self.margin = ScheduledValue("margin", tconf.margin)
        self.sched_sampling = ScheduledValue("ss", tconf.sched_sampling)
        self._scheduled_values = [self.margin, self.sched_sampling, self.enc_lrf_sv,
                                  self.dec_lrf_sv,  self.dec2_lrf_sv, self.mid_lrf_sv]
        self.reg_scores_lambda = conf.tconf.reg_scores_lambda
        # for refreshing dropouts
        self.previous_refresh_training = True

    # to be implemented
    def build_decoder(self):
        raise NotImplementedError()

    def build_slayer(self):
        return None  # None means no slayer

    # build after possible re-loading;
    # zero_extra_params: means making the extra params output 0 (zero params for the final Affine layers)
    def add_slayer(self):
        tconf = self.conf.tconf
        self.slayer = self.build_slayer()
        if self.slayer is not None:
            self.pc.optimizer_set(tconf.mid_optim.optim, self.mid_lrf_sv, tconf.mid_optim,
                                  params=self.slayer.get_parameters(), check_repeat=True, check_full=True)

    def get_extra_refresh_nodes(self):
        if self.slayer is None:
            return [self.scorer]
        else:
            return [self.scorer, self.slayer]

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
            embed_rop, other_rop = self.bter.prepare_training_rop()
            # todo(warn): once-bug, don't forget this one!!
            self.previous_refresh_training = True
        # manually refresh
        self.bter.special_refresh(embed_rop, other_rop)
        for node in self.get_extra_refresh_nodes():
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
        return self.bter.aug_words_and_embs(aug_vocab, aug_wv)

    # =====
    # common procedures
    def pred2real_labels(self, preds):
        return [self.label_vocab.trg_pred2real(z) for z in preds]

    def real2pred_labels(self, reals):
        return [self.label_vocab.trg_real2pred(z) for z in reals]

    # =====
    # decode and training for the JPos Module

    def jpos_decode(self, insts: List[ParseInstance], jpos_pack):
        # jpos prediction (directly index, no converting)
        jpos_preds_expr = jpos_pack[2]
        if jpos_preds_expr is not None:
            jpos_preds_arr = BK.get_value(jpos_preds_expr)
            for one_idx, one_inst in enumerate(insts):
                cur_length = len(one_inst) + 1  # including the artificial ROOT
                one_inst.pred_poses.build_vals(jpos_preds_arr[one_idx][:cur_length], self.bter.pos_vocab)

    def jpos_loss(self, jpos_pack, mask_expr):
        jpos_losses_expr = jpos_pack[1]
        if jpos_losses_expr is not None:
            # collect loss with mask, also excluding the first symbol of ROOT
            final_losses_masked = (jpos_losses_expr * mask_expr)[:, 1:]
            # todo(note): no need to scale lambda since already multiplied previously
            final_loss_sum = BK.sum(final_losses_masked)
            return final_loss_sum
        else:
            return None

    def reg_scores_loss(self, *scores):
        if self.reg_scores_lambda>0.:
            sreg_losses = [(z**2).mean() for z in scores]
            if len(sreg_losses)>0:
                sreg_loss = BK.stack(sreg_losses).mean() * self.reg_scores_lambda
                return sreg_loss
        return None

    # =====
    # inference and training: to be implemented
    # ...

# =====
# general label set
class DepLabelHelper:
    @staticmethod
    def get_label_list(sname):
        sname = str.lower(sname)
        if sname == "":
            sname = "none"
        return {
            "none": [],
            "punct": ['punct'],
            "func": ['aux', 'case', 'cc', 'clf', 'cop', 'det', 'mark', 'punct'],
            "all": ['punct', 'case', 'nmod', 'amod', 'det', 'obl', 'nsubj', 'root', 'advmod', 'conj', 'obj', 'cc', 'mark', 'aux', 'acl', 'nummod', 'flat', 'cop', 'advcl', 'xcomp', 'appos', 'compound', 'expl', 'ccomp', 'fixed', 'iobj', 'parataxis', 'dep', 'csubj', 'orphan', 'discourse', 'clf', 'goeswith', 'vocative', 'list', 'dislocated', 'reparandum'],
            "my1": ['aux', 'cc', 'clf', 'cop', 'det', 'mark', 'punct'],  # keep 'case'
        }[sname]

    # return {idx -> True/False} or [idx -> True/False]
    @staticmethod
    def select_label_idxes(sname_or_list, concerned_label_list: List, return_mask: bool, include_true: bool):
        if isinstance(sname_or_list, str):
            sname_or_list = DepLabelHelper.get_label_list(sname_or_list)
        tmp_set = set(sname_or_list)
        ret_set = set()
        for idx, lab in enumerate(concerned_label_list):
            lab = lab.split(":")[0]  # todo(warn): first layer of label
            in_list = lab in tmp_set
            if include_true:
                if in_list:
                    ret_set.add(idx)
            else:
                if not in_list:
                    ret_set.add(idx)
        if return_mask:
            ret_list = [False] * len(concerned_label_list)
            for idx in ret_set:
                ret_list[idx] = True
            return ret_list
        else:
            return ret_set

# b tasks/zdpar/common/model.py:215
