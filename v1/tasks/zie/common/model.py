#

# basic modules handling inputs and encodings for the IE system

from typing import List, Union
import numpy as np

from msp.utils import Conf, Random, zlog, JsonRW, zfatal, Helper
from msp.data import VocabPackage, MultiHelper
from msp.model import Model
from msp.nn import BK
from msp.nn import refresh as nn_refresh
from msp.nn.layers import BasicNode, Affine, RefreshOptions, NoDropRop
from msp.nn.modules import EmbedConf, MyEmbedder, EncConf, MyEncoder
#
from msp.zext.seq_helper import DataPadder
from msp.zext.process_train import RConf, SVConf, ScheduledValue, OptimConf
from msp.zext.ie import HLabelConf

from .data import DocInstance, Sentence
from .data_helper import KBP17_TYPES
from .dochint import DocHintConf, DocHintModule

# =====
# first the bottom part, which is the module that handles input and encoding

class BTConf(Conf):
    def __init__(self):
        # embedding and encoding layer
        # self.emb_conf = EmbedConf().init_from_kwargs(dim_word=300, dim_char=30, dim_extras='[300,50]',
        #                                              extra_names='["lemma","upos"]', emb_proj_dim=512)
        # first only use word and char
        self.emb_conf = EmbedConf().init_from_kwargs(dim_word=300, dim_char=30)
        # doc hint?
        self.use_doc_hint = False
        self.dh_conf = DocHintConf()
        self.dh_combine_method = "both"  # how to combine features from dh: add=adding, both=appending both ends, cls=replace-i0
        # shared encoder and private encoders
        self.enc_conf = EncConf().init_from_kwargs(enc_rnn_type="lstm2", enc_hidden=512, enc_rnn_layer=1)
        self.enc_ef_conf = EncConf().init_from_kwargs(enc_rnn_type="lstm2", enc_hidden=512, enc_rnn_layer=0)
        self.enc_evt_conf = EncConf().init_from_kwargs(enc_rnn_type="lstm2", enc_hidden=512, enc_rnn_layer=0)
        # stop certain gradients, detach input for the specific encoders?
        self.enc_ef_input_detach = False
        self.enc_evt_input_detach = False
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
        # =====
        # how to batch and deal with long sentences in doc
        # todo(note): decide to encode long sentences by themselves rather than split and merge
        self.enc_bucket_range = 10  # bucket the sentences (but need to do multiple calculation); also not the best choice since not smart at detecting splitting/bucketing points

# bottom part of the parsers, also include some shared procedures
class MyIEBT(BasicNode):
    def __init__(self, pc: BK.ParamCollection, bconf: BTConf, tconf: 'BaseTrainingConf', vpack: VocabPackage):
        super().__init__(pc, None, None)
        self.bconf = bconf
        # ===== Vocab =====
        self.word_vocab = vpack.get_voc("word")
        self.char_vocab = vpack.get_voc("char")
        self.lemma_vocab = vpack.get_voc("lemma")
        self.upos_vocab = vpack.get_voc("upos")
        self.ulabel_vocab = vpack.get_voc("ulabel")
        # ===== Model =====
        # embedding
        self.emb = self.add_sub_node("emb", MyEmbedder(self.pc, bconf.emb_conf, vpack))
        emb_output_dim = self.emb.get_output_dims()[0]
        self.emb_output_dim = emb_output_dim
        # doc hint
        self.use_doc_hint = bconf.use_doc_hint
        self.dh_combine_method = bconf.dh_combine_method
        if self.use_doc_hint:
            assert len(bconf.emb_conf.dim_auxes)>0
            # todo(note): currently use the concat of them if input multiple layers
            bconf.dh_conf._input_dim = bconf.emb_conf.dim_auxes[0]  # same as input bert dim
            bconf.dh_conf._output_dim = emb_output_dim  # same as emb_output_dim
            self.dh_node = self.add_sub_node("dh", DocHintModule(pc, bconf.dh_conf))
        else:
            self.dh_node = None
        # encoders
        # shared
        # todo(note): feed compute-on-the-fly hp
        bconf.enc_conf._input_dim = emb_output_dim
        self.enc = self.add_sub_node("enc", MyEncoder(self.pc, bconf.enc_conf))
        tmp_enc_output_dim = self.enc.get_output_dims()[0]
        # privates
        bconf.enc_ef_conf._input_dim = tmp_enc_output_dim
        self.enc_ef = self.add_sub_node("enc_ef", MyEncoder(self.pc, bconf.enc_ef_conf))
        self.enc_ef_output_dim = self.enc_ef.get_output_dims()[0]
        bconf.enc_evt_conf._input_dim = tmp_enc_output_dim
        self.enc_evt = self.add_sub_node("enc_evt", MyEncoder(self.pc, bconf.enc_evt_conf))
        self.enc_evt_output_dim = self.enc_evt.get_output_dims()[0]
        # ===== Input Specification =====
        # inputs (word, lemma, char, upos, ulabel) and vocabulary
        self.need_word = self.emb.has_word
        self.need_char = self.emb.has_char
        # extra fields
        # todo(warn): need to
        self.need_lemma = False
        self.need_upos = False
        self.need_ulabel = False
        for one_extra_name in self.emb.extra_names:
            if one_extra_name == "lemma":
                self.need_lemma = True
            elif one_extra_name == "upos":
                self.need_upos = True
            elif one_extra_name == "ulabel":
                self.need_ulabel = True
            else:
                raise NotImplementedError("UNK extra input name: " + one_extra_name)
        # todo(warn): currently only allow one aux field
        self.need_aux = False
        if len(self.emb.dim_auxes) > 0:
            assert len(self.emb.dim_auxes) == 1
            self.need_aux = True
        # padders
        self.word_padder = DataPadder(2, pad_vals=self.word_vocab.pad, mask_range=2)
        self.char_padder = DataPadder(3, pad_lens=(0, 0, bconf.char_max_length), pad_vals=self.char_vocab.pad)
        self.lemma_padder = DataPadder(2, pad_vals=self.lemma_vocab.pad)
        self.upos_padder = DataPadder(2, pad_vals=self.upos_vocab.pad)
        self.ulabel_padder = DataPadder(2, pad_vals=self.ulabel_vocab.pad)
        #
        self.random_sample_stream = Random.stream(Random.random_sample)
        self.train_skip_noevt_rate = tconf.train_skip_noevt_rate
        self.train_skip_length = tconf.train_skip_length
        self.train_min_length = tconf.train_min_length
        self.test_min_length = tconf.test_min_length
        self.test_skip_noevt_rate = tconf.test_skip_noevt_rate
        self.train_sent_based = tconf.train_sent_based
        #
        assert not self.train_sent_based, "The basic model should not use this sent-level mode!"

    def get_output_dims(self, *input_dims):
        return ([self.enc_ef_output_dim, self.enc_evt_output_dim], )

    #
    def refresh(self, rop=None):
        zfatal("Should call special_refresh instead!")

    # ====
    # special routines

    def special_refresh(self, embed_rop, other_rop):
        self.emb.refresh(embed_rop)
        self.enc.refresh(other_rop)
        self.enc_ef.refresh(other_rop)
        self.enc_evt.refresh(other_rop)
        if self.dh_node is not None:
            return self.dh_node.refresh(other_rop)

    def prepare_training_rop(self):
        mconf = self.bconf
        embed_rop = RefreshOptions(hdrop=mconf.drop_embed, dropmd=mconf.dropmd_embed, fix_drop=mconf.fix_drop)
        other_rop = RefreshOptions(hdrop=mconf.drop_hidden, idrop=mconf.idrop_rnn, gdrop=mconf.gdrop_rnn,
                                   fix_drop=mconf.fix_drop)
        return embed_rop, other_rop

    # =====
    # run
    def _prepare_input(self, sents: List[Sentence], training: bool):
        word_arr, char_arr, extra_arrs, aux_arrs = None, None, [], []
        # ===== specially prepare for the words
        wv = self.word_vocab
        W_UNK = wv.unk
        UNK_REP_RATE = self.bconf.singleton_unk
        UNK_REP_THR = self.bconf.singleton_thr
        word_act_idxes = []
        if training and UNK_REP_RATE>0.:    # replace unfreq/singleton words with UNK
            for one_inst in sents:
                one_act_idxes = []
                for one_idx in one_inst.words.idxes:
                    one_freq = wv.idx2val(one_idx)
                    if one_freq is not None and one_freq >= 1 and one_freq <= UNK_REP_THR:
                        if next(self.random_sample_stream) < (UNK_REP_RATE/one_freq):
                            one_idx = W_UNK
                    one_act_idxes.append(one_idx)
                word_act_idxes.append(one_act_idxes)
        else:
            word_act_idxes = [z.words.idxes for z in sents]
        # todo(warn): still need the masks
        word_arr, mask_arr = self.word_padder.pad(word_act_idxes)
        # =====
        if not self.need_word:
            word_arr = None
        if self.need_char:
            chars = [z.chars.idxes for z in sents]
            char_arr, _ = self.char_padder.pad(chars)
        # extra ones: lemma, upos, ulabel
        if self.need_lemma:
            lemmas = [z.lemmas.idxes for z in sents]
            lemmas_arr, _ = self.lemma_padder.pad(lemmas)
            extra_arrs.append(lemmas_arr)
        if self.need_upos:
            uposes = [z.uposes.idxes for z in sents]
            upos_arr, _ = self.upos_padder.pad(uposes)
            extra_arrs.append(upos_arr)
        if self.need_ulabel:
            ulabels = [z.ud_labels.idxes for z in sents]
            ulabels_arr, _ = self.ulabel_padder.pad(ulabels)
            extra_arrs.append(ulabels_arr)
        # aux ones
        if self.need_aux:
            aux_arr_list = [z.extra_features["aux_repr"] for z in sents]
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
        return input_repr, mask_arr

    # input: Tuple(Sentence, ...)
    def _bucket_sents_by_length(self, all_sents, enc_bucket_range: int, getlen_f=lambda x: x[0].length, max_bsize=None):
        # split into buckets
        all_buckets = []
        cur_local_sidx = 0
        use_max_bsize = (max_bsize is not None)
        while cur_local_sidx < len(all_sents):
            cur_bucket = []
            starting_slen = getlen_f(all_sents[cur_local_sidx])
            ending_slen = starting_slen + enc_bucket_range
            # searching forward
            tmp_sidx = cur_local_sidx
            while tmp_sidx < len(all_sents):
                one_sent = all_sents[tmp_sidx]
                one_slen = getlen_f(one_sent)
                if one_slen>ending_slen or (use_max_bsize and len(cur_bucket)>=max_bsize):
                    break
                else:
                    cur_bucket.append(one_sent)
                tmp_sidx += 1
            # put bucket and next
            all_buckets.append(cur_bucket)
            cur_local_sidx = tmp_sidx
        return all_buckets

    # todo(warn): for rnn, need to transpose masks, thus need np.array
    # TODO(+N): here we encode only at sentence level, encoding doc level might be helpful, but much harder to batch
    #  therefore, still take DOC as input, since may be extended to doc-level encoding
    # return input_repr, enc_repr, mask_arr
    def run(self, insts: List[DocInstance], training: bool):
        # make it as sentence level processing (group the sentences by length, and ignore doc level for now)
        # skip no content sentences in training?
        # assert not self.train_sent_based, "The basic model should not use this sent-level mode!"
        all_sents = []  # (inst, d_idx, s_idx)
        for d_idx, one_doc in enumerate(insts):
            for s_idx, x in enumerate(one_doc.sents):
                if training:
                    if x.length<self.train_skip_length and x.length>=self.train_min_length \
                            and (len(x.events)>0 or next(self.random_sample_stream)>self.train_skip_noevt_rate):
                        all_sents.append((x, d_idx, s_idx))
                else:
                    if x.length >= self.test_min_length:
                        all_sents.append((x, d_idx, s_idx))
        return self.run_sents(all_sents, insts, training)

    # input interested sents: Tuple[Sentence, DocId, SentId]
    def run_sents(self, all_sents: List, all_docs: List[DocInstance], training: bool, use_one_bucket=False):
        if use_one_bucket:
            all_buckets = [all_sents]  # when we do not want to split if we know the input lengths do not vary too much
        else:
            all_sents.sort(key=lambda x: x[0].length)
            all_buckets = self._bucket_sents_by_length(all_sents, self.bconf.enc_bucket_range)
        # doc hint
        use_doc_hint = self.use_doc_hint
        if use_doc_hint:
            dh_sent_repr = self.dh_node.run(all_docs)  # [NumDoc, MaxSent, D]
        else:
            dh_sent_repr = None
        # encoding for each of the bucket
        rets = []
        dh_add, dh_both, dh_cls = [self.dh_combine_method==z for z in ["add", "both", "cls"]]
        for one_bucket in all_buckets:
            one_sents = [z[0] for z in one_bucket]
            # [BS, Len, Di], [BS, Len]
            input_repr0, mask_arr0 = self._prepare_input(one_sents, training)
            if use_doc_hint:
                one_d_idxes = BK.input_idx([z[1] for z in one_bucket])
                one_s_idxes = BK.input_idx([z[2] for z in one_bucket])
                one_s_reprs = dh_sent_repr[one_d_idxes, one_s_idxes].unsqueeze(-2)  # [BS, 1, D]
                if dh_add:
                    input_repr = input_repr0 + one_s_reprs  # [BS, slen, D]
                    mask_arr = mask_arr0
                elif dh_both:
                    input_repr = BK.concat([one_s_reprs, input_repr0, one_s_reprs], -2)  # [BS, 2+slen, D]
                    mask_arr = np.pad(mask_arr0, ((0,0),(1,1)), 'constant', constant_values=1.)  # [BS, 2+slen]
                elif dh_cls:
                    input_repr = BK.concat([one_s_reprs, input_repr0[:, 1:]], -2)  # [BS, slen, D]
                    mask_arr = mask_arr0
                else:
                    raise NotImplementedError()
            else:
                input_repr, mask_arr = input_repr0, mask_arr0
            # [BS, Len, De]
            enc_repr = self.enc(input_repr, mask_arr)
            # separate ones (possibly using detach to avoid gradients for some of them)
            enc_repr_ef = self.enc_ef(enc_repr.detach() if self.bconf.enc_ef_input_detach else enc_repr, mask_arr)
            enc_repr_evt = self.enc_evt(enc_repr.detach() if self.bconf.enc_evt_input_detach else enc_repr, mask_arr)
            if use_doc_hint and dh_both:
                one_ret = (one_sents, input_repr0, enc_repr_ef[:, 1:-1].contiguous(), enc_repr_evt[:, 1:-1].contiguous(), mask_arr0)
            else:
                one_ret = (one_sents, input_repr0, enc_repr_ef, enc_repr_evt, mask_arr0)
            rets.append(one_ret)
        # todo(note): returning tuple is (List[Sentence], Tensor, Tensor, Tensor)
        return rets

    # special routine
    def aug_words_and_embs(self, aug_vocab, aug_wv):
        orig_vocab = self.word_vocab
        orig_arr = self.emb.word_embed.E.detach().cpu().numpy()
        # todo(+2): find same-spelling words in the original vocab if not-hit in the extra_embed?
        # todo(warn): here aug_vocab should be find in aug_wv
        aug_arr = aug_vocab.filter_embed(aug_wv, assert_all_hit=True)
        new_vocab, new_arr = MultiHelper.aug_vocab_and_arr(orig_vocab, orig_arr, aug_vocab, aug_arr, aug_override=True)
        # assign
        self.word_vocab = new_vocab
        self.emb.word_embed.replace_weights(new_arr)
        return new_vocab

# =====
# then the overall model (details to be implemented)

# decoding conf
class BaseInferenceConf(Conf):
    def __init__(self):
        # overall
        self.batch_size = 100
        # self.infer_single_length = 100  # single-inst batch if >= this length
        self.no_15E78 = True  # no LDC2015E78 data (but for Chinese we need this dataset)
        self.constrain_evt_types = ""  # specific for dataset?

# training conf
class BaseTrainingConf(RConf):
    def __init__(self):
        super().__init__()
        # about files
        self.no_build_dict = False
        self.load_model = False
        self.load_process = False
        #
        self.no_15E78 = True  # no LDC2015E78 data
        self.constrain_evt_types = ""  # specific for dataset?
        # batch arranger
        self.batch_size = 100  # number of sent in doc (but at doc granarity)
        self.maxibatch_size = 5  # number of batches to collect for the batcher
        # special mode
        self.train_msent_based = False  # use multi-sent based streaming in training rather than doc
        self.train_sent_based = False  # use sent based streaming in training rather than document based
        self.train_sent_shuffle = False  # if use sent based, then shuffle all at sentence level
        #
        self.train_skip_noevt_rate = 0.  # the rate to skip sentences without any event in training
        self.train_skip_length = 120  # max sentence length to consider in training
        self.train_min_length = 5  # min sentence length to consider in training
        self.test_min_length = 0  # min sentence length to consider in testing
        self.test_skip_noevt_rate = 0.  # this option is dangerous, just to take a look!
        self.shuffle_train = True
        # optimizer and lrate factor for enc&dec&sl(mid)
        self.enc_optim = OptimConf()
        self.enc_lrf = SVConf().init_from_kwargs(val=1., which_idx="eidx", mode="none", min_val=0., max_val=1.)
        self.dec_optim = OptimConf()
        self.dec_lrf = SVConf().init_from_kwargs(val=1., which_idx="eidx", mode="none", min_val=0., max_val=1.)
        # self.mid_optim = OptimConf()
        # self.mid_lrf = SVConf().init_from_kwargs(val=1., which_idx="eidx", mode="none", min_val=0., max_val=1.)
        # margin
        self.margin = SVConf().init_from_kwargs(val=0.0)
        # ===
        # overwrite default ones
        self.patience = 8
        self.anneal_times = 10
        self.max_epochs = 200

class MyIEModelConf(Conf):
    def __init__(self, iconf: BaseInferenceConf, tconf: BaseTrainingConf):
        self.iconf = iconf
        self.tconf = tconf
        # Model
        self.bt_conf = BTConf()
        # hlabel confs at the output
        self.hl_evt = HLabelConf()
        self.hl_ef = HLabelConf()
        self.hl_arg = HLabelConf()
        # special modes
        # self.exclude_nil = True  # only set False for debugging purpose (to see selection coverage)

#
class MyIEModel(Model):
    def __init__(self, conf: MyIEModelConf, vpack: VocabPackage):
        self.conf = conf
        self.vpack = vpack
        tconf = conf.tconf
        # ===== Vocab =====
        # ===== Model =====
        self.pc = BK.ParamCollection(True)
        # bottom-part: input + encoder
        self.bter: MyIEBT = self.build_encoder()
        self.lexi_output_dim = self.bter.emb_output_dim
        self.enc_ef_output_dim, self.enc_evt_output_dim = self.bter.get_output_dims()[0]
        self.enc_lrf_sv = ScheduledValue("enc_lrf", tconf.enc_lrf)
        self.pc.optimizer_set(tconf.enc_optim.optim, self.enc_lrf_sv, tconf.enc_optim,
                              params=self.bter.get_parameters(), check_repeat=True, check_full=True)
        # upper-parts: the decoders
        self.decoders: List = self.build_decoders()
        self.dec_lrf_sv = ScheduledValue("dec_lrf", tconf.dec_lrf)
        self.pc.optimizer_set(tconf.dec_optim.optim, self.dec_lrf_sv, tconf.dec_optim,
                              params=Helper.join_list(z.get_parameters() for z in self.decoders),
                              check_repeat=True, check_full=True)
        # ===== For training =====
        # schedule values
        self.margin = ScheduledValue("margin", tconf.margin)
        self._scheduled_values = [self.margin, self.enc_lrf_sv, self.dec_lrf_sv]
        # for refreshing dropouts
        self.previous_refresh_training = True
        # =====
        # others
        self.train_constrain_evt_types = {"": None, "kbp17": KBP17_TYPES}[conf.tconf.constrain_evt_types]
        self.test_constrain_evt_types = {"": None, "kbp17": KBP17_TYPES}[conf.iconf.constrain_evt_types]

    # build encoder
    def build_encoder(self) -> BasicNode:
        return MyIEBT(self.pc, self.conf.bt_conf, self.conf.tconf, self.vpack)

    # to be implemented
    def build_decoders(self) -> List[BasicNode]:
        raise NotImplementedError()

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
        for node in self.decoders:
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
                    # optional gold count
                    if len(one_losses[0][i]) >= 3:  # has gold count
                        info[f"loss_countg_{one_name}{i}"] = BK.stack([z[i][2] for z in one_losses]).sum().item()
                    # todo(note): any case that loss-count can be 0?
                    coll_sub_losses.append(this_loss_sum / (this_loss_count + 1e-5))
                # sub losses are already multiplied by sub-lambdas
                weighted_sub_loss = BK.stack(coll_sub_losses).sum() * one_lambda
                final_losses.append(weighted_sub_loss)
        if len(final_losses)>0:
            final_loss = BK.stack(final_losses).sum()
            if training and final_loss.requires_grad:
                BK.backward(final_loss, loss_factor)
