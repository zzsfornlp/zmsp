#

from typing import List
import numpy as np

from msp.utils import Conf, Random, zlog, JsonRW, zfatal, zwarn
from msp.data import VocabPackage, MultiHelper
from msp.nn import BK
from msp.nn.layers import BasicNode, Affine, RefreshOptions, NoDropRop
from msp.nn.modules import EmbedConf, MyEmbedder, EncConf, MyEncoder, Berter2Conf, Berter2
from msp.zext.seq_helper import DataPadder

# todo(note): most parts are adopted from model.py:ParserBT

# conf
class FpEncConf(Conf):
    def __init__(self):
        # embedding
        self.emb_conf = EmbedConf().init_from_kwargs(dim_word=0)  # by default no embedding inputs
        # bert
        self.bert_conf = Berter2Conf().init_from_kwargs(bert2_retinc_cls=True, bert2_training_mask_rate=0., bert2_output_mode="concat")
        # middle layer to reduce dim
        self.middle_dim = 0  # 0 means no middle one
        # encoder
        self.enc_conf = EncConf().init_from_kwargs(enc_rnn_type="lstm2", enc_hidden=1024, enc_rnn_layer=0)
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

# node
class FpEncoder(BasicNode):
    def __init__(self, pc: BK.ParamCollection, conf: FpEncConf, vpack: VocabPackage):
        super().__init__(pc, None, None)
        self.conf = conf
        # ===== Vocab =====
        self.word_vocab = vpack.get_voc("word")
        self.char_vocab = vpack.get_voc("char")
        self.pos_vocab = vpack.get_voc("pos")
        # avoid no params error
        self._tmp_v = self.add_param("nope", (1,))
        # ===== Model =====
        # embedding
        self.emb = self.add_sub_node("emb", MyEmbedder(self.pc, conf.emb_conf, vpack))
        self.emb_output_dim = self.emb.get_output_dims()[0]
        # bert
        self.bert = self.add_sub_node("bert", Berter2(self.pc, conf.bert_conf))
        self.bert_output_dim = self.bert.get_output_dims()[0]
        # make sure there are inputs
        assert self.emb_output_dim>0 or self.bert_output_dim>0
        # middle?
        if conf.middle_dim > 0:
            self.middle_node = self.add_sub_node("mid", Affine(self.pc, self.emb_output_dim + self.bert_output_dim,
                                                               conf.middle_dim, act="elu"))
            self.enc_input_dim = conf.middle_dim
        else:
            self.middle_node = None
            self.enc_input_dim = self.emb_output_dim + self.bert_output_dim  # concat the two parts (if needed)
        # encoder?
        # todo(note): feed compute-on-the-fly hp
        conf.enc_conf._input_dim = self.enc_input_dim
        self.enc = self.add_sub_node("enc", MyEncoder(self.pc, conf.enc_conf))
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
        #
        self.word_padder = DataPadder(2, pad_vals=self.word_vocab.pad, mask_range=2)
        self.char_padder = DataPadder(3, pad_lens=(0, 0, conf.char_max_length), pad_vals=self.char_vocab.pad)
        self.pos_padder = DataPadder(2, pad_vals=self.pos_vocab.pad)

    def get_output_dims(self, *input_dims):
        return (self.enc_output_dim, )

    #
    def refresh(self, rop=None):
        zfatal("Should call special_refresh instead!")

    # ====
    # special routines

    def special_refresh(self, embed_rop, other_rop):
        self.emb.refresh(embed_rop)
        self.enc.refresh(other_rop)
        self.bert.refresh(other_rop)
        if self.middle_node is not None:
            self.middle_node.refresh(other_rop)

    def prepare_training_rop(self):
        mconf = self.conf
        embed_rop = RefreshOptions(hdrop=mconf.drop_embed, dropmd=mconf.dropmd_embed, fix_drop=mconf.fix_drop)
        other_rop = RefreshOptions(hdrop=mconf.drop_hidden, idrop=mconf.idrop_rnn, gdrop=mconf.gdrop_rnn,
                                   fix_drop=mconf.fix_drop)
        return embed_rop, other_rop

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
    # run

    def prepare_inputs(self, insts, training, input_word_mask_repl=None):
        word_arr, char_arr, extra_arrs, aux_arrs = None, None, [], []
        # ===== specially prepare for the words
        wv = self.word_vocab
        W_UNK = wv.unk
        word_act_idxes = [z.words.idxes for z in insts]
        # todo(warn): still need the masks
        word_arr, mask_arr = self.word_padder.pad(word_act_idxes)
        if input_word_mask_repl is not None:
            input_word_mask_repl = input_word_mask_repl.astype(np.int)
            word_arr = word_arr * (1-input_word_mask_repl) + W_UNK * input_word_mask_repl  # replace with UNK
        # =====
        if not self.need_word:
            word_arr = None
        if self.need_char:
            chars = [z.chars.idxes for z in insts]
            char_arr, _ = self.char_padder.pad(chars)
        if self.need_pos:
            poses = [z.poses.idxes for z in insts]
            pos_arr, _ = self.pos_padder.pad(poses)
            extra_arrs.append(pos_arr)
        return word_arr, char_arr, extra_arrs, aux_arrs, mask_arr

    # todo(note): for rnn, need to transpose masks, thus need np.array
    # return input_repr, enc_repr, mask_arr
    def run(self, insts, training, input_word_mask_repl=None):
        self._cache_subword_tokens(insts)
        # prepare inputs
        word_arr, char_arr, extra_arrs, aux_arrs, mask_arr = \
            self.prepare_inputs(insts, training, input_word_mask_repl=input_word_mask_repl)
        # layer0: emb + bert
        layer0_reprs = []
        if self.emb_output_dim>0:
            emb_repr = self.emb(word_arr, char_arr, extra_arrs, aux_arrs)  # [BS, Len, Dim]
            layer0_reprs.append(emb_repr)
        if self.bert_output_dim>0:
            # prepare bert inputs
            BERT_MASK_ID = self.bert.tokenizer.mask_token_id
            batch_subword_ids, batch_subword_is_starts = [], []
            for bidx, one_inst in enumerate(insts):
                st = one_inst.extra_features["st"]
                if input_word_mask_repl is not None:
                    cur_subword_ids, cur_subword_is_start, _ = \
                        st.mask_and_return(input_word_mask_repl[bidx][1:], BERT_MASK_ID)  # todo(note): exclude ROOT for bert tokens
                else:
                    cur_subword_ids, cur_subword_is_start = st.subword_ids, st.subword_is_start
                batch_subword_ids.append(cur_subword_ids)
                batch_subword_is_starts.append(cur_subword_is_start)
            bert_repr, _ = self.bert.forward_batch(batch_subword_ids, batch_subword_is_starts,
                                                   batched_typeids=None, training=training)  # [BS, Len, D']
            layer0_reprs.append(bert_repr)
        # layer1: enc
        enc_input_repr = BK.concat(layer0_reprs, -1)  # [BS, Len, D+D']
        if self.middle_node is not None:
            enc_input_repr = self.middle_node(enc_input_repr)  # [BS, Len, D??]
        enc_repr = self.enc(enc_input_repr, mask_arr)
        mask_repr = BK.input_real(mask_arr)
        return enc_repr, mask_repr  # [bs, len, *], [bs, len]

    # =====
    # caching
    def _cache_subword_tokens(self, insts):
        for one_inst in insts:
            if "st" not in one_inst.extra_features:
                one_inst.extra_features["st"] = self.bert.subword_tokenize2(one_inst.words.vals[1:], True)
