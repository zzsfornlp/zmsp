#

# the document-hint module
# -- get doc-aware sent embeddings

from typing import List
import numpy as np

from msp.utils import Conf, Helper
from msp.nn import BK
from msp.nn.layers import BasicNode, Affine, AttConf, AttentionNode
from msp.nn.modules import EncConf, MyEncoder
from msp.zext.ie.keyword import KeyWordConf, KeyWordModel

from .data import DocInstance, Sentence

class DocHintConf(Conf):
    def __init__(self):
        # dimension specifications
        self._input_dim = 0  # dim of sent-embed CLS
        self._output_dim = 0  # dim of sent-encoder's input
        # 1. doc encoding
        self.enc_doc_conf = EncConf().init_from_kwargs(enc_rnn_type="lstm2", enc_hidden=512, enc_rnn_layer=0)
        # 2. doc hints
        # keyword model
        self.kconf = KeyWordConf()
        self.katt_conf = AttConf().init_from_kwargs(d_kqv=128, head_count=2, out_act="elu")
        # keyword part
        self.use_keyword = True
        self.keyword_min_count = 2  # indoc-freq should >=this
        self.keyword_min_rank = 100  # global-rank should >=this
        self.num_keyword_general = 3
        self.num_keyword_noun = 3
        self.num_keyword_verb = 3
        # keysent part
        self.use_keysent = True
        self.num_keysent_first = 3
        self.num_keysent_top = 3
        self.keysent_min_len = 1  # only include sent >= this len
        self.keysent_topktok_score = 10  # how many topk-tokens to include when calculating top scored sent

class DocHintModule(BasicNode):
    def __init__(self, pc, dh_conf: DocHintConf):
        super().__init__(pc, None, None)
        self.conf: DocHintConf = dh_conf
        # =====
        self.input_dim, self.output_dim = dh_conf._input_dim, dh_conf._output_dim
        # 1. doc encoding
        self.conf.enc_doc_conf._input_dim = self.input_dim
        self.enc_doc = self.add_sub_node("enc_d", MyEncoder(self.pc, self.conf.enc_doc_conf))
        self.enc_output_dim = self.enc_doc.get_output_dims()[0]
        # 2. keyword/keysent based doc hints (key/value)
        katt_conf = dh_conf.katt_conf
        self.kw_att = self.add_sub_node("kw", AttentionNode.get_att_node(
            katt_conf.type, pc, self.input_dim, self.enc_output_dim, self.input_dim, katt_conf))
        self.ks_att = self.add_sub_node("ks", AttentionNode.get_att_node(
            katt_conf.type, pc, self.input_dim, self.enc_output_dim, self.input_dim, katt_conf))
        # word model (load from outside)
        self.keyword_model = None
        if self.conf.kconf.load_file:
            self.keyword_model = KeyWordModel.load(self.conf.kconf.load_file, self.conf.kconf)
        # 3. combine
        final_input_dims = [self.enc_output_dim] + [self.input_dim] * (int(dh_conf.use_keyword) + int(dh_conf.use_keysent))
        self.final_layer = self.add_sub_node("fl", Affine(pc, final_input_dims, self.output_dim, act="elu"))

    def get_output_dims(self, *input_dims):
        return (self.output_dim, )

    # get key words and key sent reprs (dynamically) for one doc
    def _get_keyws(self, one_doc: DocInstance):
        conf = self.conf
        # -----
        # 2.0. get tf-idf for each word and collect other info
        doc_sents = []
        doc_tokens = []
        doc_reprs = []  # [NumSent] of arr[1+NumToken, D]
        doc_word_dict = {}  # w -> [List[(sid,wid(+R))], as_noun, as_verb]
        keyword_min_rank = conf.keyword_min_rank  # todo(note): use it here!
        for one_sid, one_sent in enumerate(one_doc.sents):
            one_words, one_poses = one_sent.words.vals[1:], one_sent.uposes.vals[1:]  # exclude the root!
            doc_reprs.append(one_sent.extra_features["aux_repr"])
            one_sent_tokens = []
            for one_wid in range(len(one_words)):
                w, p = one_words[one_wid], one_poses[one_wid]
                # todo(note): lowercase, exclude non-alpha, exclude global high-rank
                w = str.lower(w)
                if str.isalpha(w) and self.keyword_model.get_rank(w)>=keyword_min_rank:
                    one_sent_tokens.append(w)
                    item = doc_word_dict.get(w, None)
                    if item is None:
                        item = [[], False, False]
                        doc_word_dict[w] = item
                    item[0].append((one_sid, one_wid+1))  # original word position offset by root
                    item[1] = (item[1] or (p=="NOUN"))
                    item[2] = (item[2] or (p=="VERB"))
            doc_sents.append(one_sent_tokens)
            doc_tokens.extend(one_sent_tokens)
        # List[(w, raw-count, tf-idf, jump-score)], currently using tf-idf
        SCORE_IDX = -2
        doc_keyword_list = sorted(self.keyword_model.extract(doc_tokens), key=lambda x: x[SCORE_IDX], reverse=True)
        # -----
        # 2.1. get all the key words
        final_keyword_set = set()
        general_budget, noun_budget, verb_budget = conf.num_keyword_general, conf.num_keyword_noun, conf.num_keyword_verb
        keyword_min_count = conf.keyword_min_count
        for w, rank, raw_count, tf_idf, _ in doc_keyword_list:
            if raw_count >= keyword_min_count and rank >= keyword_min_rank:
                _, is_noun, is_verb = doc_word_dict[w]  # must be there
                # add them separately
                if general_budget>0:
                    final_keyword_set.add(w)
                    general_budget -= 1
                if is_noun and noun_budget>0:
                    final_keyword_set.add(w)
                    noun_budget -= 1
                if is_verb and verb_budget>0:
                    final_keyword_set.add(w)
                    verb_budget -= 1
                if general_budget+noun_budget+verb_budget <= 0:
                    break
        # get averaged reprs; todo(note): currently the order does not matter
        final_keyword_reprs = []  # [KW, D]
        for w in final_keyword_set:
            posi_list = doc_word_dict[w][0]
            one_repr = np.stack([doc_reprs[a][b] for a,b in posi_list], 0).mean(0)  # [D]
            final_keyword_reprs.append(one_repr)
        # -----
        # 2.2. get all keysents
        keysent_min_len = conf.keysent_min_len
        keysent_topktok_score = conf.keysent_topktok_score
        candidate_sent_ids = [i for i,z in enumerate(doc_sents) if len(z)>=keysent_min_len]
        word2score = {z[0]: z[SCORE_IDX] for z in doc_keyword_list}
        sent_id_score_pairs = []
        for s_id in candidate_sent_ids:
            s_topk_scores = sorted([word2score[t] for t in doc_sents[s_id]], reverse=True)
            s_score = np.average(s_topk_scores[:keysent_topktok_score])  # only care about high-scored words
            sent_id_score_pairs.append((s_id, s_score))
        sent_id_score_pairs.sort(key=lambda x: x[-1], reverse=True)
        # the union of first and best-score sents
        final_keysent_ids = sorted(set(candidate_sent_ids[:conf.num_keysent_first] +
                                       [z[0] for z in sent_id_score_pairs[:conf.num_keysent_top]]))
        final_keysent_reprs = [doc_reprs[z][0] for z in final_keysent_ids]  # [KS, D]
        # -----  # p final_keyword_set, final_keysent_ids
        return final_keyword_reprs, final_keysent_reprs

    # input is 2d list of 1d np.arr
    def _pad_3d_arr(self, input_arrs, pad_arr):
        dim0 = len(input_arrs)
        dim1 = max(len(z) for z in input_arrs)
        ret_list = []
        ret_mask = np.ones([dim0, dim1])
        for i, one_arrs in enumerate(input_arrs):
            valid_size = len(one_arrs)
            pad_size = dim1 - valid_size
            ret_list.extend(one_arrs)
            ret_list.extend([pad_arr] * pad_size)
            ret_mask[i][valid_size:] = 0.
        ret_arr = np.asarray(ret_list).reshape([dim0, dim1, -1])
        return ret_arr, ret_mask

    # output is [bs(doc), dlen(sent), D']
    def run(self, insts: List[DocInstance]):
        conf = self.conf
        # 1. build the query: DocEncoding
        cls_list = []  # [NumDoc, NumSent]
        for one_doc in insts:
            cls_list.append([z.extra_features["aux_repr"][0] for z in one_doc.sents])  # [0] as [CLS]
        # padding
        pad_repr = np.zeros(self.input_dim)
        cls_repr_arr, cls_mask_arr = self._pad_3d_arr(cls_list, pad_repr)  # [NumDoc, MaxDLen, D]
        enc_doc_repr = self.enc_doc(BK.input_real(cls_repr_arr), cls_mask_arr)  # [NumDoc, MaxDLen, D']
        # 2. build the doc hints
        all_keyword_reprs, all_keysent_reprs = [], []  # [NumDoc, ?, D]
        for one_doc in insts:
            one_keyword_reprs, one_keysent_reprs = self._get_keyws(one_doc)
            all_keyword_reprs.append(one_keyword_reprs)
            all_keysent_reprs.append(one_keysent_reprs)
        # 2.5 attention
        final_inputs = [enc_doc_repr]
        if conf.use_keyword:
            keyword_repr_arr, keyword_mask_arr = self._pad_3d_arr(all_keyword_reprs, pad_repr)  # [NumDoc, Kw, D]
            keyword_repr_t, keyword_mask_t = BK.input_real(keyword_repr_arr), BK.input_real(keyword_mask_arr)
            att_kw_repr = self.kw_att(keyword_repr_t, keyword_repr_t, enc_doc_repr, mask_k=keyword_mask_t)  # [NumDoc, MaxDLen, D]
            final_inputs.append(att_kw_repr)
        if conf.use_keysent:
            keysent_repr_arr, keysent_mask_arr = self._pad_3d_arr(all_keysent_reprs, pad_repr)  # [NumDoc, Ks, D]
            keysent_repr_t, keysent_mask_t = BK.input_real(keysent_repr_arr), BK.input_real(keysent_mask_arr)
            att_ks_repr = self.ks_att(keysent_repr_t, keysent_repr_t, enc_doc_repr, mask_k=keysent_mask_t)  # [NumDoc, MaxDLen, D]
            final_inputs.append(att_ks_repr)
        # 3. combine all and return
        final_repr = self.final_layer(final_inputs)  # [NumDoc, MaxDLen, Dout]
        return final_repr

# b tasks/zie/common/dochint:179
