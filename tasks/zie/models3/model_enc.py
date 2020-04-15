#

# the bert based encoder part

from typing import List
import numpy as np
from collections import namedtuple, defaultdict

from msp.utils import Conf, zlog, Constants
from msp.nn import BK
from msp.nn.layers import BasicNode, Embedding
from msp.nn.modules import Berter2, Berter2Conf
from msp.nn.modules import EncConf, MyEncoder
from msp.data import VocabPackage, MultiHelper
from msp.zext.seq_helper import DataPadder

from ..common.data import DocInstance, Sentence
from ..common.vocab import IEVocabPackage
from ..common.model import BTConf, MyIEBT
from ..common.run import IndexerHelper

from .model_dec import TaskSpecAdp

# returning info for one MultiSentence
# sents: List[Sentence], offsets: LEN=len(sents)+1 cur seq word-level offsets for each sentence (does not include ROOT),
# center_idx: idx of center sent in 'sents', subword_size: used for batching
# MultiSentItem = namedtuple('MultiSentItem', ['sents', 'offsets', 'center_idx', 'subword_size',
#                                              'fake_sent', 'arg_pack'])
class MultiSentItem:
    def __init__(self, sents: List[Sentence], offsets: List[int], center_idx: int, subword_size: int, fake_sent: Sentence,
                 arg_pack, center_word2sub):
        self.sents = sents
        self.offsets = offsets
        self.center_idx = center_idx
        self.subword_size = subword_size
        self.fake_sent = fake_sent
        self.arg_pack = arg_pack
        self.center_word2sub = center_word2sub  # word-idx -> [subword-start, subword-end)

# =====
# the encoder

#
class M3EncConf(BTConf):
    def __init__(self):
        super().__init__()
        # by default no encoders
        self.enc_conf.enc_rnn_layer = 0
        # =====
        self.m2e_use_basic = False  # use basic encoder (at least embeddings)
        self.m2e_use_basic_dep = False  # use the special dep features as a replacement of basic_plus
        self.m2e_use_basic_plus = True  # use basic encoder with the new mode
        self.bert_conf = Berter2Conf()
        self.ms_extend_step = 0  # multi-sent extending for each side (win=2*mse+1) for encoding
        self.ms_extend_budget = 510  # overall subword budget, do not make it too large
        self.benc_bucket_msize = Constants.INT_PRAC_MAX  # max bsize for one bucket
        self.benc_bucket_range = 10  # similar to enc_bucket_range, but for bert enc
        # extra encoder over bert?
        self.m3_enc_conf = EncConf().init_from_kwargs(enc_rnn_type="lstm2", enc_hidden=512, enc_rnn_layer=0)
        # simple dep based basic part
        self.dep_output_dim = 512
        self.dep_label_dim = 50
        # =====
        self.bert_use_center_typeids = True  # current(center) sent 1, others 0
        self.bert_use_special_typeids = False  # pred as 0
        # other inputs for bert
        self.bert_other_inputs = []  # names of factors in Sentence (direct field name like 'uposes', 'entity_labels', ...)

#
class M3Encoder(MyIEBT):
    def __init__(self, pc: BK.ParamCollection, conf: M3EncConf, tconf, vpack: VocabPackage):
        super().__init__(pc, conf, tconf, vpack)
        #
        self.conf = conf
        # ----- bert
        # modify bert_conf for other input
        BERT_OTHER_VSIZE = 50  # todo(+N): this should be enough for small inputs!
        conf.bert_conf.bert2_other_input_names = conf.bert_other_inputs
        conf.bert_conf.bert2_other_input_vsizes = [BERT_OTHER_VSIZE] * len(conf.bert_other_inputs)
        self.berter = self.add_sub_node("bert", Berter2(pc, conf.bert_conf))
        # -----
        # index fake sent
        self.index_helper = IndexerHelper(vpack)
        # extra encoder over bert?
        self.bert_dim, self.bert_fold = self.berter.get_output_dims()
        conf.m3_enc_conf._input_dim = self.bert_dim
        self.m3_encs = [self.add_sub_node("m3e", MyEncoder(pc, conf.m3_enc_conf)) for _ in range(self.bert_fold)]
        self.m3_enc_out_dim = self.m3_encs[0].get_output_dims()[0]
        # skip m3_enc?
        self.m3_enc_is_empty = all(len(z.layers)==0 for z in self.m3_encs)
        if self.m3_enc_is_empty:
            assert all(z.get_output_dims()[0] == self.bert_dim for z in self.m3_encs)
            zlog("For m3_enc, we will skip it since it is empty!!")
        # dep as basic?
        if conf.m2e_use_basic_dep:
            MAX_LABEL_NUM = 200  # this should be enough
            self.dep_label_emb = self.add_sub_node("dlab", Embedding(self.pc, MAX_LABEL_NUM, conf.dep_label_dim, name="dlab"))
            self.dep_layer = self.add_sub_node("dep", TaskSpecAdp(pc, [(self.m3_enc_out_dim, self.bert_fold), None],
                                                                  [conf.dep_label_dim], conf.dep_output_dim))
        else:
            self.dep_label_emb = self.dep_layer = None
        self.dep_padder = DataPadder(2, pad_vals=0)  # 0 for both head-idx and label

    # multi-sentence encoding
    def run(self, insts: List[DocInstance], training: bool):
        conf = self.conf
        BERT_MAX_LEN = 510  # save 2 for CLS and SEP
        # =====
        # encoder 1: the basic encoder
        # todo(note): only DocInstane input for this mode, otherwise will break
        if conf.m2e_use_basic:
            reidx_pad_len = conf.ms_extend_budget
            # enc the basic part + also get some indexes
            sentid2offset = {}  # id(sent)->overall_seq_offset
            seq_offset = 0  # if look at the docs in one seq
            all_sents = []  # (inst, d_idx, s_idx)
            for d_idx, one_doc in enumerate(insts):
                assert isinstance(one_doc, DocInstance)
                for s_idx, one_sent in enumerate(one_doc.sents):
                    # todo(note): here we encode all the sentences
                    all_sents.append((one_sent, d_idx, s_idx))
                    sentid2offset[id(one_sent)] = seq_offset
                    seq_offset += one_sent.length - 1  # exclude extra ROOT node
            sent_reprs = self.run_sents(all_sents, insts, training)
            # flatten and concatenate and re-index
            reidxes_arr = np.zeros(seq_offset+reidx_pad_len, dtype=np.long)  # todo(note): extra padding to avoid out of boundary
            all_flattened_reprs = []
            all_flatten_offset = 0  # the local offset for batched basic encoding
            for one_pack in sent_reprs:
                one_sents, _, one_repr_ef, one_repr_evt, _ = one_pack
                assert one_repr_ef is one_repr_evt, "Currently does not support separate basic enc in m3 mode"
                one_repr_t = one_repr_evt
                _, one_slen, one_ldim = BK.get_shape(one_repr_t)
                all_flattened_reprs.append(one_repr_t.view([-1, one_ldim]))
                # fill in the indexes
                for one_sent in one_sents:
                    cur_start_offset = sentid2offset[id(one_sent)]
                    cur_real_slen = one_sent.length - 1
                    # again, +1 to get rid of extra ROOT
                    reidxes_arr[cur_start_offset:cur_start_offset+cur_real_slen] = \
                        np.arange(cur_real_slen, dtype=np.long) + (all_flatten_offset+1)
                    all_flatten_offset += one_slen  # here add the slen in batched version
            # re-idxing
            seq_sent_repr0 = BK.concat(all_flattened_reprs, 0)
            seq_sent_repr = BK.select(seq_sent_repr0, reidxes_arr, 0)  # [all_seq_len, D]
        else:
            sentid2offset = defaultdict(int)
            seq_sent_repr = None
        # =====
        # repack and prepare for multiple sent enc
        # todo(note): here, the criterion is based on bert's tokenizer
        all_ms_info = []
        if isinstance(insts[0], DocInstance):
            for d_idx, one_doc in enumerate(insts):
                for s_idx, x in enumerate(one_doc.sents):
                    # the basic criterion is the same as the basic one
                    include_flag = False
                    if training:
                        if x.length<self.train_skip_length and x.length>=self.train_min_length \
                                and (len(x.events)>0 or next(self.random_sample_stream)>self.train_skip_noevt_rate):
                            include_flag = True
                    else:
                        if x.length >= self.test_min_length:
                            include_flag = True
                    if include_flag:
                        all_ms_info.append(x.preps["ms"])  # use the pre-calculated one
        else:
            # multisent based
            all_ms_info = insts.copy()  # shallow copy
        # =====
        # encoder 2: the bert one (multi-sent encoding)
        ms_size_f = lambda x: x.subword_size
        all_ms_info.sort(key=ms_size_f)
        all_ms_buckets = self._bucket_sents_by_length(all_ms_info, conf.benc_bucket_range, ms_size_f, max_bsize=conf.benc_bucket_msize)
        berter = self.berter
        rets = []
        bert_use_center_typeids = conf.bert_use_center_typeids
        bert_use_special_typeids = conf.bert_use_special_typeids
        bert_other_inputs = conf.bert_other_inputs
        for one_bucket in all_ms_buckets:
            # prepare
            batched_ids = []
            batched_starts = []
            batched_seq_offset = []
            batched_typeids = []
            batched_other_inputs_list: List = [[] for _ in bert_other_inputs]  # List(comp) of List(batch) of List(idx)
            for one_item in one_bucket:
                one_sents = one_item.sents
                one_center_sid = one_item.center_idx
                one_ids, one_starts, one_typeids = [], [], []
                one_other_inputs_list = [[] for _ in bert_other_inputs]  # List(comp) of List(idx)
                for one_sid, one_sent in enumerate(one_sents):  # for bert
                    one_bidxes = one_sent.preps["bidx"]
                    one_ids.extend(one_bidxes.subword_ids)
                    one_starts.extend(one_bidxes.subword_is_start)
                    # prepare other inputs
                    for this_field_name, this_tofill_list in zip(bert_other_inputs, one_other_inputs_list):
                        this_tofill_list.extend(one_sent.preps["sub_"+this_field_name])
                    # todo(note): special procedure
                    if bert_use_center_typeids:
                        if one_sid != one_center_sid:
                            one_typeids.extend([0] * len(one_bidxes.subword_ids))
                        else:
                            this_typeids = [1] * len(one_bidxes.subword_ids)
                            if bert_use_special_typeids:
                                # todo(note): this is the special mode that we are given the events!!
                                for this_event in one_sents[one_center_sid].events:
                                    _, this_wid, this_wlen = this_event.mention.hard_span.position(headed=False)
                                    for a,b in one_item.center_word2sub[this_wid-1:this_wid-1+this_wlen]:
                                        this_typeids[a:b] = [0]*(b-a)
                            one_typeids.extend(this_typeids)
                batched_ids.append(one_ids)
                batched_starts.append(one_starts)
                batched_typeids.append(one_typeids)
                for comp_one_oi, comp_batched_oi in zip(one_other_inputs_list, batched_other_inputs_list):
                    comp_batched_oi.append(comp_one_oi)
                # for basic part
                batched_seq_offset.append(sentid2offset[id(one_sents[0])])
            # bert forward: [bs, slen, fold, D]
            if not bert_use_center_typeids:
                batched_typeids = None
            bert_expr0, mask_expr = berter.forward_batch(batched_ids, batched_starts, batched_typeids,
                                                         training=training, other_inputs=batched_other_inputs_list)
            if self.m3_enc_is_empty:
                bert_expr = bert_expr0
            else:
                mask_arr = BK.get_value(mask_expr)  # [bs, slen]
                m3e_exprs = [cur_enc(bert_expr0[:,:,cur_i], mask_arr) for cur_i, cur_enc in enumerate(self.m3_encs)]
                bert_expr = BK.stack(m3e_exprs, -2)  # on the fold dim again
            # collect basic ones: [bs, slen, D'] or None
            if seq_sent_repr is not None:
                arange_idxes_t = BK.arange_idx(BK.get_shape(mask_expr, -1)).unsqueeze(0)  # [1, slen]
                offset_idxes_t = BK.input_idx(batched_seq_offset).unsqueeze(-1) + arange_idxes_t  # [bs, slen]
                basic_expr = seq_sent_repr[offset_idxes_t]  # [bs, slen, D']
            elif conf.m2e_use_basic_dep:
                # collect each token's head-bert and ud-label, then forward with adp
                fake_sents = [one_item.fake_sent for one_item in one_bucket]
                # head idx and labels, no artificial ROOT
                padded_head_arr, _ = self.dep_padder.pad([s.ud_heads.vals[1:] for s in fake_sents])
                padded_label_arr, _ = self.dep_padder.pad([s.ud_labels.idxes[1:] for s in fake_sents])
                # get tensor
                padded_head_t = (BK.input_idx(padded_head_arr) - 1)  # here, the idx exclude root
                padded_head_t.clamp_(min=0)  # [bs, slen]
                padded_label_t = BK.input_idx(padded_label_arr)
                # get inputs
                input_head_bert_t = bert_expr[BK.arange_idx(len(fake_sents)).unsqueeze(-1), padded_head_t]  # [bs, slen, fold, D]
                input_label_emb_t = self.dep_label_emb(padded_label_t)  # [bs, slen, D']
                basic_expr = self.dep_layer(input_head_bert_t, None, [input_label_emb_t])  # [bs, slen, ?]
            elif conf.m2e_use_basic_plus:
                sent_reprs = self.run_sents([(one_item.fake_sent, None, None) for one_item in one_bucket], insts, training, use_one_bucket=True)
                assert len(sent_reprs) == 1, "Unsupported split reprs for basic encoder, please set enc_bucket_range<=benc_bucket_range"
                _, _, one_repr_ef, one_repr_evt, _ = sent_reprs[0]
                assert one_repr_ef is one_repr_evt, "Currently does not support separate basic enc in m3 mode"
                basic_expr = one_repr_evt[:,1:]  # exclude ROOT, [bs, slen, D]
                assert BK.get_shape(basic_expr)[:2] == BK.get_shape(bert_expr)[:2]
            else:
                basic_expr = None
            # pack: (List[ms_item], bert_expr, basic_expr)
            rets.append((one_bucket, bert_expr, basic_expr))
        return rets

    # prepare instance
    def prepare_inst(self, inst: DocInstance):
        berter = self.berter
        conf = self.conf
        ms_extend_budget = conf.ms_extend_budget
        ms_extend_step = conf.ms_extend_step
        # -----
        # prepare bert tokenization results
        # print(inst.doc_id)
        for sent in inst.sents:
            real_words = sent.words.vals[1:]  # no special ROOT
            bidxes = berter.subword_tokenize(real_words, True)
            sent.preps["bidx"] = bidxes
            # prepare subword expanded fields
            subword2word = np.cumsum(bidxes.subword_is_start).tolist()  # -1 and +1 happens to cancel out
            for field_name in conf.bert_other_inputs:
                field_idxes = getattr(sent, field_name).idxes  # use full one since idxes are set in this way
                sent.preps["sub_"+field_name] = [field_idxes[z] for z in subword2word]
        # -----
        # prepare others (another loop since we need cross-sent bert tokens)
        for sent in inst.sents:
            # -----
            # prepare multi-sent
            # include this multiple sent pack, extend to both sides until window limit or bert limit
            cur_center_sent = sent
            cur_sid, cur_doc = sent.sid, sent.doc
            cur_doc_sents = cur_doc.sents
            cur_doc_nsent = len(cur_doc.sents)
            cur_sid_left = cur_sid_right = cur_sid
            cur_subword_size = len(cur_center_sent.preps["bidx"].subword_ids)
            for step in range(ms_extend_step):
                # first left then right
                if cur_sid_left > 0:
                    this_subword_size = len(cur_doc_sents[cur_sid_left - 1].preps["bidx"].subword_ids)
                    if cur_subword_size + this_subword_size <= ms_extend_budget:
                        cur_sid_left -= 1
                        cur_subword_size += this_subword_size
                if cur_sid_right < cur_doc_nsent - 1:
                    this_subword_size = len(cur_doc_sents[cur_sid_right + 1].preps["bidx"].subword_ids)
                    if cur_subword_size + this_subword_size <= ms_extend_budget:
                        cur_sid_right += 1
                        cur_subword_size += this_subword_size
            # List[Sentence], List[int], center_local_idx, all_subword_size
            cur_sents = cur_doc_sents[cur_sid_left:cur_sid_right + 1]
            cur_offsets = [0]
            for s in cur_sents:
                cur_offsets.append(s.length - 1 + cur_offsets[-1])  # does not include ROOT here!!
            one_ms = MultiSentItem(cur_sents, cur_offsets, cur_sid - cur_sid_left, cur_subword_size, None, None, None)
            sent.preps["ms"] = one_ms
            # -----
            # subword idx for center sent
            center_word2sub = []
            prev_start = -1
            center_subword_is_start = cur_center_sent.preps["bidx"].subword_is_start
            for cur_end, one_is_start in enumerate(center_subword_is_start):
                if one_is_start:
                    if prev_start>=0:
                        center_word2sub.append((prev_start, cur_end))
                    prev_start = cur_end
            if prev_start>=0:
                center_word2sub.append((prev_start, len(center_subword_is_start)))
            one_ms.center_word2sub = center_word2sub
            # -----
            # fake a concat sent for basic plus modeling
            if conf.m2e_use_basic_plus or conf.m2e_use_basic_dep:
                concat_words, concat_lemmas, concat_uposes, concat_ud_heads, concat_ud_labels = [], [], [], [], []
                cur_fake_offset = 0  # overall offset in fake sent
                prev_root = None
                for one_fake_inner_sent in cur_sents:  # exclude root
                    concat_words.extend(one_fake_inner_sent.words.vals[1:])
                    concat_lemmas.extend(one_fake_inner_sent.lemmas.vals[1:])
                    concat_uposes.extend(one_fake_inner_sent.uposes.vals[1:])
                    # todo(note): make the heads look like a real sent; the actual heads already +=1; root points to prev root
                    for local_i, local_h in enumerate(one_fake_inner_sent.ud_heads.vals[1:]):
                        if local_h == 0:
                            if prev_root is None:
                                global_h = cur_fake_offset + local_i + 1  # +1 here for offset
                            else:
                                global_h = prev_root
                            prev_root = cur_fake_offset + local_i + 1  # +1 here for offset
                        else:
                            global_h = cur_fake_offset + local_h  # already +=1
                        concat_ud_heads.append(global_h)
                    concat_ud_labels.extend(one_fake_inner_sent.ud_labels.vals[1:])
                    cur_fake_offset += len(one_fake_inner_sent.words.vals)-1
                one_fake_sent = Sentence(None, concat_words, concat_lemmas, concat_uposes, concat_ud_heads, concat_ud_labels,
                                         None, None)
                one_ms.fake_sent = one_fake_sent
                self.index_helper.index_sent(one_fake_sent)

    def special_refresh(self, embed_rop, other_rop):
        super().special_refresh(embed_rop, other_rop)
        self.berter.refresh(other_rop)
        for one in self.m3_encs + [self.dep_layer, self.dep_label_emb]:
            if one is not None:
                one.refresh(other_rop)

    # #
    # def get_output_dims(self, *input_dims):
    #     raise RuntimeError("Complex output, thus not using this one")

    def speical_output_dims(self):
        conf = self.conf
        if conf.m2e_use_basic_dep:
            basic_dim = conf.dep_output_dim
        elif conf.m2e_use_basic or conf.m2e_use_basic_plus:
            basic_dim = self.enc_evt_output_dim
        else:
            basic_dim = None
        # bert_outputs, basic_output
        return (self.m3_enc_out_dim, self.bert_fold), basic_dim

# long-range separate for ef/evt: ef -> is it the default one, evt -> does it need default one

# b tasks/zie/models3/model_enc:93
