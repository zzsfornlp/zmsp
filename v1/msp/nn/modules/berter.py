#

# extract features from pretrained bert

try:
    from transformers import BertModel, BertTokenizer, BertForMaskedLM
except:
    transformers = None

import numpy as np
from typing import Tuple, Iterable, List
from collections import namedtuple

from msp.utils import Conf, zcheck, zlog, zwarn, Helper
from msp.nn import BK

#
class BerterConf(Conf):
    def __init__(self):
        # basic
        self.bert_model = "bert-base-multilingual-cased"  # or "bert-base-cased", "bert-large-cased", "bert-base-chinese"
        self.bert_lower_case = False
        self.bert_layers = [-1]  # which layers to extract, use concat for final output
        self.bert_trainable = False
        self.bert_to_device = True  # perform things on default devices
        # cache dir for downloading bert models
        self.bert_cache_dir = ""
        # for feature extracting
        self.bert_sent_extend = 0  # use how many context sentence before and after (but overall is constrained within bert's input limit)
        self.bert_tok_extend = 0  # similar to the sent one, extra tok constrains
        self.bert_extend_sent_bound = True  # extend at sent boundary
        self.bert_root_mode = 0  # specially for artificial root, 0 means nothing, 1 means alwasy CLS, -1 means previous sub-token
        self.bert_previous_rate = 0.75  # if limited budget, what is the rate that we prefer for the previous contexts
        # specific
        self.bert_batch_size = 1  # forwarding batch size
        self.bert_single_len = 256  # if segement length >= this, then ignore bsize and forward one by one
        # extra special one, zero padding embedding
        self.bert_zero_pademb = False  # whether make padding embedding zero vector

#
SubwordToks = namedtuple('SubwordToks', ['subword_toks', 'subword_ids', 'subword_is_start', 'subword_typeid'])

#
class Berter(object):
    def __init__(self, pc: BK.ParamCollection, bconf: BerterConf):
        # super().__init__(pc, None, None)
        # TODO(+N): currently use freezed bert features, therefore, also not a BasicNode
        self.bconf = bconf
        assert not bconf.bert_trainable, "Currently only using this part for feature extractor"
        MODEL_NAME = bconf.bert_model
        zlog(f"Loading pre-trained bert model of {MODEL_NAME}")
        # Load pretrained model/tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=bconf.bert_lower_case,
                                                       cache_dir=None if (not bconf.bert_cache_dir) else bconf.bert_cache_dir)
        # model = BertModel.from_pretrained(MODEL_NAME, cache_dir="./")
        self.model = BertModel.from_pretrained(MODEL_NAME, output_hidden_states=True,
                                               cache_dir=None if (not bconf.bert_cache_dir) else bconf.bert_cache_dir)
        self.model.eval()
        # =====
        # for word predictions
        voc = self.tokenizer.vocab
        self.tok_word_list = [None] * len(voc)  # idx -> str
        for s, i in voc.items():
            self.tok_word_list[i] = s
        self.wp_model = BertForMaskedLM.from_pretrained(MODEL_NAME, cache_dir=None if (not bconf.bert_cache_dir) else bconf.bert_cache_dir)
        self.wp_model.eval()
        # =====
        # zero padding embeddings?
        if bconf.bert_zero_pademb:
            with BK.no_grad_env():
                # todo(warn): specific!!
                zlog(f"Unusual operation: make bert's padding embedding (idx0) zero!!")
                self.model.embeddings.word_embeddings.weight[0].fill_(0.)
        if bconf.bert_to_device:
            zlog(f"Moving bert model to default device {BK.DEFAULT_DEVICE}")
            BK.to_device(self.model)
            BK.to_device(self.wp_model)

    # if not using anymore, release the resources
    def delete_self(self):
        self.model = None
        self.wp_model = None

    # =====
    # only forward to get features

    # prepare subword-tokens and subword-word correspondences (could be document level)
    # todo(note): here we assume that the input does not have special ROOT, otherwise that will be strange to bert!
    # todo(note): here does not add CLS or SEP, since later we can concatenate things!
    def subword_tokenize(self, tokens: List[str], no_special_root: bool, mask_idx=-1, mask_mode="all", mask_repl=""):
        assert no_special_root
        return Berter._subword_tokenize(self.tokenizer, tokens, mask_idx, mask_mode, mask_repl)

    # forward with a collection of sentences, input is List[(subwords, starts)]
    # todo(note): currently all change to np.ndarray and return,
    # todo(+N): maybe more efficient to directly return tensor, but how to deal with batch?
    def extract_features(self, sents: List[Tuple]):
        # =====
        MAX_LEN = 510  # save two for [CLS] and [SEP]
        BACK_LEN = 100  # for splitting cases, still remaining some of previous sub-tokens for context
        model = self.model
        tokenizer = self.tokenizer
        CLS, CLS_IDX = tokenizer.cls_token, tokenizer.cls_token_id
        SEP, SEP_IDX = tokenizer.sep_token, tokenizer.sep_token_id
        # =====
        # prepare the forwarding
        bconf = self.bconf
        bert_sent_extend = bconf.bert_sent_extend
        bert_tok_extend = bconf.bert_tok_extend
        bert_extend_sent_bound = bconf.bert_extend_sent_bound
        bert_root_mode = bconf.bert_root_mode
        #
        num_sent = len(sents)
        all_ids = [z.subword_ids for z in sents]
        # all_subwords = [z.subword_toks for z in sents]
        all_prep_sents = []
        for sent_idx in range(num_sent):
            cur_sent = sents[sent_idx]
            center_subwords, center_ids, center_starts = cur_sent.subword_toks, cur_sent.subword_ids, cur_sent.subword_is_start
            # extend extra tokens
            # prev_all_subwords, post_all_subwords = self.extend_subwords(
            #     all_subwords, sent_idx, MAX_LEN-len(center_ids), bert_sent_extend, bert_tok_extend, bert_extend_sent_bound)
            prev_all_ids, post_all_ids = self.extend_subwords(
                all_ids, sent_idx, MAX_LEN-len(center_ids), bert_sent_extend, bert_tok_extend, bert_extend_sent_bound)
            # put them together
            cur_all_ids = [CLS_IDX] + prev_all_ids + center_ids + post_all_ids + [SEP_IDX]
            cur_all_starts = [0] * (len(prev_all_ids)+1)
            if bert_root_mode == -1:
                cur_all_starts[-1] = 1  # previous token (can also be CLS if no prev ones)
            elif bert_root_mode == 1:
                cur_all_starts[0] = 1  # always take CLS
            cur_all_starts = cur_all_starts + center_starts + [0] * (len(post_all_ids)+1)
            cur_all_len = len(cur_all_ids)
            assert cur_all_len == len(cur_all_starts)
            all_prep_sents.append((cur_all_ids, cur_all_starts, cur_all_len, sent_idx))  # put more info here
        # forward
        bert_batch_size = bconf.bert_batch_size
        bert_single_len = bconf.bert_single_len
        hit_single_thresh = (bert_batch_size==1)  # if bsize==1, directly single mode
        all_prep_sents.sort(key=lambda x: x[-2])  # sort by sent length
        ret_arrs = [None] * num_sent
        with BK.no_grad_env():
            cur_ss_idx = 0  # idx for sorted sentence
            while cur_ss_idx < len(all_prep_sents):
                if all_prep_sents[cur_ss_idx][-2] >= bert_single_len:
                    hit_single_thresh = True
                # if already hit single thresh
                if hit_single_thresh:
                    cur_single_sent = all_prep_sents[cur_ss_idx]
                    one_feat_arr = self.forward_single(cur_single_sent[0], cur_single_sent[1])
                    one_orig_idx = cur_single_sent[-1]
                    assert ret_arrs[one_orig_idx] is None
                    ret_arrs[one_orig_idx] = one_feat_arr
                    cur_ss_idx += 1
                else:
                    # collect current batch
                    cur_batch = []
                    for _ in range(bert_batch_size):
                        if cur_ss_idx >= len(all_prep_sents):
                            break
                        one_sent = all_prep_sents[cur_ss_idx]
                        if one_sent[-2] >= bert_single_len:
                            break
                        cur_batch.append(one_sent)
                        cur_ss_idx += 1
                    if len(cur_batch) > 0:
                        cur_batch_arrs = self.forward_batch([z[0] for z in cur_batch], [z[1] for z in cur_batch])
                        for one_feat_arr, one_single_sent in zip(cur_batch_arrs, cur_batch):
                            one_orig_idx = one_single_sent[-1]
                            assert ret_arrs[one_orig_idx] is None
                            ret_arrs[one_orig_idx] = one_feat_arr
        return ret_arrs

    # simple mode for extracting features, ignoring certain confs
    # add one with CLS, but not SEP
    def extract_feature_simple_mode(self, sents: List[Tuple]):
        # =====
        MAX_LEN = 510  # save two for [CLS] and [SEP]
        tokenizer = self.tokenizer
        CLS, CLS_IDX = tokenizer.cls_token, tokenizer.cls_token_id
        SEP, SEP_IDX = tokenizer.sep_token, tokenizer.sep_token_id
        # =====
        ret_arrs = []
        for one_sent in sents:
            assert len(one_sent.subword_ids) <= MAX_LEN
            cur_all_ids = [CLS_IDX] + one_sent.subword_ids + [SEP_IDX]
            cur_all_starts = [1] + one_sent.subword_is_start + [0]
            one_feat_arr = self.forward_single(cur_all_ids, cur_all_starts)
            ret_arrs.append(one_feat_arr)
        return ret_arrs

    # =====
    # helpers

    def forward_features(self, ids_expr, mask_expr, output_layers):
        # token_type_ids are by default 0
        final_layer, _, encoded_layers = self.model(ids_expr, attention_mask=mask_expr)
        concated_expr = BK.concat([encoded_layers[li] for li in output_layers], -1)  # [bsize, slen, DIM*layer]
        return concated_expr

    def forward_single(self, cur_ids: List[int], cur_starts: List[int]):
        # todo(note): we may need to split long segment in single mode
        tokenizer, model = self.tokenizer, self.model
        output_layers = self.bconf.bert_layers
        #
        MAX_LEN = 510  # save two for [CLS] and [SEP]
        BACK_LEN = 100  # for splitting cases, still remaining some of previous sub-tokens for context
        CLS, CLS_IDX = tokenizer.cls_token, tokenizer.cls_token_id
        SEP, SEP_IDX = tokenizer.sep_token, tokenizer.sep_token_id
        #
        if len(cur_ids) < MAX_LEN+2:
            final_outputs = self.forward_features(BK.input_idx(cur_ids).unsqueeze(0), None, output_layers)[0]  #[La, *]
            ret = final_outputs[BK.input_idx(cur_starts) == 1]  # direct masked select
        else:
            # forwarding (multiple times within the model max-length constrain)
            all_outputs = []
            cur_sub_idx = 0
            while cur_sub_idx < len(cur_ids)-1:  # minus 1 to ignore ending SEP
                cur_slice_start = max(1, cur_sub_idx - BACK_LEN)
                cur_slice_end = min(cur_slice_start + MAX_LEN, len(cur_ids)-1)
                cur_toks = [CLS_IDX] + cur_ids[cur_slice_start:cur_slice_end] + [SEP_IDX]
                features = self.forward_features(BK.input_idx(cur_toks).unsqueeze(0), None, output_layers)  # [bs, L, *]
                cur_features = features[0]  # [L, *]
                assert len(cur_features) == len(cur_toks)
                # only include CLS in the first run, no SEP included
                if cur_sub_idx == 0:
                    # include CLS, exclude SEP
                    all_outputs.append(cur_features[:-1])
                else:
                    # include only new ones, discard BACK ones, exclude CLS, SEP
                    all_outputs.append(cur_features[cur_sub_idx-cur_slice_start+1:-1])
                    zwarn(f"Add multiple-seg range: [{cur_slice_start}, {cur_sub_idx}, {cur_slice_end})] " 
                          f"for all-len={len(cur_ids)}")
                cur_sub_idx = cur_slice_end
            final_outputs = BK.concat(all_outputs, 0)  # [La, *]
            ret = final_outputs[BK.input_idx(cur_starts[:-1])==1]  # todo(note) SEP is surely not selected
        return BK.get_value(ret)  # directly return one arr arr(real-len, DIM*layer)

    def forward_batch(self, batched_ids: List[List], batched_starts: List[List]):
        PAD_IDX = self.tokenizer.pad_token_id
        output_layers = self.bconf.bert_layers
        #
        bsize = len(batched_ids)
        max_len = max(len(z) for z in batched_ids)
        input_shape = (bsize, max_len)
        # first collect on CPU
        input_ids_arr = np.full(input_shape, PAD_IDX, dtype=np.int64)
        input_mask_arr = np.full(input_shape, 0, dtype=np.float32)
        input_is_start = np.full(input_shape, 0, dtype=np.int64)
        for bidx in range(bsize):
            cur_ids, cur_starts = batched_ids[bidx], batched_starts[bidx]
            cur_len = len(cur_ids)
            input_ids_arr[bidx, :cur_len] = cur_ids
            input_is_start[bidx, :cur_len] = cur_starts
            input_mask_arr[bidx, :cur_len] = 1.
        # then real forward
        # todo(note): for batched mode, assume things are all within bert's max-len
        features = self.forward_features(BK.input_idx(input_ids_arr), BK.input_real(input_mask_arr), output_layers)  # [bs, slen, *]
        start_idxes, start_masks = BK.mask2idx(BK.input_idx(input_is_start).float())  # [bsize, ?]
        start_expr = BK.gather_first_dims(features, start_idxes, 1)  # [bsize, ?, DIM*layer]
        # get values
        start_expr_arr = BK.get_value(start_expr)
        start_valid_len_arr = BK.get_value(start_masks.sum(-1).int())
        return [v[:slen] for v, slen in zip(start_expr_arr, start_valid_len_arr)]  # List[arr(real-len, DIM*layer)]

    def extend_subwords(self, sents, center_sid, budget, bert_sent_extend, bert_tok_extend, bert_extend_sent_bound):
        # todo(note): we prefer previous contexts
        budget_prev = max(0, budget * self.bconf.bert_previous_rate)
        budget_prev = min(bert_tok_extend, budget_prev)
        # collect prev ones
        included_prev_sents = []
        for step in range(bert_sent_extend):
            one_sid = center_sid-1-step
            if one_sid>=0:
                prev_subwords = sents[one_sid]
                if budget_prev >= len(prev_subwords):
                    included_prev_sents.append(prev_subwords)
                    budget_prev -= len(prev_subwords)  # decrease budget
                else:
                    if not bert_extend_sent_bound:  # we can further add sub-sent
                        included_prev_sents.append(prev_subwords[-budget_prev:])
                        budget_prev = 0
                    break  # budget run out
            else:
                break  # start of doc
        prev_all_subwords = []
        for one_prev_subwords in reversed(included_prev_sents):  # remember to reverse
            prev_all_subwords.extend(one_prev_subwords)
        # collect post ones
        post_all_subwords = []
        budget_post = max(0, budget - len(prev_all_subwords))  # remaining budget for post
        budget_post = min(bert_tok_extend, budget_post)
        for step in range(bert_sent_extend):
            one_sid = center_sid+step+1
            if one_sid < len(sents):
                post_subwords = sents[one_sid]
                if budget_post >= len(post_subwords):
                    post_all_subwords.extend(post_subwords)  # driectly extend
                    budget_post -= len(post_subwords)
                else:
                    if not bert_extend_sent_bound:  # we can further add sub-sent
                        post_all_subwords.extend(post_subwords[:budget_post])
                        budget_post = 0
                    break  # not enough budget
            else:
                break  # end of doc
        return prev_all_subwords, post_all_subwords

    # =====
    # using for word prediction

    # def predict_word_simple_mode(self, sents: List[Tuple], original_sent):
    #     # =====
    #     MAX_LEN = 510  # save two for [CLS] and [SEP]
    #     tokenizer = self.tokenizer
    #     CLS, CLS_IDX = tokenizer.cls_token, tokenizer.cls_token_id
    #     SEP, SEP_IDX = tokenizer.sep_token, tokenizer.sep_token_id
    #     # =====
    #     rets = []  # (list of combined-str, list of list of str)
    #     orig_subword_ids = [CLS_IDX] + original_sent.subword_ids + [SEP_IDX]
    #     for one_sent in sents:
    #         cur_subword_ids = one_sent.subword_ids
    #         cur_subword_is_start = one_sent.subword_is_start
    #         # calculate
    #         assert len(cur_subword_ids) <= MAX_LEN
    #         cur_all_ids = [CLS_IDX] + cur_subword_ids + [SEP_IDX]
    #         word_scores, = self.wp_model(BK.input_idx(cur_all_ids).unsqueeze(0))  # [1, slen+2, Vocab]
    #         word_scores2 = BK.log_softmax(word_scores, -1)
    #         max_scores, max_idxes = word_scores2[0].max(-1)
    #         # todo(note): length mismatch if in one mode!!
    #         orig_scores = word_scores2[0].gather(-1, BK.input_idx(orig_subword_ids).unsqueeze(-1)).squeeze(-1)
    #         # then ignore CLS and SEP and group by starts
    #         max_strs = [self.tok_word_list[z] for z in BK.get_value(max_idxes)[1:-1]]
    #         max_scores_arr = BK.get_value(max_scores)[1:-1]
    #         orig_scores_arr = BK.get_value(orig_scores)[1:-1]
    #         assert len(max_strs) == len(cur_subword_is_start)
    #         one_all_strs = []
    #         one_all_lists = []
    #         one_all_scores = []
    #         one_all_orig_scores = []
    #         for this_str, this_is_start, this_score, this_orig_score in \
    #                 zip(max_strs, cur_subword_is_start, max_scores_arr, orig_scores_arr):
    #             if this_is_start:
    #                 one_all_strs.append(this_str)
    #                 one_all_lists.append([this_str])
    #                 one_all_scores.append([this_score])
    #                 one_all_orig_scores.append([this_orig_score])
    #             else:
    #                 one_all_strs[-1] = one_all_strs[-1] + this_str.lstrip("##")
    #                 one_all_lists[-1].append(this_str)  # todo(note): error if starting with not is-start
    #                 one_all_scores[-1].append(this_score)
    #                 one_all_orig_scores[-1].append(this_orig_score)
    #         rets.append((one_all_strs, one_all_lists,
    #                      [np.average(z) for z in one_all_scores], [np.average(z) for z in one_all_orig_scores]))
    #     return rets

    # mask out each word and predict
    def predict_each_word(self, tokens: List[str]):
        # todo(+N): looking inside the transformers (v2.0.0) class, may break in the future version??
        #  and for prediction, we only put one MASK for predicting one subword
        tokenizer = self.tokenizer
        CLS, CLS_IDX = tokenizer.cls_token, tokenizer.cls_token_id
        SEP, SEP_IDX = tokenizer.sep_token, tokenizer.sep_token_id
        MASK, MASK_IDX = tokenizer.mask_token, tokenizer.mask_token_id
        # two forwards: get sum-log-probs and best one word replacement
        # first do tokenization and prepare inputs
        # - original subwords
        orig_subword_lists = []  # [orig_len]
        orig_word_start = []  # [orig_len]
        orig_word_sublen = []  # [orig_len]
        orig_subword_ids = [CLS_IDX]  # [1+extened_len+1]
        for i,t in enumerate(tokens):
            cur_toks = tokenizer.tokenize(t)
            # in some cases, there can be empty strings -> put the original word
            if len(cur_toks) == 0:
                cur_toks = [t]
            orig_subword_lists.append(cur_toks)
            cut_ids = tokenizer.convert_tokens_to_ids(cur_toks)
            orig_word_start.append(len(orig_subword_ids))
            orig_word_sublen.append(len(cut_ids))
            orig_subword_ids.extend(cut_ids)
        orig_subword_ids.append(SEP_IDX)
        # forw1 (LM like)
        forw1_hiddens = []  # [extend_len]
        for one_start, one_sublen in zip(orig_word_start, orig_word_sublen):
            for j in range(one_sublen):
                cur_inputs = orig_subword_ids[:one_start+j] + [MASK_IDX] * (one_sublen-j) + orig_subword_ids[one_start+one_sublen:]
                # todo(note): not API
                cur_hidden = self.wp_model.bert(BK.input_idx(cur_inputs).unsqueeze(0))[0][0][one_start+j]  # [D]
                forw1_hiddens.append(cur_hidden)
        # todo(note): not API
        forw1_logits = self.wp_model.cls(BK.stack(forw1_hiddens, 0))  # [ELEN, V]
        forw1_all_scores = BK.log_softmax(forw1_logits, -1)
        forw1_ext_scores = forw1_all_scores.gather(-1, BK.input_idx(orig_subword_ids[1:-1]).unsqueeze(-1)).squeeze(-1)  # [ELEN]
        forw1_ext_scores_arr = BK.get_value(forw1_ext_scores)  # [ELEN]
        # todo(note): sum of logprobs, remember to minus one to exclude CLS
        forw1_scores_arr = np.asarray([np.sum(forw1_ext_scores_arr[one_start-1:one_start-1+one_sublen])
                                       for one_start, one_sublen in zip(orig_word_start, orig_word_sublen)])  # [OLEN]
        # forw2 (prediction with one word)
        forw2_hiddens = []  # [orig_len]
        for one_start, one_sublen in zip(orig_word_start, orig_word_sublen):
            cur_inputs = orig_subword_ids[:one_start] + [MASK_IDX] + orig_subword_ids[one_start+one_sublen:]
            # todo(note): not API
            cur_hidden = self.wp_model.bert(BK.input_idx(cur_inputs).unsqueeze(0))[0][0][one_start]  # [D]
            forw2_hiddens.append(cur_hidden)
        # todo(note): not API
        forw2_logits = self.wp_model.cls(BK.stack(forw2_hiddens, 0))  # [OLEN, V]
        forw2_all_scores = BK.log_softmax(forw2_logits, -1)
        forw2_max_scores, forw2_max_idxes = forw2_all_scores.max(-1)
        forw2_max_scores_arr, forw2_max_idxes_arr = BK.get_value(forw2_max_scores), BK.get_value(forw2_max_idxes)  # [OLEN]
        forw2_max_strs = [self.tok_word_list[z] for z in forw2_max_idxes_arr]
        return forw1_scores_arr, forw2_max_scores_arr, forw2_max_strs, orig_subword_lists

    # =====
    # usable helper functions
    @staticmethod
    def _subword_tokenize(bert_tokenizer, tokens: List[str], mask_idx=-1, mask_mode="all", mask_repl="", typeids: List[int]=None):
        subword_toks = []  # all sub-tokens, no CLS or SEP
        subword_is_start = []  # whether is the start of orig-token positions
        subword_ids = []
        subword_typeids = None if typeids is None else []
        for i, t in enumerate(tokens):
            cur_toks = bert_tokenizer.tokenize(t)
            # in some cases, there can be empty strings -> put the original word
            if len(cur_toks) == 0:
                cur_toks = [t]
            # =====
            # mask mode
            if i == mask_idx:
                mask_tok = mask_repl if mask_repl else bert_tokenizer.mask_token
                if mask_mode == "all":
                    cur_toks = [mask_tok] * (len(cur_toks))
                elif mask_mode == "first":
                    cur_toks[0] = mask_tok
                elif mask_mode == "one":
                    cur_toks = [mask_tok]
                elif mask_mode == "pass":
                    continue  # todo(note): special mode, ignore current tok, but need later post-processing
                else:
                    raise NotImplementedError("UNK mask mode")
            # =====
            # todo(warn): use the first BPE piece's vector for the whole word
            cur_is_start = [0]*(len(cur_toks))
            cur_is_start[0] = 1
            subword_is_start.extend(cur_is_start)
            subword_toks.extend(cur_toks)
            subword_ids.extend(bert_tokenizer.convert_tokens_to_ids(cur_toks))
            if typeids is not None:
                subword_typeids.extend([typeids[i] for _ in cur_toks])
        assert len(subword_toks) == len(subword_is_start)
        assert len(subword_toks) == len(subword_ids)
        return SubwordToks(subword_toks, subword_ids, subword_is_start, subword_typeids)
