#

# the featurer extracts the features and calculate the repr distances (influences scores)

from typing import List, Dict
import numpy as np
import os

from msp.utils import StatRecorder, Helper, zlog, zopen, Conf
from msp.nn import BK
from msp.nn.modules.berter import BerterConf, Berter
from msp.data.vocab import Vocab
from tasks.zdpar.ef.parser import G1Parser
from tasks.zdpar.common.data import ParseInstance

#
class FeaturerConf(Conf):
    def __init__(self):
        # todo(+N): later we can use other encoders, but currently only with bert
        self.use_bert = True
        self.bconf = BerterConf().init_from_kwargs(bert_sent_extend=0, bert_root_mode=-1, bert_zero_pademb=True,
                                                   bert_layers="[0,1,2,3,4,5,6,7,8,9,10,11,12]")
        # for bert
        self.b_mask_mode = "first"  # all/first/one/pass
        self.b_mask_repl = "[MASK]"  # or other things like [UNK], [PAD], ...
        # for g1p
        self.g1p_replace_unk = True  # otherwise replace with 0
        # distance (influence scores)
        self.dist_f = "cos"  # cos or mse
        # for bert repl
        self.br_vocab_file = ""
        self.br_rank_thresh = 100  # rank threshold >=this for non-topical words

    def do_validate(self):
        assert self.use_bert, "currently only implemented the bert mode"

#
class Featurer:
    def __init__(self, conf: FeaturerConf):
        self.conf = conf
        self.berter = Berter(None, conf.bconf)
        self.fold = len(conf.bconf.bert_layers)
        # used for bert-repl: self.repl_sent()
        self.br_vocab = Vocab.read(conf.br_vocab_file) if conf.br_vocab_file else None
        self.br_rthresh = conf.br_rank_thresh

    # replace (simplify) sentence by replacing with bert predictions
    # todo(note): remember that the main purpose is to remove surprising topical words but retain syntax structure
    # todo(+N): the process might be specific to en or alphabet based languages
    def repl_sent(self, sent: List[str], fixed_arr):
        sent_len = len(sent)
        orig_scores_arr, pred_scores_arr, pred_strs, orig_subword_lists = self.berter.predict_each_word(sent)
        br_vocab, br_rthresh = self.br_vocab, self.br_rthresh
        # determined method
        # 1. first extend fixed (input+unchanged) ones
        assert len(pred_strs) == sent_len
        is_fixed = [(fixed_arr[i] or sent[i]==pred_strs[i]) for i in range(sent_len)]
        # 2. check for topical words (and change it if possible)
        # 2.1 collect all input words (all, not-freq, not-subword)
        input_words_set = set()
        for sub_words in orig_subword_lists:
            w = sub_words[0]  # todo(note): only take the first one
            if br_vocab.get(w, br_rthresh) >= br_rthresh:
                input_words_set.add(w)
        # 2.2 collect all output words (changed-only, not-freq, not-subword)
        # topical words are those both in output and input
        topical_words_count = {}
        for one_pred, one_is_fix in zip(pred_strs, is_fixed):
            # todo(note): not fix and not subword and not-freq
            if not one_is_fix and not one_pred.startswith("##") and br_vocab.get(one_pred, br_rthresh)>=br_rthresh:
                topical_words_count[one_pred] = topical_words_count.get(one_pred, 0) + 1
        # only retain those >1 or hit input
        topical_words_count = {k:v for k,v in topical_words_count.items() if (v>1 or k in input_words_set)}
        # 2.3 adjust scores: 1) encourage changing input topical words, 2) disencourage changing into topical words
        # todo(+N): what strategy for the scores? currently mainly NEG orig-logprob
        changing_scores = -1 * orig_scores_arr
        for one_idx in range(sent_len):
            one_orig, one_pred, one_is_fix = sent[one_idx], pred_strs[one_idx], is_fixed[one_idx]
            if one_is_fix or (one_pred in topical_words_count) or (one_pred.startswith("##")):
                changing_scores[one_idx] = 0  # not changing fix words or into topical words or suffix
            elif one_orig in topical_words_count:
                # todo(+2): magical number 100, this should be enough
                changing_scores[one_idx] += 100*(topical_words_count[one_orig])  # encourage changing from topical words
        # in pdb: pp list(zip(is_fixed,sent,pred_strs,changing_scores))
        # 3. change one for each segment
        prev_cands = []  # (idx, score)
        ret_seq = sent.copy()
        for one_idx in range(sent_len+1):
            if one_idx>=sent_len or is_fixed[one_idx]:  # hit boundary
                if len(prev_cands)>0:
                    prev_cands.sort(key=lambda x: -x[-1])
                    togo_idx, togo_score = prev_cands[0]
                    if togo_score > 0:  # only change it if not forbidden
                        ret_seq[togo_idx] = pred_strs[togo_idx]
                    prev_cands.clear()
            else:
                prev_cands.append((one_idx, changing_scores[one_idx]))
        # todo(+N): whether fix changed word? currently nope
        return ret_seq, np.asarray(is_fixed).astype(np.bool)

    # get influence scores
    def get_scores(self, sent: List[str]):
        # get features
        conf = self.conf
        features = encode_bert(self.berter, sent, conf.b_mask_mode, conf.b_mask_repl)  # [1+slen, 1+slen, fold*D]
        orig_feature_shape = BK.get_shape(features)
        folded_shape = orig_feature_shape[:-1] + [self.fold, orig_feature_shape[-1]//self.fold]
        all_features = features.view(folded_shape)  # [1+slen, 1+slen, fold, D]
        # get scores: [slen(without-which), slen+1(for-all-tokens), fold]
        if conf.dist_f == "cos":
            # normalize and dot
            all_features = all_features / (BK.sqrt((all_features ** 2).sum(-1, keepdim=True)) + 1e-7)
            scores = 1. - (all_features[1:] * all_features[0].unsqueeze(0)).sum(-1)
        elif conf.dist_f == "mse":
            scores = BK.sqrt(((all_features[1:] - all_features[0].unsqueeze(0)) ** 2).mean(-1))
        else:
            raise NotImplementedError()
        return scores.cpu().numpy()

    def output_shape(self, slen):
        return [slen, slen+1, len(self.conf.bconf.bert_layers)]

# =====
# encoding with different models, no root in input sent

#
def encode_bert(berter: Berter, sent: List[str], b_mask_mode, b_mask_repl):
    assert berter.bconf.bert_sent_extend == 0
    assert berter.bconf.bert_root_mode == -1
    # prepare inputs
    all_sentences = [berter.subword_tokenize(sent, True)]
    for i in range(len(sent)):
        all_sentences.append(berter.subword_tokenize(sent.copy(), True, mask_idx=i, mask_mode=b_mask_mode, mask_repl=b_mask_repl))
        # all_sentences.append(berter.subword_tokenize(sent.copy(), True, mask_idx=-1, mask_mode=b_mask_mode, mask_repl=b_mask_repl))
    # get outputs
    all_features = berter.extract_features(all_sentences)  # List[arr[1+slen, D]]
    # all_features = berter.extract_feature_simple_mode(all_sentences)  # List[arr[1+slen, D]]
    # =====
    # post-processing for pass mode
    if b_mask_mode=="pass":
        for i in range(1, len(all_features)):
            all_features[i] = np.insert(all_features[i], i, 0., axis=0)  # [slen, D] -> [slen+1, D]
    return BK.input_real(np.stack(all_features, 0))  # [1(whole)+slen, 1(R)+slen, D]

#
def encode_model(model: G1Parser, sent: List[str], g1p_replace_unk):
    word_vocab = model.vpack.get_voc("word")
    REPL = word_vocab.unk if g1p_replace_unk else 0
    ROOT = word_vocab[ParseInstance.ROOT_SYMBOL]  # must be there
    sent_idxes = [ROOT] + [word_vocab.get_else_unk(z) for z in sent]  # 1(R)+slen
    # prepare inputs
    all_sentences = [sent_idxes]
    for i in range(len(sent)):
        new_one = sent_idxes.copy()
        new_one[i+1] = REPL  # here 1 as offset for ROOT
        all_sentences.append(new_one)
    # get embeddings and encodings
    word_arr = np.asarray(all_sentences)  # [1+slen, 1+slen], no need to pad
    emb_repr = model.bter.emb(word_arr=word_arr)  # no other features, [1+slen, 1+slen, D0]
    enc_repr = model.bter.enc(emb_repr)  # no needfor mask, [1+slen, 1+slen, D1]
    return enc_repr  # [1(whole)+slen, 1(R)+slen, D]

# b tasks/zdpar/stat2/helper_feature:99
