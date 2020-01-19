#

# the counting based model

import numpy as np
from typing import List
from collections import Counter
from msp.utils import zlog, Conf

from .svocab import StatVocab

# todo(+N): can be cythonized if need to speed up

# how to use the model to get pairwise scores
class StatApplyConf(Conf):
    def __init__(self):
        self.default_score = 0.  # score for non-considered ones
        # div0 = log?(COUNT + add) ** alpha
        self.feat_log = True
        self.feat_add = 1.
        self.feat_alpha = 1.
        # div1 = COUNT ** beta
        self.word_beta = 0.  # count_word ** beta
        # decay by distance
        self.dist_decay_exp = True  # True: v ** distance, False: distance ** v
        self.dist_decay_v = 0.95
        # norm scores?
        self.final_norm = False
        self.final_norm_exp = True  # True: v ** score, False: score ** v
        self.final_norm_v = np.e
        # final scale of lambda
        self.final_lambda = 1.

class StatModel:
    def __init__(self, conf, vocab: StatVocab):
        self.conf = conf
        self.vocab = vocab
        self.lc = conf.lc
        # =====
        # building distance bins (win_size is included: [1, win_size])
        win_size = conf.win_size
        win_size_p1 = win_size+1
        bins = conf.bins  # list of <= bin separators, for example: [1, 2, 4, 8, 16]
        assert bins[-1] >= win_size
        self.num_bin = len(bins) * 2  # directional
        self.dist2bin = [-1] * (2*win_size_p1)  # [0, win_size_p1) + [win_size_p1, 2*win_size_p1)
        prev_i = -1
        for bin_idx, bin_ceil in enumerate(bins):
            for x in range(prev_i+1, bin_ceil+1):
                self.dist2bin[x] = bin_idx
                self.dist2bin[x+win_size_p1] = bin_idx+len(bins)
            prev_i = bin_ceil
        assert all(z>=0 for z in self.dist2bin)
        dist2bin_str = ", ".join([f"{s}:{v}" for s,v in enumerate(self.dist2bin)])
        zlog(f"Build dist-bins with win_size={win_size}, bins=(#{len(bins)}){bins}, dist2bin={dist2bin_str}")
        # =====
        # prepare models for each word
        self.binary_nb = conf.binary_nb  # whether 0/1 for each feature and word as in binary nb
        self.num_words = len(vocab)
        self.word_counts = [0] * self.num_words
        self.feat_counts = [Counter() for i in range(self.num_words)]
        # feat storing: compact or not?
        self.feat_compact = conf.feat_compact

    def add_sent(self, tokens: List[str]):
        # =====
        # prepare
        conf = self.conf
        # nearby features
        fc_max = conf.fc_max  # near Center
        ft_max = conf.ft_max  # near conText
        lex_min = conf.lex_min
        lex_min2 = conf.lex_min2
        win_size = conf.win_size
        win_size_p1 = win_size+1
        meet_punct_max = conf.meet_punct_max
        meet_lex_thresh = conf.meet_lex_thresh
        meet_lex_freq_max = conf.meet_lex_freq_max
        vocab = self.vocab
        dist2bin = self.dist2bin
        neg_dist_adding = win_size_p1 if conf.neg_dist else 0
        # first change to idx, 0 means UNK; also collect related info
        cur_len = len(tokens)
        tok_idxes = [vocab.get(t, 0) for t in tokens]  # the token itself
        is_punct = [vocab.is_punct[i] for i in tok_idxes]  # whether is_punct
        # =====
        # start collecting: center <fc> ... -> ... <ft> context // context <ft> ... <- ... <fc> center
        collections = [[] for _ in range(cur_len)]
        for posi, tok in enumerate(tok_idxes):
            if tok >= lex_min and not is_punct[posi]:  # only consider lexicons that are not the most freq (and also exclude unk=-1)
                # the meet tokens in between
                cur_meet_punct = 0
                cur_meet_lex_min = meet_lex_thresh+1  # idx of the most freq word met
                cur_meet_freq_num = 0  # number of how many freq words met
                for posi2 in range(posi+1, min(cur_len, posi+win_size_p1)):
                    cur_meet_punct = min(cur_meet_punct + is_punct[posi2], meet_punct_max)
                    tok2 = tok_idxes[posi2]
                    if tok2 >= lex_min2 and not is_punct[posi2]:  # similarly for context token
                        cur_distance = posi2 - posi
                        cur_fleft, cur_fright = tok_idxes[posi+1], tok_idxes[posi2-1]
                        # add features: (context-tok, meet-punct, fc-tok, ft-tok, dist-bin)
                        # tok -> tok2
                        feat1 = (tok2, cur_meet_punct, min(fc_max, cur_fleft), min(ft_max, cur_fright),
                                 dist2bin[cur_distance], cur_meet_lex_min, cur_meet_freq_num)
                        feat1 = self._compact(feat1)
                        collections[posi].append(feat1)
                        # tok <- tok2
                        feat2 = (tok, cur_meet_punct, min(fc_max, cur_fright), min(ft_max, cur_fleft),
                                 dist2bin[neg_dist_adding + cur_distance], cur_meet_lex_min, cur_meet_freq_num)
                        feat2 = self._compact(feat2)
                        collections[posi2].append(feat2)
                    if tok2 > 0 and tok2 <= meet_lex_thresh:  # not UNK
                        cur_meet_lex_min = min(cur_meet_lex_min, tok2)
                        cur_meet_freq_num = min(cur_meet_freq_num+1, meet_lex_freq_max)
        # add to the repo as specific model
        for one_tok, one_collections in zip(tok_idxes, collections):
            if one_tok >= lex_min:
                if self.binary_nb:
                    wc, vs = 1, set(one_collections)  # in binary nb, only count 0/1 for each feat
                else:
                    wc, vs = len(one_collections), one_collections
                self.word_counts[one_tok] += wc
                d = self.feat_counts[one_tok]
                for z in vs:
                    d[z] += 1

    # todo(note): the input may contain a artificial root at the start, this does not matter
    #  since it will be UNK and we use relative distance
    def apply_sent(self, tokens: List[str], aconf: StatApplyConf):
        # =====
        # prepare
        conf = self.conf
        # nearby features
        fc_max = conf.fc_max  # near Center
        ft_max = conf.ft_max  # near conText
        lex_min = conf.lex_min
        lex_min2 = conf.lex_min2
        win_size = conf.win_size
        win_size_p1 = win_size + 1
        meet_punct_max = conf.meet_punct_max
        meet_lex_thresh = conf.meet_lex_thresh
        meet_lex_freq_max = conf.meet_lex_freq_max
        vocab = self.vocab
        dist2bin = self.dist2bin
        neg_dist_adding = win_size_p1 if conf.neg_dist else 0
        # first change to idx, 0 means UNK; also collect related info
        cur_len = len(tokens)
        tok_idxes = [vocab.get(t, 0) for t in tokens]  # the token itself
        is_punct = [vocab.is_punct[i] for i in tok_idxes]  # whether is_punct
        # =====
        # how to score
        scores = np.zeros([cur_len, cur_len], dtype=np.float32)
        for posi, tok in enumerate(tok_idxes):
            if tok>=lex_min and not is_punct[posi]:  # only consider lexicons that are not the most freq (and also exclude unk=-1)
                cur_meet_punct = 0
                cur_meet_lex_min = meet_lex_thresh+1  # idx of the most freq word met
                cur_meet_freq_num = 0  # number of how many freq words met
                for posi2 in range(posi+1, min(cur_len, posi+win_size_p1)):
                    cur_meet_punct = min(cur_meet_punct + is_punct[posi2], meet_punct_max)
                    tok2 = tok_idxes[posi2]
                    if tok2>=lex_min2 and not is_punct[posi2]:  # similarly for context token
                        cur_distance = posi2 - posi
                        cur_fleft, cur_fright = tok_idxes[posi+1], tok_idxes[posi2-1]
                        # add features: (context-tok, meet-punct, fc-tok, ft-tok, dist-bin, lex-min, lex-freq)
                        # tok -> tok2
                        feat1 = (tok2, cur_meet_punct, min(fc_max, cur_fleft), min(ft_max, cur_fright),
                                 dist2bin[cur_distance], cur_meet_lex_min, cur_meet_freq_num)
                        feat1 = self._compact(feat1)
                        score1 = self._score(tok, feat1, cur_distance, aconf)
                        # tok <- tok2
                        feat2 = (tok, cur_meet_punct, min(fc_max, cur_fright), min(ft_max, cur_fleft),
                                 dist2bin[neg_dist_adding+cur_distance], cur_meet_lex_min, cur_meet_freq_num)
                        feat2 = self._compact(feat2)
                        score2 = self._score(tok2, feat2, cur_distance, aconf)  # still using orig distance
                        # average and symmetric for the link score
                        final_score = (score1 + score2)/2.
                        scores[posi, posi2] = final_score
                        scores[posi2, posi] = final_score
                    if tok2 > 0 and tok2 <= meet_lex_thresh:  # not UNK
                        cur_meet_lex_min = min(cur_meet_lex_min, tok2)
                        cur_meet_freq_num = min(cur_meet_freq_num+1, meet_lex_freq_max)
        #
        if aconf.final_norm:
            if aconf.final_norm_exp:
                scores2 = aconf.final_norm_v ** scores
                scores2 = np.where(scores>0., scores2, 0.)  # 0. is still zero
            else:
                scores2 = scores ** aconf.final_norm_v
            # normalize and again average symmetrically
            scores = scores2 / (scores2.sum(-1, keepdims=True)+1e-5) + scores2 / (scores2.sum(0, keepdims=True)+1e-5)
            scores /= 2
        return scores * aconf.final_lambda

    #
    def _score(self, tok, feat, distance, aconf: StatApplyConf):
        count_feat = self.feat_counts[tok][feat] + aconf.feat_add
        if aconf.feat_log:
            div0 = aconf.feat_alpha * np.log(count_feat)
        else:
            div0 = count_feat ** aconf.feat_alpha
        if aconf.word_beta > 0.:
            count_word = self.word_counts[tok]
            div1 = count_word ** aconf.word_beta
        else:
            div1 = 1.
        if aconf.dist_decay_exp:
            dd = aconf.dist_decay_v ** distance
        else:
            dd = distance ** aconf.dist_decay_v
        if div1 == 0.:
            return 0.
        return dd * div0 / div1

    # storing feat compactly or not
    # currently: (context-tok, meet-punct, fc-tok, ft-tok, dist-bin, meet-lex-thresh, meet-lex-freq)
    def _compact(self, feat_tuple):
        if self.feat_compact:
            r = 0
            for x, bits in zip(feat_tuple, (18, 2, 10, 10, 5, 10, 3)):  # make the sum <64
            # for x, bits in zip(feat_tuple, (18, 2, 0, 0, 5, 10, 3)):  # not using fc/ft
                r = (r<<bits) + x
            return r
        else:
            return feat_tuple
