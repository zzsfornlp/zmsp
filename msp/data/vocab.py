#

from typing import Dict, Sequence

from msp.utils import zopen, zlog, zwarn, zcheck, StrHelper, FileHelper, Helper, JsonRW, PickleRW, printing, Random
from collections import Iterable, defaultdict
import numpy as np
import re

# for binary w2v loading
from gensim.models import KeyedVectors

#

class Vocab(object):
    def __init__(self, name=""):
        self.name = name
        #
        self.pre_list = []
        self.post_list = []
        #
        self.v = None                   # word -> idx (including specials, plains are sorted by vals)
        self.final_words = None         # idx -> words
        self.final_vals = None          # idx -> count (sorted highest->lowest if constructed in such way)
        self.rank_ranges_ = None        # idx -> (highest-rank, lowest-rank)
        #
        self.values = {}                # stored by outside, stats
        self.target_range_ = None       # when used as target vocab, [start, end) for predicting
        self.target_length_ = -1        # end-start

    # -------------
    # read & write
    @staticmethod
    def read(fname):
        one = Vocab()
        JsonRW.from_file(one, fname)
        zlog("-- Read Dictionary from %s: Finish %s." % (fname, len(one)), func="io")
        return one

    def write(self, fname):
        JsonRW.to_file(self, fname)
        zlog("-- Write Dictionary to %s: Finish %s." % (fname, len(self)), func="io")

    # -------------
    # queries
    def __str__(self):
        return "Dictionary %s: (final=%s, (pre + post)=%s as [%s, %s].)." \
               % (self.name, len(self.v), len(self.pre_list)+len(self.post_list), self.pre_list, self.post_list)

    def __len__(self):
        return len(self.v)

    def has_key(self, k):
        return k in self.v

    def keys(self):
        return self.final_words

    def trg_keys(self):
        start, end = self.target_range_
        return self.final_words[start:end]

    def set_value(self, k, v):
        self.values[k] = v

    def get_value(self, k):
        return self.values[k]

    # [len(pre), len(self)-len(post)
    def nonspecial_idx_range(self):
        return (len(self.pre_list), len(self)-len(self.post_list))

    # key -> idx
    def __getitem__(self, item):
        # zcheck(item in self.v, "Unknown key %s." % item)
        return self.v[item]

    def get(self, item, df=None):
        if item in self.v:
            return self.v[item]
        else:
            return df

    # todo(warn): default action is error if no unk node
    def get_else_unk(self, item):
        return self.get(item, self.unk)

    # key -> value
    def getval(self, item, df=None):
        if item in self.v:
            idx = self[item]
            return self.final_vals[idx]
        else:
            return df

    # idx -> key
    def idx2word(self, idx):
        # zcheck(0<=idx and idx<=len(self), "Out-of-range idx %d for Vocab, max %d." % (idx, len(self)))
        return self.final_words[idx]

    # idx -> value
    def idx2val(self, idx):
        return self.final_vals[idx]

    # =====
    # ranges if used as target vocab
    def trg_len(self, idx_orig):
        # if idx_orig, return more len including the start ranges to make the idxes the same
        if idx_orig:
            return self.target_length_ + self.target_range_[0]
        else:
            return self.target_length_

    # target idx to real idx
    def trg_pred2real(self, idx):
        return idx + self.target_range_[0]

    # read idx to target idx
    def trg_real2pred(self, idx):
        return idx - self.target_range_[0]

    # =====
    # words <=> indexes (be aware of lists)
    @staticmethod
    def w2i(dicts, ss, use_unk=False, add_eos=False, use_factor=False, factor_split='|'):
        # Usage: list(Vocab), list(str) => list(list(int))[use_factor] / list(int)[else]
        def _lookup(v, dict):
            if use_unk:
                return dict.get_else_unk(v)
            else:
                return dict[v]
        #
        if not isinstance(dicts, Iterable):
            dicts = [dicts]
        # lookup
        tmp = []
        for w in ss:
            if use_factor:
                idx = [_lookup(f, dicts[i]) for (i,f) in enumerate(w.split(factor_split))]
            else:
                idx = _lookup(dicts[0], w)
            tmp.append(idx)
        if add_eos:
            tmp.append([d.eos for d in dicts] if use_factor else dicts[0].eos)  # add eos
        return tmp

    @staticmethod
    def i2w(dicts, ii, rm_eos=True, factor_split='|'):
        # Usage: list(Vocab), list(int)/list(list(int)) => list(str)
        if not isinstance(dicts, Iterable):
            dicts = [dicts]
        tmp = []
        # get real list
        real_ii = ii
        if len(ii)>0 and rm_eos and ii[-1]==dicts[0].eos:
            real_ii = ii[:-1]
        # transform each token
        for one in real_ii:
            if not isinstance(one, Iterable):
                one = [one]
            zcheck(len(one) == len(dicts), "Unequal factors vs. dictionaries.")
            tmp.append(factor_split.join([v.idx2word(idx) for v, idx in zip(dicts, one)]))
        return tmp

    # filter inits for embeddings
    def filter_embed(self, wv: 'WordVectors', init_nohit=0., scale=1.0, assert_all_hit=False):
        if init_nohit <= 0.:
            get_nohit = lambda s: np.zeros((s,), dtype=np.float32)
        else:
            get_nohit = lambda s: (Random.random_sample((s,)).astype(np.float32)-0.5) * (2*init_nohit)
        #
        ret = []
        res = defaultdict(int)
        for w in self.final_words:
            hit, norm_name, norm_w = wv.norm_until_hit(w)
            if hit:
                value = np.asarray(wv.get_vec(norm_w, norm=False), dtype=np.float32)
                res[norm_name] += 1
            else:
                value = get_nohit(wv.embed_size)
                # value = np.zeros((wv.embed_size,), dtype=np.float32)
                res["no-hit"] += 1
            ret.append(value)
        #
        if assert_all_hit:
            zcheck(res["no-hit"]==0, f"Filter-embed error: assert all-hit but get no-hit of {res['no-hit']}")
        printing("Filter pre-trained embed: %s, no-hit is inited with %s." % (res, init_nohit))
        return np.asarray(ret, dtype=np.float32) * scale

#
class VocabHelper:
    # todo(0): I guess this will make them unique
    SPECIAL_PATTERN = re.compile(r"\<z_([a-zA-Z]{3})_z\>")

    @staticmethod
    def extract_name(w):
        zmatch = re.fullmatch(VocabHelper.SPECIAL_PATTERN, w)
        if zmatch:
            return zmatch.group(1)
        else:
            return None

    @staticmethod
    def convert_special_pattern(w):
        return "<z_"+w+"_z>"

# Build with counting!
class VocabBuilder(object):
    DEFAULT_PRE_LIST, DEFAULT_POST_LIST = tuple([VocabHelper.convert_special_pattern(w) for w in ["non"]]), \
                                          tuple([VocabHelper.convert_special_pattern(w) for w in ["unk", "eos", "bos", "pad", "err"]])
    DEFAULT_TARGET_RANGE = (1, None)         # only these can be predicted [a, b), default is all real words

    def __init__(self, vname="", pre_list=DEFAULT_PRE_LIST, post_list=DEFAULT_POST_LIST, default_val=None):
        v = Vocab(name=vname)
        v.pre_list = pre_list
        v.post_list = post_list
        #
        self.v = v
        self.keys_ = []     # by the order of feeding
        self.counts_ = {}
        self.default_val_ = default_val

    @staticmethod
    def _build_check(v):
        # check specials are included
        zcheck(lambda: all(x in v.v for x in v.pre_list), "Not including pre_specials")
        zcheck(lambda: all(x in v.v for x in v.post_list), "Not including post_specials")
        # check special tokens
        zcheck(lambda: all(v.v[x]<len(v.pre_list) for x in v.pre_list), "Get unexpected pre_special words in plain words!!")
        zcheck(lambda: all(v.v[x]>=len(v.v)-len(v.post_list) for x in v.post_list), "Get unexpected post_special words in plain words!!")

    @staticmethod
    def _build_prop(v):
        # build public properties
        for one_list in (v.pre_list, v.post_list):
            for name in one_list:
                zname = VocabHelper.extract_name(name)
                if zname is not None:
                    v.__dict__[zname] = v.v[name]

    @staticmethod
    def _tmp_target_idx(v, x, ndf):
        if x in v.v:
            return v.v[x]
        if x is None:
            return ndf
        else:
            return int(x)

    @staticmethod
    def _build_target_range(v, a, b):
        def _target_idx(vv, x, df):
            if x is None: return df
            else: return vv.v[x] if isinstance(x, str) else int(x)
        # todo(warn): by default, 0 means <NON>
        ia, ib = _target_idx(v, a, 1), _target_idx(v, b, len(v))
        v.target_range_ = (ia, ib)
        # zcheck(v.target_range_[0]>0, "Never use idx=0 which means 0 padding!")
        v.target_length_ = v.target_range_[1] - v.target_range_[0]

    # ================
    def has_key_currently(self, k):
        return k in self.counts_

    def feed_one(self, w, c=1):
        word_freqs = self.counts_
        if w not in word_freqs:
            self.keys_.append(w)
            word_freqs[w] = c
        else:
            word_freqs[w] += c

    def feed_stream(self, stream):
        for w in stream:
            self.feed_one(w)

    # build once from stream (maybe Iterable)
    @staticmethod
    def build_from_stream(stream, sort_by_count=False, name="Anon", pre_list=DEFAULT_PRE_LIST, post_list=DEFAULT_POST_LIST):
        builder = VocabBuilder(name, pre_list, post_list)
        builder.feed_stream(stream=stream)
        v = builder.finish(sort_by_count=sort_by_count)
        return v

    # merge multiple vocabs
    @staticmethod
    def merge_vocabs(vocabs: Sequence[Vocab], sort_by_count=False, name="Merge", pre_list=DEFAULT_PRE_LIST, post_list=DEFAULT_POST_LIST):
        builder = VocabBuilder(name, pre_list, post_list)
        for one_vocab in vocabs:
            # todo(warn): add counting
            ns_start, ns_end = one_vocab.nonspecial_idx_range()
            for idx in range(ns_start, ns_end):
                word = one_vocab.idx2word(idx)
                count = one_vocab.idx2val(idx)
                builder.feed_one(word, count)
        v = builder.finish(sort_by_count=sort_by_count)
        return v

    # =====
    # {word->vals} => new {word->vals}
    @staticmethod
    def filter_vals(word_vals, word_filter=(lambda ww, rank, val: True)):
        ranked_list = Helper.rank_key(word_vals)
        truncated_vals = {}
        for ii, ww in enumerate(ranked_list):
            rank, val = ii+1, word_vals[ww]
            if word_filter(ww, rank, val):
                truncated_vals[ww] = val
        return truncated_vals

    # {word->vals} => {word->idx}, [filtered values]
    @staticmethod
    def ranking_vals(word_vals, pre_list, post_list, default_val, word_filter=(lambda ww, rank, val: True)):
        ranked_list = Helper.rank_key(word_vals)
        #
        truncated_vals = [default_val] * len(pre_list)
        v = dict(zip(pre_list, range(len(pre_list))))
        for ii, ww in enumerate(ranked_list):
            rank, val = ii+1, word_vals[ww]
            if word_filter(ww, rank, val):
                v[ww] = len(v)
                truncated_vals.append(val)
        for one in post_list:
            v[one] = len(v)
            truncated_vals.append(default_val)
        return v, truncated_vals

    def filter(self, word_filter=(lambda ww, rank, val: True)):
        new_counts = VocabBuilder.filter_vals(self.counts_, word_filter)
        printing("Filter in VocabBuilder %s ok, from %d to %d." % (self.v.name, len(self.counts_), len(new_counts)))
        self.counts_ = new_counts

    def filter_thresh(self, rthres, fthres):
        def rf_filter(ww,rank,val): return val>=fthres and rank<=rthres
        self.filter(rf_filter)

    #
    def finish(self, word_filter=(lambda ww, rank, val: True), sort_by_count=True, target_range=DEFAULT_TARGET_RANGE):
        v = self.v
        # sort by count-value otherwise adding orders
        tmp_vals = self.counts_ if sort_by_count else {k:-i for i,k in enumerate(self.keys_) if k in self.counts_}
        v.v, v.final_vals = VocabBuilder.ranking_vals(tmp_vals, v.pre_list, v.post_list, self.default_val_, word_filter=word_filter)
        v.final_words = Helper.reverse_idx(v.v)
        printing("Build Vocab %s ok, from %d to %d, as %s." % (v.name, len(self.counts_), len(v), str(v)))
        #
        VocabBuilder._build_check(v)
        VocabBuilder._build_target_range(v, target_range[0], target_range[1])
        VocabBuilder._build_prop(v)
        return v

    def finish_thresh(self, rthres, fthres, sort_by_count=True, target_range=DEFAULT_TARGET_RANGE):
        def rf_filter(ww,rank,val): return val>=fthres and rank<=rthres
        return self.finish(rf_filter, sort_by_count, target_range)

# ===========
# todo(+1): should store {k: idx}?
class WordVectors(object):
    # ordering of special normers
    WORD_NORMERS = [
        ["orig", lambda w: w],
        # ["num", lambda w: StrHelper.norm_num(w)],
        ["lc", lambda w: str.lower(w)],
    ]

    def __init__(self, sep=" "):
        self.num_words = None
        self.embed_size = None
        self.words = []         # idx -> str
        self.wmap = {}          # str -> idx
        self.vecs = []          # idx -> vec
        self.sep = sep
        #
        self.hits = {}  # hit keys by any queries with __contains__: norm_until_hit, has_key, get_vec

    #
    def __contains__(self, item):
        if item in self.wmap:
            self.hits[item] = self.hits.get(item, 0) + 1
            return True
        else:
            return False

    # return (hit?, norm_name, normed_w)
    def norm_until_hit(self, w):
        orig_w = w
        for norm_name, norm_f in WordVectors.WORD_NORMERS:
            w = norm_f(w)
            if w in self:
                return True, norm_name, w
        return False, "", orig_w

    def has_key(self, k, norm=True):
        if k in self:
            return True
        elif norm:
            return self.norm_until_hit(k)[0]
        else:
            return False

    def get_vec(self, k, df=None, norm=True):
        if k in self:
            return self.vecs[self.wmap[k]]
        elif norm:
            hit, _, w = self.norm_until_hit(k)
            if hit:
                return self.vecs[self.wmap[w]]
        return df

    def save(self, fname):
        printing(f"Saving w2v num_words={self.num_words:d}, embed_size={self.embed_size:d} to {fname}.")
        zcheck(self.num_words == len(self.vecs), "Internal error: unmatched number!")
        with zopen(fname, "w") as fd:
            WordVectors.save_txt(fd, self.words, self.vecs, self.sep)

    def save_hits(self, fname):
        num_hits = len(self.hits)
        printing(f"Saving hit w2v num_words={num_hits:d}, embed_size={self.embed_size:d} to {fname}.")
        with zopen(fname, "w") as fd:
            tmp_words = sorted(self.hits.keys(), key=lambda k: self.wmap[k])  # use original ordering
            tmp_vecs = [self.vecs[self.wmap[k]] for k in tmp_words]
            WordVectors.save_txt(fd, tmp_words, tmp_vecs, self.sep)

    @staticmethod
    def save_txt(fd, words, vecs, sep):
        num_words = len(words)
        embed_size = len(vecs[0])
        zcheck(num_words == len(vecs), "Unmatched size!")
        fd.write(f"{num_words}{sep}{embed_size}\n")
        for w, vec in zip(words, vecs):
            zcheck(len(vec)==embed_size, "Unmatched dim!")
            print_list = [w] + ["%.6f" % float(z) for z in vec]
            fd.write(sep.join(print_list) + "\n")

    # =====
    # special methods for multi
    def aug_words(self, aug_code):
        self.words = [MultiHelper.aug_word_with_prefix(w, aug_code) for w in self.words]
        self.wmap = {w:i for i,w in enumerate(self.words)}

    def merge_others(self, others):
        embed_size = self.embed_size
        for other in others:
            zcheck(embed_size == other.embed_size, "Cannot merge two diff-sized embeddings!")
            this_all_num = other.num_words
            this_added_num = 0
            for one_w, one_vec in zip(other.words, other.vecs):
                # keep the old one!
                if one_w not in self.wmap:  # here, does not record as hits!
                    this_added_num += 1
                    self.wmap[one_w] = len(self.words)
                    self.words.append(one_w)
                    self.vecs.append(one_vec)
            zlog(f"Merge embed: add another with all={this_all_num}/add={this_added_num}")
        zlog(f"After merge, changed from {self.num_words} to {len(self.words)}")
        self.num_words = len(self.words)        # remember to change this one!
    # =====

    @staticmethod
    def load(fname, binary=False, txt_sep=" ", aug_code=""):
        if binary:
            vv = WordVectors._load_bin(fname)
        else:
            vv = WordVectors._load_txt(fname, txt_sep)
        if aug_code:
            vv.aug_words(aug_code)
        return vv

    @staticmethod
    def _load_txt(fname, sep=" "):
        printing("Going to load pre-trained (txt) w2v from %s ..." % fname)
        one = WordVectors(sep=sep)
        repeated_count = 0
        with zopen(fname) as fd:
            # first line
            line = fd.readline()
            try:
                one.num_words, one.embed_size = [int(x) for x in line.split(sep)]
                printing("Reading w2v num_words=%d, embed_size=%d." % (one.num_words, one.embed_size))
                line = fd.readline()
            except:
                printing("Reading w2v.")
            # the rest
            while len(line) > 0:
                line = line.rstrip()
                fields = line.split(sep)
                word, vec = fields[0], [float(x) for x in fields[1:]]
                # zcheck(word not in one.wmap, "Repeated key.")
                # keep the old one
                if word in one.wmap:
                    repeated_count += 1
                    zwarn(f"Repeat key {word}")
                    line = fd.readline()
                    continue
                #
                if one.embed_size is None:
                    one.embed_size = len(vec)
                else:
                    zcheck(len(vec) == one.embed_size, "Unmatched embed dimension.")
                one.vecs.append(vec)
                one.wmap[word] = len(one.words)
                one.words.append(word)
                line = fd.readline()
        # final
        if one.num_words is not None:
            zcheck(one.num_words == len(one.vecs)+repeated_count, "Unmatched num of words.")
        one.num_words = len(one.vecs)
        printing(f"Read ok: w2v num_words={one.num_words:d}, embed_size={one.embed_size:d}, repeat={repeated_count:d}")
        return one

    @staticmethod
    def _load_bin(fname):
        printing("Going to load pre-trained (binary) w2v from %s ..." % fname)
        one = WordVectors()
        #
        kv = KeyedVectors.load_word2vec_format(fname, binary=True)
        # KeyedVectors.save_word2vec_format()
        one.num_words, one.embed_size = len(kv.vectors), len(kv.vectors[0])
        for w, z in kv.vocab.items():
            one.vecs.append(kv.vectors[z.index])
            one.wmap[w] = len(one.words)
            one.words.append(w)
        printing("Read ok: w2v num_words=%d, embed_size=%d." % (one.num_words, one.embed_size))
        return one

#
class VocabPackage(object):
    def __init__(self, vocabs: Dict, embeds: Dict):
        # todo(warn): might need vocabs of word/char/pos/...
        # self.vocabs = {n:v for n,v in vocabs.items()}
        # self.embeds = {n:e for n,e in embeds.items()}
        self.vocabs = vocabs
        self.embeds = embeds

    def get_voc(self, name, df=None):
        return self.vocabs.get(name, df)

    def get_emb(self, name, df=None):
        return self.embeds.get(name, df)

    def put_voc(self, name, v):
        self.vocabs[name] = v

    def put_emb(self, name, e):
        self.embeds[name] = e

    def load(self, prefix="./"):
        for name in self.vocabs:
            fname = prefix+"vv_"+name+".txt"
            if FileHelper.exists(fname):
                self.vocabs[name] = Vocab.read(fname)
            else:
                zwarn("Cannot find Vocab " + name)
                self.vocabs[name] = None
        for name in self.embeds:
            fname = prefix+"ve_"+name+".pic"
            if FileHelper.exists(fname):
                self.embeds[name] = PickleRW.from_file(fname)
            else:
                self.embeds[name] = None

    def save(self, prefix="./"):
        for name, vv in self.vocabs.items():
            fname = prefix + "vv_" + name + ".txt"
            if vv is not None:
                vv.write(fname)
        for name, vv in self.embeds.items():
            fname = prefix+"ve_"+name+".pic"
            if vv is not None:
                PickleRW.to_file(vv, fname)


# helpers for handling multi-lingual data, simply adding lang-code prefix
class MultiHelper:
    #
    # todo(warn): change word to language specific ones
    # -- to reverse, simply ``w.split("_", 1)[1]''
    @staticmethod
    def aug_word_with_prefix(w, lang_code):
        if lang_code:
            return f"!{lang_code}_{w}"
        else:
            return w

    # return orig_w, lang_code
    # todo(warn): quite unlikely there are such normal words
    @staticmethod
    def strip_aug_prefix(w):
        if w.startswith("!"):
            fields = w[1:].split("_", 1)
            if len(fields) == 2:
                return fields[1], fields[0]
        # otherwise normal word
        return w, ""

    # ======
    # special method, combining embeddings and vocabs
    # todo(warn): slightly complex
    # keep main's pre&post, but put aug's true words before post and make corresponding changes to the arr
    # -> return (new_vocab, new_arr)
    @staticmethod
    def aug_vocab_and_arr(main_vocab, main_arr, aug_vocab, aug_arr, aug_override):
        # first merge the vocab
        new_vocab = VocabBuilder.merge_vocabs([main_vocab, aug_vocab], sort_by_count=False,
                                              pre_list=main_vocab.pre_list, post_list=main_vocab.post_list)
        # then find the arrays
        # todo(+1): which order to find words
        assert aug_override, "To be implemented for other ordering!"
        #
        new_arr = []
        main_hit = aug_hit = 0
        for idx in range(len(new_vocab)):
            word = new_vocab.idx2word(idx)
            # todo(warn): selecting the embeds in aug first (make it possible to override original ones!)
            aug_orig_idx = aug_vocab.get(word)
            if aug_orig_idx is None:
                main_orig_idx = main_vocab[word]      # must be there!
                new_arr.append(main_arr[main_orig_idx])
                main_hit += 1
            else:
                new_arr.append(aug_arr[aug_orig_idx])
                aug_hit += 1
        zlog(f"For the final merged arr, the composition is all={len(new_arr)},main={main_hit},aug={aug_hit}")
        ret_arr = np.asarray(new_arr)
        return new_vocab, ret_arr
