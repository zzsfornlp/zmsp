#

# Simple Vocab, flattened mapping between str <-> int

__all__ = [
    "VocabHelper", "Vocab", "WordVectors", "VocabPackage",
]

from collections import OrderedDict, defaultdict
from typing import Iterable, Dict, List, Callable, IO, Type
import re
import os
import numpy as np
from mspx.utils import zlog, zopen, zwarn, Random, Registrable, Serializable, default_json_serializer, default_pickle_serializer

# -----
class VocabHelper:
    # todo(note): these are reversed for other usage
    SPECIAL_PATTERN = re.compile(r"\<z_([a-zA-Z]+)_z\>")

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

    # ----
    # find word with several backoff strategies
    # ordering of special normers
    WORD_NORMERS = [
        ["orig", lambda w: w],
        # ["num", lambda w: StrHelper.norm_num(w)],
        ["lc", lambda w: str.lower(w)],
        ["cased", lambda w: str.upper(w[0])+str.lower(w[1:])]
    ]

    # return (hit?, norm_name, normed_w)
    @staticmethod
    def norm_until_hit(v, w: str):
        # orig_w = w
        for norm_name, norm_f in VocabHelper.WORD_NORMERS:
            w = norm_f(w)
            if w in v:
                return True, norm_name, w
        return False, None, None

# =====
# the simple Str<->Integer vocab: no relations among the vocab entries
@Registrable.rd('V')
class Vocab(Serializable):
    def __init__(self, name="anon"):
        self.name = name  # name for the vocab
        # todo(note): always keeps all these fields consistent: always keeping info of i2w, which has no repeats
        # fixed ones at the front and the end of the list
        self.pre_list = []
        self.post_list = []
        # these are real words
        self.w2i = {}  # word -> inner-idx (no specials)
        self.i2w = []  # inner-idx -> words
        self.counts = {}  # word -> counts
        # cache ones
        self._full_i2w = None  # outer-idx -> words

    @property
    def full_i2w(self):
        if self._full_i2w is None:
            self._full_i2w = [VocabHelper.convert_special_pattern(z) for z in self.pre_list] + self.i2w \
                             + [VocabHelper.convert_special_pattern(z) for z in self.post_list]
        return self._full_i2w

    @property
    def real_i2w(self):  # non-special real ones
        return self.i2w

    @property
    def idx_offset(self):
        return len(self.pre_list)

    def __len__(self):  # note: full length
        return len(self.i2w) + len(self.pre_list) + len(self.post_list)

    def __repr__(self):
        return f"Vocab[{self.name}]: len=({len(self.pre_list)}+{len(self.i2w)}+{len(self.post_list)})={len(self)}"

    def __contains__(self, item):
        return self.has_key(item)

    def __getitem__(self, item: str):
        assert self.has_key(item)
        return self.word2idx(item)

    def get(self, item, default=None):
        return self.word2idx(item, default)

    def has_key(self, item):
        return item in self.w2i

    # excluding pre and post ones
    def non_special_range(self):  # [)
        return (len(self.pre_list), len(self)-len(self.post_list))

    def non_speical_num(self):
        return len(self.i2w)

    def keys(self):
        return self.i2w

    # idx -> word
    def idx2word(self, idx: int):
        return self.full_i2w[idx]

    def word2idx(self, item, df=None):
        if item in self.w2i:
            return self.w2i[item] + self.idx_offset  # add offset to idx!!
        else:
            return df

    def seq_idx2word(self, idxes: List):
        return [self.full_i2w[ii] for ii in idxes]

    def seq_word2idx(self, words: List, df=None):
        return [self.word2idx(ww, df) for ww in words]

    # count related
    def word2count(self, item: str, df=0):
        if item in self.counts:
            return self.counts[item]
        else:
            return df

    def idx2count(self, idx: int):
        return self.counts[self.full_i2w[idx]]

    def get_all_counts(self):
        return sum(self.counts.values())

    # --
    # similar API to those in transformers.*Tokenizer
    def convert_tokens_to_ids(self, tokens): return self.seq_word2idx(tokens, df=self.unk)
    def convert_ids_to_tokens(self, ids): return self.seq_idx2word(ids)
    def get_vocab(self): return {vv: ii for ii, vv in enumerate(self.full_i2w)}
    @property
    def vocab_size(self): return len(self)
    @property
    def cls_token_id(self): return self.bos
    @property
    def sep_token_id(self): return self.eos
    @property
    def pad_token_id(self): return self.pad
    @property
    def mask_token_id(self): return self.mask
    @property
    def unk_token_id(self): return self.unk

    def get_toker(self, df_idx=None):
        from .toker import Toker
        return Toker(self, df_idx)
    # --

    # =====
    # building related

    # DEFAULT_PRE_LIST, DEFAULT_POST_LIST = \
    #     tuple([VocabHelper.convert_special_pattern(w) for w in ["non"]]), \
    #     tuple([VocabHelper.convert_special_pattern(w)
    #            for w in ["unk", "eos", "bos", "pad", "mask"] + [f"spe{i}" for i in range(5)]])

    DEFAULT_PRE_LIST, DEFAULT_POST_LIST = ("non", ), tuple(["unk", "eos", "bos", "pad", "mask"])

    # basic settings
    def set_name(self, name: str):
        self.name = name

    def set_pre_post(self, pre_list: List = None, post_list: List = None):
        if pre_list is not None:
            self.pre_list = pre_list
        if post_list is not None:
            self.post_list = post_list
        self._build_props()
        self._full_i2w = None  # clear!

    def set_i2w(self, new_i2w: List[str]):
        self._rebuild_items(new_i2w)

    # build public properties for the special tokens
    def _build_props(self):
        for ii, name in enumerate(self.pre_list):
            setattr(self, name, ii)
        for ii, name in enumerate(self.post_list):
            setattr(self, name, ii+len(self.pre_list)+len(self.i2w))

    # =====
    # feeding
    def feed_one(self, w: str, c=1):
        counts = self.counts
        is_new_entry = False
        if w not in counts:
            # also keep them in the adding order
            self.i2w.append(w)
            self.w2i[w] = len(self.w2i)
            counts[w] = c
            is_new_entry = True
            # --
            self.set_pre_post()  # remember to refresh special ones!!
            # --
        else:
            counts[w] += c
        return is_new_entry

    def feed_iter(self, iter: Iterable):
        rets = []  # whether add new entry
        for w in iter:
            rets.append(self.feed_one(w))
        return rets

    # filtering and sort
    def _rebuild_items(self, new_i2w: List[str], default_count=0, by_what=""):
        # todo(note): change inside!
        before_str = str(self)
        self.i2w = new_i2w
        self.w2i = {k:i for i,k in enumerate(new_i2w)}
        assert len(self.i2w) == len(self.w2i), "Err: repeated items in new_i2w!!"
        old_counts = self.counts
        self.counts = {k:old_counts.get(k, default_count) for k in new_i2w}
        after_str = str(self)
        # --
        self.set_pre_post()  # remember to refresh special ones!!
        # --
        zlog(f"Rebuild Vocab by {by_what}: {before_str} -> {after_str}")

    # filter out certain items
    def build_filter(self, word_filter=(lambda w, i, c: True)):
        _counts = self.counts
        new_i2w = [w for i,w in enumerate(self.i2w) if word_filter(w,i,_counts[w])]
        self._rebuild_items(new_i2w, by_what="filter")

    # shortcut
    def build_filter_thresh(self, rthres: int, fthres: int):
        self.build_sort()  # must sort to allow rank-thresh
        def rf_filter(w, i, c): return c>=fthres and i<=rthres
        self.build_filter(rf_filter)

    # sort items: by default sort by -count, adding-idx, word-str
    def build_sort(self, key=lambda w, i, c: (-c, w)):
        _counts = self.counts
        sorting_info = [(key(w,i,_counts[w]), w) for i,w in enumerate(self.i2w)]  # put key at first
        assert len(sorting_info) == len(self.i2w), "Inner error, repeated key!"
        sorting_info.sort()
        new_i2w = [z[1] for z in sorting_info]
        self._rebuild_items(new_i2w, by_what="sort")

    # add new tokens
    def add_tokens(self, tokens):
        ret = 0
        for t in tokens:
            ret += int(self.feed_one(t))
        zlog(f"Try to add extra_tokens {tokens}: {ret} added!")
        return ret

    # -----
    # return a pandas table
    def get_info_table(self):
        d = Vocab.create_info_table(self.i2w, [self.word2count(w) for w in self.i2w], [self.get(w) for w in self.i2w])
        return d

    @staticmethod
    def create_info_table(words, counts, idxes=None):
        import pandas as pd
        if idxes is None:
            idxes = list(range(len(words)))
        # --
        res = []
        accu_counts = 0
        for ii, w in enumerate(words):
            i, c = idxes[ii], counts[ii]
            accu_counts += c
            res.append([i, w, c, 0., accu_counts, 0.])
        d = pd.DataFrame(res, columns=["Idx", "Word", "Count", "Perc.", "ACount", "APerc."])
        d["Perc."] = d["Count"] / accu_counts
        d["APerc."] = d["ACount"] / accu_counts
        return d

    # =====
    # some shortcut buildings

    # build empty
    @staticmethod
    def build_empty(name="anon", pre_list=DEFAULT_PRE_LIST, post_list=DEFAULT_POST_LIST):
        v = Vocab(name=name)
        v.set_pre_post(pre_list, post_list)
        return v

    # build with static items
    @staticmethod
    def build_by_static(items: List[str], name="anon", pre_list=None, post_list=None):
        v = Vocab(name=name)
        v.set_i2w(items)
        v.set_pre_post(pre_list, post_list)
        return v

    # build from counting iters
    # word_filter=(lambda w, i, c: True)
    @staticmethod
    def build_from_iter(iter: Iterable, name="anon", pre_list=DEFAULT_PRE_LIST, post_list=DEFAULT_POST_LIST,
                        sorting=False, word_filter: Callable = None):
        v = Vocab(name=name)
        for one in iter:
            v.feed_one(one)
        v.set_pre_post(pre_list, post_list)
        if sorting:  # first do possible sorting!!
            v.build_sort()
        if word_filter is not None:
            v.build_filter(word_filter)
        return v

    # merge multiple vocabs
    @staticmethod
    def merge_vocabs(vocabs: Iterable['Vocab'], name="merged", pre_list=DEFAULT_PRE_LIST, post_list=DEFAULT_POST_LIST,
                     sorting=False, word_filter: Callable = None):
        v = Vocab(name=name)
        for one_vocab in vocabs:
            for w in one_vocab.keys():  # note: add counts
                v.feed_one(w, one_vocab.word2count(w))
        v.set_pre_post(pre_list, post_list)
        if sorting:  # first do possible sorting!!
            v.build_sort()
        if word_filter is not None:
            v.build_filter(word_filter)
        return v

    # filter inits for embeddings
    def filter_embed(self, wv: 'WordVectors', init_nohit=None, scale=1.0, assert_all_hit=False):
        if init_nohit is None:  # auto decide by wv
            init_nohit = np.mean([np.std(z) for z in wv.vecs]).item()
            zlog(f"Auto decide init_nohit={init_nohit}")
        if init_nohit <= 0.:
            get_nohit = lambda s: np.zeros((s,), dtype=np.float32)
        else:
            _generator = Random.get_generator("vocab")
            get_nohit = lambda s: _generator.standard_normal(s) * init_nohit
        # --
        ret = []
        res = defaultdict(int)
        embed_size = wv.get_emb_size()
        # for w in self.keys():  # todo(+N): once a bug!
        for w in self.full_i2w:
            hit, norm_name, norm_w = wv.norm_until_hit(w)
            if hit:
                value = np.asarray(wv.get_vec(norm_w, norm_name=False), dtype=np.float32)
                res[norm_name] += 1
            else:
                value = get_nohit(embed_size)
                # value = np.zeros((embed_size,), dtype=np.float32)
                res["no-hit"] += 1
            ret.append(value)
        # --
        if assert_all_hit:
            assert res["no-hit"]==0, f"Filter-embed error: assert all-hit but get no-hit of {res['no-hit']}"
        zret = np.asarray(ret, dtype=np.float32) * scale
        zlog(f"Filter pre-trained embed {self}->{zret.shape}: {res}, no-hit is inited with {init_nohit}.")
        return zret

    # ======
    # special method for combining embeddings and vocabs
    # keep main's pre&post, but put aug's true words before post and make corresponding changes to the arr
    # -> return (new_vocab, new_arr)
    @staticmethod
    def aug_vocab_and_arr(main_vocab: 'Vocab', main_arr, aug_vocab: 'Vocab', aug_arr, new_name='aug'):
        # first merge the vocab
        new_vocab = Vocab.merge_vocabs(
            [main_vocab, aug_vocab], name=new_name, sorting=False, pre_list=main_vocab.pre_list, post_list=main_vocab.post_list)
        # then find the arrays
        new_arr = [main_arr[i] for i in range(len(main_vocab.pre_list))]
        main_hit = aug_hit = 0
        for idx in range(*(new_vocab.non_special_range())):
            word = new_vocab.idx2word(idx)
            # note: selecting the embeds in aug first (make it possible to override original ones!)
            aug_orig_idx = aug_vocab.get(word)
            if aug_orig_idx is None:
                main_orig_idx = main_vocab[word]  # must be there!
                new_arr.append(main_arr[main_orig_idx])
                main_hit += 1
            else:
                new_arr.append(aug_arr[aug_orig_idx])
                aug_hit += 1
        new_arr.extend([main_arr[i] for i in range(-len(main_vocab.post_list), 0)])
        # --
        zlog(f"For the final merged arr, the composition is all={len(new_arr)},main={main_hit},aug={aug_hit}")
        ret_arr = np.asarray(new_arr)
        return new_vocab, ret_arr

# =====
class WordVectors:
    def __init__(self, words: List, vecs: Iterable):
        vecs = list(vecs)  # make it a new list!
        assert len(words) == len(vecs)
        self.vocab = Vocab.build_by_static(words)
        self.vecs = vecs  # should correspond to vocab
        # --
        self.hits = {}  # hit keys by any queries with __contains__: norm_until_hit, has_key, get_vec

    def __contains__(self, item):
        if item in self.vocab:  # extra recording!!
            self.hits[item] = self.hits.get(item, 0) + 1
            return True
        else:
            return False

    # find key possibly with norms
    def find_key(self, k: str, norm=True):
        if norm:
            success, norm_name, key = VocabHelper.norm_until_hit(self, k)
        else:
            key = k
            success = (k in self)
        return key if success else None

    def norm_until_hit(self, k: str):
        return VocabHelper.norm_until_hit(self, k)

    def get_vec(self, k: str, df=None, norm_name=True):
        key = self.find_key(k, norm_name)
        if key is None:
            return df
        else:
            return self.vecs[self.vocab[key]]

    def get_num_word(self): return len(self.vocab)
    def get_emb_size(self): return len(self.vecs[0])

    # =====
    # save and load

    @staticmethod
    def save_txt(fname: str, words: List[str], vecs: List, sep: str):
        num_words = len(words)
        embed_size = len(vecs[0])
        zlog(f"Saving w2v (in txt) num_words={num_words}, embed_size={embed_size} to {fname}.")
        assert num_words == len(vecs), "Unmatched size!"
        with zopen(fname, "w") as fd:
            fd.write(f"{num_words}{sep}{embed_size}\n")
            for w, vec in zip(words, vecs):
                assert len(vec) == embed_size, "Unmatched dim!"
                print_list = [w] + ["%.6f" % float(z) for z in vec]
                fd.write(sep.join(print_list) + "\n")

    def save(self, fname: str, sep=" "):
        WordVectors.save_txt(fname, self.vocab.i2w, self.vecs, sep)

    def clear_hits(self):
        self.hits.clear()

    def save_hits(self, fname: str, sep=" "):
        tmp_words, tmp_vecs = [], []
        for i, w in enumerate(self.vocab.i2w):
            if self.hits.get(w, 0)>0:
                tmp_words.append(w)
                tmp_vecs.append(self.vecs[i])
        WordVectors.save_txt(fname, tmp_words, tmp_vecs, sep)

    def merge_others(self, others: Iterable['WordVectors']):
        orig_num = self.get_num_word()
        embed_size = self.get_emb_size()
        self_vocab = self.vocab
        for other in others:
            assert embed_size == other.get_emb_size(), "Cannot merge two diff-sized embeddings!"
            # only merge keys that do not exist!!
            other_voc = other.vocab
            this_all_num, this_added_num = len(other_voc), 0
            for o_i, o_k in enumerate(other_voc.keys()):
                o_c = other_voc.word2count(o_k)
                is_new = self_vocab.feed_one(o_k, o_c)  # also add counts
                if is_new:
                    self.vecs.append(other.vecs[o_i])
                    this_added_num += 1
            zlog(f"Merge embed: add another with all={this_all_num}/add={this_added_num}")
        zlog(f"After merge, changed from {self.get_num_word()} to {orig_num}")

    @staticmethod
    def load(fname: str, binary=False, txt_sep=" ", return_raw=False):
        if binary:
            words, vecs = WordVectors._load_bin(fname)
        else:
            words, vecs = WordVectors._load_txt(fname, txt_sep)
        if return_raw:
            return words, vecs
        else:
            return WordVectors(words, vecs)

    @staticmethod
    def _load_txt(fname: str, sep=" "):
        zlog(f"Going to load pre-trained (txt) w2v from {fname} ...")
        repeated_count = 0
        words, vecs = [], []
        word_set = set()
        num_words, embed_size = None, None
        with zopen(fname) as fd:
            # first line
            line = fd.readline()
            try:
                num_words, embed_size = [int(x) for x in line.split(sep)]
                zlog(f"Reading w2v num_words={num_words}, embed_size={embed_size}.")
                line = fd.readline()
            except:
                zlog("Reading w2v.")
            # the rest
            while len(line) > 0:
                fields = line.rstrip().split(sep)
                word, vec = fields[0], [float(x) for x in fields[1:]]
                if word in word_set:
                    repeated_count += 1
                    zwarn(f"Repeat key {word}")
                else:  # only add the first one
                    words.append(word)
                    vecs.append(vec)
                    word_set.add(word)
                # put embed_size
                if embed_size is None:
                    embed_size = len(vec)
                else:
                    assert len(vec) == embed_size, "Unmatched embed dimension."
                line = fd.readline()
        if num_words is not None:
            assert num_words == len(vecs) + repeated_count
        num_words = len(vecs)
        # final
        zlog(f"Read ok: w2v num_words={num_words}, embed_size={embed_size}, repeat={repeated_count}")
        return words, vecs

    @staticmethod
    def _load_bin(fname: str):
        zlog(f"Going to load pre-trained (binary) w2v from {fname}")
        # --
        from gensim.models import KeyedVectors
        # --
        kv = KeyedVectors.load_word2vec_format(fname, binary=True)
        # KeyedVectors.save_word2vec_format()
        words, vecs = kv.index2word, kv.vectors
        one = WordVectors(words, vecs)
        zlog(f"Read ok with {one.vocab}")
        return words, vecs

# =====
class VocabPackage:
    def __init__(self, vocabs: Dict=None, embeds: Dict=None):
        self.vocabs = {} if vocabs is None else vocabs
        self.embeds = {} if embeds is None else embeds

    def get_voc(self, name: str, df=None):
        return self.vocabs.get(name, df)

    def get_emb(self, name: str, df=None):
        return self.embeds.get(name, df)

    def put_voc(self, name: str, v):
        self.vocabs[name] = v

    def put_emb(self, name: str, e):
        self.embeds[name] = e

    def load(self, dir_prefix="", file_prefix=""):
        for name in self.vocabs:
            fname = os.path.join(dir_prefix, "_".join([z for z in ["vv", file_prefix, name] if z]) + ".json")
            if os.path.exists(fname):
                self.vocabs[name] = default_json_serializer.from_file(fname)
            else:
                zwarn("Cannot find Vocab " + name)
                self.vocabs[name] = None
        for name in self.embeds:
            fname = os.path.join(dir_prefix, "_".join([z for z in ["ve", file_prefix, name] if z]) + ".pkl")
            if os.path.exists(fname):
                self.embeds[name] = default_pickle_serializer.from_file(fname)
            else:
                self.embeds[name] = None

    def save(self, dir_prefix="", file_prefix=""):
        for name, vv in self.vocabs.items():
            fname = os.path.join(dir_prefix, "_".join([z for z in ["vv", file_prefix, name] if z]) + ".json")
            if vv is not None:
                default_json_serializer.to_file(vv.to_dict(), fname)
        for name, vv in self.embeds.items():
            fname = os.path.join(dir_prefix, "_".join([z for z in ["ve", file_prefix, name] if z]) + ".pkl")
            if vv is not None:
                default_pickle_serializer.to_file(vv, fname)
