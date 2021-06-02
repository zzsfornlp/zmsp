#

# basic vocab

from msp.utils import zlog, zopen
from msp.data import VocabBuilder, Vocab

# ========
# vocab sorted by Freq (support ranking and tell punctuation)
# todo(note): 0 is always the <UNK>
class StatVocab:
    UNK_str = "[<myUNK>]"

    def __init__(self, name=None, pre_words=None, no_adding=False):
        self.name = str(name)
        self.count_all = 0
        self.no_adding = no_adding
        self.w2c = {}  # word -> count
        self.i2w = []  # idx -> word (only the words that survive cutting)
        self.w2i = None  # word -> idx
        self.i2c = None  # idx -> count
        self.is_punct = None  # [idx -> int(bool)]
        if pre_words is not None:
            self.i2w = list(pre_words)
            self.w2c = {w:0 for w in self.i2w}
            self._calc()
        elif no_adding:
            zlog("Warn: no adding mode for an empty Vocab!!")

    def _calc(self):
        # todo(note): put UNK as 0
        unk_str = StatVocab.UNK_str
        if unk_str in self.w2c:
            # loading, not need to add
            assert self.w2c[unk_str] == 0
            assert self.i2w[0] == unk_str
        else:
            self.w2c[unk_str] = 0
            self.i2w = [unk_str] + self.i2w
        self.w2i = {w:i for i,w in enumerate(self.i2w)}
        self.i2c = [self.w2c[w] for w in self.i2w]
        self.is_punct = [0] * len(self.i2w)
        PUNCT_LIST = """[]!"#$%&'()*+,./:;<=>?@[\\^_`{|}~„“”«»²–、。・（）「」『』：；！？〜，《》·"""
        PUNCT_SET = set(PUNCT_LIST)
        for widx, word in enumerate(self.i2w):
            if all(c in PUNCT_SET for c in word):  # if all characters are punct for this word
                self.is_punct[widx] = 1

    def __len__(self):
        return len(self.i2w)

    def __repr__(self):
        return f"Vocab {self.name}: size={len(self)}({len(self.w2c)})/count={self.count_all}."

    def __contains__(self, item):
        return item in self.w2i

    def __getitem__(self, item):
        return self.w2i[item]

    def get(self, item, d):
        return self.w2i.get(item, d)

    def copy(self):
        n = StatVocab(self.name + "_Copy")
        n.count_all = self.count_all
        n.w2c = self.w2c.copy()
        n.i2w = self.i2w.copy()
        n._calc()
        return n

    def add_all(self, ws, cc=1, ensure_exist=False):
        for w in ws:
            self.add(w, cc, ensure_exist)

    def add(self, w, cc=1, ensure_exist=False):
        orig_cc = self.w2c.get(w)
        exist = (orig_cc is not None)
        if self.no_adding or ensure_exist:
            assert exist, "Non exist key %s" % (w,)
        if exist:
            self.w2c[w] = orig_cc + cc
        else:
            self.i2w.append(w)
            self.w2c[w] = cc
        self.count_all += cc

    # cur ones that are <thresh and (only when sorting) soft-rank<=soft_cut
    def sort_and_cut(self, mincount=0, soft_cut=None, sort=True):
        zlog("Pre-cut Vocab-stat: " + str(self))
        final_i2w = [w for w, v in self.w2c.items() if v>=mincount]
        if sort:
            final_i2w.sort(key=lambda x: (-self.w2c[x], x))  # sort by (count, self) to make it fixed
            if soft_cut and len(final_i2w)>soft_cut:
                new_minc = self.w2c[final_i2w[soft_cut-1]]          # boundary counting value
                cur_idx = soft_cut
                while cur_idx<len(final_i2w) and self.w2c[final_i2w[cur_idx]]>=new_minc:
                    cur_idx += 1
                final_i2w = final_i2w[:cur_idx]     # soft cutting by ranking & keep boundary values
        #
        self.i2w = final_i2w
        self.count_all = sum(self.w2c[w] for w in final_i2w)
        self.w2c = {w:self.w2c[w] for w in final_i2w}
        self._calc()
        zlog("Post-cut Vocab-stat: " + str(self))

    #
    def yield_infos(self):
        accu_count = 0
        for i, w in enumerate(self.i2w):
            count = self.w2c[w]
            accu_count += count
            perc = count / self.count_all * 100
            accu_perc = accu_count / self.count_all * 100
            yield (i, w, count, perc, accu_count, accu_perc)

    def write_txt(self, fname):
        with zopen(fname, "w") as fd:
            for pack in self.yield_infos():
                i, w, count, perc, accu_count, accu_perc = pack
                ss = f"{i} {w} {count}({perc:.3f}) {accu_count}({accu_perc:.3f})\n"
                fd.write(ss)
        zlog("Write (txt) to %s: %s" % (fname, str(self)))

    # other i/o
    def to_zvoc(self) -> Vocab:
        builder = VocabBuilder("word", default_val=0)
        for w,c in self.w2c.items():
            builder.feed_one(w, c)
        voc: Vocab = builder.finish(sort_by_count=True)
        return voc

    def to_builtin(self):
        return (self.name, self.count_all, self.no_adding, self.w2c, self.i2w)

    def from_builtin(self, v):
        self.name, self.count_all, self.no_adding, self.w2c, self.i2w = v
        self._calc()
