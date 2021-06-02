#

# sequence labeling like data

from msp.data import Instance, Streamer, Vocab
from msp.utils import zcheck, zfatal, Random

# one field of the data
class SeqFactor(object):
    def __init__(self, vals):
        # interface: public values
        self.vals = vals
        self.idxes = None

    def __len__(self):
        return len(self.vals)

    def has_vals(self):
        return self.vals is not None

    # reset the vals
    def set_vals(self, vals):
        self.vals = vals
        self.idxes = None

    # set indexes (special situations)
    def set_idxes(self, idxes):
        zcheck(len(idxes)==len(self.vals), "Unmatched length of input idxes.")
        self.idxes = idxes

    # two directions: val <=> idx
    # init-val -> idx
    def build_idxes(self, voc: Vocab):
        self.idxes = [voc.get_else_unk(w) for w in self.vals]

    # set idx & val
    def build_vals(self, idxes, voc: Vocab):
        self.idxes = idxes
        # zcheck(self.vals is None, "Strange existed vals.")
        self.vals = [voc.idx2word(i) for i in idxes]

# initiated from words
class InputCharFactor(SeqFactor):
    def __init__(self, words):
        super().__init__(words)

    def build_idxes(self, c_voc):
        self.idxes = [[c_voc.get_else_unk(c) for c in w] for w in self.vals]

# =====
# tagging schemes for segments: BIO/BIOES
# incrementally build chunks: strictly left to right
# todo(warn): depend on convention of tag/chunk names: BIOES-type, single "O" for outside
class ChunksSeq(object):
    def __init__(self, length):
        self.length = length
        self.cur_idx = 0
        self.chunks = []        # (start-idx(inclusive), end-idx(non-inclusive), type-str)
        self.all_chunks = []    # count "O" as single chunk
        # for the building process
        self.renew_state(0)

    @property
    def finished(self):
        return self.cur_idx >= self.length

    @staticmethod
    def split_tag_type(t):
        fields = t.split(sep="-", maxsplit=1)
        fields.append("")       # if not split
        tag, type = fields[:2]
        zcheck(tag in "BIOES", "Strange tag of %s." % tag)
        return tag, type

    #
    def renew_state(self, start):
        self.prev_start = start
        self.prev_tag = "O"
        self.prev_type = ""

    def add_chunk(self, start, end, type):
        new_chunk = (start, end, type)
        zcheck((len(self.chunks)==0 and start>=0) or (start>=self.chunks[-1][1]), "Un-seq chunk!")
        zcheck((len(self.all_chunks)==0 and start==0) or (start==self.all_chunks[-1][1]), "Un-cont all-chunk!")
        if type:
            self.chunks.append(new_chunk)
        else:
            zcheck(end-start==1, "Err: continued Outside tags.")
        self.all_chunks.append(new_chunk)

    # end current continued chunk: used
    def end_prev(self):
        if self.cur_idx > self.prev_start:
            zcheck(self.prev_tag in "BI", "Strange continuing state.")
            self.add_chunk(self.prev_start, self.cur_idx, self.prev_type)
            self.renew_state(self.cur_idx)

    # BIOES tagging scheme
    def accept_tag(self, tag):
        cur_t, cur_type = ChunksSeq.split_tag_type(tag)
        # [prev_start, cur_idx)
        if cur_t in "BOS" or (self.prev_tag!="O" and (self.prev_type!=cur_type)):
            self.end_prev()     # finish previous continued ones
        # [prev_start, cur_idx+1)
        if cur_t in "EOS":
            self.add_chunk(self.prev_start, self.cur_idx+1, cur_type)
            self.renew_state(self.cur_idx+1)
        # step forward
        self.cur_idx += 1
        self.prev_tag = cur_t
        self.prev_type = cur_type

    # build and sequence
    @staticmethod
    def build(tag_seq):
        c = ChunksSeq(len(tag_seq))
        for t in tag_seq:
            c.accept_tag(t)
        c.end_prev()
        return c

    def output_tags(self, scheme):
        ret = []
        if scheme == "BIO":
            begin_tag, end_tag, single_tag = "B", "I", "B"
        elif scheme == "BIOES":
            begin_tag, end_tag, single_tag = "B", "E", "S"
        else:
            zfatal("Unknown tagging scheme")
        #
        for one in self.all_chunks:
            start, end, type = one
            if type:
                if end-start==1:
                    ret.append(single_tag+"-"+type)
                else:
                    ret.append(begin_tag+"-"+type)
                    for _ in range(end-start-2):
                        ret.append("I-"+type)
                    ret.append(end_tag+"-"+type)
            else:
                ret.append("O")
        zcheck(len(ret)==self.length, "Err length.")
        return ret

    #
    @staticmethod
    def collect_types(tag_forms):
        types = set()
        for one in tag_forms:
            one_c, one_t = ChunksSeq.split_tag_type(one)
            if one_c != "O":
                types.add(one_t)
        types = list(types)
        return types

    #
    @staticmethod
    def full_tag_forms(scheme, hit_tag_forms):
        types = ChunksSeq.collect_types(hit_tag_forms)
        full_list = ["O"]
        for t in types:
            for c in scheme:
                if c!="O":
                    full_list.append(c+"-"+t)
        return full_list

    # todo(warn): might not be efficient
    @staticmethod
    def valid_bigrams(scheme, tag_forms, bos):
        types = ChunksSeq.collect_types(tag_forms)
        #
        rets = {(bos, "O"), ("O", "O")}
        if scheme == "BIO":
            for t in types:
                rets.add((bos, "B-"+t))
                rets.add(("O", "B-"+t))
                rets.add(("B-"+t, "I-"+t))
                rets.add(("I-"+t, "I-"+t))
                rets.add(("B-"+t, "O"))
                rets.add(("I-"+t, "O"))
                for t2 in types:
                    rets.add(("B-"+t, "B-"+t2))
                    rets.add(("I-"+t, "B-"+t2))
        elif scheme == "BIOES":
            for t in types:
                rets.add((bos, "B-"+t))
                rets.add(("O", "B-"+t))
                rets.add((bos, "S-"+t))
                rets.add(("O", "S-"+t))
                rets.add(("B-"+t, "I-"+t))
                rets.add(("I-"+t, "I-"+t))
                rets.add(("B-"+t, "E-"+t))
                rets.add(("I-"+t, "E-"+t))
                rets.add(("S-"+t, "O"))
                rets.add(("E-"+t, "O"))
                for t2 in types:
                    rets.add(("S-"+t, "B-"+t2))
                    rets.add(("E-"+t, "B-"+t2))
                    rets.add(("S-"+t, "S-"+t2))
                    rets.add(("E-"+t, "S-"+t2))
        else:
            zfatal("Unknown tagging scheme")
        return rets

# =====

# chunks represented by special tagging scheme
class ChunkTaggedFactor(SeqFactor):
    def __init__(self, tags):
        super().__init__(tags)
        if tags is None:
            self.cs = None
        else:
            self.cs = ChunksSeq.build(tags)

    # set-idx & maybe to vals
    def build_vals(self, idxes, voc):
        super().build_vals(idxes, voc)
        self.cs = ChunksSeq.build(self.vals)

# =====
# sequence of equal-sized labels (segments are presented by tagging schemes if possible)
class InstanceHelper(object):
    # used for equal-sized words & labels
    @staticmethod
    def check_equal_length(factors):
        length = len(factors[0])
        for f in factors[1:]:
            zcheck((not f.has_vals()) or len(f)==length, "Unequal length")
        return length
