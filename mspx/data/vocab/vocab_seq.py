#

# special vocab for seqlabing

__all__ = [
    "SeqVocabConf", "SeqVocab", "SeqSchemeHelper", "SeqSchemeHelperIdx", "SeqSchemeHelperStr",
]

from typing import List, Tuple, Iterable
from mspx.utils import Conf
import numpy as np
from .vocab import Vocab

# =====

class SeqVocabConf(Conf):
    def __init__(self):
        self.assert_non0 = True  # assert non==0 and offset=1
        self.seq_scheme = "BIO"  # BIO/BIOES

class SeqVocab(Vocab):
    def __init__(self, base_vocab: Vocab, conf: SeqVocabConf=None, **kwargs):
        conf = SeqVocabConf.direct_conf(conf, **kwargs)
        super().__init__(f"seq_{base_vocab.name}")
        # --
        self.conf = conf
        self.base_vocab = base_vocab
        # --
        # check more
        if conf.assert_non0:
            assert base_vocab.non==0 and len(base_vocab.pre_list)==1
        # --
        self.helper_str = SeqSchemeHelperStr(conf.seq_scheme)
        self.helper_idx = SeqSchemeHelperIdx(conf.seq_scheme, o_idx=0, offset=1)
        self.base_vocab_offset = base_vocab.idx_offset
        # --
        # directly build values
        self.feed_one('O')  # o_idx=0
        for one_type in base_vocab.keys():  # from i2w
            one_count = base_vocab.word2count(one_type, None)
            for one_tag in self.helper_str.make_all_tags(one_type):
                self.feed_one(one_tag, one_count)  # this will keep the order
        # --
        assert self.helper_idx.get_num_mul() * base_vocab.non_speical_num() + 1 == self.non_speical_num()
        # --

    def _rebuild_items(self, new_i2w: List[str], default_count=0, by_what=""):
        raise RuntimeError("SeqVocab does not allow rebuilding!!")

    def get_allowed_transitions(self):
        a, b = self.base_vocab.non_special_range()
        core_mat = self.helper_idx.get_allowed_transitions(b-a)
        full_size = len(self)
        ret_mat = np.full((full_size, full_size), 0., dtype=np.float32)  # note: others are all 0.
        ret_mat[:len(core_mat),:len(core_mat)] = core_mat  # set the left-upper corner
        return ret_mat

    # from base idxes to all related idxes
    def get_range_by_basename(self, item: str):
        base_idx = self.base_vocab.get(item)
        if base_idx is None:
            return None
        num_mul = self.helper_idx.get_num_mul()
        my_idx_start = 1 + (base_idx - self.base_vocab_offset) * num_mul
        return (my_idx_start, my_idx_start+num_mul)

    # =====
    # transformations: *_idx need to handle idxes at outside for seq<->base
    def spans2tags_str(self, spans: List[Tuple], length: int, t_o=None):
        return self.helper_str.spans2tags(spans, length, t_o=t_o)

    def span2tags_str(self, label: str, length: int):  # just one span, starting from 0
        return self.spans2tags_str([(0, length, label)], length)[0][0]

    def tags2spans_str(self, tags: List):
        return self.helper_str.tags2spans(tags)

    def spans2tags_idx(self, spans: List[Tuple], length: int, t_o=None):
        offset = self.base_vocab_offset
        spans0 = [(z[0], z[1], z[2]-offset) for z in spans]
        return self.helper_idx.spans2tags(spans0, length, t_o=t_o)

    def tags2spans_idx(self, tags: List):
        offset = self.base_vocab_offset
        spans0 = self.helper_idx.tags2spans(tags)
        return [(z[0], z[1], z[2]+offset) for z in spans0]

    def output_span_idx(self, length: int, t: int):
        offset = self.base_vocab_offset
        real_t = t - offset
        return self.helper_idx.output_span(length, real_t)

    # get mapping from bio to origin
    def get_bio2origin(self):
        rets = list(range(len(self)))
        _div = self.helper_idx.get_num_mul()
        for i in range(1, len(rets)):
            rets[i] = (rets[i]+1) // _div
        return rets

# =====
# helper

class SeqSchemeHelper:
    def __init__(self, scheme: str, full_bies: Iterable, tag_o: object):
        full_bies = list(full_bies)
        # --
        scheme = scheme.upper()  # make it uppercase
        self.scheme = scheme
        self.output_bies = {
            "BIO": [full_bies[z] for z in [0,1,1,0]], "BIOES": full_bies,
            "IO": [full_bies[z] for z in [1,1,1,0]]}[scheme]  # for output
        self.parse_bies = {
            "BIO": [full_bies[0], full_bies[1], None, None], "BIOES": full_bies,
            "IO": [None, full_bies[1], None, None]}[scheme]  # for input
        self.num_mul = {"BIO": 2, "BIOES": 4, "IO": 1}[scheme]
        self.tag_o = tag_o

    def get_tag_o(self): return self.tag_o
    def get_num_mul(self): return self.num_mul  # number of multiply
    def parse_tag(self, tag): raise NotImplementedError()
    def make_tag(self, p, t): raise NotImplementedError()

    def make_all_tags(self, t):
        return [self.make_tag(p,t) for p in self.output_bies[:self.num_mul]]

    # --
    # output & input methods

    def output_span(self, length: int, t):
        assert length > 0
        t_b, t_i, t_e, t_s = [self.make_tag(p,t) for p in self.output_bies]
        if length == 1:  # single span
            one_tags = [t_s]  # make a single one
        else:  # >=2
            one_tags = [t_b] + [t_i] * (length - 2) + [t_e]
        return one_tags

    # note: return Layers[[(label)[...], (idx-back)[...]]
    def spans2tags(self, spans: List[Tuple], length: int, t_o=None) -> List[Tuple[List, List]]:
        if t_o is None:  # if not provided
            t_o = self.get_tag_o()
        # note: use the order of inputs
        layers = [([t_o]*length, [-1]*length)]  # at least one layer!
        sorted_them = sorted([(ii,ss) for ii,ss in enumerate(spans)],
                             key=lambda x: (-x[1][1], x[1][0], x[1][2]))  # sort by (len, start, label_idx)
        for one_ii, one_span in sorted_them:
            one_widx, one_wlen, one_type = one_span
            # find a layer that is compatible
            cur_li = 0
            while cur_li < len(layers):
                cur_layer = layers[cur_li][0]
                if all(cur_layer[z]==t_o for z in range(one_widx, one_widx+one_wlen)):
                    break
                cur_li += 1
            if cur_li >= len(layers):  # failed finding, make a new one
                layers.append(([t_o]*length, [-1]*length))
            # make tags and assign
            one_tags = self.output_span(one_wlen, one_type)
            layers[cur_li][0][one_widx:one_widx+one_wlen] = one_tags
            layers[cur_li][1][one_widx] = one_ii  # note: simply take the first one!
        # --
        return layers

    def tags2spans(self, tags: List):
        p_b, p_i, p_e, p_s = self.parse_bies
        # --
        spans = []
        prev_start, prev_t = 0, None
        # --
        def _close_prev(_start: int, _end: int, _t, _spans: List):
            if _t is not None:
                assert _end > _start
                _spans.append((_start, _end-_start, _t))
            return _end, None
        # --
        for idx, tag in enumerate(tags):
            cur_p, cur_t = self.parse_tag(tag)
            # use the simple but tedious way ...
            if cur_t is None:  # O
                prev_start, prev_t = _close_prev(prev_start, idx, prev_t, spans)  # try to close previous!
            elif cur_p == p_b:  # B
                prev_start, prev_t = _close_prev(prev_start, idx, prev_t, spans)  # try to close previous!
                prev_t = cur_t  # start a new one!
            elif cur_p == p_i:  # I
                # todo(+N): ignore type mismatch!!
                if prev_t != cur_t:  # act as a 'B' if type not matched!
                    prev_start, prev_t = _close_prev(prev_start, idx, prev_t, spans)  # try to close previous!
                    prev_t = cur_t  # start a new one!
            elif cur_p == p_e:  # E
                # todo(+N): ignore type mismatch!!
                if prev_t != cur_t:  # act as a 'B' if type not matched!
                    prev_start, prev_t = _close_prev(prev_start, idx, prev_t, spans)  # try to close previous!
                prev_start, prev_t = _close_prev(prev_start, idx+1, cur_t, spans)  # close current one!
            elif cur_p == p_s:  # S
                prev_start, prev_t = _close_prev(prev_start, idx, prev_t, spans)  # try to close previous!
                prev_start, prev_t = _close_prev(prev_start, idx+1, cur_t, spans)  # close current one!
            else:
                raise NotImplementedError(f"Error: UNK code {tag}=({cur_p}, {cur_t})")
        # close last one!
        _close_prev(prev_start, len(tags), prev_t, spans)
        return spans

class SeqSchemeHelperStr(SeqSchemeHelper):
    def __init__(self, scheme: str):
        super().__init__(scheme, list("BIES"), "O")

    def make_tag(self, p: str, t: str):
        return f"{p}-{t}"

    def parse_tag(self, tag: str):
        if tag == 'O':
            return 'O', None
        fields = tag.split('-', 1)
        p, t = fields
        return p, t

class SeqSchemeHelperIdx(SeqSchemeHelper):
    def __init__(self, scheme: str, o_idx=0, offset=1):
        super().__init__(scheme, range(4), o_idx)
        # --
        assert offset > o_idx
        self.offset = offset

    def parse_tag(self, tag: int):
        offset, num_p = self.offset, self.num_mul
        # --
        if tag < offset:
            assert tag == self.tag_o
            return self.tag_o, None
        # --
        N = tag - offset
        return N % num_p, N // num_p

    def make_tag(self, p: int, t: int):
        return self.offset + p + t*self.num_mul

    # get allowed transitions bigrams
    def get_allowed_transitions(self, num_type: int):
        t_o = self.tag_o
        ret_size = self.offset + num_type * self.num_mul
        ret = np.full((ret_size, ret_size), 0., dtype=np.float32)
        ret[t_o, t_o] = 1.  # O->O
        if self.scheme == "BIO":
            for t1 in range(num_type):
                t_b, t_i = [self.make_tag(p, t1) for p in self.parse_bies[:2]]
                ret[t_b, t_i] = 1.  # B->I
                ret[t_b, t_o] = 1.  # B->O
                ret[t_i, t_i] = 1.  # I->I
                ret[t_i, t_o] = 1.  # I->O
                ret[t_o, t_b] = 1.  # O->B
                for t2 in range(num_type):  # start another
                    t_b2 = self.make_tag(self.parse_bies[0], t2)
                    ret[t_b, t_b2] = 1.  # B->B
                    ret[t_i, t_b2] = 1.  # I->B
        elif self.scheme == "BIOES":
            for tt in range(num_type):
                t_b, t_i, t_e, t_s = [self.make_tag(p, tt) for p in self.parse_bies[:4]]
                ret[t_b, t_i] = 1.  # B->I
                ret[t_b, t_e] = 1.  # B->E
                ret[t_i, t_i] = 1.  # I->I
                ret[t_i, t_e] = 1.  # I->E
                ret[t_o, t_b] = 1.  # O->B
                ret[t_o, t_s] = 1.  # O->S
                ret[t_e, t_o] = 1.  # E->O
                ret[t_s, t_o] = 1.  # S->O
                for t2 in range(num_type):  # start another
                    t_b2, t_s2 = self.make_tag(self.parse_bies[0], t2), self.make_tag(self.parse_bies[3], t2)
                    ret[t_e, t_b2] = 1.  # E->B
                    ret[t_e, t_s2] = 1.  # E->S
                    ret[t_s, t_b2] = 1.  # S->B
                    ret[t_s, t_s2] = 1.  # S->S
        else:
            raise NotImplementedError(f"Error: UNK scheme {self.scheme}")
        return ret
