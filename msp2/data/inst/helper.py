#

# some helper or middle classes

__all__ = [
    "InDocInstance", "InSentInstance", "DataPadder",
    "SplitAlignInfo", "SubwordTokenizer", "CharSubwordTokenizer",
    "CharIndexer",
]

from typing import List, Iterable, Tuple
import numpy as np
from .base import DataInstance
from msp2.utils import JsonSerializable, zwarn, zlog

# =====
# instance that lives inside (is bound to / is stored inside) a Doc
class InDocInstance(DataInstance):
    def __init__(self, _id: str = None, _par: 'DataInstance' = None):
        super().__init__(_id, _par)
        # =====
        from .doc import Doc
        self._doc: Doc = None

    @property
    def doc(self):
        from .doc import Doc
        if self._doc is None:
            # find sent by searching up, maybe failed and still None
            self._doc = self.search_up_for_type(Doc)
        return self._doc

    def set_doc(self, doc: 'Doc'):
        self._doc = doc

# instance that lives inside (is bound to / is stored inside) a Sent
class InSentInstance(DataInstance):
    def __init__(self, _id: str = None, _par: 'DataInstance' = None):
        super().__init__(_id, _par)
        # =====
        from .doc import Doc, Sent
        self._doc: Doc = None
        self._sent: Sent = None

    @property
    def doc(self):
        from .doc import Doc
        if self._doc is None:
            # find sent by searching up, maybe failed and still None
            self._doc = self.search_up_for_type(Doc)
        return self._doc

    @property
    def sent(self):
        from .doc import Sent
        if self._sent is None:
            # find sent by searching up, maybe failed and still None
            self._sent = self.search_up_for_type(Sent)
        return self._sent

    def set_doc(self, doc: 'Doc'):
        self._doc = doc

    def set_sent(self, sent: 'Sent'):
        self._sent = sent

# =====
# add inst, set par to self and (optionally) register

# =====
# data padder

# prepare and truncate/pad the data along sentence-seq steps
# (batch, step, *) -> padded (step, batch, *)
# data: recursive list
# pad_lens, pad_vals: dim(data) sized list, pad_len<=0 means max
# dynamic_lens: use max_value if max_value<=pad_len
# mask_range: record mask for how many dims
class DataPadder(object):
    def __init__(self, dim: int, pad_lens:List[int]=None, pad_vals=0, dynamic_lens=True, mask_range=0):
        self.dim = dim
        self.pad_lens = [-1]*dim if pad_lens is None else pad_lens
        self.pad_vals = [pad_vals]*dim if not isinstance(pad_vals, Iterable) else pad_vals
        self.dynamic_lens = [dynamic_lens]*dim if not isinstance(dynamic_lens, Iterable) else dynamic_lens
        self.mask_range = mask_range

    # decide max sizes for all dims
    def _rec_size(self, d, cur_dim: int, sizes: List[int]):
        diff = self.dim - cur_dim
        if diff <= 0:
            return
        one_len = len(d)
        sizes[cur_dim] = max(one_len, sizes[cur_dim])
        for one_d in d:
            self._rec_size(one_d, cur_dim+1, sizes)

    # fill in the values
    def _rec_fill(self, d, cur_dim: int, sizes: List[int], strides: List[int], arr: List, arr_mask: List):
        diff = self.dim - cur_dim
        if diff <= 0:
            return
        one_len = len(d)
        cur_pad = sizes[cur_dim]  # current dim cut
        # fill
        if diff == 1:  # directly fill in the data
            arr.extend(d[:cur_pad])
        else:
            for one_d in d[:cur_pad]:
                self._rec_fill(one_d, cur_dim+1, sizes, strides, arr, arr_mask)
        # pad
        miss = cur_pad - one_len
        if miss > 0:
            arr.extend([self.pad_vals[cur_dim]] * (strides[cur_dim]*miss))
        # mask
        if cur_dim == self.mask_range-1:
            if miss > 0:
                arr_mask.extend([1.]*one_len+[0.]*miss)
            else:
                arr_mask.extend([1.]*cur_pad)

    # return numpy-arr, mask
    def pad(self, data):
        # 1. first decide the sizes for all dims
        sizes = [0] * self.dim
        self._rec_size(data, 0, sizes)  # first get max sizes
        for idx in range(self.dim):
            pad_len = self.pad_lens[idx]
            if pad_len>0:
                if self.dynamic_lens[idx]:
                    sizes[idx] = min(sizes[idx], pad_len)
                else:
                    sizes[idx] = pad_len
        # 2. then iter the data and pad/trunc
        strides = [1]
        for one_s in reversed(sizes[1:]):
            strides.append(strides[-1]*one_s)
        strides.reverse()
        arr, arr_mask = [], []
        self._rec_fill(data, 0, sizes, strides, arr, arr_mask)
        ret = np.asarray(arr).reshape(sizes)
        if self.mask_range > 0:
            ret_mask = np.asarray(arr_mask, dtype=np.float32).reshape(sizes[:self.mask_range])
        else:
            ret_mask = None
        return ret, ret_mask

    # a helper function to turn a list of lengths to MaskArr
    @staticmethod
    def lengths2mask(lengths: List[int]):
        ret = np.zeros((len(lengths), max(lengths)))
        for ii, ll in enumerate(lengths):
            ret[ii][:ll] = 1.
        return ret

    # simpler one for 2d cases
    @staticmethod
    def go_batch_2d(inputs: List, pad_val, max_len=None, dtype=None):
        bs = len(inputs)
        if max_len is None:
            max_len = 1 if bs == 0 else max(len(z) for z in inputs)
        if dtype is None:  # guess dtype
            if isinstance(pad_val, int): dtype = np.long
            elif isinstance(pad_val, float): dtype = np.float32
        arr = np.full([bs, max_len], pad_val, dtype=dtype)
        for ii, vv in enumerate(inputs):
            if len(vv) <= max_len:  # normal
                arr[ii, :len(vv)] = vv
            else:  # truncate
                arr[ii, :max_len] = vv[:max_len]
        return arr

# =====
# split alignment (a special monotonic alignment): may be useful for subtok or span-label

class SplitAlignInfo(JsonSerializable):
    def __init__(self, split_sizes: List[int]):
        # orig2sbegin(orig_len, start in split), orig2send(orig_len, end in split), split2orig(split_len, one in orig)
        orig2begin, orig2end, split2orig = [], [], []  # [) for orig2s
        self.orig_len = len(split_sizes)
        split_len = 0
        for i, s in enumerate(split_sizes):
            assert s>=0
            orig2begin.append(split_len)
            split_len += s
            orig2end.append(split_len)
            split2orig.extend([i]*s)
        self.split_len = split_len
        self.split_sizes = split_sizes
        self.orig2begin, self.orig2end, self.split2orig = orig2begin, orig2end, split2orig

# Actual tokenizer
class SubwordTokenizer:
    def __repr__(self):
        return f"{self.__class__.__name__}"

    # sub-tokenize one token
    def sub_tok(self, tok: str) -> List[str]:
        raise NotImplementedError("To be implemented!")

    # sub-tokenize List[str] with align-info: by default sub_tok each one!
    def sub_vals(self, vals: List[str]):
        split_sizes = []
        sub_vals: List[str] = []  # flattened ones
        sub_idxes = None
        for v in vals:
            one_sub_vals = self.sub_tok(v)
            sub_vals.extend(one_sub_vals)
            split_sizes.append(len(one_sub_vals))
        return sub_vals, sub_idxes, SplitAlignInfo(split_sizes)

# simple splitting into chars
class CharSubwordTokenizer(SubwordTokenizer):
    def sub_tok(self, w: str) -> List[str]:
        return list(w)

# --
# char indexer: from char indexes to token indexes
class CharIndexer:
    def __init__(self, full_char_idxes: List, offset_str: str, sent_tokens: List[List[str]], sent_positions: List[List], doc=None):
        self.full_char_idxes = full_char_idxes
        self.offset_str = offset_str
        self.sent_tokens = sent_tokens
        self.sent_positions = sent_positions
        self.doc = doc

    @staticmethod
    def build(offset_str: str, sent_tokens: List[List[str]], sent_positions: List[List], doc=None):
        full_char_idxes = [None] * len(offset_str)
        assert len(sent_tokens) == len(sent_positions)
        # --
        for sid, one_toks in enumerate(sent_tokens):
            one_sent_positions = sent_positions[sid]
            assert len(one_toks) == len(one_sent_positions)
            wid = 0
            for one_tok, one_posi in zip(one_toks, one_sent_positions):
                cstart, clen = one_posi
                assert ''.join(one_tok.split()) == ''.join(offset_str[cstart:cstart+clen].split())
                for cc in range(cstart, cstart+clen):
                    assert full_char_idxes[cc] is None  # None means blank!!
                    full_char_idxes[cc] = (sid, wid)  # (sid, wid)
                wid += 1
        return CharIndexer(full_char_idxes, offset_str, sent_tokens, sent_positions, doc=doc)

    @staticmethod
    def build_from_doc(doc: 'Doc', offset_str: str = None):
        if offset_str is None:
            offset_str = doc.get_text()
        ret = CharIndexer.build(offset_str, [s.seq_word.vals for s in doc.sents], [s.word_positions for s in doc.sents], doc=doc)
        return ret

    # (cidx, clen) -> List[(sid, wid)]
    def collect_tokens(self, char_idx: int, char_len: int):
        # collect all tokens
        index_chars = self.full_char_idxes
        tokens = []
        for ii in range(char_idx, char_idx+char_len):
            vv = index_chars[ii]
            if vv is not None:
                if len(tokens) == 0 or vv != tokens[-1]:  # find a new one
                    assert len(tokens)==0 or (vv[0]==tokens[-1][0] and vv[1]==tokens[-1][1]+1) \
                           or (vv[0]==tokens[-1][0]+1 and vv[1]==0)  # assert continuing span
                    tokens.append(vv)
        # --
        # check
        str0 = ''.join(self.offset_str[char_idx:char_idx+char_len].split())
        str1 = ''.join([''.join(self.sent_tokens[sid][wid].split()) for sid,wid in tokens])
        if str0 not in str1:
            # note: a very strange 'ar' case ...
            if str1 == ''.join(str0.split("_")) or set(str0).difference(set(str1))==set(chr(1618)):
                zwarn(f"=> Slightly unmatch: {str0} vs {str1}")
            else:
                raise RuntimeError()
        return tokens

    # (cidx, clen) -> (sid, widx, wlen)
    # -- hint_sidx helps to select the sentence if split-sent (mostly from head_posi)
    def get_posi(self, char_idx: int, char_len: int, hint_sid=None):
        assert char_len > 0
        positions = self.collect_tokens(char_idx, char_len)  # List[sid, widx]
        # check it and report error?
        ret_code = ""
        if len(positions) == 0:
            ret_code = "ErrNotFound"  # no non-blank words at those position (maybe annotations in tag or ignored tag)
            ret_posi = None
        else:
            if len(set([z[0] for z in positions])) > 1:  # more than 1 sentence
                hint_positions = [z for z in positions if z[0]==hint_sid]
                if len(hint_positions) == 0:
                    ret_code = "ErrDiffSent"  # sent splitting error
                    ret_posi = None
                else:
                    ret_code = "WarnDiffSent"  # accept this part
                    ret_posi = (hint_positions[0][0], hint_positions[0][1], len(hint_positions))
            else:
                ret_posi = (positions[0][0], positions[0][1], len(positions))  # (sid, widx, wlen)
                # check boundaries
                ret_cidx = self.sent_positions[ret_posi[0]][ret_posi[1]][0]
                ret_cridx = sum(self.sent_positions[ret_posi[0]][ret_posi[1]+ret_posi[2]-1])
                # inside or the more ones should be empty
                assert all(str.isspace(c) for c in self.offset_str[char_idx:ret_cidx]+self.offset_str[ret_cridx:char_idx+char_len])
                if ret_cidx < char_idx:
                    ret_code += "WarnLeft"
                if ret_cridx > (char_idx+char_len):
                    if ret_cridx-(char_idx+char_len)==1 and self.offset_str[ret_cridx-1]==".":  # simply a dot
                        ret_code += "WarnRDot"
                    else:
                        ret_code += "WarnRight"
        # --
        # print the mismatches
        if ret_posi is None:
            zlog(f"=> Cannot find span: {self.offset_str[char_idx:char_idx+char_len]}")
        elif ret_code != "":
            str0 = ' '.join(self.offset_str[char_idx:char_idx+char_len].split())
            str1 = ' '.join(self.sent_tokens[ret_posi[0]][ret_posi[1]:ret_posi[1]+ret_posi[2]])
            zlog(f"=> Span mismatch ({ret_code}): {str0} ||| {str1}")
        # --
        return ret_posi, ret_code

    # get position and return a mention!
    def get_mention(self, char_idx: int, char_len: int, hint_sid=None):
        from .frame import Mention
        ret_posi, ret_code = self.get_posi(char_idx, char_len, hint_sid=hint_sid)
        if ret_posi is None:
            return None, ret_code
        else:
            sid, widx, wlen = ret_posi
            m = Mention.create(self.doc.sents[sid], widx, wlen)
            m.info["_code"] = ret_code
            return m, ret_code
# --
