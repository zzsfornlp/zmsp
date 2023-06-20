#

# some helpers

__all__ = [
    "DataPadder", "CharIndexer",
]

from typing import List, Sequence
import numpy as np
from mspx.utils import zwarn, zlog

# --
# data padding: from lists to np.arr
class DataPadder:
    # a helper function to turn a list of lengths to MaskArr
    @staticmethod
    def len2mask(lengths: List[int]):
        ret = np.zeros((len(lengths), max(lengths)))
        for ii, ll in enumerate(lengths):
            ret[ii][:ll] = 1.
        return ret

    # for 2d case
    @staticmethod
    def batch_2d(inputs: Sequence[Sequence], pad_val, max_len=None, dtype=None, ret_mask=False, ret_tensor=False):
        bs = len(inputs)
        if max_len is None:
            max_len = max((len(z) for z in inputs), default=1)
        if dtype is None:  # guess dtype
            if isinstance(pad_val, int):
                dtype = np.int64
            elif isinstance(pad_val, float):
                dtype = np.float32
        arr = np.full([bs, max_len], pad_val, dtype=dtype)
        if ret_mask:
            arr_m = np.zeros([bs, max_len], dtype=np.float32)
        else:
            arr_m = None
        for ii, vv in enumerate(inputs):
            _ll = min(len(vv), max_len)
            arr[ii, :_ll] = vv[:_ll]
            if ret_mask:
                arr_m[ii, :_ll] = 1.
        if ret_tensor:
            from mspx.nn import BK
            return BK.input_tensor(arr), BK.input_real(arr_m) if ret_mask else None
        else:
            return arr, arr_m
        # --

    # for 3d case
    @staticmethod
    def batch_3d(inputs: Sequence[Sequence[Sequence]], pad_val,
                 max_len1=None, max_len2=None, dtype=None, ret_mask=False, ret_tensor=False):
        bs = len(inputs)
        if max_len1 is None:
            max_len1 = max((len(z) for z in inputs), default=1)
        if max_len2 is None:
            max_len2 = max((len(b) for a in inputs for b in a), default=1)
        arr = np.full([bs, max_len1, max_len2], pad_val, dtype=dtype)
        if ret_mask:
            arr_m = np.zeros([bs, max_len1, max_len2], dtype=np.float32)
        else:
            arr_m = None
        for ii1, vv1 in enumerate(inputs):
            for ii2, vv2 in enumerate(vv1):
                _ll = min(len(vv2), max_len2)
                arr[ii1, ii2, :_ll] = vv2[:_ll]
                if ret_mask:
                    arr_m[ii1, ii2, :_ll] = 1.
        if ret_tensor:
            from mspx.nn import BK
            return BK.input_tensor(arr), BK.input_real(arr_m) if ret_mask else None
        else:
            return arr, arr_m
        # --

# --
# char indexer: from char indexes to token indexes
class CharIndexer:
    def __init__(self, full_char_idxes: List, offset_str: str, sent_tokens: List[List[str]], sent_positions: List[List], doc=None):
        self.full_char_idxes = full_char_idxes  # char-idx -> (sid, wid)
        self.offset_str = offset_str  # original str for indexing
        self.sent_tokens = sent_tokens  # List[sent] of List[token]
        self.sent_positions = sent_positions  # List[Sent] of List[Token] of (cstart, clen)
        self.doc = doc  # doc?

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
            zwarn(f"=> Cannot find span: {self.offset_str[char_idx:char_idx+char_len]}")
        elif ret_code != "":
            str0 = ' '.join(self.offset_str[char_idx:char_idx+char_len].split())
            str1 = ' '.join(self.sent_tokens[ret_posi[0]][ret_posi[1]:ret_posi[1]+ret_posi[2]])
            zwarn(f"=> Span mismatch ({ret_code}): {str0} ||| {str1}")
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
            m = Mention(widx, wlen, par=self.doc.sents[sid])
            m.info["_code"] = ret_code
            return m, ret_code

# --
# b mspx/data/inst/helper:?
