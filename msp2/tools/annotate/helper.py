#

# some common helpers

from typing import List, Union
from msp2.data.inst import DataInstance, Doc, Sent, yield_sents

# =====
# input strategies
class InputHelper:
    def prepare(self, insts: List[Union[Doc, Sent]]) -> List:
        raise NotImplementedError()

    @staticmethod
    def get_input_helper(input_mode: str):
        return {"raw": RawInputHelper, "ssplit": SsplitInputHelper, "tokenized": TokenizedInputHelper}[input_mode]()

# simply get all the texts
class RawInputHelper:
    def prepare(self, insts: List[Union[Doc, Sent]]):
        return [(inst, inst.get_text()) for inst in insts]  # List[Inst, str]

# get list fo sents
class SsplitInputHelper:
    def prepare(self, insts: List[Union[Doc, Sent]]):
        ret_sents, ret_texts = [], []
        for s in yield_sents(insts):
            ret_sents.append(s)
            ret_texts.append(s.get_text())
        return [(ret_sents, ret_texts)]  # List[List[Sent], List[str]]

# already tokenized, get all tokens
class TokenizedInputHelper:
    def prepare(self, insts: List[Union[Doc, Sent]]):
        ret_sents, ret_toks = [], []
        for s in yield_sents(insts):
            ret_sents.append(s)
            ret_toks.append(s.seq_word.vals)
        return [(ret_sents, ret_toks)]  # List[List[Sent], List[Toks(List[str])]]
