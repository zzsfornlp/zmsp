#

# annotate with nltk

__all__ = [
    "AnnotatorNltkConf", "AnnotatorNltk",
]

from typing import List, Union
from mspx.data.inst import Doc, Sent, NLTKTokenizer
from mspx.utils import zwarn
from .annotator import *

@AnnotatorConf.rd('nltk')
class AnnotatorNltkConf(AnnotatorConf):
    def __init__(self):
        super().__init__()
        # --
        self.ann_input_mode = "raw"  # input mode: raw / ssplit / tokenized
        self.ann_rm_text = True  # remove text after tokenization
        # --
        self.nltk_scheme = 'default'

@AnnotatorNltkConf.conf_rd()
class AnnotatorNltk(Annotator):
    def __init__(self, conf: AnnotatorNltkConf):
        super().__init__(conf)
        conf: AnnotatorNltkConf = self.conf
        self.ann_ff = getattr(self, f"ann_{conf.ann_input_mode}")
        self.toker = NLTKTokenizer(scheme=conf.nltk_scheme)

    def annotate(self, insts: List[Doc]):
        return self.ann_ff(insts)

    def ann_raw(self, insts: List[Doc]):
        _ann_rm_text = self.conf.ann_rm_text
        # --
        for doc in insts:  # note: process one by one!
            text = doc.get_text()
            all_tokens, all_token_spans, _ = self.toker.tokenize(text, return_posi_info=True)
            # --
            sent_positions = []
            for _tokens, _spans in zip(all_tokens, all_token_spans):
                if len(_tokens) == 0: continue
                sent_start_char, sent_end_char = _spans[0][0], _spans[-1][-1]
                new_s = Sent(_tokens, text=text[sent_start_char:sent_end_char])
                doc.add_sent(new_s)  # add sent
                sent_positions.append((sent_start_char, sent_end_char-sent_start_char))
                if _ann_rm_text:
                    new_s.build_text(None)
                else:
                    new_s.build_word_positions([(a, b-a) for a,b in _spans])
            if _ann_rm_text:
                doc.build_text(None)
            else:
                doc.build_sent_positions(sent_positions)  # put positions
        # --

    def ann_ssplit(self, insts: List[Doc]):
        _ann_rm_text = self.conf.ann_rm_text
        # --
        _toker = self.toker
        for d in insts:
            for s in d.sents:
                tokens = _toker.tokenize(s.get_text(), split_sent=False)
                s.build_words(tokens)
                if _ann_rm_text:
                    s.build_text(None)
        # --

    def ann_tokenized(self, insts: List[Doc]):
        pass
