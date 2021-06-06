#

# basic formator

__all__ = [
    "DataFormator", "ZJsonDataFormator", "PlainDocDataFormator", "PlainSentDataFormator",
]

from typing import Type, Callable, Union, Dict
import json
from msp2.utils import Registrable
from msp2.data.inst import DataInstance, Doc, Sent

# =====
# python-obj <-> DataInstance
class DataFormator(Registrable):
    def to_obj(self, inst: DataInstance) -> object: raise NotImplementedError()
    def from_obj(self, s: object) -> DataInstance: raise NotImplementedError()

# very basic json one
@DataFormator.reg_decorator("zjson")
class ZJsonDataFormator(DataFormator):
    def __init__(self, cls: Type = None):
        assert cls is None or issubclass(cls, DataInstance)
        self.cls = cls  # if None, then guessing

    def to_obj(self, inst: DataInstance) -> str:
        return json.dumps(inst.to_json(), ensure_ascii=False)

    def from_obj(self, s: str):
        d = json.loads(s)
        if self.cls is None:
            # todo(+N): guess according to simple feature
            if "sents" in d:
                guessing_cls = Doc
            else:
                guessing_cls = Sent
        else:
            guessing_cls = self.cls
        # -----
        ret = guessing_cls.cls_from_json(d)
        ret.deref()  # deref at top reading level
        return ret

# shortcuts
DataFormator.reg("zjson_doc", lambda: ZJsonDataFormator(cls=Doc))
DataFormator.reg("zjson_sent", lambda: ZJsonDataFormator(cls=Sent))

# some simple plain text reader
@DataFormator.reg_decorator("plain_sent")
class PlainSentDataFormator(DataFormator):
    def __init__(self, do_tok_sep: bool = True, tok_sep: str = None):
        self.do_tok_sep = do_tok_sep
        self.tok_sep = tok_sep

    def to_obj(self, inst: Sent) -> str:
        if self.do_tok_sep:
            sep = " " if self.tok_sep is None else self.tok_sep
            return sep.join(inst.seq_word.vals)
        else:
            return inst.get_text()

    def from_obj(self, s: str):
        ret = Sent.create(text=s)
        if self.do_tok_sep:  # simple split
            words = s.split(self.tok_sep)
            ret.build_words(words)
        return ret

@DataFormator.reg_decorator("plain_doc")
class PlainDocDataFormator(DataFormator):
    def __init__(self, do_sent_sep: bool = True, sent_sep: str = "\n", do_tok_sep: bool = True, tok_sep: str = None):
        self.do_sent_sep = do_sent_sep
        self.sent_sep = sent_sep
        self.sent_formator = PlainSentDataFormator(do_tok_sep=do_tok_sep, tok_sep=tok_sep)

    def to_obj(self, doc: Doc) -> str:
        if self.do_sent_sep:
            sent_strs = [self.sent_formator.to_obj(s) for s in doc.sents]
            return self.sent_sep.join(sent_strs)
        else:
            return doc.get_text()

    def from_obj(self, s: str):
        ret = Doc.create(text=s)
        if self.do_sent_sep:
            s = s.strip(self.sent_sep)  # exclude boundary "\n"
            for line in s.split(self.sent_sep):
                sent = self.sent_formator.from_obj(line)
                ret.add_sent(sent)
        return ret

# shortcuts
DataFormator.reg("raw_sent", lambda: PlainSentDataFormator(do_tok_sep=False))
DataFormator.reg("raw_doc", lambda: PlainDocDataFormator(do_sent_sep=False, do_tok_sep=False))
DataFormator.reg("raw_doc_splitline", lambda: PlainDocDataFormator(do_sent_sep=True, do_tok_sep=False))
