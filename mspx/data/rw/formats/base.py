#

# basic formator

__all__ = [
    "DataFormator", "ZJsonDataFormator", "PlainDocDataFormator", "PlainSentDataFormator",
]

from typing import Type, List, Union, Dict
import json
from mspx.data.inst import Doc, Sent, DataInst
from mspx.data.stream import FWrapperStreamer, FWrapperDumper
from mspx.utils import Registrable

# =====
# python-obj <-> DataInstance
@Registrable.rd('DF')
class DataFormator(Registrable):
    def to_obj(self, inst: DataInst) -> object: raise NotImplementedError()
    def from_obj(self, s: object) -> DataInst: raise NotImplementedError()

    def yield_objs(self, gen):
        for d in gen:
            yield self.from_obj(d)

# very basic json one
@DataFormator.rd('zjson')
class ZJsonDataFormator(DataFormator):
    def __init__(self, cls: Type = None):
        assert cls is None or issubclass(cls, DataInst)
        self.cls = cls  # if None, then guessing

    def to_obj(self, inst: DataInst) -> str:
        return json.dumps(inst.to_dict(), ensure_ascii=False)

    def from_obj(self, s: str):
        d = json.loads(s)
        if self.cls is None:
            guessing_cls = Doc
        else:
            guessing_cls = self.cls
        ret = guessing_cls.create_from_dict(d)
        return ret

# some simple plain text reader
@DataFormator.rd("plain_sent")
class PlainSentDataFormator(DataFormator):
    def __init__(self, do_tok_sep=True, tok_sep: str = None, do_sent_strip=True):
        self.do_tok_sep = do_tok_sep
        self.tok_sep = tok_sep
        self.do_sent_strip = do_sent_strip

    def to_obj(self, inst: Union[Sent, Doc]) -> str:
        inst = Doc.get_sent_single(inst)
        if self.do_tok_sep:
            sep = " " if self.tok_sep is None else self.tok_sep
            ret = sep.join(inst.seq_word.vals)
        else:
            ret = inst.get_text()
        if self.do_sent_strip:
            ret = ret.strip()
        return ret

    def sent_from_obj(self, s: str):
        if self.do_sent_strip:
            s = s.strip()
        sent = Sent(text=s)
        if self.do_tok_sep:  # simple split
            words = s.split(self.tok_sep)
            sent.build_words(words)
        return sent

    def from_obj(self, s: str):
        sent = self.sent_from_obj(s)
        doc = sent.make_singleton_doc()  # note: make a doc!
        return doc

@DataFormator.rd("plain_doc")
class PlainDocDataFormator(DataFormator):
    def __init__(self, do_sent_sep=True, sent_sep="\n", do_doc_strip=True, **sent_kwargs):
        self.do_sent_sep = do_sent_sep
        self.sent_sep = sent_sep
        self.sent_formator = PlainSentDataFormator(**sent_kwargs)
        self.do_doc_strip = do_doc_strip

    def to_obj(self, doc: Doc) -> str:
        if self.do_sent_sep:
            sent_strs = [self.sent_formator.to_obj(s) for s in doc.sents]
            ret = self.sent_sep.join(sent_strs)
        else:
            ret = doc.get_text()
        if self.do_doc_strip:
            ret = ret.strip()
        return ret

    def from_obj(self, s: str):
        if self.do_doc_strip:
            s = s.strip()
        ret = Doc(text=s)
        if self.do_sent_sep:
            for line in s.split(self.sent_sep):
                sent = self.sent_formator.sent_from_obj(line)
                ret.add_sent(sent)
        return ret

@DataFormator.rd("list_doc")
class ListDocDataFormator(DataFormator):
    def __init__(self, sent_formator=None):
        if sent_formator is None:
            sent_formator = PlainSentDataFormator()  # by default, this simple one!
        self.sent_formator = sent_formator

    def to_obj(self, doc: Doc) -> List:
        ret = [self.sent_formator.to_obj(s) for s in doc.sents]
        return ret

    def from_obj(self, ss: List):
        ret = Doc()
        for line in ss:
            sent = self.sent_formator.sent_from_obj(line)
            ret.add_sent(sent)
        return ret

# --
# shortcuts
DataFormator.reg((lambda: PlainSentDataFormator(do_tok_sep=False, do_sent_strip=False)), "raw_sent")
DataFormator.reg((lambda: PlainDocDataFormator(do_sent_sep=False, do_doc_strip=False)), "raw_doc")
DataFormator.reg((lambda: PlainDocDataFormator(do_sent_sep=True, do_tok_sep=False, do_sent_strip=False)), "raw_doc_splitline")
DataFormator.reg((lambda: PlainDocDataFormator(sent_sep="|||")), "doc3b")
# --
