#

# Doc and Sent

__all__ = [
    "Doc", "Sent", "Token",
]

from typing import List, Dict, Iterable, Union, Tuple
from itertools import chain
from .base import DataInstance, DataInstanceComposite, SubInfo
from .helper import InDocInstance, InSentInstance
from .field import PlainSeqField
from .frame import Frame, Mention
from .tree import DepTree, PhraseTree

# =====
# Composite ones

# Document
class Doc(DataInstanceComposite):
    @classmethod
    def _get_sub_map(cls) -> Dict[str, SubInfo]:
        return {
            "text": SubInfo(str, df_val=None),
            "sent_positions": SubInfo(list, df_val=None),  # positions for words: List[(widx, wlen)]
            "sents": SubInfo(Sent, wrapper_type=list, needs_reg=True, reg_sname='s', df_val=[]),
        }

    def __init__(self, _id: str = None, _par: 'DataInstance' = None):
        super().__init__(_id, _par)
        # --
        self.text: str = None  # original text
        self.sent_positions: List[Tuple[int, int]] = None
        self.sents: List[Sent] = []  # list of Sent inside the Doc

    @classmethod
    def create(cls, sents: Iterable['Sent']=None, text=None, id: str = None, par: 'DataInstance' = None):
        inst: Doc = super().create(id, par)
        if text is not None:
            inst.build_text(text)
        if sents is not None:  # then add the list
            for s in sents:
                inst.add_sent(s)
        return inst

    def __repr__(self):
        return f"DOC({self.id},S={len(self.sents)})"

    def build_text(self, text: str):
        self.text = text

    def get_text(self):
        if self.text is not None:
            return self.text
        else:
            return "\n".join([z.get_text() for z in self.sents])

    def build_sent_positions(self, sent_positions: List[tuple]):
        assert len(sent_positions) == len(self.sents), "Error: unmatched seq length!"
        self.sent_positions = sent_positions  # here, directly assign!
        return self.sent_positions

    # add sent
    def add_sent(self, sent: 'Sent'):
        self.add_and_reg_inst(sent, 's')
        self.sents.append(sent)
        return sent

    # clean all sents and idxes
    def clear_sents(self):
        self.sents.clear()
        self.clear_insts('s')

    # special method
    def assign_sids(self):
        for i, s in enumerate(self.sents):
            s._sid = i

# Sentence
# todo(note): be careful about the multi-inheritance!!
class Sent(DataInstanceComposite, InDocInstance):
    @classmethod
    def _get_sub_map(cls) -> Dict[str, SubInfo]:
        return {
            # text
            "text": SubInfo(str, df_val=None),  # original str
            "word_positions": SubInfo(list, df_val=None),  # positions for words: List[(widx, wlen)]
            # sequences
            "seq_word": SubInfo(PlainSeqField),
            "seq_lemma": SubInfo(PlainSeqField),
            "seq_upos": SubInfo(PlainSeqField),
            # trees
            "tree_dep": SubInfo(DepTree),
            "tree_phrase": SubInfo(PhraseTree),
            # --
            "entity_fillers": SubInfo(Frame, wrapper_type=list, needs_reg=True, reg_sname='ef', df_val=[]),
            "events": SubInfo(Frame, wrapper_type=list, needs_reg=True, reg_sname='evt', df_val=[]),
        }

    def __init__(self, _id: str = None, _par: 'DataInstance' = None):
        super().__init__(_id, _par)
        # --
        self._sid: int = None  # to be set by Doc if needed
        self.text: str = None  # original str
        self.word_positions: List[Tuple[int, int]] = None  # word positions
        # --
        # seq
        self.seq_word: PlainSeqField = None  # tokens(words)
        self.seq_lemma: PlainSeqField = None
        self.seq_upos: PlainSeqField = None
        # --
        # tree
        self.tree_dep: DepTree = None
        self.tree_phrase: PhraseTree = None
        # --
        # frames
        self.entity_fillers: List[Frame] = []
        self.events: List[Frame] = []
        # --
        # cached tokens
        self._cached_toks: List[Token] = None  # ordinary ones
        self._cached_htoks: List[Token] = None  # +1 for ArtiRoot
        # externally set prev_sent and next_sent
        self._prev_sent = None
        self._next_sent = None

    @classmethod
    def create(cls, words: List[str] = None, text: str = None, id: str = None, par: 'DataInstance' = None):
        inst: Sent = super().create(id, par)
        # build components
        if text is not None:
            inst.build_text(text)
        if words is not None:
            inst.build_words(words)
        return inst

    def __repr__(self):
        return f"SENT({self.id},L={len(self)})"

    def __len__(self):
        return len(self.seq_word)

    def clear_caches(self):
        self._cached_toks = None
        self._cached_htoks = None

    @property
    def sid(self):  # try to find this sentence's sid in Doc!
        if self._sid is None:
            d: Doc = self.doc
            if d is not None:
                d.assign_sids()
        return self._sid

    @property
    def sent(self):
        return self  # shortcut!

    @property
    def prev_sent(self):
        if self._prev_sent is None:  # try to get
            doc = self.doc
            if doc is not None:
                sid = self.sid
                if sid > 0:
                    self._prev_sent = doc.sents[sid-1]
        return self._prev_sent

    @property
    def next_sent(self):
        if self._next_sent is None:  # try to get
            doc = self.doc
            if doc is not None:
                sid = self.sid
                if sid < len(doc.sents)-1:
                    self._next_sent = doc.sents[sid+1]
        return self._next_sent

    @classmethod
    def assign_prev_next(cls, s0, s1):
        assert s0.next_sent is None and s1.prev_sent is None, "Already there!!"
        s0._next_sent = s1
        s1._prev_sent = s0

    @property
    def tokens(self):
        if self._cached_toks is None:
            self._cached_toks = [Token.create(self, i, par=self) for i in range(len(self.seq_word))]
        return self._cached_toks

    @property
    def htokens(self):
        if self._cached_htoks is None:
            plain_tokens = self.tokens
            root_tok = Token.create(self, -1, par=self)
            self._cached_htoks = [root_tok] + plain_tokens  # +1 for ArtiRoot
        return self._cached_htoks

    def get_tokens(self, start: int = 0, end: int = None):  # return all tokens
        return self.tokens[start:end]

    # =====
    # building/adding various components

    def build_text(self, text: str):
        self.text = text
        self.clear_caches()

    def get_text(self):
        if self.text is not None:
            return self.text
        else:
            return " ".join([z for z in self.seq_word.vals])

    def build_word_positions(self, word_positions: List[tuple]):
        assert len(word_positions) == len(self.seq_word), "Error: unmatched seq length!"
        self.word_positions = word_positions  # here, directly assign!
        self.clear_caches()
        return self.word_positions

    def build_words(self, words: List[str]):
        self.seq_word = PlainSeqField.create(vals=words, par=self)
        self.clear_caches()
        return self.seq_word

    def build_lemmas(self, lemmas: List[str]):
        assert len(lemmas) == len(self.seq_word), "Error: unmatched seq length!"
        self.seq_lemma = PlainSeqField.create(vals=lemmas, par=self)
        self.clear_caches()
        return self.seq_lemma

    def build_uposes(self, uposes: List[str]):
        assert len(uposes) == len(self.seq_word), "Error: unmatched seq length!"
        self.seq_upos = PlainSeqField.create(vals=uposes, par=self)
        self.clear_caches()
        return self.seq_upos

    def build_dep_tree(self, heads: List[int], labels: List[str] = None):
        cur_len = len(self.seq_word)
        assert len(heads) == cur_len and (labels is None or len(labels)==cur_len), "Error: unmatched tree length!"
        self.tree_dep = DepTree.create(heads, labels, sent=self, par=self)
        self.clear_caches()
        return self.tree_dep

    def build_phrase_tree(self, *args, **kwargs):
        self.clear_caches()
        raise NotImplementedError()

    # =====
    # frames

    _FRAME_MAP = {"ef": "entity_fillers", "evt": "events"}

    def _get_frame_list(self, tag: str):
        tag = Sent._FRAME_MAP.get(tag, tag)
        return getattr(self, tag)

    def _set_frame_list(self, tag: str, flist: List):
        tag = Sent._FRAME_MAP.get(tag, tag)
        setattr(self, tag, flist)

    # get frames
    def get_frames(self, tag: str, copy=False):
        flist: List[Frame] = self._get_frame_list(tag)
        if copy and flist is not None:
            return flist.copy()
        else:
            return flist

    def set_frames(self, tag: str, flist: List, direct=False):
        self.delete_frames(tag)
        if direct:
            self._set_frame_list(tag, flist)
        else:
            orig_flist = self._get_frame_list(tag)
            orig_flist.clear()
            orig_flist.extend(flist)

    # add an existing frame
    def add_frame(self, f: 'Frame', tag: str):
        self.add_and_reg_inst(f, tag)
        flist = self._get_frame_list(tag)
        flist.append(f)
        return f

    # make a new frame and (by default) add it
    def make_frame(self, widx: int, wlen: int, tag: str, adding=True, **frame_kwargs):
        m = Mention.create(self, widx, wlen)  # no id/par
        f = Frame.create(mention=m, **frame_kwargs)
        if adding:
            self.add_frame(f, tag)
        return f

    # delete one frame
    def delete_frame(self, f: Frame, tag: str):
        # first delete args
        f.clear_args()
        f.clear_as_args()
        # then delete self
        flist = self._get_frame_list(tag)
        if f in flist:  # todo(+W): how to deal with double delete??
            flist.remove(f)
        self.del_inst(f)

    # delete all frames
    def delete_frames(self, tag: str):
        flist = self._get_frame_list(tag)
        if flist is None:  # set empty frames
            self._set_frame_list(tag, [])
        else:
            for f in list(flist):  # remember to copy!
                self.delete_frame(f, tag)
            flist.clear()
            self.clear_insts(tag)

    # mark no frames
    def mark_no_frames(self, tag: str):
        self.delete_frames(tag)
        self._set_frame_list(tag, None)

    # --
    # shortcuts
    def add_entity_filler(self, ef: 'Frame'): return self.add_frame(ef, 'ef')
    def add_event(self, evt: 'Frame'): return self.add_frame(evt, 'evt')
    def make_entity_filler(self, widx: int, wlen: int, adding=True, **frame_kwargs):
        return self.make_frame(widx, wlen, 'ef', adding, **frame_kwargs)
    def make_event(self, widx: int, wlen: int, adding=True, **frame_kwargs):
        return self.make_frame(widx, wlen, 'evt', adding, **frame_kwargs)
    def clear_entity_fillers(self): self.delete_frames('ef')
    def clear_events(self): self.delete_frames('evt')
    def mark_no_entity_fillers(self): self.mark_no_frames('ef')
    def mark_no_events(self): self.mark_no_frames('evt')

# Token
class Token(InSentInstance):
    @classmethod
    def _get_sub_map(cls) -> Dict[str, SubInfo]:
        return {}

    def __init__(self, _id: str = None, _par: 'DataInstance' = None):
        super().__init__(_id, _par)
        # --
        self.widx: int = None  # word idx
        self._word: str = None
        self._upos: str = None
        self._lemma: str = None
        self._deplab: str = None
        self._htok: Token = None  # cached head tok
        self._chtoks: List[Token] = None  # cached children toks

    @classmethod
    def create(cls, sent: Sent, widx: int, id: str = None, par: 'DataInstance' = None):
        inst: Token = super().create(id, par)
        inst._sent = sent  # diectly assign
        # --
        if widx == -1:  # special one as ArtiRoot, usually we will not directly create this one!
            inst._word = inst._upos = inst._lemma = inst._deplab = "<R>"
        else:
            # check range
            slen = len(sent)
            assert widx>=0 and widx<slen, "Bad span idxes for the Mention!"
        inst.widx = widx
        return inst

    @property
    def is_arti_root(self):
        return self.widx < 0

    def __repr__(self):
        return f"Token({self.widx}): `{self.word}'"

    @property
    def word(self):
        if self._word is None:
            self._word = self.sent.seq_word.vals[self.widx]
        return self._word

    @property
    def lemma(self):
        if self._lemma is None and self.sent.seq_lemma is not None:
            self._lemma = self.sent.seq_lemma.vals[self.widx]
        return self._lemma

    @property
    def upos(self):
        if self._upos is None and self.sent.seq_upos is not None:
            self._upos = self.sent.seq_upos.vals[self.widx]
        return self._upos

    @property
    def deplab(self):
        if self._deplab is None and self.sent.tree_dep is not None:
            self._deplab = self.sent.tree_dep.seq_label.vals[self.widx]
        return self._deplab

    @property
    def head_tok(self):
        if self._htok is None and self.sent.tree_dep is not None:  # note: -1 for real widx, ROOT will handled in 'create'
            if self.is_arti_root:
                return None  # no head_tok for this!
            h = self.sent.tree_dep.seq_head.vals[self.widx]
            self._htok = self.sent.htokens[h]  # including the +1 offset
        return self._htok

    @property
    def head_idx(self):
        h = self.head_tok
        return -1 if h is None else h.widx

    @property
    def ch_toks(self):
        if self._chtoks is None and self.sent.tree_dep is not None:  # note: +1 since chs_lists include +1 offset
            sent_toks = self.sent.tokens
            self._chtoks = [sent_toks[i] for i in self.sent.tree_dep.chs_lists[self.widx+1]]
        return self._chtoks
