#

# Doc and Sent
# note: the convention is only deleting par->item links for "del_*" and do "clear" when adding item to par!

__all__ = [
    "Doc", "Sent", "Token",
]

import itertools
from typing import List, Dict, Iterable, Union, Tuple
from mspx.utils import ZHelper, InfoField
from collections import defaultdict
from .base import DataInst
from .field import SeqField
from .tree import DepTree, PhraseTree
from .frame import Frame, Mention, ItemCollection

# Document
@DataInst.rd('doc')
class Doc(DataInst):
    def __init__(self, sents: Iterable['Sent'] = None, text: str = None, id: str = None):
        super().__init__(id=id)
        # --
        self.sents: List[Sent] = []  # list of Sent inside the Doc
        self.text: str = text  # original text
        self.sent_positions: List[Tuple[int, int]] = None  # sent char positions (cidx, clen)
        self._fmap: Dict[str, 'Frame'] = {}  # fid -> Frame
        self._frames = defaultdict(list)  # cate -> List[Frame]
        # --
        if sents:
            self.add_sents(sents)
        # --

    def to_dict(self, store_type=True):
        ret = super().to_dict(store_type)
        ret['sents'] = [v.to_dict(store_type=False) for v in self.sents]
        if len(self._fmap) > 0:  # if we have frames
            ret['_frames'] = {k: [v.to_dict(store_type=False) for v in vs] for k,vs in self._frames.items() if len(vs)>0}
        return ret

    def from_dict(self, data: Dict):  # also as "finish_from_dict"
        super().from_dict(data)
        # --
        # sents
        self.sents = []
        self.add_sents([Sent.create_from_dict(v) for v in data['sents']])
        # frames
        self._frames = defaultdict(list)
        _fs = data.get('_frames')
        if _fs is not None:
            for cate, vs in _fs.items():
                for v in vs:
                    frame = Frame.create_from_dict(v)
                    frame._cate = cate
                    if frame.mention is not None:  # mention -> sent
                        frame.mention.set_par(self.sents[frame.mention._sid])
                        delattr(frame.mention, '_sid')
                    self.add_frame(frame)
        # arglink
        for frame in self._fmap.values():
            _tmp_arglinks = frame.args
            frame.args = []
            for alink in _tmp_arglinks:  # arglink -> frame(s)
                alink._arg = self._fmap[alink._arg]
                frame.add_arglink(alink)
        # --

    def __repr__(self):
        return f"DOC({self.id},S={len(self.sents)})"

    def __len__(self):
        return len(self.sents)

    def remove_all_text(self):  # remove all info with regard to original texts
        for sent in self.sents:
            sent.build_text(None)
            sent.build_word_positions(None)
        self.build_text(None)
        self.build_sent_positions(None)

    def build_text(self, text: str):
        self.text = text

    def get_text(self):
        if self.text is not None:
            return self.text
        else:
            return "\n".join([z.get_text() for z in self.sents])

    def build_sent_positions(self, sent_positions: List[tuple]):
        if sent_positions is not None:
            assert len(sent_positions) == len(self.sents), "Error: unmatched seq length!"
        self.sent_positions = sent_positions  # here, directly assign!
        return self.sent_positions

    def get_sent_positions(self, save=False):
        if self.sent_positions is not None:
            return self.sent_positions
        else:
            texts = [z.get_text() for z in self.sents]
            _positions = []
            _curr = 0
            for t in texts:
                _positions.append((_curr, len(t)))
                _curr += len(t) + 1  # extra one for '\n'
            if save:
                self.build_sent_positions(_positions)
            return _positions
        # --

    # note: only allow append sent!
    def add_sent(self, sent: 'Sent'):
        sent.clear_cached_vals()
        self.sents.append(sent)
        sent.set_par(self)
        return sent

    def clear_sents(self):
        self.sents = []  # simply reset!

    def _clear_frame_col(self, frame: Frame):
        # clear cached col due to changes; todo(+W): simply clear them all!
        if frame.sent is not None:
            frame.sent._frame_cols.clear()
        # --

    def add_frame(self, frame: Frame, force_re_id=False):
        frame.clear_cached_vals()
        _id = frame.id
        if force_re_id or _id is None:
            _id = ZHelper.get_new_key(self._fmap, 'f')
            frame.set_id(_id)
        assert _id not in self._fmap
        self._fmap[_id] = frame
        frame.set_par(self)
        self._frames[frame.cate].append(frame)
        self._clear_frame_col(frame)
        return frame

    def del_frame(self, frame: Frame):
        del self._fmap[frame.id]
        self._frames[frame.cate].remove(frame)
        self._clear_frame_col(frame)

    def find_frame(self, frame_id: str):
        return self._fmap.get(frame_id, None)

    def yield_frames(self, *other_filters, label_prefix: str = None, cates=None):
        if cates is None:
            cands = self._fmap.values()
        else:
            if isinstance(cates, str):
                cates = [cates]
            cands = itertools.chain.from_iterable((self._frames.get(cc, []) for cc in cates))
        yield from ItemCollection.yield_items(cands, *other_filters, label_prefix=label_prefix)

    def get_frames(self, *args, **kwargs):  # make a copy for the list
        return list(self.yield_frames(*args, **kwargs))

    def get_frame_cates(self):
        return list(self._frames.keys())

    # special method to re-assign all sids
    def assign_sids(self):
        for i, s in enumerate(self.sents):
            s._sid = i

    # shortcuts
    def add_sents(self, sents: Iterable['Sent']): [self.add_sent(s) for s in sents]
    def add_frames(self, frames: Iterable[Frame], **kwargs): [self.add_frame(f, **kwargs) for f in frames]
    def del_frames(self, frames: Iterable[Frame]): [self.del_frame(f) for f in frames]
    # --

    # --
    # note: for simplicity, make interfaces for "SentPair" & "Singleton"
    @staticmethod
    def make_sent_pair(src: 'Sent', trg: 'Sent', **kwargs): return Doc([src, trg], **kwargs)
    @property
    def sent_src(self): return self.sents[0]
    @property
    def sent_trg(self): return self.sents[1]
    @property
    def sent_single(self): return self.sents[0]
    # --

    @staticmethod
    def get_sent_single(d):
        if isinstance(d, Sent): return d
        elif isinstance(d, Doc): return d.sent_single
        else: return d.sent

    @staticmethod
    def merge_docs(docs: Iterable['Doc'], force_re_id=True, new_doc_id=None):
        ret = Doc(id=new_doc_id)
        for d in docs:  # note: simply add them all, the old docs are not well-handled and should not be used!
            ret.add_sents(d.sents)
            ret.add_frames(d.yield_frames(), force_re_id=force_re_id)
        return ret

    def split_docs(self, force_re_id=True):
        ret = []
        for s in self.sents:  # note: this has severe risks of dangling arg-links!
            s_frames = s.get_frames()
            d = Doc([s])
            self.del_frames(s_frames)
            d.add_frames(s_frames, force_re_id=force_re_id)
            ret.append(d)
        return ret

# Sentence
@DataInst.rd('sent')
class Sent(DataInst):
    def __init__(self, words: Iterable[str] = None, text: str = None, par: Doc = None, make_singleton_doc=False):
        super().__init__(par=par)
        # --
        self._sid: int = None  # to be set by Doc if needed
        self.text: str = text  # original str
        self.word_positions: List[Tuple[int, int]] = None  # word positions (cidx, clen)
        # --
        # seq
        self.seq_word: SeqField = None if words is None else SeqField(words, par=self)  # tokens(words)
        self.seq_lemma: SeqField = None
        self.seq_upos: SeqField = None
        # --
        # tree
        self.tree_dep: DepTree = None
        self.tree_phrase: PhraseTree = None
        # --
        # frames & graph
        # --
        # cached tokens
        self._cached_toks: List[Token] = None  # ordinary ones
        self._cached_htoks: List[Token] = None  # +1 for ArtiRoot
        # specially set prev_sent and next_sent
        self._prev_sent = None
        self._next_sent = None
        # frames
        self._frame_cols = {}  # prefix -> Col
        # --
        if make_singleton_doc:  # make a container for frames
            self.make_singleton_doc()
        # --

    def __repr__(self):
        return f"SENT({self.id},L={len(self)})"

    def __len__(self):
        return len(self.seq_word) if self.seq_word is not None else 0

    def clear_cached_vals(self):
        super().clear_cached_vals()
        self._cached_toks = None
        self._cached_htoks = None
        for z in [self.seq_word, self.seq_upos, self.seq_lemma]:
            if z is not None:
                z.clear_cached_vals()
        self._frame_cols.clear()
        self._sid = None
        # --

    @classmethod
    def _info_fields(cls):
        return {'seq_word': InfoField(inner_type=SeqField),
                'seq_lemma': InfoField(inner_type=SeqField),
                'seq_upos': InfoField(inner_type=SeqField),
                'tree_dep': InfoField(inner_type=DepTree),
                'tree_phrase': InfoField(inner_type=PhraseTree)}

    def finish_from_dict(self):
        # special handling!
        for z in [self.seq_word, self.seq_upos, self.seq_lemma, self.tree_dep, self.tree_phrase]:
            if z is not None:
                z.set_par(self)
        # --

    def make_singleton_doc(self, **kwargs):
        assert self.par is None
        doc = Doc([self], **kwargs)
        return doc

    @property
    def sid(self):  # try to find this sentence's sid in Doc!
        if self._sid is None:
            d: Doc = self.doc
            if d is not None:
                d.assign_sids()
        return self._sid

    @property
    def dsids(self):  # shortcut!
        return (self.doc.id, self.sid)

    @property
    def sig(self):
        return self.sid

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

    def get_sent_win(self, win: int):  # sentence window
        center = self
        all_sents = []
        ss = center
        for ii in range(win):
            ss = ss.prev_sent
            if ss is None: break
            all_sents.append(ss)
        all_sents.reverse()
        all_sents.append(center)
        ss = center
        for ii in range(win):
            ss = ss.next_sent
            if ss is None: break
            all_sents.append(ss)
        return all_sents

    @classmethod
    def assign_prev_next(cls, s0, s1):
        assert s0.next_sent is None and s1.prev_sent is None, "Already there!!"
        s0._next_sent = s1
        s1._prev_sent = s0

    @property
    def tokens(self):
        if self._cached_toks is None:
            self._cached_toks = [Token(i, par=self) for i in range(len(self.seq_word))]
        return self._cached_toks

    @property
    def htokens(self):
        if self._cached_htoks is None:
            plain_tokens = self.tokens
            root_tok = Token(-1, par=self)
            self._cached_htoks = [root_tok] + plain_tokens  # +1 for ArtiRoot
        return self._cached_htoks

    def get_tokens(self, start: int = 0, end: int = None):  # return all tokens
        return self.tokens[start:end]

    # =====
    # building/adding various components

    def build_text(self, text: str):
        self.text = text

    def get_text(self):
        if self.text is not None:
            return self.text
        else:
            return " ".join([z for z in self.seq_word.vals])

    def build_word_positions(self, word_positions: List[tuple]):
        if word_positions is not None:
            assert len(word_positions) == len(self.seq_word), "Error: unmatched seq length!"
        self.word_positions = word_positions  # here, directly assign!
        return self.word_positions

    def get_word_positions(self, save=False):
        if self.word_positions is not None:
            return self.word_positions
        else:
            _curr = self.doc.get_sent_positions(save)[self.sid][0]  # note: doc-level
            _positions = []
            for t in self.seq_word.vals:
                _positions.append((_curr, len(t)))
                _curr += len(t) + 1  # extra one for ' '
            if save:
                self.build_word_positions(_positions)
            return _positions
        # --

    def build_words(self, words: List[str]):
        self.seq_word = SeqField(words, par=self)
        self.clear_cached_vals()  # this is like changing a sentence
        return self.seq_word

    def build_lemmas(self, lemmas: List[str]):
        assert len(lemmas) == len(self.seq_word), "Error: unmatched seq length!"
        self.seq_lemma = SeqField(lemmas, par=self)
        return self.seq_lemma

    def build_uposes(self, uposes: List[str]):
        assert len(uposes) == len(self.seq_word), "Error: unmatched seq length!"
        self.seq_upos = SeqField(uposes, par=self)
        return self.seq_upos

    def build_dep_tree(self, heads: List[int], labels: List[str] = None):
        self.tree_dep = DepTree(heads, labels, par=self)
        return self.tree_dep

    def build_phrase_tree(self, parse_str: str):
        self.tree_phrase = PhraseTree(parse_str, par=self)
        return self.tree_phrase

    # --
    # frames related

    # frame collections
    def get_frame_col(self, key: str = None, *other_filters, label_prefix: str = None, cates: str = None):
        if len(other_filters) > 0:
            assert key is not None
        if key is None:
            key = (label_prefix, cates)
        ret = self._frame_cols.get(key)
        if ret is None:
            ret = ItemCollection.create(
                self.doc.yield_frames(*other_filters, label_prefix=label_prefix, cates=cates), in_sent=self)
            self._frame_cols[key] = ret
        return ret

    def yield_frames(self, *other_filters, label_prefix: str = None, cates=None):
        yield from ItemCollection.yield_items(
            self.doc.yield_frames(*other_filters, label_prefix=label_prefix, cates=cates), in_sent=self)

    def get_frames(self, *args, **kwargs):  # make a copy for the list
        return list(self.yield_frames(*args, **kwargs))

    # make a new in-sent frame
    def make_frame(self, widx: int, wlen: int, label: str, cate: str, **frame_kwargs):
        m = Mention(widx, wlen, par=self)
        f = Frame(m, label=label, cate=cate, **frame_kwargs)
        self.doc.add_frame(f)
        return f

    # --
    @staticmethod
    def combine_sents(sents: Iterable['Sent']):
        sents = list(sents)
        ret = Sent()
        _offsets = [0]
        for s in sents:
            _offsets.append(_offsets[-1]+len(s))
        ret.info['combine_offsets'] = _offsets
        ret.cache['combine_sents'] = sents  # store cache
        # --
        # note: directly assign
        if all(s.seq_word is not None for s in sents):
            ret.seq_word = SeqField.combine([s.seq_word for s in sents])
        if all(s.seq_lemma is not None for s in sents):
            ret.seq_lemma = SeqField.combine([s.seq_lemma for s in sents])
        if all(s.seq_upos is not None for s in sents):
            ret.seq_upos = SeqField.combine([s.seq_upos for s in sents])
        # todo(+N): combine other fields?
        return ret

# Token (on the fly)
@DataInst.rd('tok')
class Token(DataInst):
    def __init__(self, widx: int = None, par: Sent = None):
        super().__init__(par=par)
        self.widx: int = widx  # word idx

    @property
    def sig(self):
        return self.widx

    @property
    def is_arti_root(self):
        return self.widx < 0

    def __repr__(self):
        return f"Token({self.widx}): `{self.word}'"

    def _get_one(self, seq, arti_root_ret, noseq_ret):
        _widx = self.widx
        if _widx < 0:
            return arti_root_ret
        elif seq is not None:
            return seq.vals[_widx]
        else:
            return noseq_ret

    @property
    def word(self): return self._get_one(self.sent.seq_word, '<R>', None)
    @property
    def lemma(self): return self._get_one(self.sent.seq_lemma, '<R>', None)
    @property
    def upos(self): return self._get_one(self.sent.seq_upos, '<R>', None)

    @property
    def deplab(self):
        _tree = self.sent.tree_dep
        return None if _tree is None else self._get_one(_tree.seq_label, '<R>', None)

    @property
    def head_tok(self):
        _tree = self.sent.tree_dep
        if _tree is None or self.is_arti_root:
            return None
        h = _tree.seq_head.vals[self.widx]
        ret = self.sent.htokens[h]  # including the +1 offset
        return ret

    @property
    def head_idx(self):
        h = self.head_tok
        return -1 if h is None else h.widx

    @property
    def ch_toks(self):
        _tree = self.sent.tree_dep
        if _tree is None: return None
        _toks = self.sent.tokens
        _chs_lists = _tree.chs_lists
        ret = [_toks[i] for i in _chs_lists[self.widx+1]]  # note: +1 since chs_lists include +1 offset
        return ret
