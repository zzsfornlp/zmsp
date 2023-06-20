#

# Mentions and Frames

__all__ = [
    "Mention", "Frame", "ArgLink", "ItemCollection",
]

from typing import Tuple, List, Dict, Iterable, Union
from mspx.utils import InfoField, zwarn
from .base import DataInst

# =====
# Mention and Relation and Semantic-Frames

# Mention to locate the surface range (in a sent)
# todo(+N): currently we only support continuous span ranged mention
@DataInst.rd('ment')
class Mention(DataInst):
    def __init__(self, widx=None, wlen=None, shidx=None, par: 'Sent' = None):
        super().__init__(par=par)
        # --
        self.widx: int = None  # starting word idx
        self.wlen: int = None  # length in words
        self.shidx: int = None  # single-head word idx
        # --
        if widx is not None:
            self.set_span(widx, wlen)
        if shidx is not None:
            self.set_span(shidx, 1, shead=True)
        # --

    def to_dict(self, store_type=True):
        ret = super().to_dict(store_type=store_type)
        ret['_sid'] = self.sid
        return ret

    def __repr__(self):
        return f"M({self.sid}/[{self.widx},{self.wlen}]): `{self.text}'"

    # =====
    # used for comparison
    def __eq__(self, other: 'Mention'):
        return self.par is not None and self.get_sig() == other.get_sig()

    @property
    def sig(self):
        return (self.widx, self.wlen)  # full span!

    @property
    def sid(self):
        return self.sent.sid

    # shortcuts
    @property
    def text(self):
        s = self.sent
        if s is None:
            return None
        return " ".join(s.seq_word.vals[self.widx:self.wridx])

    @property
    def wridx(self):  # right end
        return self.widx + self.wlen

    @property
    def shead_widx(self):
        # note: special treatment!!
        if self.shidx is not None:
            return self.shidx
        if self.wlen == 1:
            return self.widx
        return None

    @property
    def shead_token(self):
        return self.sent.tokens[self.shead_widx]

    def get_tokens(self):
        return self.sent.tokens[self.widx:self.wridx]

    def get_words(self, concat=False):
        ret = self.sent.seq_word.vals[self.widx:self.wridx]
        if concat:
            ret = " ".join(ret)
        return ret

    def overlap_tokens(self, mention: 'Mention'):
        left, right = 0, 0
        if self.sent is mention.sent:
            left = max(self.widx, mention.widx)
            right = min(self.wridx, mention.wridx)
        if right > left:
            return self.sent.tokens[left:right]
        else:
            return []

    # =====
    # note: the following ones use sent-level idxes for all!!

    def get_span(self, shead: bool=False) -> Tuple[int, int]:
        if shead:
            return (self.shead_widx, 1)  # always len=1
        else:
            return (self.widx, self.wlen)

    def set_span(self, widx: int, wlen: int, shead: bool=False):
        # check range
        _sent = self.sent
        if _sent is not None:
            slen = len(_sent)
            assert wlen > 0 and widx >= 0 and widx + wlen <= slen, "Bad span idxes for the Mention!"
        # --
        if shead:
            assert wlen==1
            self.shidx = widx
        else:
            self.widx, self.wlen = widx, wlen

    @staticmethod
    def create_span_getter(mode: str):
        return {
            "span": lambda m: m.get_span(),  # overall span
            "shead": lambda m: m.get_span(True),  # single head
        }[mode]

    @staticmethod
    def create_span_setter(mode: str):
        return {
            "span": lambda m, *args: m.set_span(*args),  # overall span
            "shead": lambda m, *args: m.set_span(*args, True),  # single head
        }[mode]

# Frame belongs to Doc!
@DataInst.rd('frame')
class Frame(DataInst):
    def __init__(self, mention: Mention = None, label: str = None, cate: str = None, score: float = None,
                 id: str = None, par: 'Doc' = None):
        super().__init__(id=id, par=par)
        # --
        cate, label = Frame.correct_cate_label(cate, label)  # note: fix for easier processing
        # --
        self.mention: Mention = mention  # None if no mention!
        self.label: str = label
        self._cate: str = cate  # category
        self.score: float = score
        self.args: List[ArgLink] = []  # self -> other
        self._as_args: List[ArgLink] = []  # other -> self (self as args)
        self._arg_col: ItemCollection = None  # argument collection
        # --

    def _search_sent(self):
        if self.mention is None:  # no mention!
            return None
        return self.doc.sents[self.mention.sid]

    def clear_cached_vals(self):
        super().clear_cached_vals()
        if self.mention is not None:
            self.mention.clear_cached_vals()
        for a in self.args:
            a.clear_cached_vals()
        for a in self.as_args:
            a.clear_cached_vals()
        # --

    def __repr__(self):
        return f"F({self.id},{self.label})[{self.mention}]"

    @classmethod
    def _info_fields(cls):
        return {'mention': InfoField(inner_type=Mention),
                'args': InfoField(inner_type=ArgLink, wrapper_type=list, no_store_f='len0')}

    def set_mention(self, mention: Mention):
        self.mention = mention
        self._sent = self._search_sent()  # reset sent!
        return mention

    def set_cate(self, cate: str):
        # note: keeping the same id will be fine!
        self.doc.del_frame(self)
        self._cate = cate
        self.doc.add_frame(self)

    def del_self(self):
        self.del_args()
        self.del_as_args()
        self.doc.del_frame(self)

    def add_arglink(self, arglink: 'ArgLink'):
        arglink.set_par(self)
        self.args.append(arglink)  # add args at this end
        arglink.arg.as_args.append(arglink)  # add as_args to the other end
        self._arg_col = None  # clear
        return arglink

    def add_arg(self, arg: 'Frame', label: str, score: float = None):
        arglink = ArgLink(arg, label, score, par=self)  # will be linked back
        return self.add_arglink(arglink)

    def del_args(self):
        for a in list(self.args):  # remember to copy!
            a.del_self()
        self.args.clear()

    # note: also propagate to delete the arg itself!!
    def del_as_args(self):
        for a in list(self.as_args):  # remember to copy!
            a.del_self()
        self.as_args.clear()

    def get_args(self):
        return list(self.args)  # copy to make it easier to del!

    @property
    def arg_col(self):
        if self._arg_col is None:
            self._arg_col = ItemCollection(self.args)
        return self._arg_col

    @staticmethod
    def get_cate_label(cate: str, label: str):
        return f"{cate}___{label}"  # full one!

    @staticmethod
    def parse_cate_label(cate_label: str, quiet=True):
        sep = "___"
        if sep in cate_label:
            cate, label = cate_label.split(sep, 1)
        else:
            if not quiet:
                zwarn(f"Cannot parser cate_label of {cate_label}")
            cate, label = None, cate_label
        return cate, label

    @staticmethod
    def correct_cate_label(cate, label: str):  # make it easier to make_frame!
        if label is None:
            _cate = _label = None
        else:
            _cate, _label = Frame.parse_cate_label(label, quiet=True)  # only parse the label
        if _cate is not None:  # seems a good parse?
            if cate is None or (isinstance(cate, (list, tuple)) and _cate in cate):
                return _cate, _label
        if isinstance(cate, (list, tuple)):  # simply choose one!
            cate = cate[0]
        return cate, label  # original one!

    # shortcuts
    @property
    def as_args(self): return self._as_args
    @property
    def cate(self): return self._cate
    @property
    def type(self): return self.label
    @property
    def cate_label(self): return self.get_cate_label(self.cate, self.label)
    def set_label(self, label: str): self.label = label
    # --

# Arg (or relation)
@DataInst.rd('arg')
class ArgLink(DataInst):
    def __init__(self, arg: Frame = None, label: str = None, score: float = None,
                 id: str = None, par: Frame = None):
        super().__init__(id=id, par=par)
        # --
        self._arg: Frame = arg  # the argument, which is also a frame
        self.label: str = label
        self.score: float = score
        # --

    def _search_sent(self):
        if self.mention is None:  # no mention!
            return None
        return self.mention.sent

    def to_dict(self, store_type=True):
        ret = super().to_dict(store_type)
        ret.update({'_arg': self._arg.id})  # only store id!!
        return ret

    def from_dict(self, data: Dict):
        super().from_dict(data)
        self._arg = data['_arg']  # note: need to resolve at Doc-level!

    def __repr__(self):
        return f"ArgLink({self.label})[{self.mention}]"

    # delete the links
    def del_self(self):
        # must be there
        self.main.args.remove(self)
        self.arg.as_args.remove(self)
        self.main._arg_col = None  # clear arg_col

    def get_spath(self, sent=None, level=1):
        if sent is None:
            sent = self.arg.sent
        ret = sent.tree_dep.get_path_between_mentions(self.main.mention, self.arg.mention, level=level)
        return ret

    # shortcuts
    @property
    def mention(self): return self.arg.mention
    @property
    def main(self): return self._par  # simply the par!
    @property
    def arg(self): return self._arg
    @property
    def role(self): return self.label
    def set_label(self, label: str): self.label = label
    # --

# --
# Item Collection: potentially with a light-weight graph-manager
class ItemCollection:
    def __init__(self, items: Union[Iterable[Frame], Iterable[ArgLink]]):
        self.items = list(items)

    @staticmethod
    def yield_items(items, *other_filters, label_prefix: str = None, cates: str = None, in_sent: 'Sent' = None):
        all_filters = []
        if label_prefix is not None:
            all_filters.append((lambda x: x.label.startswith(label_prefix)))
        if cates is not None:
            if isinstance(cates, str):
                all_filters.append(lambda x: x.cate == cates)
            else:  # also allow a list
                cates = list(cates)
                all_filters.append(lambda x: x.cate in cates)
        if in_sent is not None:
            all_filters.append((lambda x: x.sent is in_sent))  # note: directly judge "is"
        all_filters.extend(other_filters)
        yield from (z for z in items if all(f(z) for f in all_filters))

    @staticmethod
    def create(items, *other_filters, **kwargs):
        return ItemCollection(ItemCollection.yield_items(items, *other_filters, **kwargs))

# --
# b mspx/data/inst/frame:?
