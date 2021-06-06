#

# Mentions and Frames

__all__ = [
    "Mention", "Frame", "ArgLink",
]

from typing import List, Dict, Iterable, Union, Tuple
from .base import DataInstance, DataInstanceComposite, SubInfo
from .helper import InDocInstance, InSentInstance

# =====
# Mention and Relation and Semantic-Frames

# todo(note): how about the consistency!?
# 1. When creating ArgLink, no actual links are added
# 2. When add_arg, both args and as_args are added!
# 3. When ArgLink.delete_self, removed from both args and as_args.

# Mention to locate the surface range
# todo(+N): currently we only support continuous span ranged mention
class Mention(InSentInstance):
    @classmethod
    def _get_sub_map(cls) -> Dict[str, SubInfo]:
        return {
            "hidx": SubInfo(int, df_val=None),
            "hlen": SubInfo(int, df_val=None),
            "shidx": SubInfo(int, df_val=None),
            "soft_idxes": SubInfo(list, df_val=None),
        }

    def __init__(self, _id: str = None, _par: 'DataInstance' = None):
        super().__init__(_id, _par)
        # --
        self.widx: int = None  # starting word idx
        self.wlen: int = None  # length in words
        # self.hoff: int = None  # (relative) starting head-span offset inside
        # self.hlen: int = None  # length of head-span
        # self.shoff: int = None  # (relative) single head offset inside, len=1
        self.hidx: int = None  # head-span word idx
        self.hlen: int = None  # head-span length
        self.shidx: int = None  # single-head word idx
        self.soft_idxes: List[float] = None  # soft ones

    @classmethod
    def create(cls, sent: 'Sent', widx: int, wlen: int, id: str = None, par: 'DataInstance' = None):
        inst: Mention = super().create(id, par)
        inst._sent = sent  # directly assign
        inst.set_span(widx, wlen)
        return inst

    def __repr__(self):
        sid = self.sent.id if self.sent is not None else "?"
        return f"M({sid}/[{self.widx},{self.wlen}]): `{self.text}'"

    # shortcuts
    @property
    def text(self):
        s = self.sent
        if s is None:
            return None
        return " ".join(s.seq_word.vals[self.widx:self.widx+self.wlen])

    @property
    def wridx(self):  # right end
        return self.widx + self.wlen

    @property
    def hspan_widx(self): return self.hidx

    @property
    def hspan_wlen(self): return self.hlen

    @property
    def shead_widx(self):
        # note: special treatment!!
        if self.shidx is not None:
            return self.shidx
        if self.wlen == 1:
            return self.widx
        return None

    def get_shoff(self): return self.shead_widx - self.widx  # single-head's offset inside larger span

    def get_tokens(self):
        return self.sent.tokens[self.widx, self.wridx]

    @property
    def shead_token(self):
        return self.sent.tokens[self.shead_widx]

    # =====
    # note: the following ones use sent-level idxes for all!!

    def get_span(self, hspan: bool=False, shead: bool=False) -> Tuple[int, int]:
        if shead:
            return (self.shead_widx, 1)  # always len=1
        elif hspan:
            return (self.hidx, self.hlen)
        else:
            return (self.widx, self.wlen)

    def set_span(self, widx: int, wlen: int, hspan: bool=False, shead: bool=False):
        # check range
        slen = len(self.sent)
        assert wlen > 0 and widx >= 0 and widx + wlen <= slen, "Bad span idxes for the Mention!"
        # --
        if shead:
            assert wlen==1
            self.shidx = widx
        elif hspan:
            self.hidx, self.hlen = widx, wlen
        else:
            self.widx, self.wlen = widx, wlen

    @staticmethod
    def create_span_getter(mode: str):
        return {
            "span": lambda m: m.get_span(),  # overall span
            "hspan": lambda m: m.get_span(True, False),  # head span
            "shead": lambda m: m.get_span(True, True),  # single head
        }[mode]

    @staticmethod
    def create_span_setter(mode: str):
        return {
            "span": lambda m, *args: m.set_span(*args),  # overall span
            "hspan": lambda m, *args: m.set_span(*args, True, False),  # head span
            "shead": lambda m, *args: m.set_span(*args, True, True),  # single head
        }[mode]

    # =====
    # used for comparison
    def __eq__(self, other: "Mention"):
        return self.is_equal(other)

    def get_sig(self, **kwargs):
        sid = self.sent.sid  # note: simply use SID
        return (sid,) + self.get_span(**kwargs)

    def is_equal(self, other: "Mention", **kwargs):
        if not isinstance(other, Mention):
            return False
        return self.get_sig(**kwargs) == other.get_sig(**kwargs)

# Frame: (Mention: Mention) + (Type: str) + (Args: List[Links])
# todo(note): be careful about the multi-inheritance!!
class Frame(DataInstanceComposite, InSentInstance):
    @classmethod
    def _get_sub_map(cls) -> Dict[str, SubInfo]:
        return {
            "mention": SubInfo(Mention),
            "args": SubInfo(ArgLink, wrapper_type=list, needs_reg=True, reg_sname='arg', df_val=[]),
            "as_args": SubInfo(ArgLink, wrapper_type=list, is_ref=True, df_val=[]),
        }

    def __init__(self, _id: str = None, _par: 'DataInstance' = None):
        super().__init__(_id, _par)
        # --
        self.mention: Mention = None
        self.type: str = None
        self.score: float = 0.
        self._type_idx: Union[int, object] = None  # idx can be a simple int or more complex index
        self.args: List[ArgLink] = []  # self -> other
        self.as_args: List[ArgLink] = []  # other -> self (self as args)

    @classmethod
    def create(cls, mention: Mention = None, type: str = None, score: float = 0.,
               id: str = None, par: 'DataInstance' = None):
        inst: Frame = super().create(id, par)
        if mention is not None:
            inst.set_mention(mention)
        inst.type = type
        inst.score = score
        return inst

    def __repr__(self):
        return f"F({self.id},{self.type})[{self.mention}]"

    def set_mention(self, mention: Mention):
        self.add_inst(mention)
        self.mention = mention
        return mention

    # not public one!!
    def _add_arglink(self, arglink: 'ArgLink'):
        self.add_and_reg_inst(arglink, 'arg')
        self.args.append(arglink)  # add args at this end
        arglink.arg.as_args.append(arglink)  # add as_args to the other end
        return arglink

    def add_arg(self, arg: 'Frame', role: str, score: float = 0.):
        arglink = ArgLink.create(self, arg, role, score)  # will be linked back
        self._add_arglink(arglink)  # add it in
        return arglink

    def mark_no_args(self):
        self.clear_args()
        self.args = None

    def clear_args(self):
        if self.args is None:
            self.args = []
        else:
            for a in list(self.args):  # remember to copy!
                a.delete_self()
            self.args.clear()
            self.clear_insts('arg')

    # note: also propagate to delete the arg itself!!
    def clear_as_args(self):
        for a in list(self.as_args):  # remember to copy!
            a.delete_self()
        self.as_args.clear()

    # shortcut
    @property
    def label(self):
        return str(self.type)

    @property
    def label_idx(self):
        return self._type_idx

    def set_label(self, label: str): self.type = label
    def set_label_idx(self, idx): self._type_idx = idx
    def set_score(self, score: float): self.score = score

# ArgLink: link between Frames (eg, frame->args, rel->entities, ...)
class ArgLink(DataInstance):
    @classmethod
    def _get_sub_map(cls) -> Dict[str, SubInfo]:
        return {  # these two should only be ref
            "main": SubInfo(Frame, is_ref=True),
            "arg": SubInfo(Frame, is_ref=True),
        }

    def __init__(self, _id: str = None, _par: 'DataInstance' = None):
        super().__init__(_id, _par)
        # --
        self.main: Frame = None  # the main frame, which should contain this ArgLink
        self.arg: Frame = None  # the argument, which is also a frame
        self.role: str = None
        self.score: float = 0.
        self._role_idx: Union[int, object] = None

    @classmethod
    def create(cls, main: Frame, arg: Frame, role: str, score: float = 0.,
               id: str = None, par: 'DataInstance' = None):
        inst: ArgLink = super().create(id, par)
        inst.main = main  # parent, upper frame
        inst.arg = arg  # actual argument
        inst.role = role
        inst.score = score
        return inst

    def __repr__(self):
        return f"ArgLink({self.role})[{self.mention}]"

    # delete the links
    def delete_self(self):
        # must be there
        self.main.args.remove(self)
        self.arg.as_args.remove(self)
        self.main.del_inst(self)

    # shortcut
    @property
    def mention(self):
        return self.arg.mention

    @property
    def label(self):
        return str(self.role)

    @property
    def label_idx(self):
        return self._role_idx

    def set_label(self, label: str): self.role = label
    def set_label_idx(self, idx): self._role_idx = idx
    def set_score(self, score: float): self.score = score
