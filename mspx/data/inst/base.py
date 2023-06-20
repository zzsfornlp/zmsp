#

# the basic one (keep it simple!)
# -> simplifications to msp2:
# 0) Every sent belong to a Doc, even single ones.
# 1) type specific to/from -> specific for each class
# 2) lightweight indexing -> all frames now belong to Doc (but may support Sent finding)!

__all__ = [
    "DataInst",
]

from typing import Dict, List, Union, Type, Callable
from mspx.utils import Registrable, Serializable, InfoField, ZArr

"""
# 'par' structures
Doc -> {Sent, Frame}
Sent -> {(*Mention), Token, SeqField, Tree}
Frame -> {ArgLink}
"""

@Registrable.rd('D')
class DataInst(Serializable):
    def __init__(self, id: str = None, par: 'DataInst' = None):
        self.id = id  # id
        self._par = par  # parent node (if there is)
        # set on need!
        # self.info_ = {}  # for general info
        # self.arrs_ = {}  # for np.ndarray with ZArr
        # self.cache = {}  # other (temp) info
        # --

    def seria_fields(self, store_all_fields=False):
        assert not store_all_fields, "Not storing all fields in this mode!"
        # note: ignore None fields!
        return [k for k, v in self.__dict__.items() if (v is not None) and (not k.startswith("_"))]

    @classmethod
    def _info_fields(cls):
        return {'arrs_': InfoField(to_f=(lambda x: {k: ZArr.arr2obj(v) for k,v in x.items()}),
                                   from_f=(lambda d: {k: ZArr.obj2arr(v) for k,v in d.items()}),
                                   no_store_f='len0'),
                'info_': InfoField(no_store_f='len0')}

    # get or set
    def get_or_set_field(self, name, cls):
        ret = getattr(self, name, None)
        if ret is None:
            ret = cls()
            setattr(self, name, ret)
        return ret

    @property
    def par(self):
        return self._par

    def set_par(self, par: 'DataInst'):
        self._par = par

    def set_id(self, id: str = None):
        self.id = id

    @property
    def read_idx(self):  # todo(+N): here assign read idx to support in-order writing
        return self._read_idx

    def set_read_idx(self, idx: int):
        self._read_idx = idx

    @property
    def cache(self): return self.get_or_set_field('_cache', dict)  # no store!
    @property
    def info(self): return self.get_or_set_field('info_', dict)  # store!
    @property
    def arrs(self): return self.get_or_set_field('arrs_', dict)  # store!

    @property
    def sig(self):
        return self.id  # by default just id

    def clear_cached_vals(self):
        if hasattr(self, '_sent'): delattr(self, '_sent')
        if hasattr(self, '_doc'): delattr(self, '_doc')

    def __repr__(self):
        # return f"{self.__class__.__name__}(id={self.id},par={self.par})"
        return f"{self.__class__.__name__}({self.id})"

    def pprint(self):
        from .printer import MyPrettyPrinter
        return MyPrettyPrinter().str_auto(self)

    # --
    # paths

    # get parent path
    def get_par_spine(self, inc_self=True, top_down=True, step=-1):
        path: List[DataInst] = []
        c: DataInst = self if inc_self else self._par
        while c is not None:
            if step == 0: break  # out of budget, thus -1 means INF
            path.append(c)
            step -= 1
            c = c._par
        if top_down:
            path.reverse()
        return path

    # get full id
    def get_sig(self, ret_str=False, ignore_starting_none=True, **kwargs):
        spine = self.get_par_spine(**kwargs)
        ret = [z.sig for z in spine]
        ii = 0  # note: ignore starting None
        if ignore_starting_none:
            while ii < len(ret) and ret[ii] is None:
                ii += 1
        if ret_str:
            ret = "/".join([str(z) for z in ret[ii:]])
        return ret

    # search up parent path for certain type
    def search_up_for_type(self, trg_type: Type, inc_self=True):
        c = self if inc_self else self._par
        while c is not None:
            if isinstance(c, trg_type):
                return c
            c = c._par
        return None

    # --
    # special shortcuts
    def _search_doc(self):
        from .doc import Doc
        return self.search_up_for_type(Doc)

    def _search_sent(self):
        from .doc import Sent
        return self.search_up_for_type(Sent)

    @property
    def doc(self):
        if getattr(self, '_doc', None) is None:
            self._doc = self._search_doc()
        return self._doc

    @property
    def sent(self):
        if getattr(self, '_sent', None) is None:
            self._sent = self._search_sent()
        return self._sent
    # --

    def str_auto(self, printer=None, **kwargs):
        from .printer import MyPrettyPrinter
        if printer is None:
            printer = MyPrettyPrinter(**kwargs)
        return printer.str_auto(self)
