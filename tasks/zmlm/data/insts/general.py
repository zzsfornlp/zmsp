#

from typing import Dict, List, Tuple, Union
from msp.utils import Helper
from .base import BaseDataItem, data_type_reg
from .field import DataField, SeqField, InputCharSeqField, DepTreeField

# =====
# instances may auto recursively judge IO in some way
# -- two ways of building: cls.create(...) or cls.from_builtin(d)
# todo(note): how to deal with cycles?
#  => no cycles in the "General" structures but put cycle links inside "Field" and specifically handle them

# recursive general instances
@data_type_reg
class GeneralInstance(BaseDataItem):
    def __init__(self):
        self.items: Dict = {}  # main items (serializable)
        self.info: Dict = {}  # other contents (serializable)
        # -----
        self.features = {}  # running/extra/tmp/cached features (non-serializable)

    # common
    def _add_sth(self, name: str, sth, v, assert_non_exist: bool):
        if assert_non_exist:
            assert not hasattr(self, name), f"Inside the class: {name} already exists"
            assert name not in v, f"Inside the collection: {name} already exists!"
        v[name] = sth
        setattr(self, name, sth)

    # dynamically adding items
    def add_item(self, name: str, item: Union[BaseDataItem, List, Tuple], assert_non_exist=True):
        self._add_sth(name, item, self.items, assert_non_exist)
        return item

    # adding info (should be builtin-type)
    def add_info(self, name: str, info, assert_non_exist=True):
        self._add_sth(name, info, self.info, assert_non_exist)
        return info

    # =====
    # creating

    # general creating
    @classmethod
    def create(cls, *args, **kwargs):
        x = cls()
        x._init(*args, **kwargs)
        x._finish_create()
        return x

    # real creating (to be overridden, replacing original __init__)
    def _init(self, *args, **kwargs):
        pass

    # finish up init (to be overridden, common routine used in both create and from_builtin)
    def _finish_create(self):
        pass

    # =====
    # general IO

    @classmethod
    def get_unwrap_name_mappings(cls):
        return {}  # name to class to avoid wrapping (str -> FieldType)

    @classmethod
    def get_unwrap_middle_mappings(cls):
        return {}  # name to extra middle builtin types, usually List (str -> type)

    def to_builtin(self, *args, **kwargs):
        unwrap_name_mappings = self.__class__.get_unwrap_name_mappings()
        # only consider items and info
        ret = {}
        if len(self.info) > 0:
            ret["_info"] = self.info
        for k, v in self.items.items():
            # same inside the whole layer
            if k in unwrap_name_mappings:  # no need to record type info
                _f = lambda _v, *_args, **_kwargs: _v.to_builtin(*_args, **_kwargs)
            else:
                _f = lambda _v, *_args, **_kwargs: _v.to_builtin_wrapped(*_args, **_kwargs)
            # can be different types; todo(note): can be only one layer, typically usage is List
            if isinstance(v, BaseDataItem):
                d = _f(v, *args, **kwargs)
            elif isinstance(v, (List, Tuple)):
                d = [_f(v2, *args, **kwargs) for v2 in v]
            else:
                raise NotImplementedError(f"Unknown class for 'to_builtin': {v.__class__}")
            ret[k] = d
        return ret

    @classmethod
    def from_builtin(cls, data: Dict):
        unwrap_name_mappings = cls.get_unwrap_name_mappings()
        unwrap_middle_mappings = cls.get_unwrap_middle_mappings()
        ret = cls()  # blank init
        # recursively add items (corresponding to _init)
        for k, v in data.items():
            if k == "_info":
                for k2, v2 in v.items():
                    ret.add_info(k2, v2)
            else:
                # same inside the whole layer
                if k in unwrap_name_mappings:  # no need to record type info
                    trg_cls = unwrap_name_mappings[k]
                    _f = trg_cls.from_builtin
                else:
                    _f = BaseDataItem.from_builtin_wrapped
                middle_type = unwrap_middle_mappings.get(k)
                # can be different types
                # todo(+N): more middle types?
                if middle_type is not None and issubclass(middle_type, (List, Tuple)):
                    item = [_f(v2) for v2 in v]
                else:
                    item = _f(v)
                # else:
                #     raise NotImplementedError(f"Unknown type for 'from_builtin': {v.__class__}")
                ret.add_item(k, item)
        # finish up
        ret._finish_create()
        return ret

#
@data_type_reg
class GeneralSentence(GeneralInstance):
    @classmethod
    def get_unwrap_name_mappings(cls):
        return {"word_seq": SeqField, "pos_seq": SeqField, "pred_pos_seq": SeqField,
                "dep_tree": DepTreeField, "pred_dep_tree": DepTreeField}

    def _init(self, words: List[str]):
        self.add_item("word_seq", SeqField(words))

    def __len__(self):
        return len(self.word_seq.vals)  # length of original input

    @property
    def char_seq(self):
        if not hasattr(self, "_char_seq"):  # create from "word_seq" on demand
            self._char_seq = InputCharSeqField(self.word_seq.vals)
        return self._char_seq

#
@data_type_reg
class GeneralParagraph(GeneralInstance):
    @classmethod
    def get_unwrap_name_mappings(cls):
        return {"sents": GeneralSentence}

    @classmethod
    def get_unwrap_middle_mappings(cls):
        return {"sents": list}

    def _init(self, sents: List[GeneralSentence]):
        self.add_item("sents", sents)

#
@data_type_reg
class GeneralDocument(GeneralInstance):
    @classmethod
    def get_unwrap_name_mappings(cls):
        return {"paras": GeneralParagraph}

    @classmethod
    def get_unwrap_middle_mappings(cls):
        return {"paras": list}

    def _init(self, paras: List[GeneralParagraph]):
        self.add_item("paras", paras)

    @property
    def sents(self):
        # on-the-fly joining all (no cache)
        return Helper.join_list(z.sents for z in self.paras)
