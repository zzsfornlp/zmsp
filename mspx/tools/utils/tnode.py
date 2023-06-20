#

from typing import Dict, List, Callable, Union
from mspx.utils import IdAssignable

__all__ = [
    "TNode", "TNodeVisitor", "RecordNode",
]

# note: node_id and ch_key are bonded together!!
class TNode(IdAssignable):
    def __init__(self, id=None, par: 'TNode' = None, **kwargs):
        self.id = self.get_new_id() if id is None else id  # otherwise automatic id
        self.par: TNode = None
        self.chs_map: Dict = {}  # id -> node
        self.__dict__.update(kwargs)  # extra properties
        # --
        if par is not None:
            par.add_ch(self)
        # --

    # has ch?
    def has_ch(self, node: 'TNode'):
        return node.id in self.chs_map

    # get ch?
    def get_ch(self, id, df=None):
        return self.chs_map.get(id, df)

    # add one children
    def add_ch(self, node: 'TNode'):
        assert node.par is None, "Node already has parent!"
        assert node.id not in self.chs_map, "Node already in chs_map"
        self.chs_map[node.id] = node
        node.par = self  # link both ways

    # del one child from self
    def del_ch(self, node: 'TNode'):
        assert node.par is self
        del self.chs_map[node.id]
        node.par = None

    # del from parent
    def del_from_par(self):
        self.par.del_ch(self)

    # =====
    # like a dictionary
    def __getitem__(self, item):
        return self.chs_map[item]

    def __contains__(self, item):
        return item in self.chs_map

    def ch_keys(self):
        return self.chs_map.keys()

    def ch_values(self):
        return self.chs_map.values()

    def is_root(self):
        return self.par is None

    # get nodes of descendents
    def get_descendants(self, recursive=True, key: Union[str,Callable]=None, preorder=True, include_self=True):
        _chs_f = TNode.get_enum_f(key)
        # --
        def _get_descendants(_n: 'TNode'):
            _ret = [_n] if (include_self and preorder) else []
            ch_list = _chs_f(_n)
            if recursive:
                for _n2 in ch_list:
                    _ret.extend(_get_descendants(_n2))
            else:  # only adding direct ones
                _ret.extend(ch_list)
            if include_self and (not preorder):
                _ret.extend(_n)
            return _ret
        # --
        return _get_descendants(self)

    # get parent, grandparent, etc; h2l means high to low
    def get_antecedents(self, max_num=-1, include_self=False, h2l=True):
        ret, cur_num = [], 0
        _cur = self if include_self else self.par
        while cur_num != max_num and _cur is not None:
            ret.append(_cur)
            _cur = _cur.par
            cur_num += 1
        if h2l:  # high-to-low?
            ret.reverse()
        return ret

    # helper for enum children
    @staticmethod
    def get_enum_f(key: Union[str, Callable]=None):
        # prepare get_values
        if not isinstance(key, Callable):
            if key is None:
                _chs_f = lambda x: x.ch_values()  # order does not matter!
            else:
                assert isinstance(key, str)
                _chs_f = lambda x: sorted(x.ch_values(), key=lambda v: getattr(v, key))
        else:
            _chs_f = key
        return _chs_f

    # recursively visit (updating)
    def rec_visit(self, visitor: 'TNodeVisitor'):
        # pre-visit self
        pre_value = visitor.pre_visit(self)
        # first visit all the children nodes
        ch_values = [n.rec_visit(visitor) for n in visitor.enum_chs(self)]
        # post-visit self
        return visitor.post_visit(self, pre_value, ch_values)

# --
class TNodeVisitor:
    # which order to see chs: by default no specific order
    def enum_chs(self, node: TNode):
        return node.ch_values()

    # pre_visit: by default do nothing
    def pre_visit(self, node: TNode):
        return None

    # post_visit (with chs's values)
    def post_visit(self, node: TNode, pre_value, ch_values: List):
        raise NotImplementedError()

# --

class RecordNode(TNode):
    def __init__(self, path=(), par: 'RecordNode' = None, **kwargs):
        super().__init__(id=('R' if len(path)==0 else path[-1]), par=par, **kwargs)
        # --
        # path info
        self.path = tuple(path)
        self.name = ".".join([str(z) for z in path])
        self.level = len(path)  # starting from 0
        # content info
        self.count = 0  # all the ones that go through this node
        self.count_end = 0  # only those ending at this node
        self.objs: List = []  # only added to the end points!
        # --

    @property
    def ncount(self): return -self.count
    def get_content(self): return None

    def record(self, seq, count=1, obj=None):
        assert self.is_root(), "Currently only support adding from ROOT"
        # make it iterable
        if not isinstance(seq, (list, tuple)):
            seq = [seq]
        # recursive adding
        cur_node = self
        cur_path = []
        while True:
            # update for current node
            cur_node.count += count
            if obj is not None:
                cur_node.objs.append(obj)
            # next one
            if len(seq) <= 0:
                cur_node.count_end += count
                break
            seq0, seq = seq[0], seq[1:]
            cur_path.append(seq0)
            next_node = cur_node.get_ch(seq0)  # try to get children
            if next_node is None:
                next_node = RecordNode(cur_path, par=cur_node)  # no need copy, since changed to a new tuple later.
            cur_node = next_node
        # --
