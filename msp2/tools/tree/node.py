#

from typing import Dict, List, Callable, Union
from msp2.utils import IdAssignable

__all__ = [
    "TreeNode", "TreeNodeVisitor",
]

# todo(note): node_id and ch_key are bonded together!!
class TreeNode(IdAssignable):
    def __init__(self, id=None, **kwargs):
        if id is None:  # automatic id
            self.id = self.__class__.get_new_id()
        else:
            self.id = id
        # --
        self.par: TreeNode = None
        self.chs_map: Dict = {}  # id -> node
        # extra properties
        self.props = kwargs

    # has ch?
    def has_ch(self, node: 'TreeNode'):
        return node.id in self.chs_map

    # get ch?
    def get_ch(self, id, df=None):
        return self.chs_map.get(id, df)

    # add one children
    def add_ch(self, node: 'TreeNode'):
        assert node.par is None, "Node already has parent!"
        assert node.id not in self.chs_map, "Node already in chs_map"
        self.chs_map[node.id] = node
        node.par = self  # link both ways

    # detach one child from self
    def detach_ch(self, node: 'TreeNode'):
        assert node.par is self
        del self.chs_map[node.id]
        node.par = None

    # detach from parent
    def detach_par(self):
        self.par.detach_ch(self)

    # =====
    # like a dictionary
    def __getitem__(self, item):
        return self.chs_map[item]

    def __contains__(self, item):
        return item in self.chs_map

    def keys(self):
        return self.chs_map.keys()

    def values(self):
        return self.chs_map.values()

    def is_root(self):
        return self.par is None

    def __getattr__(self, item):
        if item in self.props:
            return self.props[item]
        else:
            raise AttributeError()

    # get nodes of descendents
    def get_descendants(self, recursive=True, key: Union[str,Callable]=None, preorder=True, include_self=True):
        _chs_f = TreeNode.get_enum_f(key)
        # --
        def _get_descendants(_n: 'TreeNode'):
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
                _chs_f = lambda x: x.chs_map.values()  # order does not matter!
            else:
                assert isinstance(key, str)
                _chs_f = lambda x: sorted(x.chs_map.values(), key=lambda v: getattr(v, key))
        else:
            _chs_f = key
        return _chs_f

    # recursively visit (updating)
    def rec_visit(self, visitor: 'TreeNodeVisitor'):
        # pre-visit self
        pre_value = visitor.pre_visit(self)
        # first visit all the children nodes
        ch_values = [n.rec_visit(visitor) for n in visitor.enum_chs(self)]
        # post-visit self
        return visitor.post_visit(self, pre_value, ch_values)

# -----
class TreeNodeVisitor:
    # which order to see chs: by default no specific order
    def enum_chs(self, node: TreeNode):
        return node.values()

    # pre_visit: by default do nothing
    def pre_visit(self, node: TreeNode):
        return None

    # post_visit (with chs's values)
    def post_visit(self, node: TreeNode, pre_value, ch_values: List):
        raise NotImplementedError()
