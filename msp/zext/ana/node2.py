#

# an updated version of RecNode: recording nodes
# making it easier to use

from typing import List, Iterable

#
class ZObject(object):
    def __init__(self, m=None):
        if m is not None:
            self.update(m)

    def update(self, m):
        for k, v in m.items():
            setattr(self, k, v)

class ZRecNode:
    def __init__(self, parent: 'ZRecNode', path: List):
        self.path = tuple(path)
        self.name = ".".join([str(z) for z in path])
        self.level = len(path)  # starting from 0
        self.parent = parent
        self.count = 0  # all the ones that go through this node
        self.count_end = 0  # only those ending at this node
        self.elems = {}  # to subnodes: key -> ZRecNode
        #
        self.objs: List = []  # only added to the end points
        self.props = {}  # other to-be-set properties

    # =====
    # like a dictionary
    def __getitem__(self, item):
        return self.elems[item]

    def __contains__(self, item):
        return item in self.elems

    def keys(self):
        return self.elems.keys()

    def values(self):
        return self.elems.values()

    def is_root(self):
        return self.parent is None

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        elif item in self.props:
            return self.props[item]
        else:
            raise AttributeError()

    # get nodes of descendents
    def get_descendants(self, recursive=True, key="count", preorder=True):
        ret = [self] if preorder else []
        if recursive:
            for v in sorted(self.elems.values(), key=lambda v: getattr(v, key)):
                ret.extend(v.get_descendants(recursive, key, preorder))
        if not preorder:
            ret.extend(self)
        return ret

    # get parants, grandparents, etc; h2l means high to low
    def get_antecedents(self, h2l=True):
        ret = []
        n = self
        while n.parent is not None:
            ret.append(n.parent)
            n = n.parent
        if h2l:
            ret.reverse()
        return ret

    # =====
    # recording specific
    def add_seq(self, seq, count=1, obj=None):
        assert self.parent is None, "Currently only support adding from ROOT"
        # make it list
        if not isinstance(seq, (list, tuple)):
            seq = [seq]
        # recursive adding
        cur_node = self
        cur_path = []
        while 1:
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
            next_node = cur_node.elems.get(seq0, None)
            if next_node is None:
                next_node = ZRecNode(cur_node, cur_path)  # no need copy, since changed to a new tuple later.
                cur_node.elems[seq0] = next_node
            cur_node = next_node

    # recursively visit (updating)
    def rec_visit(self, visitor: 'ZNodeVisitor'):
        # first visit all the children nodes
        values = []
        for node in visitor.sort(self.values()):
            values.append(node.rec_visit(visitor))
        return visitor.visit(self, values)

#
class ZNodeVisitor:
    def sort(self, nodes: Iterable[ZRecNode]):
        return nodes

    # process one Node
    def visit(self, node: ZRecNode, values: List):
        raise NotImplementedError("To be overridden!")

    # show result for one Node (mainly used for printing)
    def show(self, node: ZRecNode):
        raise NotImplementedError("To be overridden!")
