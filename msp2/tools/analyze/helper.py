#

# some helpers

__all__ = [
    "CmdLineConf", "CmdLineParser", "RecordNode", "RecordNodeVisitor",
]

from shlex import split as sh_split
import traceback
from typing import List
from msp2.utils import Conf, zlog, zfatal
from msp2.data.stream import Streamer
from msp2.data.rw import LineStreamer
from msp2.tools.tree import TreeNode, TreeNodeVisitor

# =====
# command-line parser (reader)

class CmdLineConf(Conf):
    def __init__(self):
        self.cmd_input = ""  # by default from input
        self.assign_target = True  # allow "[target-name=]args..."
        self.target_sep = "="  # sep target and cmd
        self.kwargs_sep = "--"  # sep args and kwargs
        self.kwargs_kv_sep = ":"  # sep k:v for kwargs

class CmdLineParser(Streamer):
    def __init__(self, conf: CmdLineConf, **kwargs):
        super().__init__()
        self.conf = conf.direct_update(**kwargs)
        # --
        if self.conf.cmd_input in ["", "-"]:
            self.streamer = None
        else:
            self.streamer = LineStreamer(conf.cmd_input)

    def _next(self):
        conf = self.conf
        while True:  # allow several failures
            # read one line
            is_end = False
            if self.streamer is None:
                # read from input
                try:
                    line = input(">> ")
                    if line.strip() == "":
                        continue
                except EOFError:
                    line, is_end = None, True
                    break
                except KeyboardInterrupt:
                    continue
            else:
                line, is_end = self.streamer.next_and_check()
            if is_end:
                return self.eos
            else:
                target, args, kwargs = None, [], {}
                cmd = line.strip()
                # find target
                if conf.assign_target:
                    tmp_fields = cmd.split(conf.target_sep, 1)
                    if len(tmp_fields) == 2:
                        tmp_target, remainings = [x.strip() for x in tmp_fields]
                        if str.isidentifier(tmp_target):  # make sure it is identifier
                            target = tmp_target  # assign target
                            line = remainings
                # split into args
                try:  # try shell splitting
                    tmp_args = sh_split(line)
                    cur_i = 0
                    # collect *args
                    while cur_i < len(tmp_args):
                        cur_a = tmp_args[cur_i]
                        if cur_a == conf.kwargs_sep:
                            cur_i += 1  # skip this one!
                            break
                        else:
                            args.append(cur_a)
                        cur_i += 1
                    # collect **kwargs
                    while cur_i < len(tmp_args):
                        _k, _v = tmp_args[cur_i].split(conf.kwargs_kv_sep)
                        kwargs[_k] = _v
                        cur_i += 1
                    return (cmd, target, args, kwargs)
                except:
                    zlog(f"Err in CMD-Parsing: {traceback.format_exc()}")

    def _restart(self):
        if self._restart_times == 0:
            if self.streamer is not None:
                self.streamer.restart()
        else:
            zfatal("Cannot restart CMDParser!")

# =====
# record node

class RecordNode(TreeNode):
    def __init__(self, par: 'RecordNode', path: List, **kwargs):
        super().__init__(id=('R' if len(path)==0 else path[-1]), **kwargs)
        # --
        # path info
        self.path = tuple(path)
        self.name = ".".join([str(z) for z in path])
        self.level = len(path)  # starting from 0
        # content info
        self.count = 0  # all the ones that go through this node
        self.count_end = 0  # only those ending at this node
        self.objs: List = []  # only added to the end points!
        # add par
        if par is not None:
            par.add_ch(self)

    @classmethod
    def new_root(cls):
        return cls(None, [])

    # =====
    # recording a seq and add/extend node if needed
    def record_seq(self, seq, count=1, obj=None):
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
                next_node = RecordNode(cur_node, cur_path)  # no need copy, since changed to a new tuple later.
            cur_node = next_node

    # content for printing
    def get_content(self):
        return None

class RecordNodeVisitor(TreeNodeVisitor):
    pass
