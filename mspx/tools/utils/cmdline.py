#

# cmdline reader

__all__ = [
    "CmdLineConf", "CmdLineParser",
]

from shlex import split as sh_split
import traceback
from typing import List
from mspx.utils import Conf, zlog, zfatal, zglob1, zopen, Configurable
from mspx.data.stream import Streamer

# --
# command-line parser (reader)

class CmdLineConf(Conf):
    def __init__(self):
        self.cmd_input = ""  # by default from input
        self.assign_target = True  # allow "[target-name=]args..."
        self.target_sep = "="  # sep target and cmd
        self.kwargs_sep = "--"  # sep args and kwargs
        self.kwargs_kv_sep = ":"  # sep k:v for kwargs

class CmdLineParser(Streamer, Configurable):
    def __init__(self, conf: CmdLineConf = None, **kwargs):
        Streamer.__init__(self)
        Configurable.__init__(self, conf, **kwargs)
        # --
        self.src = None
        # --

    def _yield(self, f: str):
        if f in ["", "-"]:
            while True:
                try:
                    line = input(">> ")
                    yield line
                except EOFError:
                    break
                except KeyboardInterrupt:
                    continue
        else:
            with zopen(zglob1(f)) as fd:
                yield from fd

    def _restart(self):
        self.src = self._yield(self.conf.cmd_input)

    def _next(self):
        conf: CmdLineConf = self.conf
        while True:  # allow several failures
            # read one line
            is_end = False
            try:
                line = next(self.src)
                if line.strip() == "":
                    continue
            except:
                is_end = True
            if is_end:
                return self.eos
            else:
                try:  # try shell splitting
                    return self.parse_cmd(line, assign_target=conf.assign_target, target_sep=conf.target_sep, kwargs_sep=conf.kwargs_sep, kwargs_kv_sep=conf.kwargs_kv_sep)
                except:
                    zlog(f"Err in CMD-Parsing: {traceback.format_exc()}")

    @staticmethod
    def parse_cmd(line: str, assign_target=True, target_sep="=", kwargs_sep="--", kwargs_kv_sep=":"):
        target, args, kwargs = None, [], {}
        cmd = line.strip()
        # find target
        if assign_target:
            tmp_fields = cmd.split(target_sep, 1)
            if len(tmp_fields) == 2:
                tmp_target, remainings = [x.strip() for x in tmp_fields]
                if str.isidentifier(tmp_target):  # make sure it is identifier
                    target = tmp_target  # assign target
                    line = remainings
        tmp_args = sh_split(line)
        cur_i = 0
        # collect *args
        while cur_i < len(tmp_args):
            cur_a = tmp_args[cur_i]
            if cur_a == kwargs_sep:
                cur_i += 1  # skip this one!
                break
            else:
                args.append(cur_a)
            cur_i += 1
        # collect **kwargs
        while cur_i < len(tmp_args):
            _k, _v = tmp_args[cur_i].split(kwargs_kv_sep)
            kwargs[_k] = _v
            cur_i += 1
        return (cmd, target, args, kwargs)
