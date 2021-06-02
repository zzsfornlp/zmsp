#

# base IO with BaseDataItem using "to_builtin" and "from_builtin"

import os
import re
import json
from typing import Iterable
from msp.utils import zopen
from msp.data import Streamer, FileOrFdStreamer
from ..insts import BaseDataItem, GeneralSentence

# =====
# an improved FileStreamer: read files for dir_mode, read lines for file_mode
class PathOrFdStreamer(Streamer):
    def __init__(self, path_or_fd: str, is_dir=False, re_pat=""):
        super().__init__()
        self.path_or_fd = path_or_fd
        # for dir mode
        self.is_dir = is_dir
        if is_dir:
            file_list0 = os.listdir(path_or_fd)
            if re_pat != "":
                if isinstance(re_pat, str):
                    re_pat = re.compile(re_pat)
                file_list0 = [f for f in file_list0 if re.fullmatch(re_pat, f)]
            file_list0.sort()  # sort by string name
            file_list = [os.path.join(path_or_fd, f) for f in file_list0]
        else:
            file_list = None
        self.dir_file_list = file_list
        self.dir_file_ptr = 0
        # -----
        self.input_is_fd = not isinstance(path_or_fd, str)
        if self.input_is_fd:
            self.fd = path_or_fd
        else:
            self.fd = None

    def __del__(self):
        if (self.fd is not None) and (not self.input_is_fd) and (not self.fd.closed):
            self.fd.close()

    def _restart(self):
        if self.is_dir:
            self.dir_file_ptr = 0
        elif not self.input_is_fd:
            if self.fd is not None:
                self.fd.close()
            self.fd = zopen(self.path_or_fd)
        else:
            assert self.restart_times_==0, "Cannot restart (read multiple times) a FdStreamer"

    def _next(self):
        if self.is_dir:
            cur_ptr = self.dir_file_ptr
            if cur_ptr >= len(self.dir_file_list):
                return None
            else:
                with zopen(self.dir_file_list[cur_ptr]) as fd:
                    ss = fd.read()
                self.dir_file_ptr = cur_ptr + 1
                return ss
        else:
            line = self.fd.readline()
            if len(line) == 0:
                return None
            else:
                return line

# further using json to load
class BaseDataReader(PathOrFdStreamer):
    def __init__(self, cls, path_or_fd: str, is_dir=False, re_pat="", cut=-1):
        super().__init__(path_or_fd, is_dir, re_pat)
        #
        assert issubclass(cls, BaseDataItem)
        self.cls = cls  # the main BaseDataItem type
        self.cut = cut

    def _next(self):
        if self.count_ == self.cut:
            return None
        ss = super()._next()
        if ss is None:
            return None
        else:
            d = json.loads(ss)
            return self.cls.from_builtin(d)

# writing
class BaseDataWriter:
    def __init__(self, path_or_fd: str, is_dir=False, fname_getter=None, suffix=""):
        self.path_or_fd = path_or_fd
        # dir mode
        self.is_dir = is_dir
        self.anon_counter = -1
        self.fname_getter = fname_getter if fname_getter else self.anon_name_getter
        self.suffix = suffix
        # otherwise
        if isinstance(path_or_fd, str):
            self.fd = zopen(path_or_fd, "w")
        else:
            self.fd = path_or_fd

    def anon_name_getter(self):
        self.anon_counter += 1
        return f"anon.{self.anon_counter}{self.suffix}"

    def write_list(self, insts: Iterable[BaseDataItem]):
        for inst in insts:
            self.write_one(inst)

    def write_one(self, inst: BaseDataItem):
        if self.is_dir:
            fname = self.fname_getter(inst)
            with zopen(os.path.join(self.path_or_fd, fname), 'w') as fd:
                json.dump(inst.to_builtin(), fd)
        else:
            self.fd.write(json.dumps(inst.to_builtin()))
            self.fd.write("\n")

    def finish(self):
        if self.fd is not None and (not self.fd.closed):
            self.fd.close()
            self.fd = None

    def __del__(self):
        self.finish()

# =====
# plain reader

class PlainTextReader(FileOrFdStreamer):
    def __init__(self, file_or_fd, aug_code="", cut=-1, sep=None):
        super().__init__(file_or_fd)
        self.aug_code = aug_code
        self.cut = cut
        self.sep = sep

    def _next(self):
        if self.count_ == self.cut:
            return None
        line = self.fd.readline()
        if len(line) == 0:
            return None
        tokens = line.rstrip("\n").split(self.sep)
        one = GeneralSentence.create(tokens)
        one.add_info("sid", self.count())
        one.add_info("aug_code", self.aug_code)
        return one
