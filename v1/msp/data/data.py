# Instances: data-set and date-instance

import re
from .streamer import Streamer
from msp.utils import zopen, zfatal, zcheck

# ===== Data & DataSet
class Instance(object):
    def __init__(self):
        self.init_idx = -1

    @property
    def inst_idx(self):
        return self.init_idx

# normalize words
class WordNormer:
    DIGIT_PATTERN = re.compile(r"\d")

    def __init__(self, lower_case, norm_digit):
        self.lower_case = lower_case
        self.norm_digit = norm_digit

    def norm_one(self, w):
        if self.lower_case:
            w = str.lower(w)
        if self.norm_digit:
            w = WordNormer.DIGIT_PATTERN.sub("0", w)
        return w

    def norm_stream(self, s):
        return [self.norm_one(w) for w in s]

#
class LineReader(Streamer):
    def __init__(self, fd):
        super().__init__()
        self.fd = fd

    def _restart(self):
        raise NotImplementedError()

    def _obtain(self):
        line = self.fd.readline()
        if len(line)==0:
            return None
        else:
            one = Instance()
            # todo(+1): specific splitter?
            one.tokens = line.split()
            one.init_idx = self.count()
            return one

class TextReader(LineReader):
    def __init__(self, file):
        super().__init__(None)
        self.file = file

    def _restart(self):
        if self.fd is not None:
            self.fd.close()
        self.fd = zopen(self.file)

# cannot restart!!
class FdReader(LineReader):
    def __init__(self, fd):
        super().__init__(fd)
        self.started = False

    def _restart(self):
        if self.started:
            zfatal("Cannot restart this fd stream")
        self.started = True
