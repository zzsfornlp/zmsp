#

# some useful generators

__all__ = [
    'yield_with_f', 'yield_with_flist', 'yield_forever',
    'yield_lines', 'yield_multilines', 'yield_files', 'yield_filenames',
    'Yielder', 'WrapperYielder', 'FWrapperYielder',
]

from typing import Union, IO, Iterable, Callable
import re
import os
from mspx.utils import zopen, zopen_withwrapper

# =====
# compositional ones

# yield with func (similar to FWrapperStreamer)
def yield_with_f(base: Iterable, func: Callable, inplaced: bool, **kwargs):
    for one in base:
        z = func(one, **kwargs)
        yield (one if inplaced else z)

# yield from with func
def yield_with_flist(base: Iterable, func: Callable, **kwargs):
    for one in base:
        yield from func(one, **kwargs)

# yield forever
def yield_forever(base: Iterable):
    while True:
        yield from base

# =====
# file reading related

# yield lines from fd for path
def yield_lines(fd_or_path: Union[IO, str]):
    with zopen_withwrapper(fd_or_path) as fd:
        yield from fd

# yield multilines from fd or path
def yield_multilines(fd_or_path: Union[IO, str], sep_f=None, inc_sep=False):
    if sep_f is None:  # a default one!
        sep_f = (lambda x: len(x.rstrip()) == 0)
    line_gen = yield_lines(fd_or_path)
    lines = []
    for line in line_gen:
        if sep_f(line):
            if inc_sep:
                lines.append(line)
            if len(lines) > 0:
                yield ''.join(lines)
                lines.clear()
        else:
            lines.append(line)
    if len(lines) > 0:
        yield ''.join(lines)
    # --

# yield files that satisfy certain pattern from a dir
def yield_files(fd_or_paths: Iterable[Union[IO, str]], yield_lines=False):
    for f in fd_or_paths:
        with zopen_withwrapper(f) as fd:
            if yield_lines:
                yield from fd
            else:
                yield fd.read()

# yield files under a dir
def yield_filenames(dir: str, re_pat: str = None, sorted=True):
    files = os.listdir(dir)
    if sorted:
        files.sort()
    if re_pat is None:
        yield from files
    else:
        if isinstance(re_pat, str):
            re_pat = re.compile(re_pat)
        for f in files:
            if re.fullmatch(re_pat, f):
                yield os.path.join(dir, f)

# --
# simpler than streamer

class Yielder:
    def yf(self):  # function to get generator
        raise NotImplementedError()

class WrapperYielder(Yielder):
    def __init__(self, base: Yielder):
        self.base = base

    def yf(self):
        yield from self.base.yf()

class FWrapperYielder(WrapperYielder):
    def __init__(self, base: Yielder, func: Callable, inplaced=False):
        super().__init__(base)
        self.func = func
        self.inplaced = inplaced

    def yf(self):
        for one in self.base.yf():
            z = self.func(one)
            yield one if self.inplaced else z
