#

# some useful generators

__all__ = [
    'yield_with_f', 'yield_with_flist', 'yield_forever',
    'yield_lines', 'yield_multilines', 'yield_files', 'yield_filenames',
    'Yielder', 'FWrapperYielder',
    "Dumper", "WrapperDumper", "FWrapperDumper", "MultiDumper",
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
    def __iter__(self):
        return self.yf()

    def yf(self):  # function to get generator
        raise NotImplementedError()

class FWrapperYielder(Yielder):
    def __init__(self, base, func: Callable=None, inplaced=False):
        self.base = base
        self.func = func
        self.inplaced = inplaced

    def yf(self):
        for one in self.base:
            if self.func is None:
                yield one
            else:
                z = self.func(one)
                yield one if self.inplaced else z

# ==
# just make it simple
class Dumper:
    def __init__(self):
        pass

    def dump_one(self, obj: object):
        raise NotImplementedError()

    def dump_iter(self, iter: Iterable):
        for one in iter:
            self.dump_one(one)

    def close(self):
        pass

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def __del__(self): self.close()

# with a wrapper
class WrapperDumper(Dumper):
    def __init__(self, base_dumper: Dumper):
        super().__init__()
        self._base_dumper = base_dumper

    def close(self):
        self._base_dumper.close()

class FWrapperDumper(WrapperDumper):
    def __init__(self, base_dumper: Dumper, func: Callable, inplaced=False):
        super().__init__(base_dumper)
        self.func = func
        self.inplaced = inplaced

    def dump_one(self, obj: object):
        z = self.func(obj)
        out = obj if self.inplaced else z
        self._base_dumper.dump_one(out)

# inverse of zip
class MultiDumper(Dumper):
    def __init__(self, base_dumpers: Iterable[Dumper]):
        super().__init__()
        # --
        self._base_dumpers = list(base_dumpers)

    def close(self):
        for d in self._base_dumpers:
            d.close()

    def dump_one(self, obj: Iterable):
        objs = list(obj)
        assert len(objs) == len(self._base_dumpers)
        for v, d in zip(objs, self._base_dumpers):
            d.dump_one(v)
