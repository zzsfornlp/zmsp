#

# some useful generators
from typing import Union, IO, Iterable, Callable
import re
import os
from msp2.utils import zopen, zopen_withwrapper

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

# yield files that satisfy certain pattern from a dir
def yield_files(fd_or_paths: Iterable[Union[IO, str]], yield_lines=False):
    for f in fd_or_paths:
        with zopen_withwrapper(f) as fd:
            if yield_lines:
                yield from fd
            else:
                yield fd.read()

# yield files under a dir
def yield_filenames(dir: str, re_pat: str = None, sorted=False):
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
