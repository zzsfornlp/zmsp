#

# about files

__all__ = [
    "zopen", "WithWrapper", "zopen_withwrapper", "dir_msp",
]

from typing import IO, Union, Callable, Iterable
import gzip, bz2
import os, sys
import glob

# open various kinds of files
def zopen(filename: str, mode='r', encoding="utf-8", check_zip=True):
    suffix = "" if ('b' in mode) else "t"
    if 'b' in mode:
        encoding = None
    fd = None
    if check_zip:
        if filename.endswith('.gz'):
            fd = gzip.open(filename, mode+suffix, encoding=encoding)
        elif filename.endswith('.bz2'):
            fd = bz2.open(filename, mode+suffix, encoding=encoding)
    if fd is None:
        return open(filename, mode, encoding=encoding)
    else:
        return fd

# a simple wrapper class for with expression
class WithWrapper:
    def __init__(self, f_start: Callable = None, f_end: Callable = None, item=None):
        self.f_start = f_start
        self.f_end = f_end
        self.item: object = item

    def __enter__(self):
        if self.f_start is not None:
            self.f_start()
        # return self if self.item is None else self.item
        return self.item

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.f_end is not None:
            self.f_end()

# open (possibly with fd)
def zopen_withwrapper(fd_or_path: Union[str, IO], empty_std=False, **kwargs):
    if empty_std and fd_or_path == '':
        fd_or_path = sys.stdout if ('w' in kwargs.get('mode')) else sys.stdin
    if isinstance(fd_or_path, str):
        return zopen(fd_or_path, **kwargs)
    else:
        # assert isinstance(fd_or_path, IO)
        return WithWrapper(None, None, fd_or_path)

# get msp's directory
def dir_msp(absolute=False):
    dir_name = os.path.dirname(os.path.abspath(__file__))  # msp2/utils
    dir_name2 = os.path.join(dir_name, "..")  # msp?
    if absolute:
        dir_name2 = os.path.abspath(dir_name2)
    return dir_name2
