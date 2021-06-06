#

# about files

__all__ = [
    "zopen", "WithWrapper", "zopen_withwrapper",
    "dir_msp2",
    "zglob", "zglob1",
    "mkdir_p",
]

from typing import IO, Union, Callable
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
        return self if self.item is None else self.item

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.f_end is not None:
            self.f_end()

# open (possibly with fd)
def zopen_withwrapper(fd_or_path: Union[str, IO], **kwargs):
    if isinstance(fd_or_path, str):
        return zopen(fd_or_path, **kwargs)
    else:
        # assert isinstance(fd_or_path, IO)
        return WithWrapper(None, None, fd_or_path)

# get msp2's directory
def dir_msp2(absolute=False):
    dir_name = os.path.dirname(os.path.abspath(__file__))  # msp2/utils
    dir_name2 = os.path.join(dir_name, "..")  # msp2
    if absolute:
        dir_name2 = os.path.abspath(dir_name2)
    return dir_name2

# glob
def zglob(pathname: str, check_prefix="..", check_iter=0, assert_exist=False, assert_only_one=False, sort=False):
    if pathname == "":
        return []
    files = glob.glob(pathname)
    if len(check_prefix)>0:
        while len(files)==0 and check_iter>0:
            pathname = os.path.join(check_prefix, pathname)
            files = glob.glob(pathname)
            check_iter -= 1  # limit for checking
    if assert_only_one:
        assert len(files) == 1
    elif assert_exist:  # only_one leads to exists
        assert len(files) > 0
    return sorted(files) if sort else files

# assert there should be only one
def zglob1(pathname: str, raise_error=False, **kwargs):
    rets = zglob(pathname, **kwargs)
    if len(rets) == 1:
        return rets[0]
    else:
        assert not raise_error
        return pathname  # return pathname by default!

# mkdir -p path
def mkdir_p(path: str, raise_error=False):
    if os.path.exists(path):
        if os.path.isdir(path):
            return True
        if raise_error:
            raise FileExistsError(f"Failed mkdir: {path} exists and is not dir!")
        return False
    else:
        os.mkdir(path)
        return True
