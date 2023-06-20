#

# about system information

__all__ = [
    "system", "get_statm", "get_sysinfo",
    "zglob", "zglob1", "zglobs", "mkdir_p", "auto_mkdir", "resymlink",
]

import sys, os, subprocess, traceback, glob, platform
from typing import Iterable
from .log import zopen, zlog, zcheck

# performing system CMD
def system(cmd: str, pp=False, ass=False, popen=False, return_code=False):
    if pp:
        zlog(f"Executing cmd: {cmd}")
    if popen:
        try:
            tmp_out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
            n = 0
            output = str(tmp_out.decode())  # byte->str
        except subprocess.CalledProcessError as grepexc:
            n = grepexc.returncode
            output = grepexc.output
        # p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        # n = p.wait()
        # output = str(p.stdout.read().decode())      # byte -> str
    else:
        n = os.system(cmd)
        output = None
    if pp:
        zlog(f"Output is: {output}")
    if ass:
        assert n == 0, f"Executing previous cmd returns error {n}"
    if return_code:
        return output, n
    else:
        return output
    # --

# get mem info
def get_statm():
    with zopen("/proc/self/statm") as f:
        rss = (f.read().split())        # strange!! readline-nope, read-ok
        mem0 = str(int(rss[1])*4//1024) + "MiB"
    try:
        line = system("nvidia-smi | grep -E '%s.*MiB'" % os.getpid())
        mem1 = line[-1].split()[-2]
    except:
        mem1 = "0MiB"
    return mem0, mem1

# get sys info
def get_sysinfo(ret_str=True, get_gpu_info=False):
    ret = {'uname': platform.uname(), 'gpu': system("nvidia-smi", popen=True) if get_gpu_info else None}
    if ret_str:
        ret = f"#== Sysinfo: {ret['uname']}\n{ret['gpu']}"
    return ret

# traceback info
# traceback.format_exc(limit=None)
# traceback.format_stack(limit=None)

# glob
def zglob(pathname: str, check_prefix="..", check_iter=0, sorted=True):
    if pathname == "":
        return []
    if pathname.startswith("__"):  # note: special semantics!!
        pathname = pathname[2:]
        check_iter = max(check_iter, 10)
    files = glob.glob(pathname)
    if len(check_prefix)>0:
        while len(files)==0 and check_iter>0:
            pathname = os.path.join(check_prefix, pathname)
            files = glob.glob(pathname)
            check_iter -= 1  # limit for checking
    if sorted:
        files.sort()
    return files

# assert there should be only one
def zglob1(pathname: str, err_act='warn', **kwargs):
    rets = zglob(pathname, **kwargs)
    if len(rets) == 1:
        return rets[0]
    else:
        zcheck(False, s=f'Not one item {pathname} -> {rets}', err_act=err_act)
        # breakpoint()
        return pathname  # return pathname by default!

# zglob for iterable
def zglobs(pathnames: Iterable[str], err_act='warn', **kwargs):
    ret = []
    if isinstance(pathnames, str):
        pathnames = [pathnames]
    for path in pathnames:
        one_ret = zglob(path, **kwargs)
        ret.extend(one_ret)
        zcheck(len(one_ret)>0, s=f'Cannot find {path}', err_act=err_act)
    return ret

# mkdir -p path
def mkdir_p(path: str, err_act='warn'):
    if os.path.exists(path):
        if os.path.isdir(path):
            return True
        zcheck(False, f"Failed mkdir: {path} exists and is not dir!", err_act=err_act)
        return False
    else:
        # os.mkdir(path)
        os.makedirs(path)
        return True

# auto mkdir
def auto_mkdir(path: str, **kwargs):
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        mkdir_p(dir_name, **kwargs)

# relink file: delete if existing
def resymlink(src, dst):
    if os.path.islink(dst):
        os.unlink(dst)
    os.symlink(src, dst)
