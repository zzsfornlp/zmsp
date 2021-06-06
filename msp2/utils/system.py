#

# about system information

__all__ = [
    "system", "get_statm",
]

import sys, os, subprocess, traceback, glob
from .log import zopen, zlog

# performing system CMD
def system(cmd: str, pp=False, ass=False, popen=False):
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
    return output

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

# traceback info
# traceback.format_exc(limit=None)
# traceback.format_stack(limit=None)
