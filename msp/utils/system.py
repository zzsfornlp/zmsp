# about system information

import sys, os, subprocess, traceback, glob
from .log import zopen, zlog
from .check import zcheck

def system(cmd, pp=False, ass=False, popen=False):
    if pp:
        zlog("Executing cmd: %s" % cmd)
    if popen:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        n = p.wait()
        output = str(p.stdout.read().decode())      # byte -> str
    else:
        n = os.system(cmd)
        output = None
    if pp:
        zlog("Output is: %s" % output)
    if ass:
        zcheck(n==0, "Executing previous cmd returns error %s." % n)
    return output

def dir_msp():
    dir_name = os.path.dirname(os.path.abspath(__file__))
    dir_name2 = os.path.join(dir_name, "..")
    return dir_name2

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

class FileHelper(object):
    isfile = os.path.isfile
    exists = os.path.exists
    listdir = os.listdir
    #
    path_join = os.path.join
    path_basename = os.path.basename
    path_dirname = os.path.dirname

    @staticmethod
    def read_multiline(fd, skip_f=lambda x: len(x.strip()) == 0, ignore_f=lambda x: False):
        lines = []
        line = None
        # skip continuous empty lines
        while True:
            line = fd.readline()
            if len(line) == 0:
                return None
            if not skip_f(line):
                break
        # collect non-empty lines
        while not skip_f(line):
            if not ignore_f(line):
                lines.append(line)
            line = fd.readline()
            if len(line) == 0:
                break
        return lines

    @staticmethod
    def glob(pathname, assert_exist=False, assert_only_one=False):
        files = glob.glob(pathname)
        if assert_only_one:
            assert len(files) == 1
        elif assert_exist:  # only_one leads to exists
            assert len(files) > 0
        return files

# get tracebacks
def extract_stack(num=-1):
    # exclude this frame
    frames = traceback.extract_stack()[1:]
    if num > 0:
        frames = frames[:num]
    return frames
