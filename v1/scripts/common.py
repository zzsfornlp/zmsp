#

# helpers from msp

import os, subprocess, sys
from msp.utils import system, zlog, dir_msp

printing = zlog

def dir_scripts():
    dir_name = os.path.dirname(os.path.abspath(__file__))
    return dir_name

dir_root = dir_msp
