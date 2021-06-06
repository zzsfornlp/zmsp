#!/bin/python3

# run scripts in other packages

import sys
import importlib

def main():
    # get running module
    mname = sys.argv[1]
    # assert mname in all_runs
    module = importlib.import_module(mname)
    # run it
    args = sys.argv[2:]
    module.main(*args)

if __name__ == '__main__':
    main()
