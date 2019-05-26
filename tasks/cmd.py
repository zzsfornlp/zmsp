#!/bin/python3

import sys
import importlib

all_runs = [
    # dep parsing
    "zdpar.main.train",
    "zdpar.main.test",
]

def main():
    # get running module
    mname = sys.argv[1]
    # assert mname in all_runs
    module = importlib.import_module(mname)
    # run it
    args = sys.argv[2:]
    module.main(args)

if __name__ == '__main__':
    main()
