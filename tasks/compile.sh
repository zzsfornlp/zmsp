#!/usr/bin/env bash

# compile the cython modules

# 1st order algorithms
python zdpar/algo/setup.py build_ext
# high order algorithm
bash zdpar/algo/gp2/compile.sh
