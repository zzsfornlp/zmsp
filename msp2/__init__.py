#

# The Mingled Structured Prediction v2 (zmsp2)
# author: zzs
# time: 2020.06 - now

# dependencies: pytorch, numpy, scipy, gensim, cython, pybind11, pandas
# conda install pytorch=1.5.0 -c pytorch
# conda install numpy scipy cython pybind11 pandas pip
# gensim cannot use conda?
# (optional): pip install transformers==3.1.0
# (optional): pip install stanza

VERSION_MAJOR = 0
VERSION_MINOR = 2
VERSION_PATCH = 0
VERSION_STATUS = "dev"

def version(level=3):
    return (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, VERSION_STATUS)[:level]

__version__ = ".".join(str(z) for z in version())

# file structures
"""
msp2:
# core
- data: Data Instances & ReaderWriter & Vocabs
- utils: utils
# peripherals
- tools: some useful tools
# others (not packages)
- cli: command line runners
- scripts: various helping scripts (for pre-processing or post-processing ...)
"""

# formats of todos
"""
-> TODO(!): immediate todos
-> TODO(+W): wait for maybe near future
-> todo(+N): difficult todos
-> todo(+[int]): plain ones with levels
-> todo(note): just noting
"""
