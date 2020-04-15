#

# The Mingled Structured Prediction (v0plus) package
# author: zzs
# time: 2018.02 - now

# dependencies: pytorch, numpy, scipy, gensim, cython, pybind11, pandas
# conda install pytorch numpy scipy gensim cython pybind11 pandas

VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 1
VERSION_STATUS = "dev"

# specific todos
# nn optimizer / param groups?
# check nn module (for simplification?)
# new model/training/testing scheme -> make it more modularized
# Conf's init: what types
# dropout setting: use training/testing(with/wo)-mode
# easy-to-use calculations result-representing tools for analysis
# various tools for python as the replacement of direct bash shell
# gru and cnn have problems?
# ----
# -- Next Version principles and goals:
# nlp data types
# use type hint
# checkings and reportings
# use eval for Conf
# io and serialization
# summarize more common patterns, including those in scripts
# everything has (more flexible) conf
# more flexible save/load for part of model; (better naming and support dynamic adding and deleting components!!)

def version():
    return (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, VERSION_STATUS)

__version__ = ".".join(str(z) for z in version())

# basic principles
# 0. shared pattern: make it re-usable
# 1. lazy eval: calc as need & with cache
# 2. cpp style: maybe transfer to other lang (c++) in the future
# 3. search oriented: focus should be at the searching part
# 4. others: avoid-early-opt, checking&snapshot, clear-code&locality, table-driven&oo
# !!: (renewed) composition rather than inheritance, that is, some useful pieces rather than a framework
# !!: (again-corrected) The goal is not to build a whole framework, but several useful pieces.

# conventions
# todo(0)/todo(warn)/todo(note): simply warning or noting
# todo(+N): need what level of efforts, +N means lots of efforts
# TODO: unfinished, real todo

# hierarchically: msp -> scripts / tasks, no reverse ref allowed!

"""
Driver -- Utils
Data -- Model (nn) -- Search
"""

# -- full-usage init order (example: tasks.zdpar.common.confs.init_everything)
# ** no need to init if only use default ones
# top-level
# utils
# nn
