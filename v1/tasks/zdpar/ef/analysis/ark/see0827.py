#

try:
    from tasks.zdpar.ef.analysis.common import *
except:
    from common import *

import sys

#
def main(args):
    pass

if __name__ == '__main__':
    main(*sys.argv[1:])

# todo(note): error fixing with multiple passes of fixing templates
# 0. ignore punct, 1. mwe/complex-np errors (usually local), 2. attachment error (PP, NP, CONJ, ...)
# looking at the relations of the errored nodes in the original tree (according to the frame-structure of the original tree)
# bottom-up or top-down fixing?
# what is the actual relation between the wrongly predicted edge (mod, pred-head, correct-head)?
# -> the "important" errors are the one that creates cycles if added to the original tree
# --> attachment error vs. head error (phrase-internal error and phrase-external error)
# =====
