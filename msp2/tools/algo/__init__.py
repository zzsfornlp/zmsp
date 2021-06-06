#

# some algorithms for dep-parsing, some can be implemented with cython to speed-up

# for the algorithms, mainly adapted from NeuroNLP2/nnpgdparser/MaxParser

# CPU versions
from .nmst import mst_unproj, mst_proj, mst_greedy
from .nmst import marginal_unproj, marginal_proj, marginal_greedy
# Tensor versions (can be simply wrappers)
from .nmst import nmst_unproj, nmst_proj, nmst_greedy
from .nmst import nmarginal_unproj, nmarginal_proj, nmarginal_greedy
