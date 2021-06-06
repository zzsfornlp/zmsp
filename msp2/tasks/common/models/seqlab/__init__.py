#

# labeling and especially sequence labeling
"""
# some variations (currently):
1. pairwise: inputs (whether need input_pair)
2. decoder: whether need cache and incremental decoding
3. transition-matrix: bigram output, whether need last-step predictions
"""

from .bigram import *
from .inference import *
from .seq import *
from .simple import *
