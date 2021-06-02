#

# the final parser (hopefully a simple one)

#
# 0. labeled+recursive, direct embeddings; what object?
# 1. score as confidence for the combination and used for later calculation
# 2. encourage sparsity (with NOP op)
# 3. how to reduce?
# 4. the connections should also be distributed?

# todo(note):
# =====
# 20.01
# look at word pairs for syntax path -> tools/see_pairs.py (co-heads?)
# word drop and word pred loss: joint-training as aux loss -> directly multi-task training is not helpful
# pre-training + fine-tune?
# meaning grouping by picking and cutting the important dep links
# predict combined path as new label?
# slayer: d-self-att
