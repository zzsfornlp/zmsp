#

# the MuRec encoder

# -----
# for the murec-att-node, it aims to collect attentinal evidences:
# c1_scorer / c2_normer / c3_collector
# for the murec-layer, it is composed of murec-att & rec-wrapper

from .vrec import VRecEncoderConf, VRecEncoder, VRecCache, VRecConf, VRecNode
