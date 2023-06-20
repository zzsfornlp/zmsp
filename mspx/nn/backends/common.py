#

from typing import List, Union, Tuple
import numpy as np
from mspx.utils import Random, Conf

# Common NN Init
class NIConf(Conf):
    pass

# optimizer conf
class OptimConf(Conf):
    def __init__(self):
        self.optim_type = "adam"
        self.sgd_momentum = 0.85  # for "sgd"
        # self.adam_betas = [0.9, 0.999]  # for "adam"
        self.adam_betas = [0.9, 0.98]  # for "adam"
        self.adam_eps = 1e-8  # for "adam"s
        self.weight_decay = 0.
