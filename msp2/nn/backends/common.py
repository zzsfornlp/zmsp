#

from typing import List, Union, Tuple
import numpy as np
from msp2.utils import Random, Conf

# Common NN Init
class NIConf(Conf):
    def __init__(self):
        self.use_my_init = False
        # nolookup / lookup
        self.init_def_nl = "glorot"
        self.init_def_l = "glorot"
        self.init_scale_nl = 1.
        self.init_scale_l = 1.
        # --
        self.random_seed = 9347
        self.random_cuda_seed = 9349
        self.num_threads = 4  # maximum NUM_THREADS if using cpu
        self.device = -1  # -1: cpu, [0,): gpu

# first init params with np: return is C-order as the default of numpy
# -- also fix some hyper-params here
def _my_get_params_init(conf: NIConf, shape: Union[List[int], Tuple[int]], init: Union[str, object], lookup: bool):
    # shape is a tuple of dims
    assert init in ["default", "random", "glorot", "ortho", "gaussian", "zeros"], f"Unknown init method {init}"
    poss_scale = conf.init_scale_l if lookup else conf.init_scale_nl
    if len(shape) == 1:  # set bias to 0
        return np.zeros((shape[0],))
    else:
        # get defaults
        if init == "default":
            init = conf.init_def_l if lookup else conf.init_def_nl
        _gen = Random.get_generator("param")
        # specifics
        if init == "glorot":
            if lookup:  # special for lookups
                shape_g = (shape[-1], )  # fan-out for lookup
            else:
                shape_g = shape
            w0 = _gen.random_sample(shape)  # [0,1)
            w0 = (w0-0.5)*(2*(np.sqrt(3.0*len(shape_g)/(sum(shape_g)))))
            return w0*poss_scale
        elif init == "random":
            w0 = _gen.random_sample(shape)  # [0,1)
            w0 = (w0-0.5)*2
            return w0*poss_scale
        elif init == "gaussian":
            w0 = _randn_clip(_gen, shape, 2.)  # clip to [-2, 2]
            return w0*poss_scale
        elif init == "ortho":
            # todo(note): always assume init square matrices
            assert len(shape)==2 and (shape[0] % shape[1] == 0 or shape[1] % shape[0] == 0), f"Bad shape {shape} for ortho_init!"
            orig_num = shape[0] // shape[1]
            if orig_num == 0:
                num = shape[1] // shape[0]
            else:
                num = orig_num
            if num == 1:
                w0 = _ortho_weight(_gen, shape[1])
            else:
                w0 = np.concatenate([_ortho_weight(_gen, shape[1]) for _ in range(num)])
            if orig_num == 0:  # reverse it!
                w0 = np.transpose(w0)
            return w0*poss_scale
        elif init == "zeros":
            return np.zeros(shape)

# helper functions
def _randn_clip(gen, shape, r: float):
    w0 = gen.standard_normal(shape)  # N(0,1)
    w0 = w0.clip(-r, r)  # clip to [-2, 2]
    return w0

def _ortho_weight(gen, size: int):
    W = _randn_clip(gen, size, 2.)
    u, s, v = np.linalg.svd(W)
    return u.astype(np.float32)

# =====
# optimizer conf
class OptimConf(Conf):
    def __init__(self):
        self.optim_type = "adam"
        self.sgd_momentum = 0.85  # for "sgd"
        # self.adam_betas = [0.9, 0.999]  # for "adam"
        self.adam_betas = [0.9, 0.98]  # for "adam"
        self.adam_eps = 1e-9  # for "adam"
        self.adadelta_rho = 0.95  # for "adadelta"
        self.adadelta_eps = 1e-6
        self.grad_clip = 5.0
        self.weight_decay = 0.
