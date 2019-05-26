#

from msp.utils import Random, Conf
import numpy as np

# Common NN Init
class NIConf(Conf):
    def __init__(self):
        # nolookup / lookup
        self.use_my_init = False
        # current init
        self.init_def_nl = "glorot"
        self.init_def_l = "glorot"
        self.init_scale_nl = 1.
        self.init_scale_l = 1.
        #
        self.random_seed = 9347
        self.random_cuda_seed = 9349
        #
        self.num_threads = 4    # maximum NUM_THREADS if using cpu
        self.device = -1        # -1: cpu, [0,): gpu
        # toolkit specific

# global one (default one)
COMMON_CONFIG = NIConf()

# first init params with np: return is C-order as the default of numpy
# -- also fix some hyper-params here
def _my_get_params_init(shape, init, lookup):
    # shape is a tuple of dims
    assert init in ["default", "random", "glorot", "ortho", "gaussian", "zeros"], "Unknown init method %s" % init
    poss_scale = COMMON_CONFIG.init_scale_l if lookup else COMMON_CONFIG.init_scale_nl
    if len(shape) == 1:     # set bias to 0
        return np.zeros((shape[0],))
    else:
        # get defaults
        if init == "default":
            init = COMMON_CONFIG.init_def_l if lookup else COMMON_CONFIG.init_def_nl
        # specifics
        if init == "glorot":
            if lookup:  # special for lookups
                shape_g = (shape[-1], )  # fan-out for lookup
            else:
                shape_g = shape
            w0 = Random.random_sample(shape, "winit")  # [0,1)
            w0 = (w0-0.5)*(2*(np.sqrt(3.0*len(shape_g)/(sum(shape_g)))))
            return w0*poss_scale
        elif init == "random":
            w0 = Random.random_sample(shape, "winit")  # [0,1)
            w0 = (w0-0.5)*2
            return w0*poss_scale
        elif init == "gaussian":
            w0 = Random.randn_clip(shape, "winit")
            return w0*poss_scale
        elif init == "ortho":
            assert len(shape)==2 and shape[0] % shape[1] == 0, "Bad shape %s for ortho_init" % shape
            num = shape[0] // shape[1]
            if num == 1:
                w0 = Random.ortho_weight(shape[1], "winit")
            else:
                w0 = np.concatenate([Random.ortho_weight(shape[1], "winit") for _ in range(num)])
            return w0*poss_scale
        elif init == "zeros":
            return np.zeros(shape)

# helper
# get name
def get_unique_name(d, name):
    if name in d:
        d[name] += 1
    else:
        d[name] = 0
    return name + str(d[name])
