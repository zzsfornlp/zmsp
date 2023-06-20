#

# random with np

__all__ = ["Random"]

import numpy as np
from .log import zlog, zwarn

# useful when we need groups of random generators
class Random:
    _init_times = 0
    _init_seed = 9341
    _curr_seed = None
    _seeds = {}

    @staticmethod
    def get_generator(task=''):
        g = Random._seeds.get(task, None)
        if g is None:
            if Random._init_times == 0:
                Random.init(None)   # default
            one = np.random.randint(1, 10000)
            # casual init seed according to the task name
            div = (2<<30)
            for t in task:
                one = one * ord(t) % div
            one += 1
            g = np.random.RandomState(one)  # use it as seed
            Random._seeds[task] = g
        return g

    # separate one
    @staticmethod
    def get_np_generator(seed: int = None):
        if seed is None:
            seed = Random.get_curr_seed()
        return np.random.RandomState(seed)

    # init overall
    @staticmethod
    def init(seed: int = None, quite=False):
        # assert not Random._init_flag, "Cannot init twice Random!"
        if not quite:
            zlog(f"Initial msp2.utils.random for Time={Random._init_times}")
        Random._init_times += 1
        if seed is None or seed == Random._init_seed:
            seed = Random._init_seed
        elif not quite:
            zlog(f"Manually Random init with seed={seed}.")
        Random._curr_seed = seed
        np.random.seed(seed)

    @staticmethod
    def get_curr_seed():
        return Random._curr_seed

    # =====
    # batched forever streamer
    @staticmethod
    def stream(f, *args, size=1024, **kwargs):
        while True:
            for one in f(*args, **kwargs, size=size):
                yield one
