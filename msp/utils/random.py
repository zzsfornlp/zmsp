import numpy as np

from .log import printing
from .check import zwarn

# random from numpy

class Random(object):
    _init_times = 0
    _init_seed = 9341
    _seeds = {}

    @staticmethod
    def get_generator(task):
        g = Random._seeds.get(task, None)
        if g is None:
            if Random._init_times==0:
                Random.init(None)   # default
            one = np.random.randint(0, 10000)
            # casual init seed according to the task name
            div = (2<<30)
            for t in task:
                one = one * ord(t) % div
            one += 1
            g = np.random.RandomState(one)
            Random._seeds[task] = g
        return g

    @staticmethod
    def init(seed=None):
        # assert not Random._init_flag, "Cannot init twice Random!"
        if Random._init_times > 0:
            zwarn("Initial utils.random more than once (%d)!!" % Random._init_times)
        Random._init_times += 1
        if seed is None or seed==Random._init_seed:
            seed = Random._init_seed
        else:
            printing("Manually Random init with seed=%s." % seed)
        np.random.seed(seed)

    # ====================
    # mostly adopting numpy

    # todo(warn): inplaced
    @staticmethod
    def shuffle(x, task=""):
        return Random.get_generator(task).shuffle(x)

    @staticmethod
    def binomial(n, p, size=None, task=""):
        return Random.get_generator(task).binomial(n, p, size)

    @staticmethod
    def ortho_weight(n, task=""):
        W = Random.randn_clip((n, n), task=task)
        u, s, v = np.linalg.svd(W)
        return u.astype(np.float32)

    @staticmethod
    def random_sample(size, task=""):
        return Random.get_generator(task).random_sample(size)

    @staticmethod
    def randn_clip(size, cc=2, task=""):
        w = Random.get_generator(task).standard_normal(size)
        if cc > 0:
            w.clip(-cc, cc)   # clip [-2*si, 2*si]
        return w

    @staticmethod
    def multinomial(n, p, size=None, task=""):
        return Random.get_generator(task).multinomial(n, p, size)

    @staticmethod
    def multinomial_select(n, p, size=None, task=""):
        # once selection
        x = Random.multinomial(n, p, size, task)
        return np.argmax(x, axis=-1)

    @staticmethod
    def choice(a, size=None, replace=True, p=None, task=""):
        if size is not None and not replace and len(a) <= size:
            return list(a)
        return Random.get_generator(task).choice(a, size, replace, p)

    @staticmethod
    def random_bool(true_rate, size=None, task=""):
        x = Random.random_sample(size, task)
        return x < true_rate

    @staticmethod
    def randint(low, high=None, size=None, task=""):
        return Random.get_generator(task).randint(low, high, size)
