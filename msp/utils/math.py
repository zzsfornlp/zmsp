#
import numpy as np
from scipy.misc import logsumexp as scipy_logsumexp
from math import isclose as math_isclose, exp as math_exp

class MathHelper(object):
    #
    @staticmethod
    def softmax(vals):
        exps = np.exp(vals)
        probs = exps / np.sum(exps, axis=0)
        return probs

    @staticmethod
    def logsumexp(a):
        return float(scipy_logsumexp(a))

    @staticmethod
    def isclose(a, b):
        # following numpy's default
        return math_isclose(a, b, rel_tol=1.e-5, abs_tol=1.e-8)

    exp = math_exp

    @staticmethod
    def sigmoid(vals):
        exps = np.exp(-1*vals)
        return 1. / (1. + exps)

    @staticmethod
    def upper_int(x):
        y = int(x)
        return y if x==y else y+1
