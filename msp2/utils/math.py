#

# About Math

__all__ = [
    "MathHelper", "DivNumber",
]

#
import numpy as np
try:
    from scipy.misc import logsumexp as scipy_logsumexp
except:
    from scipy.special import logsumexp as scipy_logsumexp
from math import isclose as math_isclose, exp as math_exp
from typing import Union

class MathHelper(object):
    #
    @staticmethod
    def softmax(vals, axis=0):
        exps = np.exp(vals)
        probs = exps / np.sum(exps, axis=axis, keepdims=True)
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

    @staticmethod
    def safe_div(x, y):
        return x if y==0 else x/y

# div number which can be used for eval
class DivNumber:
    _REPR_DIGIT = 4

    def __init__(self, x: Union[int,float], y: Union[int,float]):
        self.x = x
        self.y = y

    @property
    def res(self):
        return MathHelper.safe_div(self.x, self.y)

    @property
    def details(self):
        return (self.x, self.y, self.res)

    def add_x(self, d: Union[int,float]): self.x += d
    def add_y(self, d: Union[int,float]): self.y += d

    def add_xy(self, dx: Union[int,float], dy: Union[int,float]):
        self.x += dx
        self.y += dy

    def __float__(self):
        return self.res

    def __repr__(self):
        r = f"%.{DivNumber._REPR_DIGIT}f" % self.res
        return f"{self.x}/{self.y}={r}"

    def copy(self):
        return DivNumber(self.x, self.y)

    def scale(self, alpha: float):
        self.x *= alpha
        self.y *= alpha

    def combine(self, other: 'DivNumber', scale=1.):
        self.add_xy(other.x*scale, other.y*scale)

    @staticmethod
    def combine_two(d1: 'DivNumber', d2: 'DivNumber'):
        ret = d1.copy()
        ret.combine(d2)
        return ret

    @staticmethod
    def zero():  # get zero start!
        return DivNumber(0., 0.)
