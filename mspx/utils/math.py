#

# About Math

__all__ = [
    "MathHelper", "DivNumber",
]

#
import numpy as np
from math import isclose as math_isclose, exp as math_exp
from typing import Union
from .reg import Registrable
from .seria import Serializable

class MathHelper:
    @staticmethod
    def softmax(vals, masks=None, axis=-1):
        exps = np.exp(vals - vals.max(axis, keepdims=True))
        if masks is not None:
            exps *= masks  # 0s are masked out!
        probs = exps / np.sum(exps, axis=axis, keepdims=True)
        return probs

    @staticmethod
    def logsumexp(a):
        from scipy.special import logsumexp as scipy_logsumexp
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
@Registrable.rd('divN')
class DivNumber(Serializable):
    def __init__(self, x: Union[int,float]=0, y: Union[int,float]=0, repr_digit=4):
        self.x = x
        self.y = y
        self._repr_digit = repr_digit

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
        return float(self.res)

    def __repr__(self):
        r = f"%.{self._repr_digit}f" % self.res
        return f"{self.x}/{self.y}={r}"

    def scale(self, alpha: float):
        self.x *= alpha
        self.y *= alpha

    def combine(self, other: 'DivNumber', scale=1.):
        self.add_xy(other.x*scale, other.y*scale)

    @staticmethod
    def combine_two(d1: 'DivNumber', d2: 'DivNumber'):
        ret = DivNumber(d1.x, d1.y)
        ret.combine(d2)
        return ret
