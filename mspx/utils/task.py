#

# sth related with task recording

__all__ = [
    "Timer", "AccEvalEntry", "F1EvalEntry", "StatRecorder", "get_singleton_global_recorder", "MyCounter",
]

import time
from typing import Union, Callable, Dict
from collections import Counter, defaultdict, OrderedDict
from copy import deepcopy
from .file import WithWrapper
from .log import zlog
from .math import MathHelper, DivNumber
from .reg import Registrable
from .seria import Serializable

# =====
# Timer: record about time
class Timer:
    _START = 0.

    @staticmethod
    def init():  # record program start time
        Timer._START = time.time()

    @staticmethod
    def time():
        return time.time()-Timer._START

    @staticmethod
    def ctime():
        return time.ctime()

    def __init__(self, info="ANON", print_date=True, quite=False, callback_f=None):
        self.info = info
        self.print_date = print_date
        self.quite = quite
        self.callback_f = callback_f
        # --
        self.accu = 0.  # accumulated time
        self.start = Timer.time()  # start time
        self.paused = True  # whether paused

    def __repr__(self):
        return f"Timer {self.info} (accu_time={self.accu}, paused={self.paused})"

    def clear(self):
        self.accu = 0.

    def pause(self):
        if not self.paused:
            cur = Timer.time()
            self.accu += cur - self.start  # record this piece
            self.paused = True

    def resume(self):
        if self.paused:
            self.start = Timer.time()  # a new start
            self.paused = False

    def get_accu_time(self):  # get accumulated time
        return self.accu

    def begin(self):
        if not self.quite:
            cur_date = time.ctime() if self.print_date else ""
            zlog(f"Start timer: {self.info} at {self.start:.3f}. ({cur_date})", func="time")
        self.resume()

    def end(self):
        self.pause()
        if not self.quite:
            cur_date = time.ctime() if self.print_date else ""
            zlog(f"End timer: {self.info} at {Timer.time():.3f}, the period is {self.accu:.3f} seconds. ({cur_date})", func="time")
        if self.callback_f is not None:
            self.callback_f(self)  # pass the Timer self!!

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

# =====
# Helpers for eval
@Registrable.rd('e_acc')
class AccEvalEntry(Serializable):
    def __init__(self):
        self.v = DivNumber(0, 0)

    def record(self, corr: Union[int, float], all=1):
        self.v.add_xy(corr, all)

    def combine_dn(self, d: DivNumber, scale=1.):
        self.v.combine(d, scale=scale)

    def combine(self, e: 'AccEvalEntry', scale=1.):
        self.combine_dn(e.v, scale=scale)

    def scale(self, alpha: float):
        self.v.scale(alpha)

    # forward others
    @property
    def res(self): return self.v.res
    @property
    def details(self): return self.v.details
    def __float__(self): return float(self.v)
    def __repr__(self): return repr(self.v)

@Registrable.rd('e_f1')
class F1EvalEntry(Serializable):
    def __init__(self):
        self.p = AccEvalEntry()
        self.r = AccEvalEntry()

    def record_p(self, corr: int, all=1):
        self.p.record(corr, all)

    def record_r(self, corr: int, all=1):
        self.r.record(corr, all)

    def record_pr(self, corr: int, all=1):
        self.p.record(corr, all)
        self.r.record(corr, all)

    def combine_dn(self, dp: DivNumber, dr: DivNumber, scale=1.):
        if dp is not None:
            self.p.combine_dn(dp, scale=scale)
        if dr is not None:
            self.r.combine_dn(dr, scale=scale)

    def combine_acc(self, ap: AccEvalEntry, ar: AccEvalEntry, scale=1.):
        if ap is not None:
            self.p.combine(ap, scale=scale)
        if ar is not None:
            self.r.combine(ar, scale=scale)

    def combine(self, e: 'F1EvalEntry', scale=1.):
        self.combine_acc(e.p, e.r, scale=scale)

    def scale(self, alpha: float):
        self.p.scale(alpha)
        self.r.scale(alpha)

    # =====
    @property
    def res(self):
        return float(self.prf[-1])

    @property
    def prf(self):
        P, R = self.p.res, self.r.res
        F1 = MathHelper.safe_div(2 * P * R, P + R)
        # P, R, F1 = [round(z, 4) for z in (P, R, F1)]
        return P, R, F1

    @property
    def details(self):
        return self.p.details + self.r.details + (self.res, )

    def __float__(self): return float(self.res)
    def __repr__(self): return f"{str(self.p)}; {str(self.r)}; {float(self):.4f}"

# =====
# About Stat Recording

# Counter with percentages counting
class MyCounter(Counter):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)

    def __str__(self):
        return self.summary_str(30, None)

    def summary_str(self, topk=None, maxc=None):
        classname = self.__class__.__name__
        info0 = f"[{self.all_counts()}/{len(self)}]"
        info1 = "{" + ", ".join(self.topk_strs(topk)) + "}"
        return f"{classname}{info0}{info1}"[:maxc]

    def all_counts(self):
        return sum(self.values())  # all counts

    # get topk keys
    def topk_keys(self, topk: int = None, key: Callable = None):
        if key is None:
            key = lambda x: -self[x]  # by default sort by larger value
        all_keys = sorted(self.keys(), key=key)[:topk]
        return all_keys

    # get topk values with percs
    def topk_entries(self, topk: int = None, key: Callable = None):
        topk_keys = self.topk_keys(topk, key)
        # --
        res = OrderedDict()
        accu_counts = 0
        _div_all = max(self.all_counts(), 1)
        for k in topk_keys:
            c = self[k]
            accu_counts += c
            res[k] = (k, c, c/_div_all, accu_counts, accu_counts/_div_all)
        return res

    # get pd
    def topk_pb(self, topk: int = None, key: Callable = None):
        import pandas as pd
        res = self.topk_entries(topk, key)
        d = pd.DataFrame(res.values(), columns=["Key", "Count", "Perc.", "ACount", "APerc."])
        return d

    # get topk str
    def topk_strs(self, topk: int = None, key: Callable = None):
        res = self.topk_entries(topk, key)
        all_ss = []
        for key, count, perc, a_count, a_perc in res.values():
            all_ss.append(f"{key}: {count}/{perc:.3f}({a_count}/{a_perc:.3f})")
        return all_ss

    # todo(+N): currently just put it as this!
    def summary(self, **kwargs):
        return self.topk_entries(**kwargs)

# simple recoder, mainly based on Counter
class StatRecorder:
    def __init__(self, report_key: str = None, report_interval=0, report_start_timer=True):
        self.timers = defaultdict(lambda: Timer(quite=True))
        self.report_key = report_key
        self.report_interval = report_interval
        self.plain_values = Counter()  # plain values
        self.special_values: [str, MyCounter] = {}  # special ones with MyCounter
        self.reset()
        # --
        if report_key and report_start_timer:  # start this one!
            x = self.timers[report_key]
        # --

    @property
    def timer(self):  # a default timer
        return self.timers['']

    # get a simple 'with' env
    def go(self, timer=''):
        t = self.timers[timer]
        return WithWrapper(lambda: t.resume(), lambda: t.pause())

    # reset
    def reset(self):
        self.timers.clear()
        self.plain_values.clear()
        self.special_values.clear()

    # record
    # functions are optionally provided for accu & init: f(old, v)
    def record_kv(self, k: str, v: object, f_accu: Callable = None, f_init: Callable = None):
        if k not in self.plain_values and f_init is not None:
            self.plain_values[k] = f_init()
        if f_accu is None:  # by default
            self.plain_values[k] += v
        else:
            self.plain_values[k] = f_accu(self.plain_values.get(k), v)
        # --
        if k is not None and k == self.report_key:
            _vv = self.plain_values[k]
            if self.report_interval > 0 and _vv % self.report_interval == 0:
                self.summary('Process')
        # --

    # special record
    def srecord_kv(self, k: str, v: object, c=1):
        if k not in self.special_values:
            self.special_values[k] = MyCounter()
        self.special_values[k][v] += c

    def record(self, res: dict):
        for name, v in res.items():
            self.record_kv(name, v)

    # return a deep copy
    def summary(self, report=''):
        _report = (report and self.report_key)
        r = deepcopy(self.plain_values)
        if _report:
            self.timers[self.report_key].pause()  # accumulate in!
        for k, t in self.timers.items():
            r[f'_time_{k}'] = round(t.get_accu_time(), 2)
        if _report:
            _t = self.timers[self.report_key]
            zlog(f"{report}: {r} [{_t.get_accu_time():.2f}s]"
                 f"[{(r.get(self.report_key,0)/_t.get_accu_time()):.2f}d/s]", timed=True)
            _t.resume()
        return r

    def summary_special(self, k: str, **kwargs):
        if k in self.special_values:
            return self.special_values[k].summary(**kwargs)
        else:
            return None

# singleton global recorder
_singleton_global_recorder = StatRecorder()
def get_singleton_global_recorder(): return _singleton_global_recorder
