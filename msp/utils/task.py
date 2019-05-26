# performing something and recording
import time
from .log import zlog

from msp.cmp import RecNode

from collections import defaultdict, OrderedDict, Iterable

# Timer: record about time
class Timer(object):
    START = 0.

    @staticmethod
    def init():
        Timer.START = time.time()

    @staticmethod
    def systime():
        return time.time()-Timer.START

    # constructions
    @staticmethod
    def obtain_anon_timer():
        return Timer("anon", "whatever", print_date=False, quiet=True)

    def __init__(self, tag, info, print_date=True, quiet=False, callback_f=None):
        self.tag = tag
        self.info = info
        self.print_date = print_date
        self.quiet = quiet
        self.info = info
        self.accu = 0.   # accumulated time
        self.paused = False
        self.start = Timer.systime()
        self.callback_f = callback_f    # callback at the end

    def pause(self):
        if not self.paused:
            cur = Timer.systime()
            self.accu += cur - self.start
            self.start = cur
            self.paused = True

    def resume(self):
        if not self.paused:
            zlog("Timer should be paused to be resumed.", func="warn")
        else:
            self.start = Timer.systime()
            self.paused = False

    def get_time(self):
        origin_paused = self.paused
        self.pause()
        if not origin_paused:
            self.resume()
        return self.accu

    def begin(self):
        self.start = Timer.systime()
        if not self.quiet:
            cur_date = time.ctime() if self.print_date and not self.quiet else ""
            zlog("Start timer %s: %s at %.3f. (%s)" % (self.tag, self.info, self.start, cur_date), func="time")

    def end(self):
        self.pause()
        if not self.quiet:
            cur_date = time.ctime() if self.print_date and not self.quiet else ""
            zlog("End timer %s: %s at %.3f, the period is %.3f seconds. (%s)" % (self.tag, self.info, Timer.systime(), self.accu, cur_date), func="time")
        if self.callback_f is not None:
            self.callback_f(self.tag, self.info, self.accu)

    def __enter__(self):
        self.begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()

#
class AccuItem(object):
    def accept(self, v):
        raise NotImplementedError()

    def value(self):
        raise NotImplementedError()

class AccuItemNumber(AccuItem):
    def __init__(self, dtype):
        self._v = dtype(0)

    def accept(self, v):
        self._v += v

    def value(self):
        return self._v

    def __str__(self):
        return str(self._v)

    def __int__(self):
        return int(self._v)

    def __float__(self):
        return float(self._v)

class AccuItemDistr(AccuItem):
    def __init__(self):
        self.node = RecNode()

    def accept(self, v):
        self.node.add_seq(v)

    def value(self):
        thems = self.node.ks_info(sort_by=lambda x: -x["num"])
        ordered_val = OrderedDict()
        for one in thems:
            ordered_val[one["key"]] = one
        return ordered_val

    def __str__(self):
        return self.node.ks_str(sort_by=lambda x: -x["num"])

#
# todo(warn): may depend on specific field name
class StatRecorder(object):
    DEFAULT_SPECIAL_HANDLERS = {
        # count & distribution
        "distr": lambda: AccuItemDistr(),
    }

    @staticmethod
    def guess_default_handler(name, v):
        if isinstance(name, Iterable):
            name = str(name[0])
        fileds = name.split("_")
        start_name = fileds[0]
        if start_name in StatRecorder.DEFAULT_SPECIAL_HANDLERS and len(fileds)>1:
            return StatRecorder.DEFAULT_SPECIAL_HANDLERS[start_name]
        else:
            return lambda: AccuItemNumber(type(v))

    # =====
    def __init__(self, timing, item_handlers=None):
        # running timer
        self.timing = timing
        self.item_handlers = item_handlers if item_handlers is not None else {}
        # self.avg_names = set(avg_names)     # average all numbers over this
        #
        self.timer = None
        self.values = {}
        self.reset()

    def go(self):
        return StatRecorder._Recorder(self)

    class _Recorder(object):
        def __init__(self, r):
            self.r = r

        def __enter__(self):
            self.r.before()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.r.after()

    # before each recording, mainly start Timer
    def before(self):
        if self.timing:
            self.timer.resume()

    def after(self):
        if self.timing:
            self.timer.pause()

    # record one, recording is driven by res
    # -> res should be a dictionary

    def record_kv(self, name, v):
        if name not in self.values:
            # init the entry
            if name in self.item_handlers:
                ff = self.item_handlers[name]
            else:
                ff = StatRecorder.guess_default_handler(name, v)
                self.item_handlers[name] = ff
            self.values[name] = ff()
        self.values[name].accept(v)

    def record(self, res):
        for name, v in res.items():
            self.record_kv(name, v)

    # get calculated stat
    # -> todo(warn): average explicitly at outside
    def summary(self, get_v=True, get_str=False):
        vals = {}
        # time and base items
        base_vals = {}
        if self.timing:
            accu_time = self.timer.get_time()
            vals["time"] = accu_time
            # self.avg_names.add("time")        # avg over time
        for name, item in self.values.items():
            base_vals[name] = item.value()
        if get_str:
            for n, v in self.values.items():
                # todo(warn): hard to repeat name
                vals["_str_" + str(n)] = str(v)
        if get_v:
            vals.update(base_vals)
        return vals

    # reset
    def reset(self):
        if self.timing:
            self.timer = Timer.obtain_anon_timer()
            self.timer.pause()
        self.values.clear()

    # in fact, should be method for dict
    # div_pair = [(div_name, iter-of-to_div_name), ...]
    @staticmethod
    def div_values(val, div_pairs):
        for div_name, to_div_set in div_pairs:
            if div_name in val:
                dv = val[div_name]
                if dv == 0:
                    dv = 1e-6
                for name in to_div_set:
                    if name in val:
                        cname = name+"_"+div_name
                        val[cname] = val[name] / dv
