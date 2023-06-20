#

__all__ = [
    "MyResource", "MyEngineConf", "MyEngine",
]

import os
import time
import traceback
from typing import List, Iterable
from collections import Counter, defaultdict
from threading import Thread, Lock
from multiprocessing import Process, cpu_count
from mspx.utils import zlog, zopen, Conf, Configurable, ConfEntryList
from mspx.tools.utils import *
from mspx.tools.analyze import *
from .task import *

# resource
class MyResource:
    def __init__(self, rtype: str, rid: int, center=None):
        assert rtype in ['gpu', 'cpu']
        self.rtype = rtype
        self.rid = rid
        self.in_usage = False  # whether in usage
        self.center = center  # belongs to which center?

    @property
    def status(self):
        ret = "B" if self.in_usage else "I"  # budy or idle?
        return ret

    def __repr__(self):
        return f"R{self.rtype}:{self.rid}[{self.status}]"

class MyResourceCenter:
    def __init__(self, **kwargs):
        self.rs = defaultdict(dict)  # rs -> {i: MyResource}
        self.rs.update({'cpu': {}, 'gpu': {}})
        self.lock = Lock()
        self.add_rs(**kwargs)
        # --

    def __repr__(self):
        return f"ResourceCenter: {self.get_status()}"

    def get_status(self):
        ret = {}
        with self.lock:
            for k, rs in self.rs.items():
                if len(rs) > 0:
                    rs_alloc, rs_idle = [], []
                    for r in rs.values():
                        if r.in_usage:
                            rs_alloc.append(r.rid)
                        else:
                            rs_idle.append(r.rid)
                        ret[k] = f"A[{len(rs_alloc)}]={rs_alloc},I[{len(rs_idle)}]={rs_idle}"
        return ret

    def _spec_rs(self, v):
        if isinstance(v, int):
            ret = [v]
        elif isinstance(v, str):
            ret = ConfEntryList.list_convert(v, int)
        else:
            ret = list(v)
        return ret

    def add_rs(self, **kwargs):  # rname -> rspec
        add_counts = Counter()
        with self.lock:
            for k, rs in kwargs.items():
                if k not in self.rs:
                    self.rs[k] = {}
                dd = self.rs[k]
                rs = self._spec_rs(rs)
                for rr in rs:
                    if rr not in dd:
                        dd[rr] = MyResource(k, rr, center=self)
                        add_counts[k] += 1
        # --
        zlog(f"Added {add_counts}, now: {self}")
        # --

    def del_rs(self, **kwargs):  # rname -> rspec
        del_counts = Counter()
        with self.lock:
            for k, rs in kwargs.items():
                if k not in self.rs: continue
                dd = self.rs[k]
                rs = self._spec_rs(rs)
                for rr in rs:
                    if rr in dd:
                        dd[rr].center = None
                        del dd[rr]
                        del_counts[k] += 1
        # --
        zlog(f"Deleted {del_counts}, now: {self}")
        # --

    def allocate(self, **kwargs):  # rname -> int
        with self.lock:
            rsd = {}
            for k, n in kwargs.items():
                n = int(n)
                idle_rs = [z for z in self.rs[k].values() if not z.in_usage]
                if n > 0 and len(self.rs[k]) == 0:
                    raise RuntimeError(f"No resources available {k} for allocating {kwargs}")
                if len(idle_rs) < n:
                    return None  # not enough!
                alloc_rs = idle_rs[:n]
                for rr in alloc_rs:
                    rr.in_usage = True
                rsd[k] = [z.rid for z in alloc_rs]
                # # --
                # if k == 'cpu':
                #     rsd['_prefix'] += f" OMP_NUM_THREADS={n} MKL_NUM_THREADS={n}"
                # elif k == 'gpu':
                #     rsd['_prefix'] += f" CUDA_VISIBLE_DEVICES={','.join(rsd[k])}"
                # # --
            return rsd

    def release(self, rsd: dict):
        with self.lock:
            for k, rs in rsd.items():
                if k in self.rs:
                    dd = self.rs[k]
                    for rid in rs:
                        if rid in dd:
                            dd[rid].in_usage = False
        # --

# engine
class MyEngineConf(AnalyzerConf):
    def __init__(self):
        super().__init__()
        # --
        self.name = ''
        self.para = True  # allow parallel running (using threads)
        self.wait_time = 5.  # how many seconds?
        self.res_prefix = "_res."
        self.extra_info = {}
        self.loop_as_daemon = False
        # --
        self.cpus = 0.9  # number of total cpu available or (c*num_process)
        self.gpus = []  # gpu ids
        # --

    @property
    def real_cpus(self):
        if self.cpus < 1.:
            return int(cpu_count() * self.cpus)
        else:
            return int(self.cpus)

class MyEngine(Analyzer):
    def __init__(self, conf: MyEngineConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: MyEngineConf = self.conf
        # --
        self.rc = MyResourceCenter(cpu=list(range(conf.real_cpus)), gpu=[int(z) for z in conf.gpus])
        self.tasks: List[MyTask] = []
        # --

    def __repr__(self):
        return f"MyEngine (rc={self.rc})"

    # --
    # cmd

    def list_tasks(self, *tids, keyer=None, verbose=False):
        # --
        if len(tids) == 0:
            tasks = list(self.tasks)  # by default list all
        else:
            ss = set(tids)
            tasks = [t for tii, t in enumerate(self.tasks) if str(t.id) in ss or str(tii) in ss]
        # --
        if keyer:
            if isinstance(keyer, str):
                keyer = eval(keyer)
            tasks.sort(key=keyer)
        return [t.descr(verbose) for t in tasks]

    def do_jobs(self, *args, **kwargs):
        tasks = self.list_tasks(*args, **kwargs)
        zlog(f"List tasks ({len(tasks)}) by cmd jobs {args} {kwargs}:")
        for t in tasks:
            zlog(t)
        # --

    def do_rs_add(self, **kwargs): self.rc.add_rs(**kwargs)
    def do_rs_del(self, **kwargs): self.rc.del_rs(**kwargs)
    def do_rs(self): zlog(self.rc)

    def write_results(self):
        if self.conf.name and self.conf.res_prefix:
            with zopen(f"{self.conf.res_prefix}{self.conf.name}", "w") as fd:
                for one in sorted(self.tasks, reverse=True, key=lambda t: -100. if t.result is None else float(t.result)):
                    if one.ended:
                        fd.write(one.descr(True)+"\n")
        # --

    def wait(self):
        # wait for all tasks to be finished
        while True:
            flag_all_done = all(t.ended for t in self.tasks)
            if flag_all_done:
                break
            time.sleep(self.conf.wait_time)

    def run_tasks(self, task_iter: Iterable[MyTask]):
        for task in task_iter:
            while True:
                # try to release finished jobs
                for t in self.tasks:
                    if t.ended and t.rsd is not None:
                        self.rc.release(t.rsd)
                        t.rsd = None
                # try to get resources
                required_rs = task.required_rs
                rsd = self.rc.allocate(**required_rs)
                if rsd is not None:  # success
                    task.rsd = rsd
                    break
                else:  # wait
                    time.sleep(self.conf.wait_time)
            # running
            self.tasks.append(task)
            task.run(use_thread=self.conf.para)
        zlog("!! Loop ended, wait for all to finish.")
        self.wait()
        self.write_results()
        zlog("!! All done.")

    def main(self, task_iter: Iterable[MyTask]):
        conf: MyEngineConf = self.conf
        # --
        run_thread = None
        if conf.para:
            run_thread = Thread(target=self.run_tasks, args=(task_iter, ))
            run_thread.start()
        else:
            self.run_tasks(task_iter)
        # --
        if conf.do_loop:
            self.loop()  # self's thread do parse-cmd
        else:
            zlog("No cmd loop, wait for all tasks to be done!!")
        if run_thread is not None:
            run_thread.join()
        zlog("Finished, all done!!")
