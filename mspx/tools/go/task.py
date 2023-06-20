#

__all__ = [
    "MyTaskConf", "MyTask", "MyTaskSeries",
]

import os
import sys
import re
import traceback
import time
from threading import Thread
from typing import List, Dict
from collections import OrderedDict, Counter
from mspx.utils import ZResult, InfoField, Conf, Configurable, IdAssignable, zlog, Timer, zopen
from mspx.utils import system as my_srun

# conf
class MyTaskConf(Conf):
    def __init__(self):
        super().__init__()
        # basic
        self.id = ""  # to be filled
        self.req_rs = {}  # required resources, key->int, e.g., {'cpu':2,'gpu':1}
        self.verbose = ''  # whether print details
        self.no_popen = 'Y'  # whether not using popen
        # execution
        self.cmd = ""  # running cmd
        self.cmd_res = ""  # how to get the sys-run result?

    def new_with_repl(self, repl_dict):
        ret = self.copy()
        for k in list(ret.__dict__.keys()):
            if k.startswith("cmd"):
                ss = getattr(ret, k)
                for rk, rv in repl_dict.items():
                    ss = ss.replace(rk, rv)
                setattr(ret, k, ss)
        return ret

# one task
class MyTask(Configurable, IdAssignable):
    def __init__(self, conf: MyTaskConf = None, **kwargs):
        super().__init__(conf, **kwargs)
        conf: MyTaskConf = self.conf
        if not conf.id:  # assign an auto ID
            conf.id = f"T{self.get_new_id()}"
        # --
        # runtime info
        self.status = 'init'  # init, run, fin, err
        self.result = None
        self.output = ""
        self.rsd = None  # allocated resources
        self.thread = None  # thread for running
        # --

    @staticmethod
    def make_task(d: Dict):
        if 'tasks' in d:
            sub_tasks = [MyTask.make_task(z) for z in d['tasks']]
            del d['tasks']
            conf = MyTaskConf.create_from_dict(d)
            ret = MyTaskSeries(sub_tasks, conf)
        else:
            conf = MyTaskConf.create_from_dict(d)
            ret = MyTask(conf)
        return ret

    def descr(self, verbose=False):
        s = f"[{self.id if self.id else ''}:{self.status}:{self.rsd if self.rsd else ''}]"
        if verbose:
            s += f" {self.conf.cmd} -> {str(self.result)}"
        return s

    def __repr__(self): return self.descr()
    @property
    def id(self): return self.conf.id
    def set_id(self, id: str): self.conf.id = id
    @property
    def required_rs(self): return self.conf.req_rs
    @property
    def ended(self): return self.status in ['fin', 'err']

    def run(self, use_thread: bool):
        if use_thread:
            self.thread = Thread(target=self._run)
            self.thread.start()
        else:
            self._run()

    def _run(self):
        conf: MyTaskConf = self.conf
        with Timer(f"Run task {self}:"):
            self.status = 'run'
            try:
                if conf.verbose:
                    zlog(f"Execute: {conf.cmd}")
                main_res, self.out = self._run_cmd(conf.cmd, (not conf.no_popen))
                if conf.verbose:
                    zlog(f"Finish: {conf.cmd} -> {main_res}")
                    zlog(self.out)
                if conf.cmd_res:
                    self.result, _ = self._run_cmd(conf.cmd_res)
                else:
                    self.result = main_res
                self.status = 'fin'
            except:
                zlog(f"Exception when running {self}: \n{traceback.format_exc()}")
                self.status = 'err'
        # --

    def _run_cmd(self, c: str, popen=True):
        c = c.strip()  # note: check method
        if c.startswith('[') and ']' in c:
            _end = c.index(']')
            method = c[1:_end]
            c = c[_end+1:]
        else:  # a default one
            method = "sys"
        # --
        res = None
        output = None
        if method == "py":
            try:
                res = eval(c)  # first try eval
            except:
                # assert 'res=' in c
                exec(c, globals(), locals())  # otherwise exec, should assign "res" explicitly
                res = locals().get('res')
        elif method == "sys":
            _prefix_envs = []
            if 'cpu' in self.rsd:
                n_cpu = len(self.rsd['cpu'])
                _prefix_envs.append(f"OMP_NUM_THREADS={n_cpu} MKL_NUM_THREADS={n_cpu}")
            if 'gpu' in self.rsd:
                _prefix_envs.append(f"CUDA_VISIBLE_DEVICES={','.join([str(z) for z in self.rsd['gpu']])}")
            if _prefix_envs:
                c = f"export {' '.join(_prefix_envs)}; {c}"
            output, code = my_srun(c, popen=popen, return_code=True)
            assert code == 0
        else:
            raise NotImplementedError(f"UNK method of {method}")
        return res, output

    # --
    # other helpers

    def _read_res(self, file: str, prefix="zzzzzfinal"):
        result_dict = None
        with zopen(file) as fd:
            for line in fd:
                if prefix in line:
                    result_dict = re.search(": (\{.*\})", line).groups()[0]
                    result_dict = eval(result_dict)
        if result_dict is not None:
            return ZResult(result_dict)
        else:
            return None
        # --

# a series of task
class MyTaskSeries(MyTask):
    def __init__(self, tasks: List[MyTask], conf: MyTaskConf = None, **kwargs):
        super().__init__(conf, **kwargs)
        self.tasks = tasks

    def descr(self, verbose=False):
        ret = super().descr(verbose)
        ds = [z.descr(verbose) for z in self.tasks]
        if verbose:
            ret += "\n" + "\n".join(["\t"+z for z in ds])
        else:
            ret += str(ds)
        return ret

    def _run(self):
        conf: MyTaskConf = self.conf
        assert conf.cmd == "", "No cmd for the Composite itself!"
        # simply run them in series
        for tii, t in enumerate(self.tasks):
            t.set_id(f"{self.id}_{tii}")
            t.rsd = self.rsd
            t._run()
            t.rsd = None
        if conf.cmd_res:  # but allow cmd_res
            self.result = self._run_cmd(conf.cmd_res)
        # --
