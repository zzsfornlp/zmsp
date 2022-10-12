#

from typing import List, Union, Tuple, Sequence, Iterable
from threading import Thread, Lock
import sys, time, traceback
import os, subprocess
from msp2.utils import zlog, system, Random
from msp2.tools.analyze import AnalyzerConf, Analyzer

# =====
class Result(object):
    UNFINISHED_V = -12345678.0
    NONE_RES_V = -12345677.0

class Task(object):
    def __init__(self):
        self.id = None
        self.finished = False
        self.error = False
        self.result = None

    @property
    def ended(self):
        return self.finished or self.error

    def descr(self, verbose=False):
        s = f"[{self.id}:{self.status}] {self.__class__.__name__}"
        if self.ended and verbose:
            s += "\n-> " + str(self.result)
        return s

    def __repr__(self):
        return self.descr()

    @property
    def status(self):
        return "fin" if self.finished else ("err" if self.error else "run")

    @property
    def v(self):
        r = self.result
        if not self.ended:
            return Result.UNFINISHED_V
        elif r is None:
            return Result.NONE_RES_V
        else:
            return float(r)

    # to be implemented
    def execute(self, env_prefix: str):
        raise NotImplementedError()

class Worker(object):
    def __init__(self, rgpu=-1, ncore=1):
        self.attached_task = None
        self.thread = None
        self.deleted = False
        # --
        # resource properties
        self.rgpu = int(rgpu)
        self.ncore = int(ncore)

    def __repr__(self):
        return f"[{self.status}] (RGPU={self.rgpu},NCORE={self.ncore}), attach -> {str(self.attached_task)}"

    def env_prefix(self):
        return f"CUDA_VISIBLE_DEVICES={self.rgpu} OMP_NUM_THREADS={self.ncore} MKL_NUM_THREADS={self.ncore}"

    def mark_delete(self):
        self.deleted = True
        # todo(+N): to kill the task?

    @property
    def status(self):
        if self.deleted:
            return "dele" if self.attached_task is None else "zomb"
        else:
            return "idle" if self.attached_task is None else "work"

    @property
    def idle(self):
        return self.status == "idle"

    def run(self, task: Task, id):
        assert self.idle
        self.attached_task = task
        self.thread = Thread(target=self._run, args=(task, id))
        self.thread.start()

    def _run(self, task: Task, id):
        try:
            task.id = id
            zlog("-- Start task %s." % task)
            t0 = time.time()
            task.result = task.execute(env_prefix=self.env_prefix())
            t1 = time.time()
            zlog("-- End task %s, result %s, time %.2f sec." % (task, str(task.result), t1-t0))
            task.finished = True
            if hasattr(task.result, 'get'):
                return_code = task.result.get('return_code', None)
                if return_code is not None and return_code != 0:
                    task.error = True
        except:
            zlog("-- Exception task %s ->\n%s" % (task, traceback.format_exc()))
            task.error = True
        self.attached_task = None

    # =====
    @staticmethod
    def get_gpu_workers(gpu_ids, ncore=1):
        return [Worker(gi, ncore) for gi in gpu_ids]

class WorkerPool(object):
    def __init__(self, init_workers: Iterable[Worker] = ()):
        self.workers = list(init_workers)
        # --
        self.lock = Lock()

    def add(self, worker: Worker):
        self.lock.acquire()
        self.workers.append(worker)
        self.lock.release()

    def delete(self, wid: int):
        self.lock.acquire()
        if wid >= 0 and wid < len(self.workers):
            self.workers[wid].mark_delete()
        else:
            zlog(f"Err in WPool: invalid worker-id {wid}.")
        self.lock.release()

    def list(self, status=None):
        if status is None:
            return [str(w) for w in self.workers]
        else:
            return [str(w) for w in self.workers if w.status in status]

    def get_idle_worker(self):
        ret = None
        self.lock.acquire()
        for w in self.workers:
            if w.idle:
                ret = w
                break
        self.lock.release()
        return ret

# ==
class TuneConf(AnalyzerConf):
    def __init__(self):
        super().__init__()
        # --
        self.wait_time = 5.0
        self.res_prefix = "_res."
        self.do_loop = True  # read cmd

class TuneDriver(Analyzer):
    def __init__(self, conf: TuneConf, init_workers: Iterable[Worker], name: str, extra_info=None):
        super().__init__(conf)
        conf: TuneConf = self.conf
        # --
        self.name = name
        self.pool = WorkerPool(init_workers)
        self.tasks = []
        self.extra_info = extra_info

    # =====
    # cmd
    def do_jobs(self, *args):
        tasks = self.list_tasks(*args)
        zlog(f"List tasks ({len(tasks)}) by cmd jobs {args}:")
        for t in tasks:
            zlog(t)

    def do_show(self, *args):
        for id in args:
            t = self.tasks[int(id)]
            zlog(t.descr(True))

    def do_listw(self, *args):
        select_status = args[0].split(",") if len(args)>0 else None
        zlog(f"List all workers with status {select_status}")
        for wid, ss in enumerate(self.pool.list()):
            zlog(f"{wid}: {ss}")

    def do_addw(self, *args, **kwargs):
        new_worker = Worker(*args, **kwargs)
        zlog(f"Add new worker {new_worker}")
        self.pool.add(new_worker)

    def do_delw(self, wid: int):
        wid = int(wid)
        zlog(f"Mark wid={wid} as deleted.")
        self.pool.delete(wid)

    # =====
    def list_tasks(self, select_flag="all", key_flag="i", verbose=False):
        #
        if select_flag == "all":
            tasks = self.tasks.copy()
        else:
            tasks = list(filter(lambda x: x.status==select_flag, self.tasks))
        #
        keyer = {"i": lambda x: x.id, "v": lambda x: x.v, "s": lambda x: x.status}[key_flag]
        tasks.sort(key=keyer, reverse=True)
        return [t.descr(verbose) for t in tasks]

    def write_results(self):
        if self.name and self.conf.res_prefix:
            with open(f"{self.conf.res_prefix}{self.name}", "w") as fd:
                for one in self.list_tasks(key_flag="v", verbose=True):
                    fd.write(one+"\n")
                fd.write(f"##ExtraInfo: {self.extra_info}\n")

    # =====
    def wait(self):
        # wait for all tasks to be finished
        while True:
            flag_all_done = True
            for t in self.tasks:
                if not t.ended:
                    flag_all_done = False
                    break
            if flag_all_done:
                break
            time.sleep(self.conf.wait_time)

    def run_tasks(self, task_iter: Iterable):
        for task in task_iter:
            # get idle worker
            while True:
                worker = self.pool.get_idle_worker()
                if worker is not None:
                    self.write_results()
                    break
                time.sleep(self.conf.wait_time)
            # running
            self.tasks.append(task)
            worker.run(task, len(self.tasks)-1)
        zlog("!! Loop ended, wait for all to finish.")
        self.wait()
        self.write_results()
        zlog("!! All done.")

    def main(self, task_iter: Iterable):
        run_thread = Thread(target=self.run_tasks, args=(task_iter, ))
        run_thread.start()  # one thread do task managing
        if self.conf.do_loop:
            self.loop()  # self's thread do parse-cmd
        else:
            zlog("No cmd loop, wait for all tasks to be done!!")
            run_thread.join()
            zlog("Finished, all done!!")

# =====
# extra common helpers: iterate tasks with different args
def iter_arg_choices(m: List, repeat=True, shuffle=True, max_num=-1):
    _gen = Random.get_generator("tune")
    # --
    idx = 0
    # expand fully
    args_pool = None
    if not repeat:
        args_pool = [[]]
        for cur_items in m:
            new_pool = []
            for a in args_pool:
                for one_idx in range(len(cur_items)):
                    new_pool.append(a+[one_idx])
            args_pool = new_pool
        # --
        zlog("** Arrange non-repeat iter, sized %d." % len(args_pool))
        if shuffle:
            for _ in range(10):
                _gen.shuffle(args_pool)
        else:
            args_pool.reverse()  # later using pop
    while True:
        if idx == max_num:
            break
        if repeat:
            sel_idxes = [_gen.randint(len(one)) for one in m]
        else:
            if len(args_pool) > 0:
                sel_idxes = args_pool.pop()
            else:
                break
        # -----
        yield sel_idxes  # return selection idxes
        idx += 1

# ======
def test():
    class TestTask(Task):
        def __init__(self, sec: float):
            super().__init__()
            self.sec = sec

        def execute(self, env_prefix: str):
            zlog(f"Executing task with env={env_prefix} sec={self.sec}")
            time.sleep(self.sec)
            return self.sec

    x = TuneDriver(TuneConf(), Worker.get_gpu_workers([0,1,2]), "test")
    x.main([TestTask(i) for i in range(5)])

if __name__ == '__main__':
    test()
