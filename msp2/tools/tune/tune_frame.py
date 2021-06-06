#

# tune the frame parser

import os
import re
import sys
from typing import Dict, List
from msp2.utils import zlog, zopen, Conf, system, init_everything, default_json_serializer
from .engine import Result, Task, TuneConf, TuneDriver, Worker, iter_arg_choices

# -----
class MyResult(Result):
    def __init__(self, result_res: float, result_dict: Dict):
        self.result_res = result_res
        self.result_dict = result_dict

    def __float__(self):
        return float(self.result_res)

    def __repr__(self):
        return f"RES={self.result_res}: {self.result_dict}"

class MyTaskConf(Conf):
    def __init__(self):
        self.conf_infile = "../_conf"
        self.conf_outfile = "./_conf"
        self.task_name = "zsfp"

class MyTask(Task):
    def __init__(self, gid: int, dir_predix: str, sel_idxes: List, args: List, conf: MyTaskConf):
        super().__init__()
        self.gid = gid
        self.sel_str = str(sel_idxes)
        self.arg_str = " ".join(args)
        self.dir_name = f"run_{dir_predix}_{self.gid}"
        self.conf = conf

    def descr(self, verbose=False):
        s = super().descr(verbose)
        if verbose:
            s += "\n-> (%s) [%s] %s" % (self.dir_name, self.sel_str, self.arg_str)
        return s

    # read the training zfinal line
    def _read_results(self, file):
        output = system(f"cat {file} | grep zzzzzfinal", popen=True)
        result_res, result_dict = re.search("Result\(([0-9.]+)\): (\{.*\})", output).groups()
        result_res, result_dict = eval(result_res), eval(result_dict)
        return MyResult(result_res, result_dict)

    def execute(self, env_prefix: str):
        conf = self.conf
        # --
        dir_name = self.dir_name
        system(f"mkdir -p {dir_name}")
        # train
        system(f"cd {dir_name}; {env_prefix} PYTHONPATH=../src:../../src/:../../../src python3 -m msp2.tasks.{conf.task_name}.main.train {conf.conf_infile} {self.arg_str} conf_output:{conf.conf_outfile} >_log_train 2>&1;")
        rr = self._read_results(dir_name+"/_log_train")
        # test
        # system(f"cd {dir_name}; {env_prefix} python3 PYTHONPATH=../../src/ python3 -m msp2.tasks.zsfp.main.test {conf.conf_outfile} device:0 {self.arg_str} >_log_test 2>&1")
        return rr

# --
class MainConf(Conf):
    def __init__(self):
        self.name = "tune"
        self.input_table_file = ""
        self.gpus = []
        self.ncore = 4
        self.repeat = False
        self.max_count = -1
        self.tune_conf = TuneConf()
        self.task_conf = MyTaskConf()
        self.shuffle = True

# =====
def main(args):
    conf: MainConf = init_everything(MainConf(), args, add_nn=False)
    zlog(f"** Run with {conf.to_json()}")
    # --
    if conf.input_table_file:
        with zopen(conf.input_table_file) as fd:
            s = fd.read()
            table = eval(s)
            mm = table[conf.name]  # read the table from file
    else:
        mm = globals()[conf.name]
    # --
    workers = Worker.get_gpu_workers([int(z) for z in conf.gpus], ncore=int(conf.ncore))
    # --
    x = TuneDriver(conf.tune_conf, workers, conf.name, extra_info=mm)
    task_iter = [MyTask(gid, conf.name, sel_idxes, [a[i] for a,i in zip(mm, sel_idxes)], conf.task_conf)
                 for gid, sel_idxes in enumerate(iter_arg_choices(mm, repeat=conf.repeat,
                                                                  shuffle=conf.shuffle, max_num=conf.max_count))]
    x.main(task_iter)

# PYTHONPATH=../src/ python3 -m pdb -m msp2.tools.tune.tune_frame input_table_file:./t.py gpus:0,1 name:??
if __name__ == '__main__':
    main(sys.argv[1:])
