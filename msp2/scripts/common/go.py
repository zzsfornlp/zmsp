#

# one script that can handle multiple things
# -- like train+test(_all), tune, ...

"""
# note: add this to sub-script for convenience
# --
# add path!!
import sys
sys.path.extend(["../"*i+"src" for i in range(5)])
# --
"""

__all__ = [
    "MyResult", "MyTask", "MyTaskConf", "main",
]

import os
from copy import deepcopy
from msp2.utils import Conf, ConfEntryChoices, zlog, init_everything, mkdir_p, zglob1, system, zopen
from msp2.tools.tune import engine as te

# ==
class MyResult(te.Result):
    def __init__(self, result_res: float, result_dict: dict):
        self.result_res = result_res
        self.result_dict = result_dict

    def __float__(self):
        return float(self.result_res)

    def __repr__(self):
        return f"RES={self.result_res}: {self.result_dict}"

class MyTaskConf(Conf):
    def __init__(self):
        # basic
        self.conf_input = ""  # conf input for training
        self.conf_output = "_conf"  # conf output for training
        self.debug = False  # -m pdb
        self.do_train = True  # training
        self.do_test = True  # testing
        self.do_test2 = False  # special testing for convenience: only test2 and without logging!
        self.do_test_all = False  # testing all_df_files
        self.train_pyopts = ""  # python options for training
        self.train_extras = ""
        self.test_extras = ""
        self.quite = False  # no screen output
        # paths
        self.run_dir = ""  # by default current one!
        self.src_dir = "src"  # where is msp?
        self.log_prefix = "_log"
        self.out_prefix = "_zout"  # todo(+N): add this?
        # --
        # not to be configured but to be set
        self._module = ""
        self._task_cls = MyTask  # for convenience
        self._env_prefix = ""
        self._id_str = ""
        self._train_extras = ""  # these are set but not configured!
        self._test_extras = ""
        # --

    def make_task(self, _copy_conf=True, **kwargs):
        conf = self
        if _copy_conf:
            conf = deepcopy(conf)
        conf.direct_update(**kwargs)
        return self._task_cls(conf)

# note: use a single class with various template methods to make it simple
class MyTask(te.Task):
    def __init__(self, conf: MyTaskConf):
        super().__init__()
        self.conf = conf  # info are in conf!
        # --

    def descr(self, verbose=False):
        conf = self.conf
        s = super().descr(verbose)
        if verbose:
            s += f"\n-> ({conf.run_dir}) [{conf._id_str}] {conf._train_extras}"
        return s

    def execute(self, env_prefix: str):
        conf = self.conf
        # first change path if needed!
        RUN_DIR = conf.run_dir
        if RUN_DIR:
            mkdir_p(RUN_DIR, raise_error=True)
            # os.chdir(RUN_DIR)  # change to it!!
        else:
            conf.run_dir = "."
        # directly change conf
        conf.src_dir = zglob1(conf.src_dir, check_prefix="..", check_iter=10)
        conf.src_dir = os.path.abspath(conf.src_dir)  # get absolute path!!
        conf._env_prefix = env_prefix
        # --
        self.run()
        res = self.get_result()
        return res

    def _get_gpus_from_env_prefix(self, ep: str):
        rets = []
        for ss in ep.split():
            if ss.startswith("CUDA_VISIBLE_DEVICES="):
                rets = [int(z) for z in ss.split("=", 1)[1].split(",") if z]
        return rets

    def run(self):
        conf = self.conf
        # --
        train_base_opt = self.get_train_base_opt()
        test_base_opt = self.get_test_base_opt()
        _L_PRE = conf.log_prefix
        DEBUG_OPTION = "-m pdb" if conf.debug else ""
        PRINT_OPTION = "log_stderr:0" if conf.quite else ""
        DEVICE_OPTION = "nn.device:0" if self._get_gpus_from_env_prefix(conf._env_prefix) else ""
        # --
        # special mode for convenience
        if conf.do_test2:
            # test without logging?
            TEST2_CMD = f"cd {conf.run_dir}; {conf._env_prefix} PYTHONPATH={conf.src_dir}:$PYTHONPATH python3 {DEBUG_OPTION} -m {conf._module}.test {conf.conf_output} {test_base_opt} log_file: log_stderr:1 {DEVICE_OPTION} {conf.test_extras} {conf._test_extras}"
            system(TEST2_CMD, pp=(not conf.quite))
        else:
            # train?
            TRAIN_CMD = f"cd {conf.run_dir}; {conf._env_prefix} PYTHONPATH={conf.src_dir}:$PYTHONPATH python3 {conf.train_pyopts} {DEBUG_OPTION} -m {conf._module}.train {conf.conf_input} {train_base_opt} log_file:{_L_PRE}_train {PRINT_OPTION} {DEVICE_OPTION} conf_output:{conf.conf_output} {conf.train_extras} {conf._train_extras}"
            if conf.do_train:
                system(TRAIN_CMD, pp=(not conf.quite))
            # test?
            TEST_CMD = f"cd {conf.run_dir}; {conf._env_prefix} PYTHONPATH={conf.src_dir}:$PYTHONPATH python3 {DEBUG_OPTION} -m {conf._module}.test {conf.conf_output} {test_base_opt} log_file:{_L_PRE}_test {PRINT_OPTION} {DEVICE_OPTION} {conf.test_extras} {conf._test_extras}"
            if conf.do_test:
                system(TEST_CMD, pp=(not conf.quite))
            # test-all?
            if conf.do_test_all:
                for extras in self.get_all_dt_opts():
                    # note: mainly input/output/log
                    _TMP_CMD = TEST_CMD + f" {extras}"
                    system(_TMP_CMD, pp=True)
        # --

    # --
    # template methods

    def get_result(self):
        return MyResult(0., {})

    def get_all_dt_opts(self):
        return []

    def get_train_base_opt(self):
        return ""

    def get_test_base_opt(self):
        return ""

# --
# main runner

class MainConf(Conf):
    def __init__(self):
        self.gpus = []
        # tuning if we can read tune_name
        self.tune_name = ""
        self.tune_table_file = ""
        self.ncore = 4
        self.repeat = False
        self.max_count = -1
        self.tune_conf = te.TuneConf()
        self.task_conf: MyTaskConf = None
        self.shuffle = True
        self.task_sels = []  # only run sel ids

def main(tconf: MyTaskConf, args):
    conf = MainConf()
    conf.task_conf = tconf
    conf: MainConf = init_everything(conf, args, add_nn=False)
    zlog(f"** Run with {conf.to_json()}")
    # --
    if conf.tune_table_file:
        with zopen(conf.tune_table_file) as fd:
            s = fd.read()
            table = eval(s)
            mm = table.get(conf.tune_name)  # read the table from file
    else:
        mm = globals().get(conf.tune_name)
    # --
    if mm is not None:  # if we can read it!
        # note: if tuning, we want it to be quiet
        conf.task_conf.quite = True
        # --
        workers = te.Worker.get_gpu_workers([int(z) for z in conf.gpus], ncore=int(conf.ncore))
        x = te.TuneDriver(conf.tune_conf, workers, conf.tune_name, extra_info=mm)
        task_iter = []
        # --
        all_runs = enumerate(te.iter_arg_choices(mm, repeat=conf.repeat, shuffle=conf.shuffle, max_num=conf.max_count))
        if not conf.repeat:
            all_runs = list(all_runs)
            orig_all_runs = list(all_runs)
        else:
            orig_all_runs = None
        if len(conf.task_sels)>0:
            assert not conf.repeat and not conf.shuffle
            _sels = [int(z) for z in conf.task_sels]
            all_runs = list(all_runs)
            zlog(f"Select {len(_sels)}/{len(all_runs)}: {_sels}")
            all_runs = [all_runs[z] for z in _sels]
        # --
        if not conf.repeat:
            _max_gid = 0 if len(orig_all_runs)==0 else max(z[0] for z in orig_all_runs)
            _padn = len(str(_max_gid))
            _pads = f"%0{_padn}d"
        else:
            _pads = "%d"
        # --
        for gid, sel_idxes in all_runs:
            # note: override run_dir!!
            s_gid = _pads % gid
            one = conf.task_conf.make_task(run_dir=f"run_{conf.tune_name}_{s_gid}", _id_str=f"{s_gid}:{sel_idxes}",
                                           _train_extras=" ".join([a[i] for a,i in zip(mm, sel_idxes)]))
            task_iter.append(one)
        x.main(task_iter)
    else:  # otherwise single run
        assert not conf.tune_name
        task = conf.task_conf.make_task()  # no setting here!!
        task.execute(f"CUDA_VISIBLE_DEVICES={','.join(conf.gpus)} OMP_NUM_THREADS={conf.ncore} MKL_NUM_THREADS={conf.ncore}")
    # --

# --
if __name__ == '__main__':
    import sys
    main(MyTaskConf(), sys.argv[1:])
