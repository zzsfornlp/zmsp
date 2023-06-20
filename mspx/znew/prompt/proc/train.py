#

# model training

import sys
from mspx.utils import init_everything, zlog
from ..data import DatasetConf, MyDataset
from .common import CommonConf, AccelerateConf
from .run import RunnerConf, MyRunner

class MainConf(CommonConf):
    def __init__(self):
        super().__init__()

def extra_args_for_mp(accelerator):
    args = []
    _par_is_main, _par_rank = accelerator.is_main_process, accelerator.process_index
    if not _par_is_main:
        args.extend(["log_stderr:0", "log_magic_file:0", "conf_output:"])
        log_file_item = None
        for a in args:
            if "log_file:" in a:
                log_file_item = a
        if log_file_item is not None:
            args.extend([f"{log_file_item}{_par_rank}"])  # put into another file!
    return args

# --
def main(args):
    # special handling for multiprocess running
    accelerator = AccelerateConf.get_acce(args)
    args.extend(extra_args_for_mp(accelerator))
    # TODO(!): how to specify multi-gpus per process?
    # --
    conf: MainConf = init_everything(MainConf(), args)
    # --
    # model & task
    model = conf.model_conf.make_node()
    task = conf.task_conf.make_node(model=model)
    # --
    # data & loader
    data_train0, data_dev0 = [(MyDataset(z) if z.has_data() else None) for z in [conf.train0, conf.dev0]]
    loader_train0, loader_dev0 = [(task.get_dataloader(z, False) if z is not None else None) for z in [data_train0, data_dev0]]
    # --
    # train
    runner = MyRunner(conf.run_conf, model, task, accelerator)
    if conf.run_conf.model_load_name:
        runner.load_model(model, conf.run_conf.model_load_name)
    best_state_dict = runner.do_train(loader_train0, loader_dev0)
    # --
    # test
    if conf.test0.has_data():
        if best_state_dict is not None:
            runner.load_model(model, best_state_dict)
        data_test0 = MyDataset(conf.test0, name='test0')
        loader_test0 = task.get_dataloader(data_test0, False)
        test_res = runner.do_test(model, loader_test0)  # note: simply use raw model for testing!
        zlog(f"zzzzztestfinal: {test_res.to_dict(store_all_fields=True)}")
    # --

# python3 -m mspx.znew.prompt.proc.train ...
if __name__ == '__main__':
    main(sys.argv[1:])
