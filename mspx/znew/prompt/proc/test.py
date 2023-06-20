#

# model testing

import sys
import torch
from mspx.utils import init_everything, zlog, zwarn
from ..data import DatasetConf, MyDataset
from .common import CommonConf
from .run import RunnerConf, MyRunner

class MainConf(CommonConf):
    def __init__(self):
        super().__init__()

# --
def main(args):
    conf: MainConf = init_everything(MainConf(), args)
    accelerator = conf.accelerate_conf.get_accelerator()
    # --
    # model & task
    model = conf.model_conf.make_node()
    task = conf.task_conf.make_node(model=model)
    # load model
    runner = MyRunner(conf.run_conf, model, task, accelerator)
    if conf.run_conf.model_load_name:
        runner.load_model(model, conf.run_conf.model_load_name)
    else:
        zwarn("No model loading for testing, is this OK?")
    # data
    data_test0 = MyDataset(conf.test0, name='test0')
    loader_test0 = task.get_dataloader(data_test0, False)
    if conf.accelerate_conf.mixed_precision:
        with torch.autocast("cuda" if torch.cuda.is_available() else 'cpu'):
            test_res = runner.do_test(model, loader_test0)  # note: simply use raw model for testing!
    else:
        test_res = runner.do_test(model, loader_test0)  # note: simply use raw model for testing!
    zlog(f"zzzzztestfinal: {test_res.to_dict(store_all_fields=True)}")
    # --

# python3 -m mspx.znew.prompt.proc.test ...
if __name__ == '__main__':
    main(sys.argv[1:])
