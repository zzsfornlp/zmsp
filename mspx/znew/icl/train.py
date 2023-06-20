#

# training

import sys
import pandas
from collections import Counter
from copy import deepcopy
from mspx.utils import Conf, zlog, init_everything, ConfEntryCallback, default_json_serializer, ZHelper, ZResult
from mspx.nn import BK
from .data import IclDataConf, prepare_icl_data
from .task import TaskHelper
from .models import NewBaseModelConf
from .task import TaskConf
from .select import IclSelectConf, select_demonstrations
from .format import IclFormatConf, format_queries
from .pred import IclPredConf, pred_results
from .eval import IclEvalConf, eval_results
from .modified_run_center import ZRunCenterConf, ZRunCenter
from .test import CommonConf, pred_and_eval

class MainConf(CommonConf):
    def __init__(self):
        super().__init__()
        # --
        self.run_conf = ZRunCenterConf()
        self.train_batch_size = 4
        # data
        self.data_train = IclDataConf()
        self.data_dev = IclDataConf()
        self.data_test = IclDataConf()

class MyRunCenter(ZRunCenter):
    def __init__(self, conf: ZRunCenterConf, model, task_helper, conf_c):
        super().__init__(conf, model)
        self.task_helper = task_helper
        self.conf_c = conf_c  # common conf

    def do_test(self, test_data):
        conf_c = self.conf_c
        self.model.eval()  # note: remember to make it eval!
        # --
        _data, _queries = test_data
        pred_data, eval_res = pred_and_eval(_data, _queries, self.model, conf_c.pred_conf, conf_c.eval_conf, self.task_helper)
        return ZResult(eval_res)

    # note: this can be changed in sub-class
    def forward_model(self, model, ibatch):
        model = self.model
        # --
        all_insts = [d for query in ibatch for d in query['data']]
        inputs0, inputs1 = [z[0] for z in all_insts], [z[1] for z in all_insts]
        t_tok, t_mask, t_tid = model.make_logprob_inputs(inputs0, inputs1)
        _logprobs = model.forward_logprob(t_tok, t_mask, t_tid)  # [bs, L-1]
        loss = (-_logprobs.sum(-1) / t_tid.sum(-1)).mean()  # []
        return loss, {'bs': len(ibatch), 'bsi': len(all_insts)}

def prepare_query(data_conf, task_helper, conf, label_options=None):
    orig_data = prepare_icl_data(data_conf, task_helper)
    data_test_paired = [([], z) for z in orig_data]
    queries = format_queries(data_test_paired, conf.format_conf, label_options, task_helper)
    return orig_data, queries

def batch_data(queries, batch_size):
    while 1:
        yield from ZHelper.yield_batches(queries, batch_size)

def main(args):
    conf: MainConf = init_everything(MainConf(), args)
    task_helper = TaskHelper(conf.task_conf)
    # --
    # model & task
    model = conf.mod_conf.make_node()
    model.to(BK.DEFAULT_DEVICE)
    run_center = MyRunCenter(conf.run_conf, model, task_helper, conf)
    # --
    # prepare data
    train_data, train_queries = prepare_query(conf.data_train, task_helper, conf)
    label_options = task_helper.get_label_options(train_data)
    dev_data, dev_queries = prepare_query(conf.data_dev, task_helper, conf, label_options)
    # --
    # train
    train_stream = batch_data(train_queries, conf.train_batch_size)
    run_center.do_train(train_stream, (dev_data, dev_queries))
    # --
    if conf.data_test.path:
        zlog("Reload model")
        BK.load_mod(run_center.model, conf.run_conf.model_load_name)
        data_test = prepare_query(conf.data_test, task_helper, conf, label_options)
        test_res = run_center.do_test(data_test)
        zlog(f"zzzzztestfinal: {test_res.to_dict(store_all_fields=True)}")
    # --

# python3 -m mspx.znew.icl.train ...
if __name__ == '__main__':
    main(sys.argv[1:])

"""
python3 -m mspx.znew.icl.train task:eae data_train.path:ace05_attack.train.json data_dev.path:ace05_attack.dev.json data_test.path:ace05_attack.test2.json cache_dir:./_cache/ "sel0.sig_f:(x['label'],)" data_train.shuffle_times:1 data_dev.shuffle_times:1 data_train.sample_k:50 data_dev.sample_k:50 device:1 template:eae score_mode:first model_name:gpt2 fp16:1 use_torch_amp:1 max_uidx:500 msp_seed:13
"""
