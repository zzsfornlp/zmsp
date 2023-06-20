#

# testing

import sys
import pandas
from collections import Counter
from copy import deepcopy
from mspx.utils import Conf, zlog, init_everything, ConfEntryCallback, default_json_serializer
from mspx.nn import BK, NnConf
from .data import IclDataConf, prepare_icl_data
from .task import TaskHelper
from .models import NewBaseModelConf
from .task import TaskConf
from .select import IclSelectConf, select_demonstrations
from .format import IclFormatConf, format_queries
from .pred import IclPredConf, pred_results
from .eval import IclEvalConf, eval_results

class CommonConf(Conf):
    def __init__(self):
        # model
        self.mod_conf = ConfEntryCallback((lambda s: self.callback_entry(s, T=NnConf)), default_s='nm_trf')
        self.task_conf = TaskConf()
        # select
        self.sel_conf = IclSelectConf()
        # format
        self.format_conf = IclFormatConf()
        # predict
        self.pred_conf = IclPredConf()
        # eval
        self.eval_conf = IclEvalConf()

class MainConf(CommonConf):
    def __init__(self):
        super().__init__()
        # data
        self.data_pool = IclDataConf()
        self.data_test = IclDataConf()
        # output
        self.output_train = ""
        self.output_pred = ""
        self.output_gold = ""

def pred_and_eval(data, queries, model, pred_conf, eval_conf, task_helper):
    # model prediction
    results = pred_results(queries, model, pred_conf, task_helper)
    # output & eval
    pred_data = deepcopy(data)
    for one_res, one_dp in zip(results, pred_data):
        one_dp.update(one_res)
    eval_res = eval_results(data, pred_data, eval_conf, task_helper)
    return pred_data, eval_res

def main(args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    # model & task
    model = conf.mod_conf.make_node()
    task_helper = TaskHelper(conf.task_conf)
    # data
    data_pool = prepare_icl_data(conf.data_pool, task_helper)
    data_test = prepare_icl_data(conf.data_test, task_helper)
    # demonstration selection (pairing test instances with train ones)
    data_test_paired = select_demonstrations(data_test, data_pool, conf.sel_conf, task_helper)
    # format
    label_options = task_helper.get_label_options(data_pool)
    queries = format_queries(data_test_paired, conf.format_conf, label_options, task_helper)
    # pred
    data_test_pred, eval_res = pred_and_eval(
        data_test, queries, model, conf.pred_conf, conf.eval_conf, task_helper)
    # assuming store the first inst's paired training insts
    for dps, path in zip([data_test, data_test_pred, data_test_paired[0][0]],
                         [conf.output_gold, conf.output_pred, conf.output_train]):
        if path:
            default_json_serializer.save_iter(dps, path)
    zlog(f"zzzzztestfinal: {eval_res}")
    # --
    # breakpoint()

# python3 -m mspx.znew.icl.test ...
if __name__ == '__main__':
    main(sys.argv[1:])

"""
python3 -m mspx.znew.icl.test task:sst data_pool.path::glue:sst2:train data_test.path::glue:sst2:validation cache_dir:./_cache/ "sel0.sig_f:(x['label'],)" data_pool.shuffle_times:1 device:0 fp16:1 "sel0.sig_budgets:4" template:sst templateC:sstC
# --
model_name:gpt2-large
mod_conf:api model_name:local-llama
sel0.sig_budgets:4 sel0.k:8
"label_f:x.lower()"
"eval_bd:min(4,len(x['spath']))"
[previous] sel1.k:8 sel1.score_stra:random
# --
python3 -m mspx.znew.icl.test task:eae data_pool.path:ace05_attack.train.json data_test.path:ace05_attack.test.json cache_dir:./_cache/ "sel0.sig_f:(x['label'],)" data_pool.shuffle_times:1 device:3 fp16:1 model_name:gpt2-xl "sel0.sig_f:(len_spath(x),sel_spath(x),x['label'])" sel0.sig_budgets:10000,10000,10000 template:eaeR batch_size:4 msp_seed:13 sel0.k:100 sel1.k:12 "sel1.sig_f:(len_spath(x),sel_spath(x),x['label'])" sel1.sig_budgets:100,100,3 sel1.score_stra:sbert sel1.random_delta:0.01 f_repr_sent:spine sel1.repr_sent:pair,spine print_first:100
# --
"sel0.sig_f:(len_spath(x),sel_spath(x),x['label'])" sel0.sig_budgets:100,100,3
"sel0.filter_f:x['spath']==['nsubj']"
sel0.k:-1 sel1.k:12 "sel1.sig_f:(len_spath(x),sel_spath(x),x['label'])" sel1.sig_budgets:100,100,3 sel1.score_stra:spath sel1.random_delta:0.01
sel1.score_stra:sbert,spath sel1.score_flatratio:2,-1
sel1.score_stra:sbert,spath "sel1.score_w:float(len(x['spath'])!=1),float(len(x['spath'])==1)"
"""
