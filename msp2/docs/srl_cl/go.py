#

# one easy-to-run script to run them all

import os

# --
# add path!!
import sys
sys.path.extend(["../"*i+"src" for i in range(5)])
# --

import re
from collections import OrderedDict
from msp2.scripts.common.go import *
from msp2.utils import system

class Zmtl2Conf(MyTaskConf):
    def __init__(self):
        super().__init__()
        # preset!
        self._module = "msp2.tasks.zmtl2.main"
        self._task_cls = Zmtl2Task
        # --
        # settings (main one)
        self.settings = "blank"  # udep/ud2/pbsrl/clsrl0/clsrl1/clsrl2/clsrl3(s)
        # extra one (active when > 1)
        self.aug_ud2 = -1
        self.aug_pb3 = -1
        self.aug_fn = -1
        # individual specific settings
        self.cl2_lang = 'UNK'  # zh/ar?
        self.cl3_lang = 'UNK'  # en/zh/cs/ca/es?
        self.cl4_lang = 'UNK'  # ??
        # --
        self.ud2_langs = ['en_ewt']  # what ud2 langs to handle
        self.pb3_datasets = ['ewt', 'ontoC']  # there are two datasets!
        self.fn_version = '17'  # 15/17?
        self.fn_use_exemplars = False
        # --

class Zmtl2Task(MyTask):
    def __init__(self, conf: Zmtl2Conf):
        super().__init__(conf)

    def get_result(self):
        output = system(f"cat {self.conf.run_dir}/_log_train | grep zzzzzfinal", popen=True)
        result_res, result_dict = re.search("\"Result\(([0-9.]+)\): (.*)\"", output).groups()
        result_res, result_dict = eval(result_res), eval(result_dict)
        return MyResult(result_res, result_dict)

    def get_all_dt_opts(self):
        return []

    def get_train_base_opt(self):
        conf: Zmtl2Conf = self.conf
        # --
        args = ""
        # --
        # 1.task
        args += " tconf.enc:bert"
        # --
        # 3.run
        UPE = 1000
        args += " optim_type:adam"
        args += " lrate.val:0.00002"
        args += f" lrate.mode:linear lrate.which_idx:uidx lrate.b:1 lrate.k:-1 lrate.m:1 lrate.idx_bias:{10*UPE} lrate.idx_scale:{100*UPE}"  # derterminstic annealing to 0.1
        args += f" valid_ufreq:{UPE} max_uidx:{UPE*100} lrate_warmup_uidx:{0*UPE} lrate_decrease_alpha:0."
        args += " df_hdrop:0.2"  # general dropout
        args += " train0.batch_size:1024 dev0.batch_size:256 test0.batch_size:256"
        # --
        _settings = conf.settings
        if _settings == "udep":  # ud parsing
            # 1.5.model
            # args += " idec_upos.app_layers:12"
            args += " tconf.upos:no tconf.udep:yes"
            args += " idec_udep_lab.app_layers:12 idec_udep_root.app_layers:"
            # 2.data
            args += " train0.input_format:conllu dev0.input_format:conllu test0.input_format:conllu"
            args += " train0.filter_max_length:128"
            args += " train0.input_dir:ud dev0.input_dir:ud test0.input_dir:ud"
            args += " train0.group_name:ud train0.group_files:_ud14/en0 train0.group_tasks:udep"
            args += " dev0.group_name:ud dev0.group_files:_ud14/en1 dev0.group_tasks:udep"
            _test_files = ','.join([f'_ud14/{cl}2' for cl in ['en', 'es', 'zh', 'fi', 'fr', 'de', 'it', 'pt_bosque']])
            args += f" test0.group_name:ud test0.group_files:{_test_files} test0.group_tasks:udep"
        elif _settings == "ud2":  # get good ud2 parsers
            # model
            args += " lrate.val:0.00002"
            args += " bert_model:xlm-roberta-base"  # use xlmr
            args += " tconf.upos:yes tconf.udep:yes upos.loss_upos:0.5"
            args += " idec_upos.app_layers:12 idec_udep_lab.app_layers:12 idec_udep_root.app_layers:"
            # data
            # --
            def _getfs(_ls, _ns):
                return ",".join([f"_ud27/{_l}{_n}" for _l in _ls for _n in _ns])
            # --
            args += " train0.input_format:conllu dev0.input_format:conllu test0.input_format:conllu"
            args += " train0.input_dir:ud dev0.input_dir:ud test0.input_dir:ud"
            args += " train0.filter_max_length:150 dev0.filter_max_length:512 test0.filter_max_length:512"
            args += f" train0.group_name:ud train0.group_files:{_getfs(conf.ud2_langs, [0])} train0.group_tasks:upos,udep"
            args += f" dev0.group_name:ud dev0.group_files:{_getfs(conf.ud2_langs, [1])} dev0.group_tasks:upos,udep"
            args += f" test0.group_name:ud test0.group_files:{_getfs(conf.ud2_langs, [1,2])} test0.group_tasks:upos,udep"
        elif _settings == "pbsrl":  # pb-srl on standard datasets!
            # model
            args += " tconf.pb1:yes"
            args += " pb1.idec_evt.app_layers:12 pb1.idec_arg.app_layers:12"
            args += " pb1.loss_arg_boundary:0.5 pb1.arg_boundary:yes"  # use boundary
            # data
            args += " train0.filter_max_length:128"
            for wset in ["train", "dev", "test"]:
                args += f" {wset}0.input_dir:ud {wset}0.group_name:pb {wset}0.group_tasks:pb1"
            # which data (conll05 or conll12)
            args += " train0.group_files:_pb/pb050 dev0.group_files:_pb/pb051 test0.group_files:_pb/pb051,_pb/pb052,_pb/pb053"
            # args += " train0.group_files:_pb/en0 dev0.group_files:_pb/en1 test0.group_files:_pb/en1,_pb/en2"
        elif _settings in ["midtrain", "blank"]:  # mid-pretrain or blank
            pass
        elif _settings == "clsrl0":  # cl-srl set0
            # model
            args += " bert_model:bert-base-multilingual-cased"
            args += " tconf.pb1:yes"
            args += " pb1.idec_evt.app_layers:12 pb1.idec_arg.app_layers:12"
            args += " pb1.loss_arg_boundary:0. pb1.arg_boundary:no"  # no need for boundary since dep-srl!
            # given evt position
            args += " pb1.pred_given_evt:1 pb1.pred_evt_nil_add:-100. pb1.srl_pred_clear_evt:0"
            # or full
            # args += " pb1.pred_given_evt:0 pb1.pred_evt_nil_add:0. pb1.srl_pred_clear_evt:1"
            # has evt/pred.
            # args += " srl_eval.weight_frame:1."  # micro-avg with pred.
            # or no evt/pred.
            args += " srl_eval.weight_frame:0. pb1.loss_evt:0."
            # data
            args += " train0.filter_max_length:128 train1.filter_max_length:128 train2.filter_max_length:128"
            args += " train0.preprocessors:pb_delete_argv"  # delete V for training!
            for wset in ["train", "dev", "test"]:
                args += f" {wset}0.input_dir:ud/cl0 {wset}0.group_name:pb {wset}0.group_tasks:pb1"
            # which data
            args += " train0.group_files:en.ewt.train.json dev0.group_files:en.ewt.dev.json"
            args += " test0.input_dir:ud test0.group_files:_up/de2,_up/fr2,_up/it2,_up/es22,_up/pt_bosque2,_up/fi2"
            args += " record_best_start_cidx:80"
        elif _settings == "clsrl1":  # cl-srl set1
            # model
            # args += " bert_model:bert-base-multilingual-cased"
            args += " bert_model:xlm-roberta-base"
            for cc in ['pb1', 'pb2']:
                args += f" tconf.{cc}:yes"
                args += f" {cc}.idec_evt.app_layers:12 {cc}.idec_arg.app_layers:12"
                args += f" {cc}.loss_arg_boundary:0. {cc}.arg_boundary:no"  # no need for boundary since dep-srl!
                args += f" {cc}.binary_evt:1"  # binary target
            # data (train0:en-ewt, train1:fipb, train2:ud)
            args += " train0.filter_max_length:128 train1.filter_max_length:128 train2.filter_max_length:128"
            args += " train0.preprocessors:pb_delete_argv"  # delete V for training!
            for wset in ["train", "dev", "test"]:
                args += f" {wset}0.input_dir:ud/cl1 {wset}0.group_name:pb1 {wset}0.group_tasks:pb1"
                args += f" {wset}1.input_dir:ud/cl1 {wset}1.group_name:pb2 {wset}1.group_tasks:pb2"
                args += f" {wset}2.input_dir:ud {wset}2.group_name:ud {wset}2.group_tasks:udep {wset}2.input_format:conllu"
            for ii, dname in enumerate(["en.ewt", "fipb"]):
                args += f" train{ii}.batch_size:1024 dev{ii}.batch_size:256 test{ii}.batch_size:256"
                args += f" train{ii}.group_files:{dname}.train.json dev{ii}.group_files:{dname}.dev.json" \
                        f" test{ii}.group_files:{dname}.dev.json,{dname}.test.json"
            args += " train1.presample:1.0 dev1.presample:1.0"  # sample it!
            args += " record_best_start_cidx:80"
            # simply combine data to pb1?
            # args += " train1.group_tasks:pb1 dev1.group_tasks:pb1 test1.group_tasks:pb1"
            args += " train1.group_tasks:pb1 dev1.group_tasks:pb1 test1.group_tasks:pb1 dev0.group_files:"
        elif _settings == "clsrl2":  # cl-srl set2
            # model
            args += " bert_model:xlm-roberta-base"
            for cc in ['pb1', 'pb2']:
                args += f" tconf.{cc}:yes"
                args += f" {cc}.idec_evt.app_layers:12 {cc}.idec_arg.app_layers:12"
                args += f" {cc}.loss_arg_boundary:0.5 {cc}.arg_boundary:yes"  # need boundary!
                args += f" {cc}.binary_evt:1"  # binary target
            # data (train0:en, train1:ar/zh?, train2:ud)
            for ii in [0,1,2]:  # specifically ar has long sentences
                args += f" train{ii}.filter_max_length:200 dev{ii}.filter_max_length:500 test{ii}.filter_max_length:500"
            for wset in ["train", "dev", "test"]:
                args += f" {wset}0.input_dir:ud/cl2 {wset}0.group_name:pb1 {wset}0.group_tasks:pb1"
                args += f" {wset}1.input_dir:ud/cl2 {wset}1.group_name:pb2 {wset}1.group_tasks:pb2"
                args += f" {wset}2.input_dir:ud {wset}2.group_name:ud {wset}2.group_tasks:udep {wset}2.input_format:conllu"
            for ii, dname in enumerate(["en", conf.cl2_lang]):
                args += f" train{ii}.batch_size:1024 dev{ii}.batch_size:256 test{ii}.batch_size:256"
                args += f" train{ii}.batch_size_f:ftok dev{ii}.batch_size_f:ftok test{ii}.batch_size_f:ftok"
                args += f" train{ii}.group_files:{dname}.train.ud.json dev{ii}.group_files:{dname}.dev.ud.json" \
                        f" test{ii}.group_files:{dname}.dev.ud.json,{dname}.test.ud.json"
            args += " train1.presample:1.0 dev1.presample:1.0"  # sample it!
            args += " record_best_start_cidx:80"
        elif _settings in ["clsrl3", "clsrl3s"]:  # cl-srl set3
            # model
            args += " bert_model:xlm-roberta-base"
            _pb1, _pb2 = ('pbS', 'pbS.2') if _settings[-1]=='s' else ('pb1', 'pb2')
            for cc in [_pb1, _pb2]:
                if '.' in cc: continue
                args += f" tconf.{cc}:yes"
                args += f" {cc}.idec_evt.app_layers:12 {cc}.idec_arg.app_layers:12"
                args += f" {cc}.loss_arg_boundary:0. {cc}.arg_boundary:no"  # no boundary since dep-srl!
                args += f" {cc}.binary_evt:1"  # binary target
            # data (train0:en, train1:other, train2/train3:ud)
            for ii in [0,1,2,3]:
                args += f" train{ii}.filter_max_length:200 dev{ii}.filter_max_length:500 test{ii}.filter_max_length:500"
            for wset in ["train", "dev", "test"]:
                args += f" {wset}0.input_dir:ud/cl3 {wset}0.group_name:pb1 {wset}0.group_tasks:{_pb1}"
                args += f" {wset}1.input_dir:ud/cl3 {wset}1.group_name:pb2 {wset}1.group_tasks:{_pb2}"
                args += f" {wset}2.input_dir:ud/cl3 {wset}2.group_name:ud {wset}2.group_tasks:udep"
                args += f" {wset}3.input_dir:ud/cl3 {wset}3.group_name:ud2 {wset}3.group_tasks:udep2"
            for ii, dname in enumerate(["en", conf.cl3_lang]):
                args += f" train{ii}.batch_size:1024 dev{ii}.batch_size:256 test{ii}.batch_size:256"
                args += f" train{ii}.group_files:{dname}.train.ud.json dev{ii}.group_files:{dname}.dev.ud.json" \
                        f" test{ii}.group_files:{dname}.dev.ud.json,{dname}.test.ud.json"
            args += " train1.presample:1.0 dev1.presample:1.0"  # sample it!
            args += " record_best_start_cidx:80"
        elif _settings == "clsrl4":  # cl-srl set4
            args += " bert_model:xlm-roberta-base"
            for cc in ['pb1', 'pb2']:
                args += f" tconf.{cc}:yes"
                args += f" {cc}.idec_evt.app_layers:12 {cc}.idec_arg.app_layers:12"
                args += f" {cc}.loss_arg_boundary:0. {cc}.arg_boundary:no"  # no need for boundary since dep-srl!
                args += f" {cc}.binary_evt:1"  # binary target
            # data (train0:en-ewt, train1:target, train2:ud)
            args += " train0.filter_max_length:128 train1.filter_max_length:128 train2.filter_max_length:128"
            args += " train0.preprocessors:pb_delete_argv"  # delete V for training!
            for wset in ["train", "dev", "test"]:
                args += f" {wset}0.input_dir:ud/cl4 {wset}0.group_name:pbS {wset}0.group_tasks:pb1"
                args += f" {wset}1.input_dir:ud/cl4 {wset}1.group_name:pbT {wset}1.group_tasks:pb1"  # shared pb1!
                args += f" {wset}2.input_dir:ud/cl4 {wset}2.group_name:ud {wset}2.group_tasks:udep"
            for ii, dname in enumerate(["en.ewt", conf.cl4_lang]):
                args += f" train{ii}.batch_size:1024 dev{ii}.batch_size:256 test{ii}.batch_size:256"
                args += f" train{ii}.group_files:{dname}.train.json dev{ii}.group_files:{dname}.dev.json" \
                        f" test{ii}.group_files:{dname}.dev.json,{dname}.test.json"
            args += " train1.presample:1.0 dev1.presample:1.0"  # sample it!
            args += " record_best_start_cidx:80"
        else:
            raise RuntimeError(f"UNK setting: {_settings}")
        # --
        return args

    def get_test_base_opt(self):
        conf: Zmtl2Conf = self.conf
        args = ""
        if conf.settings == "clsrl0":  # cl-srl set0
            # note: specific for cl-dep-srl!
            args += " srl_eval.no_join_c:1"  # no join c-* like in span-based ones, more like dep!
            args += " label_budget:1 strategy_c:strip strategy_r:strip"
            args += " srl_delete_noarg_evts:1"  # maybe ann errors
        elif conf.settings in ["clsrl3", "clsrl4"]:
            args += " pb1.srl_eval.no_join_c:1 pb2.srl_eval.no_join_c:1"  # no join c-* like in span-based ones
        elif conf.settings == "clsrl3s":
            args += " pbS.srl_eval.no_join_c:1"
        # --
        return args

# --
if __name__ == '__main__':
    import sys
    main(Zmtl2Conf(), sys.argv[1:])
