#

# one easy-to-run script to run them all

# --
# add path!!
import sys
sys.path.extend(["../"*i+"src" for i in range(5)])
# --

import re
from collections import OrderedDict, Counter
from msp2.scripts.common.go import *
from msp2.utils import system

class ZFs1Conf(MyTaskConf):
    def __init__(self):
        super().__init__()
        # preset!
        self._module = "msp2.tasks.zmtl3.main"
        self._task_cls = ZFs1Task
        # --
        # self.dataset = 'ace'  # which dataset to run?

class ZFs1Task(MyTask):
    def __init__(self, conf: ZFs1Conf):
        super().__init__(conf)

    def get_result(self):
        output = system(f"cat {self.conf.run_dir}/_log_train | grep zzzzzfinal", popen=True)
        result_res, result_dict = re.search("\"Result\(([0-9.]+)\): (.*)\"", output).groups()
        result_res, result_dict = eval(result_res), eval(result_dict)
        return MyResult(result_res, result_dict)

    def get_train_base_opt(self):
        conf: ZFs1Conf = self.conf
        # --
        opts = ""
        # opts = f""" "conf_sbase:data:{conf.dataset}" """
        return opts

    def do_extra_test(self):
        conf: ZFs1Conf = self.conf
        # --
        return True

# --
if __name__ == '__main__':
    import sys
    main(ZFs1Conf(), sys.argv[1:])

# --
# python3 _go.py run_dir:debug gpus:0 "train_extras:"
# python3 _go.py shuffle:0 gpus:1,2,3 tune_table_file:t.py tune_name:??
# python3 _go.py gpus:0 do_test2:1 run_dir:??
