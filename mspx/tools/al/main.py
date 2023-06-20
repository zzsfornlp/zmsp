#

import os
import pandas as pd
from mspx.utils import Conf, default_json_serializer, init_everything, Timer, zlog
from .core import ALConf, ALProject

# need a wrapper for hold utils
class MainConf(Conf):
    def __init__(self):
        self.al = ALConf()

# --
def main(*args):
    list_args = list(args)
    # special start: handling "file_conf" and "file_log"
    conf0 = ALConf()
    conf0.update_from_args(list_args, quite=True, check=False, add_global_key='')  # init0
    if conf0.file_log:
        list_args.append(f"log_file:{conf0.file_log}")   # note: overwrite!
    # --
    conf = MainConf()  # a global one
    if os.path.isfile(conf0.file_conf) and (not conf0.no_load_conf):
        conf_external = default_json_serializer.from_file(conf0.file_conf)  # load it!
        conf.al.direct_update_from_other(conf_external)
    conf: MainConf = init_everything(conf, list_args)  # real init!
    zlog("# =====")
    with Timer("RUN AL"):
        project = ALProject(conf.al)
        project.run()
    zlog("# =====")
    # --

# PYTHONPATH=../src/ python3 -m mspx.tools.al.main ...
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
# --

"""
# debug
python3 -m mspx.tools.al.main setup_dataU:__data/ner/data/en.dev.json al_task:zext frame_cate:ef al_task.name:ext0
# extra ones
query_i0_model:__tmp/zmodel.best.m query_emb_bname:xlm-roberta-base query_emb_use_model:1
train_st:2 "specs_train:conf_sbase:bert_name:xlm-roberta-base dev0.group_files:__data/ner/data/en.dev.json max_uidx:10 valid_first:1"
# query with clustering
cluster_k:100 query_emb_bname:xlm-roberta-base
# stat
python3 -m mspx.tools.al.main special:stat al_task:zext al_task.name:ext0 input_path:iter0/data.query.json
python3 -m pdb -m mspx.cli.analyze frame gold:iter0/data.query.json frame_cate:ef
# check
python3 -m pdb -m mspx.cli.analyze frame gold:iter01/data.comb.json preds:iter01/data.ann.json frame_cate:evt 'filter_frame:lambda x: x.label.strip("*")!="_NIL_"' 'filter_arg:lambda x: x.label.strip("*")!="_NIL_"' 'labf_frame:lambda x: x.label.strip("*")' 'labf_arg:lambda x: x.label.strip("*")'
"""
