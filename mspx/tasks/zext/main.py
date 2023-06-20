#

# for extraction

import pandas
import numpy
from mspx.cli.main import main as cli_main
from mspx.utils import Timer
from .mod import *  # note: include the Mod!

def conf_getter(bert_name: str, total_epoch=10, **kwargs):
    args = []
    # task & model
    args += "tcs:ext0:ext".split()
    if bert_name:
        args += f"ext0.sconf:{bert_name}".split()
    # data
    for wset in ["train0", "train1", "dev0", "test0"]:
        args += f"{wset}.group_files: {wset}.tasks:ext0 {wset}.batch_size:512".split()
        if bert_name:
            args += f"{wset}.len_f:subword:{bert_name}".split()
    # training
    UPE = 1000
    EPOCH = int(total_epoch)
    args += f" optim_type:adam lrate.val:0.00001 lrate.val_range:0.1,1".split()
    args += f" lrate.ff:1-(i-{EPOCH*0.1})/{EPOCH} record_best_start_cidx:{int(EPOCH*0.5)}".split()
    args += f" valid_ufreq:{UPE} max_uidx:{UPE*EPOCH}".split()
    args += "model_save_suffix_curr: model_save_suffix_best: save_bestn:1 model_save_suffix_bestn:.best".split()
    # --
    return args

def main(args):
    cli_main(args, sbase_getter=conf_getter)

# python3 -m mspx.tasks.zext.main ...
if __name__ == '__main__':
    import sys
    with Timer(info=f"Main", print_date=True) as et:
        main(sys.argv[1:])

"""
RARGS="conf_output:_conf log_file:_log device:0 conf_sbase:bert_name:xlm-roberta-base d_input_dir:__data/ner/data train0.group_files:en.train.json dev0.group_files:en.dev.json test0.group_files:en.dev.json,en.test.json fs:build,train,test"
"""
