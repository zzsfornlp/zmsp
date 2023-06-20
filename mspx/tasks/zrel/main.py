#

# for relation (arg)

import pandas
import numpy
from mspx.cli.main import main as cli_main
from mspx.utils import Timer
from .mod import *  # note: include the Mod!
from mspx.tasks.zext.mod import *  # note: include the Mod!

def conf_getter(bert_name: str, total_epoch=10, base_layer=0, mix_layer=-1, total_layer=12,
                cateH='evt', cateT='ef', modH='evt,ef', modT='', **kwargs):
    # --
    total_epoch, base_layer, mix_layer, total_layer = \
        [int(z) for z in [total_epoch, base_layer, mix_layer, total_layer]]
    # --
    args = []
    # prep
    e_n_layers = max(base_layer, mix_layer)
    e_layers = ','.join([str(z) for z in range(e_n_layers)])  # common encoding
    t_layers = ','.join([str(z) for z in range(base_layer, total_layer)])  # task specific
    # task & model
    a_tasks = ["enc0"]
    a_tcs = "tcs:enc0:sb"  # common encoder
    args += f"enc0.bconf:bmod2 enc0.remain_toks:1 enc0.sconf:{bert_name} enc0.enc.n_layers:{e_n_layers} enc0.enc.n_layer_lidxes:{e_layers}".split()
    if modH:  # extracting Head
        a_tcs += ",extH:ext"
        a_tasks.append("extH")
        args += f"extH.frame_cate:{modH} extH.input_name:enc0:enc:E".split()
        args += f"extH.sconf:{bert_name} extH.b_inc_emb:0 extH.enc.n_layers:{total_layer-base_layer} extH.enc.n_layer_lidxes:{t_layers}".split()
        if mix_layer >= 0:
            args += f"extH.bout.extra_names:enc0:bout:hidden_states:{mix_layer}".split()
    if modT:
        a_tcs += ",extT:ext"
        a_tasks.append("extT")
        args += f"extT.frame_cate:{modT} extT.input_name:enc0:enc:E".split()
        args += f"extT.sconf:{bert_name} extT.b_inc_emb:0 extT.enc.n_layers:{total_layer-base_layer} extT.enc.n_layer_lidxes:{t_layers}".split()
        if mix_layer >= 0:
            args += f"extT.bout.extra_names:enc0:bout:hidden_states:{mix_layer}".split()
    a_tcs += ",rel0:rel"  # relation
    a_tasks.append("rel0")
    args += f"rel0.cateHs:{cateH} rel0.cateTs:{cateT} rel0.input_name:enc0:enc:E".split()
    args += f"rel0.sconf:{bert_name} rel0.b_inc_emb:0 rel0.enc.n_layers:{total_layer-base_layer} rel0.enc.n_layer_lidxes:{t_layers}".split()
    if mix_layer >= 0:
        args += f"rel0.bout.extra_names:enc0:bout:hidden_states:{mix_layer}".split()
    args += [a_tcs]  # all the tasks
    # --
    # data
    for wset in ["train0", "train1", "dev0", "test0", "test1"]:
        args += f"{wset}.group_files: {wset}.tasks:{','.join(a_tasks)} {wset}.batch_size:512".split()
        if bert_name:
            args += f"{wset}.len_f:subword:{bert_name}".split()
    # training
    UPE = 1000
    EPOCH = int(total_epoch)
    args += f" optim_type:adam lrate.val:0.00002 lrate.val_range:0.1,1".split()
    args += f" lrate.ff:1-(i-{EPOCH*0.1})/{EPOCH} record_best_start_cidx:{int(EPOCH*0.5)}".split()
    args += f" valid_ufreq:{UPE} max_uidx:{UPE*EPOCH}".split()
    args += "model_save_suffix_curr: model_save_suffix_best: save_bestn:1 model_save_suffix_bestn:.best".split()
    # --
    return args

def main(args):
    cli_main(args, sbase_getter=conf_getter)

# python3 -m mspx.tasks.zrel.main ...
if __name__ == '__main__':
    import sys
    with Timer(info=f"Main", print_date=True) as et:
        main(sys.argv[1:])

"""
RARGS="conf_output:_conf log_file:_log device:0 conf_sbase:bert_name:xlm-roberta-base d_input_dir:__data/evt/dataS train0.group_files:en.ace05.train.json dev0.group_files:en.ace05.dev.json test0.group_files:en.ace05.dev.json,en.ace05.test.json fs:build,train,test"
# group al '(lambda x: sum(z.arg is x.arg for z in x.main.args))(d[0])'  # in ace05, <1%
"""
