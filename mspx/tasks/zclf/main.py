#

# for convenience

from mspx.cli.main import main as cli_main
from mspx.utils import Timer
from .mod import *  # note: include the Mod!

def conf_getter(bert_name: str, bmod2=False, **kwargs):
    args = []
    # task & model
    args += "tcs:clf0:clf".split()
    if bert_name:
        bmod2 = bool(int(bmod2))
        if bmod2:
            args += f"clf0.bconf:bmod2 clf0.b_vpath:{bert_name} clf0.sconf:{bert_name} clf0.init_with_bmodel:{bert_name}".split()
        else:
            args += f"clf0.b_model:{bert_name}".split()
    # data
    for wset in ["train0", "dev0", "test0"]:
        args += f"{wset}.group_files: {wset}.tasks:clf0".split()
        if bert_name:
            args += f"{wset}.len_f:subword:{bert_name}".split()
    # training
    UPE = 1000
    args += " optim_type:adam lrate.val:0.00002 lrate.ff:1-(i-2)/20 lrate.val_range:0.1,1".split()
    args += f" valid_ufreq:{UPE} max_uidx:{UPE*20}".split()
    # --
    return args

def main(args):
    cli_main(args, sbase_getter=conf_getter)

# python3 -m mspx.tasks.zclf.main ...
if __name__ == '__main__':
    import sys
    with Timer(info=f"Main", print_date=True) as et:
        main(sys.argv[1:])

# --
"""
# --
# test0 (base)
RARGS="conf_output:_conf log_file:_log device:0 conf_sbase:bert_name:roberta-base train0.group_files:../data/glue/sst2.train.json dev0.group_files:../data/glue/sst2.dev.json test0.group_files:../data/glue/sst2.test.json fs:build,train,test"
for bmod2 in 0 1; do
for lrate in 1 2 3; do
python3 -m mspx.tasks.zclf.main $RARGS conf_sbase:bert_name:roberta-base::bmod2:${bmod2} lrate.val:0.0000${lrate} log_file:_log${bmod2}${lrate}
done; done
# => 0: [14]0.9427, [4]0.9427, [2]0.9381;; 1: [4]0.9461, [0]0.9415, [4]0.9381
# --
# test1 (ddp & amp)
{
python3 -m mspx.tasks.zclf.main $RARGS
python3 -m mspx.tasks.zclf.main $RARGS use_torch_amp:1
python3 -m mspx.tasks.zclf.main $RARGS amp_opt_level:O2 fp16:1
CUDA_VISIBLE_DEVICES=0,1 python3 -m mspx.cli.run_ddp mspx.tasks.zclf.main $RARGS train0.batch_size:256
CUDA_VISIBLE_DEVICES=0,1 python3 -m mspx.cli.run_ddp mspx.tasks.zclf.main $RARGS use_torch_amp:1 train0.batch_size:256
CUDA_VISIBLE_DEVICES=0,1 python3 -m mspx.cli.run_ddp mspx.tasks.zclf.main $RARGS amp_opt_level:O2 fp16:1 train0.batch_size:256
} |& tee _logD
"""
