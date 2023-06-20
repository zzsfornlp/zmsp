#

# for convenience

import numpy as np
from mspx.cli.main import main as cli_main
from mspx.utils import Timer
from .mod import *  # note: include the Mod!

def conf_getter(bert_name: str, do_sub: str, **kwargs):
    do_sub = int(do_sub)
    # --
    args = []
    # task & model
    args += "tcs:mlm0:mlm".split()
    args += f"mlm0.do_sub_split:{do_sub}".split()  # add lmhead!
    args += "mlm0.b_inc_lmhead:1".split()  # add lmhead!
    if bert_name:  # note: no init by default!
        args += f"mlm0.b_vpath:{bert_name} mlm0.sconf:{bert_name}".split()
        args += f"mlm0.init_with_bmodel:".split()  # no init!
    # data
    args += "mlm0.max_seq_len:128".split()
    for wset in ["train0", "dev0", "test0"]:
        args += f"{wset}.tasks:mlm0 {wset}.batch_size:{512*128} {wset}.bucket_max_length:128".split()
        # --
        if do_sub:
            args += f"{wset}.inst_f:sentA:60:128".split()
            if bert_name:
                args += f"{wset}.len_f:subword:{bert_name}".split()
        # --
        if wset.startswith("train"):
            args += f"{wset}.do_cache_insts:0 {wset}.cache_mul:20".split()  # no cache!
    # training
    UPE = 1000  # update per "epoch"
    args += f" optim_type:adamw adam_betas:0.9,0.98 adam_eps:1e-6 weight_decay:0.01".split()
    args += f" lrate.val:0.0001 lrate.which_idx:uidx lrate.ff:1-(i-{UPE*10})/{UPE*500} lrate.val_range:0.1,1".split()
    args += f" valid_ufreq:{UPE*5} lrate_warmup_uidx:{UPE*10} max_uidx:{UPE*500}".split()
    # --
    return args

def main(args):
    cli_main(args, sbase_getter=conf_getter)

# python3 -m mspx.tasks.zmlm.main ...
if __name__ == '__main__':
    import sys
    with Timer(info=f"Main", print_date=True) as et:
        main(sys.argv[1:])

# --
"""
# --
CUDA_VISIBLE_DEVICES=0 python3 -m mspx.tasks.zmlm.main conf_sbase:bert_name:bert-base-cased::do_sub:1 zdir:tmp2 conf_output:_conf log_file:_log device:0 train0.group_files:__train0.tok.bz2 dev0.group_files:__dev.tok.json fs:build,train train0.batch_size:6400 valid_ufreq:100 max_uidx:200
CUDA_VISIBLE_DEVICES=0 python3 -m mspx.tasks.zmlm.main conf_sbase:bert_name:bert-base-cased::do_sub:0 zdir:tmp2 conf_output:_conf log_file:_log device:0 train0.group_files:__train0.tok.txt.bz2 dev0.group_files:__dev.tok.txt train0.input_format:plain_sent dev0.input_format:plain_sent fs:build,train train0.batch_size:6400 valid_ufreq:100 max_uidx:200
# --
# first run with .bz2 & dyn. chunk/sub-word 
# --
# for bs in 6400; do
for bs in 6400 25600; do
for ddp in "" "mspx.cli.run_ddp"; do
for ddd in "0" "0,1" "0,1,2,3"; do
if [[ "$ddp" != "" || "$ddd" == "0" ]]; then
for extra in "" "use_torch_amp:1" "amp_opt_level:O2 fp16:1"; do
echo ZRUN $ddp $ddd $extra
for _ in {0..4}; do
CUDA_VISIBLE_DEVICES=$ddd python3 -m $ddp mspx.tasks.zmlm.main conf_sbase:bert_name:bert-base-cased::do_sub:1 zdir:tmp2 conf_output:_conf log_file:_log device:0 train0.group_files:__train0.tok.bz2 dev0.group_files:__dev.tok.json fs:build,train train0.batch_size:$bs valid_ufreq:100 max_uidx:200 $extra
done; done; fi; done; done; done |& tee _log2
# --
# time(s) (plain/torch/apex)[train,fetch,update]
# W(11g-1080Ti)[(ufreq=100,bs=6400,ddp=)]:  (41,14,2.8) (45,13,2.7) (47,13,1.5)
# W(11g-1080Ti)[(ufreq=100,bs=6400,ddp=2)]: (43,30,11.0) (47,30,11.0) (70,30,1.3)
# W(11g-1080Ti)[(ufreq=100,bs=6400,ddp=4)]: (43,70,22.0) (47,70,23.0) (110,70,1.3)
# T(11g-2080Ti)[(ufreq=100,bs=6400,ddp=)]:  (30,22,2.0) (14,22,2.2) (12,22,1.3)
# T(11g-2080Ti)[(ufreq=100,bs=6400,ddp=2)]: (32,48,11.0) (17,45,11.0) (37,46,2.0)
# T(11g-2080Ti)[(ufreq=100,bs=6400,ddp=4)]: (32,90,20.0) (17,90,20.0) (80,90,1.5)
# T(48g-A6000)[(ufreq=100,bs=6400,ddp=)]:  (11,17,1.3) (9,18,1.4) (7.7,18,1.3)
# T(48g-A6000)[(ufreq=100,bs=6400,ddp=2)]: (13,37,3.6) (10,37,5.0) (24,39,1.3)
# T(48g-A6000)[(ufreq=100,bs=6400,ddp=4)]: (13,80,10.0) (10,80,11.0) (70,80,1.3)
# T(48g-A6000)[(ufreq=100,bs=25600,ddp=)]:  (37,70,1.5) (26,70,1.5) (23,70,1.2)
# T(48g-A6000)[(ufreq=100,bs=25600,ddp=2)]: (39,140,8.8) (28,140,7.0) (90,140,1.3)
# T(48g-A6000)[(ufreq=100,bs=25600,ddp=4)]: (40,280,20) (28,280,20) (100,280,1.4)
# => apex slightly faster when ddp=1, certain less mem than torch-amp, but seems sometimes slower when ddp?
# --
# with preprocessing
# for bs in 6400 10000; do
for bs in 25600 51200; do
for ddp in "" "mspx.cli.run_ddp"; do
for ddd in "0" "0,1" "0,1,2,3"; do
if [[ "$ddp" != "" || "$ddd" == "0" ]]; then
for extra in "" "use_torch_amp:1" "amp_opt_level:O2 fp16:1"; do
echo ZRUN $ddp $ddd $extra
CUDA_VISIBLE_DEVICES=$ddd python3 -m $ddp mspx.tasks.zmlm.main conf_sbase:bert_name:bert-base-cased::do_sub:0 zdir:tmp2 conf_output:_conf log_file:_log device:0 train0.group_files:__train0.tok.txt.bz2 dev0.group_files:__dev.tok.txt train0.input_format:plain_sent dev0.input_format:plain_sent fs:build,train train0.batch_size:$bs valid_ufreq:100 max_uidx:400 $extra
done; fi; done; done; done |& tee _log2
# cat _log | grep -E "ZRUN|Train-Info|END"
# W(11g-1080Ti)D04: (43+0.4+2.5)(47+0.4+2.7)(50+0.4+1.5) (44+1.5+17)(47+1.5+20)(66+1.5+1.3)
# T(11g-2080Ti)D04: (30+0.5+2)(14+0.5+2)(12+0.5+1.5) (32+2.2+15)(17+2.2+9)(19+2.2+1.7)
# T(48g-A6000)D024: (38+1+1)(27+1+1)(22+1+1) (40+3+9)(30+3+7)(30+3+1) (40+5+17)(30+5+14)(36+5+1)
# T(48g-A6000)D024: (OOM)(50+3+1)(41+3+1) (OOM)(53+5+10)(52+5+1) (OOM)(52+12+20)(66+12+1)
# => now apex generally slightly faster?
"""
