#!/bin/python3

# run cmd with ddp

import sys
import importlib
import os
import numpy as np
import torch.multiprocessing as mp

def main_run(rank: int, world_size: int, mname: str, args):
    # add confs for ddp
    args.extend([f"nn.dist_rank:{rank}", f"nn.dist_world_size:{world_size}", f"nn.device:{rank}"])
    if rank > 0:  # disable log_stderr and change log_file's names
        args.extend(["log_stderr:0", "log_magic_file:0", "conf_output:"])
        log_file_item = None
        for a in args:
            if "log_file:" in a:
                log_file_item = a
        if log_file_item is not None:
            args.extend([f"{log_file_item}{rank}"])  # put into another file!
    # --
    module = importlib.import_module(mname)
    # module.main(*args)
    module.main(args)

def main():
    mname = sys.argv[1]
    args = sys.argv[2:]
    # --
    devices = [int(z) for z in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    assert len(devices)>=2, "Otherwise do not need to use DDP!"
    world_size = len(devices)
    mp.spawn(main_run, args=(world_size, mname, args), nprocs=world_size, join=True)
    # --

# CUDA_VISIBLE_DEVICES=?? PYTHONPATH=?? python3 -m mspx.cli.run_ddp ??(real_main) ...(args)
if __name__ == '__main__':
    main()
