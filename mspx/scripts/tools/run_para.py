#

# simple parallel running of certain commands

import argparse
import os
import multiprocessing

# --
try:
    from mspx.utils import zlog, system
except:
    print("CANNOT find mspx!")
    zlog = print
    system = lambda cmd, **kwargs: os.system(cmd)
# --

# --
# global lock!
_global_lock = multiprocessing.Lock()
manager = multiprocessing.Manager()
global_resources = manager.dict()
# --

def parse_args():
    parser = argparse.ArgumentParser("Run things in parallel!")
    parser.add_argument("-n", "--nworkers", type=int, default=4)
    parser.add_argument("-i", "--inputs", type=str, nargs="+", required=True)
    # note: to run CMD: [[IN]]: input file, [[OUT]]: output name; for example: grep ZZ [[IN]] >[[OUT]]
    parser.add_argument("-c", "--cmd", type=str, required=True)
    parser.add_argument("-o", "--output_pattern", type=str, default="''")
    # --
    args = parser.parse_args()
    return args

def run(aa):
    fin, args = aa
    # --
    _ff = args.output_pattern.strip()
    if not _ff.startswith('lambda'):
        _ff = "lambda x: " + _ff
    fout = eval(_ff)(fin)
    CMD = args.cmd
    CMD = CMD.replace("[[IN]]", fin)
    CMD = CMD.replace("[[OUT]]", fout)
    with _global_lock:  # get resource
        cur_idx = list(global_resources.keys())[0]
        del global_resources[cur_idx]
        zlog(f"Run with [[IDX]]={cur_idx}: {CMD} ")
    try:
        CMD = CMD.replace("[[IDX]]", str(cur_idx))
        system(CMD, pp=True)
    except:
        import traceback
        traceback.print_exc()
        raise RuntimeError()
    finally:  # put resource back
        with _global_lock:
            global_resources[cur_idx] = 1
    # --

def main():
    args = parse_args()
    # --
    inputs = args.inputs
    zlog(f"Run {len(inputs)} inputs with {args.nworkers} workers! Inputs={inputs}")
    global_resources.update({z:1 for z in range(args.nworkers)})  # init resources
    with multiprocessing.Pool(processes=args.nworkers) as p:
        p.map(run, [(f,args) for f in inputs], chunksize=1)
    # --

# python3 -m mspx.scripts.tools.run_para -i ... -c "??" -o "??"
# python3 run_para.py -i ... -c "??" -o "??"
if __name__ == '__main__':
    main()
