#

# easier printing of many results

import re
import sys
import numpy as np
import pandas as pd
from collections import Counter
from mspx.utils import zlog, zwarn, ZResult, zopen, zopen_withwrapper, Conf, zglobs, init_everything, ZHelper, Serializable

# --
class MainConf(Conf):
    def __init__(self):
        self.input_path = []
        self.mode = 'table'  # table/avg/...
        self.table_r1 = True  # each file get one res!
        self.table_default = [-10000.]  # default item
        # for table mode
        self.t_lp = r'zboth'  # line pattern for the res lines
        self.t_ff0 = 'ZResult(eval(x))'  # conversion0 to be compatible with previous
        self.t_ff = "float(x)"  # conversion for each res match
        self.t_dims = [-1]  # reshape
        self.t_mdim = -1  # merge (avg) dim?
        self.t_cdims = [-1]  # concat dim?
        self.breakpoint = True

# --
# from resP.py
def get_res(vs):
    v_mean = np.mean(vs).item()
    v_std = np.std(vs).item()
    # ret = f"{v_mean:.2f}[{v_std:.2f}]"
    ret = f"{v_mean:.4f}[{v_std:.4f}]"
    return ret

def main_avg(conf):
    printing, warning = zlog, zwarn
    P = r'([-+]?(?:\d*\.\d+|\d+))'
    pat = None
    for f in conf.input_path:
        with zopen(f) as fd:
            # read current pat
            cur_pat = []
            for line in fd:
                line = line.strip()
                # looking for numbers
                fields = re.split(P, line)
                fields2 = [[float(z)] if re.fullmatch(P, z) else z.strip() for z in fields]
                cur_pat.append(fields2)
            # add pat
            if pat is None:
                pat = cur_pat
            else:  # match and add
                if len(pat) != len(cur_pat):
                    warning(f"Unmatched lines with {f}: {len(pat)} vs {len(cur_pat)}")
                for fs0, fs1 in zip(pat, cur_pat):
                    for ii in range(max(len(fs0), len(fs1))):
                        v0, v1 = fs0[ii] if ii<len(fs0) else None, fs1[ii] if ii<len(fs1) else None
                        if isinstance(v0, list):
                            if isinstance(v1, list):
                                assert len(v1) == 1
                                v0.append(v1[0])
                            else:
                                warning(f"Unmatched entry0: {v0} vs {v1}")
                        else:
                            if isinstance(v1, list):
                                warning(f"Unmatched entry1: {v0} vs {v1}")
                            elif v0 != v1:
                                warning(f"Unmatched str: {v0} vs {v1}")
            # --
    # --
    # final print
    for fs in pat:
        items = [(get_res(z) if isinstance(z, list) else z) for z in fs]
        printing(" ".join(items))
    # --
# --

# helper
def gi(x, *args):
    ret = x
    flat_args = sum([z.split("...") for z in args], [])
    for a in flat_args:
        ret = ret[a]
    return ret

# --
# from res2.py
def main_table(conf: MainConf):
    # first load all res
    all_res = []
    _inputs = conf.input_path if conf.input_path else [sys.stdin]
    _ff0 = ZHelper.eval_ff(conf.t_ff0, 'x', locals=locals().copy(), globals=globals().copy())
    _ff_orig = ZHelper.eval_ff(conf.t_ff, 'x', locals=locals().copy(), globals=globals().copy())
    _ff = (lambda x: conf.table_default if x is None else _ff_orig(x))  # handling empty ones
    if conf.t_lp.endswith("ztest"):  # short-cut!
        conf.t_lp = r"zzzzztestfinal: (\{.*\})"
    if conf.t_lp.endswith("zdev"):  # short-cut!
        conf.t_lp = r"zzzzzdevfinal: (\{.*\})"
    if conf.t_lp.endswith("zboth"):  # short-cut!
        conf.t_lp = r"zzzzz.*final: (\{.*\})"
    _lp = re.compile(conf.t_lp)
    no_res_files = []
    for f in _inputs:
        with zopen_withwrapper(f) as fd:
            f_rr = None
            for line in fd:
                match = re.match(_lp, line)
                if match:
                    res0 = match.groups()[0]  # first match!
                    rr = _ff0(res0)  # first conversion
                    if isinstance(rr, ZResult):
                        rr.file = f
                    f_rr = rr
                    all_res.append(rr)
                    if conf.table_r1:
                        break
            if conf.table_r1 and f_rr is None:
                no_res_files.append(f)
                all_res.append(None)  # add a padding of None
    for rr in all_res:
        zlog(f"Read [{len(all_res)}] {_ff(rr)} from {getattr(rr, 'file', None)}")
    zlog(f"No res files: {no_res_files}")
    # --
    # read inner results
    all_items = [_ff(z) for z in all_res]  # [N, ??]
    all_items = [([z] if not isinstance(z, list) else z) for z in all_items]  # [N, K]
    arr = np.asarray(all_items)  # [N, K]
    # judge shape
    _arr_size = np.prod(arr.shape[:-1])
    t_dims = conf.t_dims
    if -1 in t_dims:  # fill in auto one!
        assert t_dims.count(-1) == 1
        t_ii = t_dims.index(-1)
        t_dims[t_ii] = int(_arr_size // (np.prod(t_dims[:t_ii]) * np.prod(t_dims[t_ii+1:])))
    _missing = np.prod(t_dims) - _arr_size
    if _missing > 0:  # pad!
        zwarn("Padding at the final for missing fields!")
        arr = np.pad(arr, [(0, _missing), (0, 0)])
    arr = arr.reshape(t_dims + [arr.shape[-1]])  # [..., K]
    # --
    # merge things!
    mdim = conf.t_mdim
    if mdim >= 0:
        print(f"Avg at dim={mdim}")
        arr_avg = arr.mean(axis=mdim)
        arr_std = arr.std(axis=mdim)
    else:
        arr_avg = arr
        arr_std = None
    # --
    # concat things!
    _m_shape = arr_avg.shape
    _cdims = [(z if z>=0 else (len(_m_shape)+z)) for z in conf.t_cdims]  # dims to disappear
    _dims0, _dims1 = [ii for ii,zz in enumerate(_m_shape) if ii not in _cdims], \
                     [ii for ii,zz in enumerate(_m_shape) if ii in _cdims]
    _f_shape = [_m_shape[z] for z in _dims0] + [-1]
    arr_avg = np.transpose(arr_avg, _dims0+_dims1).reshape(_f_shape)
    if arr_std is not None:
        arr_std = np.transpose(arr_std, _dims0+_dims1).reshape(_f_shape)
    # --
    # final prep things
    _shape0 = arr_avg.shape[:-1]
    _final_items = []
    for ii, vv_avg in enumerate(arr_avg.reshape([-1, arr_avg.shape[-1]])):
        _ss = []
        vv_std = None if arr_std is None else arr_std.reshape([-1, arr_avg.shape[-1]])[ii]
        for jj, vv2 in enumerate(vv_avg):
            if vv_std is None:
                _ss.append(f"{vv2:.4f}")
            else:
                _ss.append(f"{vv2:.4f}({vv_std[jj]:.4f})")
        _final_items.append("/".join(_ss))
    _final_arr = np.asarray(_final_items, dtype=object).reshape(_shape0)
    # --
    df = pd.DataFrame(_final_arr)
    print(df.to_string())
    if conf.breakpoint:
        breakpoint()
    # --
# --

def main(args):
    conf: MainConf = init_everything(MainConf(), args)
    conf.input_path = zglobs(conf.input_path)
    # --
    globals()[f'main_{conf.mode}'](conf)
    # --

# python3 -m mspx.scripts.tools.print_res input_path:??
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])

# --
# examples
"""
python3 -m mspx.scripts.tools.print_res input_path:./run*/_log t_dims:2,6
python3 -m mspx.scripts.tools.print_res input_path:./run*/_log t_dims:2,6 "t_ff:[x[f'test0_{z}']['zres'] for z in [2,3]]" t_lp:zzztestfinal
python3 -m mspx.scripts.tools.print_res "t_ff:[x[f'test0_{z}']['zres'] for z in [0,1]]" t_lp:zzztestfinal input_path:?? t_dims:2,6 
python3 -m mspx.scripts.tools.print_res "t_ff:[float(Serializable.create(x[f'test0_{z}']['dpar0']).uas) for z in [0,1]]" t_lp:zzztestfinal input_path:?? t_dims:2,6 
"""
