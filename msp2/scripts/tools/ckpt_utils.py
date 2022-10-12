#

# some tools to manipulate ckpt models!

import sys
import json
import numpy as np
from collections import Counter, OrderedDict
import re
import torch
from msp2.utils import Conf, init_everything, zopen, zlog, Random, zwarn, zglob

class MainConf(Conf):
    def __init__(self):
        # input
        self.inputs = []  # input models
        self.output = ""  # output
        # --
        self.action = ''
        # a1: extract model
        self.extract_filter = '.*'  # re pattern
        self.extract_sub = ''  # '...=>...'
        # a2: compare m0 & m1
        self.cmp_break = False
        # a3: average models
        # --

    @property
    def all_inputs(self):
        ret = sum([zglob(z) for z in self.inputs], [])
        return ret

# --
def _load_model(f: str):
    if f:
        m = torch.load(f, map_location='cpu')
        zlog(f"Load from {f}")
    else:
        m = None
    return m
def _save_model(m, f: str):
    if f:
        torch.save(m, f)
        zlog(f"Save to {f}")
# --
def _extract_model(conf, mname=None, save=True):
    if mname is None:
        mname = conf.all_inputs[0]
    m = _load_model(mname)  # only take the first one
    # step 1: filter
    pat = re.compile(conf.extract_filter)
    orig_keys = list(m.keys())
    del_keys = []
    for k in orig_keys:
        if pat.fullmatch(k):
            pass
        else:
            del m[k]
            del_keys.append(k)
    zlog(f"Filter by ``{conf.extract_filter}'': del={len(del_keys)},kept={len(m)}")
    # step 2: change name
    if conf.extract_sub:
        s1, s2 = conf.extract_sub.split("=>")
        zlog(f"Sub with {s1} => {s2}")
        m_new = OrderedDict()
        for k, v in m.items():
            k_new = re.sub(s1, s2, k)
            m_new[k_new] = v
            if k != k_new:
                zlog(f"Change name {k} => {k_new}")
        m = m_new
    # save
    if save:
        _save_model(m, conf.output)
    # --
    return m
# --
def _cmp_model(conf):
    m0, m1 = [_load_model(z) for z in conf.all_inputs[:2]]  # only get the first two!
    keys0, keys1 = set(m0.keys()), set(m1.keys())
    # extra ones?
    extra0, extra1 = keys0-keys1, keys1-keys0
    zlog(f"Extra in M0 [{len(extra0)}/{len(keys0)}]: {extra0}")
    zlog(f"Extra in M1 [{len(extra1)}/{len(keys1)}]: {extra1}")
    # common ones
    diffs = []
    for k in sorted(keys0 & keys1):
        v0, v1 = m0[k], m1[k]
        if not torch.allclose(v0, v1):
            if conf.cmp_break:
                breakpoint()
            diffs.append(k)
    zlog(f"Diff are [{len(diffs)}]: {diffs}")
# --
def _avg_model(conf):
    ms = [_extract_model(conf, f, False) for f in conf.all_inputs]
    out = OrderedDict()
    for m in ms:
        for k, v in m.items():
            if k not in out:
                out[k] = [v]
            else:
                out[k].append(v)
    for k in list(out.keys()):
        vs = out[k]
        if len(vs) < len(ms):
            zwarn(f"Warning: {k} has less params: {(len(vs))} < {len(ms)}")
        v2 = (torch.stack(vs, 0).sum(0) / len(vs)).to(vs[0].dtype)
        out[k] = v2
    # save
    _save_model(out, conf.output)
    zlog(f"Finished avg_model with #param={len(out)}")
# --
def _count_model(conf):
    for ff in conf.all_inputs:
        m = _load_model(ff)
        cc = sum(np.prod(v.shape) for v in m.values())
        zlog(f"Count {ff} = {cc}")
# --

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # go!!
    action_map = {'extract': _extract_model, 'cmp': _cmp_model, 'avg': _avg_model, 'count': _count_model}
    zlog(f"Go with action: {conf.action}")
    action_map[conf.action](conf)
    # --

# PYTHONPATH=../src/ python3 ckpt_utils.py
if __name__ == '__main__':
    main(*sys.argv[1:])

"""
# cmp
python3 -m msp2.scripts.tools.ckpt_utils action:cmp inputs:./zmodel.curr.m0,./zmodel.curr.m1 cmp_break:1
# extract
python3 -m msp2.scripts.tools.ckpt_utils action:extract inputs:./zmodel.curr.m0 'extract_filter:^Menc.*' 'extract_sub:Menc.=>' 'extract_out:_tmp.m'
# average
python3 -m msp2.scripts.tools.ckpt_utils action:avg inputs:...
"""
