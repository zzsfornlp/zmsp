#

# map and check pb2fn (with trained model)

import os
import re
import math
from collections import Counter, defaultdict
from msp2.data.inst import yield_sents, yield_frames
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import Conf, zlog, zopen, zwarn, init_everything, default_json_serializer, OtherHelper, zglob
from msp2.tasks.zmtl3.mod.extract.evt_arg import onto as zonto

# --
class MainConf(Conf):
    def __init__(self):
        # step 1: read input instances and eval
        self.input_files = []
        self.output_stat = ""
        self.input_stat = ""  # shortcut for reading stat
        # step 2: map and add to onto
        self.input_onto = ""
        self.output_onto = ""
        self.topk = 1
        self.topp = 0.
        # --

#
def get_stat(files):
    cc = Counter()
    ret = {}  # frame_name -> {'count': int, 'roles': ARG? -> {'count': int, 'maps': {Role -> [num, sum_prob]}}}
    _pat = re.compile(r'ARG[0-9].*')
    # --
    for f in files:
        cc['all_file'] += 1
        reader = ReaderGetterConf().get_reader(input_path=f)
        for inst in reader:
            cc['all_inst'] += 1
            for sent in yield_sents(inst):
                cc['all_sent'] += 1
                for evt in sent.events:
                    cc['all_evt'] += 1
                    frame_name = evt.label
                    if frame_name not in ret:
                        ret[frame_name] = {'count': 0, 'roles': {}}
                    ret[frame_name]['count'] += 1
                    for arg in evt.args:
                        orig_label_name = arg.info['orig_label']
                        label_name = arg.label
                        cc['all_arg'] += 1
                        if not _pat.fullmatch(orig_label_name):
                            cc['all_arg0'] += 1
                            continue  # ignore others
                        else:
                            cc['all_arg1'] += 1
                        # --
                        _rr = ret[frame_name]['roles'].get(orig_label_name)
                        if _rr is None:
                            _rr = {'count': 0, 'maps': {}}
                            ret[frame_name]['roles'][orig_label_name] = _rr
                        _rr['count'] += 1
                        if label_name not in _rr['maps']:
                            _rr['maps'][label_name] = [0, 0.]
                        _rr['maps'][label_name][0] += 1  # count
                        _rr['maps'][label_name][1] += math.exp(arg.score)  # prob
    # --
    zlog(f"Read all files from {files}: {cc}")
    return ret
    # --

def show_stat(stat):
    ret = {
        'n_frames': len(stat), 'n_roles': sum(len(ff['roles']) for ff in stat.values()),
        'c_frames': sum(ff['count'] for ff in stat.values()),
        'c_roles': sum(rr['count'] for ff in stat.values() for rr in ff['roles'].values()),
    }
    zlog(f"Stat = {ret}")
    return ret

def role2np_fn(role: str):
    ret = role.lower()
    ret = " ".join(ret.split("_"))
    return ret

# --
def convert_roles_pb2fn(onto, pb2fn, conf):
    cc = Counter()
    for ff in onto.frames:
        cc['all_frame'] += 1
        _map = pb2fn.get(ff.name)
        if _map is None:
            cc['all_frame_map0'] += 1
            continue
        else:
            cc[f'all_frame_map1'] += 1
        # only core_roles!
        for cr in ff.core_roles:
            if cr.name.startswith("ARGM"):
                cc['all_roleM'] += 1
            else:
                cc['all_roleC'] += 1
                # --
                _rr = _map['roles'].get(cr.name)
                if _rr is None:
                    cc['all_roleC_miss'] += 1
                    continue
                # --
                cc['all_roleCH'] += 1  # hit
                # find the best one!
                _cands = [(k, v[0], v[1]) for k,v in _rr['maps'].items()]
                _best0 = max(_cands, key=lambda x: (x[1], x[2]))  # best by count
                _best1 = max(_cands, key=lambda x: (x[2], x[1]))  # best by prob
                if _best0 is not _best1:
                    cc['all_roleCH_c01'] += 1  # disagree?
                _final_best = _best0
                _rr_count = _rr['count']
                cc[f'all_roleCH_perc={int(4*_final_best[1]/_rr_count)}'] += 1  # perc by count
                # --
                _counts = sorted(_cands, reverse=True, key=lambda x: (x[1], x[2]))[:conf.topk]
                _fns = []
                _budget = int(sum([x[1] for x in _cands]) * conf.topp)
                while _budget >= 0 and len(_counts)>0:
                    _one = _counts.pop(0)
                    _fns.append(_one[0])
                    _budget -= _one[1]
                assert len(_fns) > 0
                # breakpoint()
                # assign
                cc[f'all_roleCH_NUM={len(_fns)}'] += 1
                cr.info['np_fn'] = [role2np_fn(z) for z in _fns]
    # --
    zlog(f"Convert roles:")
    OtherHelper.printd(cc, try_div=True)
    # --
    return cc

# --
def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    # step 1: get stat
    if conf.input_stat:  # read
        stat = default_json_serializer.from_file(conf.input_stat)
        zlog(f"Read stat from {conf.input_stat}")
    else:  # build on the fly
        _input_files = sum([zglob(z) for z in conf.input_files], [])
        stat = get_stat(_input_files)
        zlog("Build stat finished.")
    show_stat(stat)
    # breakpoint()
    if conf.output_stat:
        default_json_serializer.to_file(stat, conf.output_stat)
    # --
    # step 2: put onto
    if conf.input_onto:
        onto = zonto.Onto.load_onto(conf.input_onto)
        convert_roles_pb2fn(onto, stat, conf)
        if conf.output_onto:
            default_json_serializer.to_file(onto.to_json(), conf.output_onto, indent=2)
    # --

# python3 -m msp2.tasks.zmtl3.scripts.srl.s22_map_fn input_onto:... output_onto:
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# --
"""
1. first train a label clf model
# -> see "run2201:zgo0111_fnlab"
2. test with pb files (@frames)
for vv in 0 1; do
odir=fn_pred_files${vv}
mkdir -p $odir
for ff in ../../events/data/data21f/en.{ontoC,ewt}.* ; do
mdir=../_zrun_zgo0111_fnlab_${vv}/
CUDA_VISIBLE_DEVICES=$vv python3 -m msp2.tasks.zmtl3.main.test $mdir/_conf model_load_name:$mdir/zmodel.best.m vocab_load_dir:$mdir/ log_stderr:1 log_file: test0.output_dir:$odir "test0.group_files:`basename $ff`" arg0.default_frame_name:Event
done |& tee $odir/_log &
done
3. obtain stat & mapped-onto with s22_map_fn.py
..., see 'prepare.sh'
for eg: python3 -m msp2.tasks.zmtl3.scripts.srl.s22_map_fn 'input_files:fn_pred_files0/*.json' input_onto:./f_pbA.json
# --
python3 -m msp2.tasks.zmtl3.scripts.srl.s22_map_fn 'input_files:fn_pred_files0/*.json' output_stat:fn_pred_files0.json
python3 -m msp2.tasks.zmtl3.scripts.srl.s22_map_fn 'input_files:fn_pred_files1/*.json' output_stat:fn_pred_files1.json
# Read all files from [...]: Counter({'all_arg': 1009760, 'all_arg0': 552896, 'all_arg1': 456864, 'all_evt': 327355, 'all_inst': 110848, 'all_sent': 110848, 'all_file': 6})
# Stat = {'n_frames': 6043, 'n_roles': 12403, 'c_frames': 327355, 'c_roles': 456864}
"""
# with xlm
"""
1. first train a label clf model
# -> see "zzf_xfnl"
2. test with pb files
mkdir _xfnl
ii=0
for ff in zh.c09.all.ud.json es.c09.all.ud.json ; do
mdir=../run_zzf_xfnl_ALL/run_zzf_xfnl_0/
CUDA_VISIBLE_DEVICES=$ii python3 -m msp2.tasks.zmtl3.main.test $mdir/_conf model_load_name:$mdir/zmodel.best.m vocab_load_dir:$mdir/ log_stderr:1 log_file: test0.input_dir:./ test0.output_dir:_xfnl test0.group_files:$ff arg0.default_frame_name:Event |& tee _xfnl/_log_${ff:0:2} &
ii=$((ii+1))
done
# 3. stat and map
python3 -m msp2.tasks.zmtl3.scripts.srl.s22_map_fn 'input_files:_xfnl/_zout*.json' input_onto:merged_cl3sem.json output_onto:_xfnl/fn_pred_onto.json
python3 -m msp2.tasks.zmtl3.scripts.misc.onto_utils input_ontos:_xfnl/fn_pred_onto.json output_onto:_xfnl/merged_cl3sem.json
"""
