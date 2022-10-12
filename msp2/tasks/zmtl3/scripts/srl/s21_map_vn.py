#

# map and check pb2vn

import os
from collections import Counter, defaultdict
from msp2.data.inst import yield_sents, yield_frames
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import Conf, zlog, zopen, zwarn, init_everything, default_json_serializer, OtherHelper, zglob
from msp2.tasks.zmtl3.mod.extract.evt_arg import onto as zonto

class MainConf(Conf):
    def __init__(self):
        self.input_onto = ""
        self.output_onto = ""
        self.pb2vn = ""
        self.check_input_files = []  # check input data
        # --

#
def convert_roles_pb2vn(onto, pb2vn):
    cc = Counter()
    for ff in onto.frames:
        cc['all_frame'] += 1
        _m0 = pb2vn.get(ff.name)
        if _m0 is None:
            _map = {}
            cc['all_frame_map0'] += 1
        else:
            cc[f'all_frame_map{min(2, len(_m0))}'] += 1
            _map = {}
            for _mm in _m0.values():
                _map.update(_mm)
            _conflict = any(_map[k]!=v for _mm in _m0.values() for k,v in _mm.items())
            cc['all_frame_mapC'] += int(_conflict)
        for cr in ff.core_roles:
            if cr.name.startswith("ARGM"):
                cc['all_roleM'] += 1
            else:
                cc['all_roleC'] += 1
                _orig_vn = cr.info.get("np_vn")
                _map_vn = _map.get(cr.name)
                cc[f'all_roleC_O{int(_orig_vn is not None)}M{int(_map_vn is not None)}'] += 1
                _conflict = (_orig_vn is not None and _map_vn is not None and _orig_vn != _map_vn)
                cc['all_roleC_C'] += int(_conflict)
    # --
    zlog(f"Convert roles:")
    OtherHelper.printd(cc, try_div=True)
    # --
    return cc

def stat_file(file, onto):
    cc = Counter()
    reader = ReaderGetterConf().get_reader(input_path=file)
    # --
    # note: special ones for fn
    def _change_fn_label(_label: str):
        if _label.endswith("_2"):
            return "_zz_ignore_this_one"
        elif _label.endswith("_1"):
            return (_label[:-3] + 'ies') if _label[-3] == 'y' else (_label[:-2] + 's')
        else:
            return _label
    # --
    for inst in reader:
        cc['all_inst'] += 1
        for sent in yield_sents(inst):
            cc['all_sent'] += 1
            for evt in sent.events:
                cc['all_frame'] += 1
                ff = onto.find_frame(evt.label)
                if ff is None:
                    cc['all_frame_None'] += 1
                    continue
                for arg in evt.args:
                    cc['all_arg'] += 1
                    _role, _iscore = ff.find_role(_change_fn_label(arg.label))
                    if _iscore and _role is not None:
                        cc['all_arg_hit'] += 1
                        cc['all_arg_hasvn'] += int(_role.info.get("np_vn") is not None)
    # --
    zlog(f"Read {file}: {cc}")
    return cc

# --
def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    onto = zonto.Onto.load_onto(conf.input_onto)
    pb2vn = default_json_serializer.from_file(conf.pb2vn)
    convert_roles_pb2vn(onto, pb2vn)
    if conf.output_onto:
        default_json_serializer.to_file(onto.to_json(), conf.output_onto, indent=2)
    _input_files = sum([zglob(z) for z in conf.check_input_files], [])
    if _input_files:
        cc = Counter()
        for file in _input_files:
            cc += stat_file(file, onto)
        zlog(f"Read all files: {cc}")
    # --

# python3 -m msp2.tasks.zmtl3.scripts.srl.s21_map_vn input_onto:... output_onto: pb2vn:... check_input_files:...
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

