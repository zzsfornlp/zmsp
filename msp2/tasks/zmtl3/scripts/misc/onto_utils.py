#

# utils for onto

import sys
import re
from collections import Counter, defaultdict
import numpy as np
from msp2.nn import BK
from msp2.data.inst import DataPadder
from msp2.utils import init_everything, Conf, zlog, zwarn, zopen, default_json_serializer, default_pickle_serializer
from msp2.tasks.zmtl3.mod.extract.evt_arg.onto import Onto
from msp2.tasks.zmtl3.mod.extract.evt_arg.m_query import build_onto_reprs

# --
class MainConf(Conf):
    def __init__(self):
        self.input_ontos = []  # input ontos
        self.output_onto = ""
        self.output_repr = ""
        self.repr_bname = "roberta-base"
        # --

def merge_ontos(ontos):
    # simple merge
    if len(ontos) > 1:
        all_frames = sum([z.frames for z in ontos], [])
        all_roles = sum([z.roles for z in ontos], [])
        final_onto = Onto(all_frames, all_roles)
        zlog(f"Simply merge into {final_onto}")
    else:
        final_onto = ontos[0]
    return final_onto

# --
def main(*args):
    conf = MainConf()
    conf: MainConf = init_everything(conf, args)
    # load ones
    ontos = []
    for one_input in conf.input_ontos:
        one_onto = Onto.load_onto(one_input)
        ontos.append(one_onto)
    # simply merge
    final_onto = merge_ontos(ontos)
    # write it
    if conf.output_onto:
        default_json_serializer.to_file(final_onto.to_json(), conf.output_onto)
        zlog(f"Write {final_onto} to {conf.output_onto}")
        # --
        # also print pp_str format
        inc_noncore = {'Place', 'Instrument', 'Weapon', 'Vehicle', 'ARGM-LOC'}
        pp_s = final_onto.pp_str(inc_noncore)
        with zopen(conf.output_onto + ".txt", 'w') as fd:
            fd.write(pp_s)
        # --
    # get repr
    if conf.output_repr:
        # repr_f, repr_r = repr_ontos(final_onto, conf)
        repr_f, repr_r = build_onto_reprs(final_onto, conf.repr_bname)
        default_pickle_serializer.to_file((repr_f, repr_r), conf.output_repr)
        zlog(f"Write {(repr_f.shape, repr_r.shape)} to {conf.output_repr}")
    # --

# --
if __name__ == '__main__':
    main(*sys.argv[1:])

# --
"""
python3 -m msp2.tasks.zmtl3.scripts.misc.onto_utils input_ontos:ace output_repr:repr_ace.pkl device:0
python3 -m msp2.tasks.zmtl3.scripts.misc.onto_utils input_ontos:ere output_repr:repr_ere.pkl device:0
"""
