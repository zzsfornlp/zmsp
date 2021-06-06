#

# check the recorded losses

import sys
from typing import List
from collections import Counter
import numpy as np
from msp2.utils import Conf, zlog, OtherHelper
from msp2.data.inst import yield_sents, set_ee_heads
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

class MainConf(Conf):
    def __init__(self):
        self.input = ReaderGetterConf()

def main(args):
    conf = MainConf()
    conf.update_from_args(args)
    zlog(f"Ready to analyze with {conf.input}")
    # --
    input_sents = list(yield_sents(conf.input.get_reader()))
    all_events = []
    all_loss_avgs = []
    cc = Counter()
    for s in input_sents:
        for f in s.events:
            one_loss_avg = f.info.get("loss_avg")
            if one_loss_avg is not None:
                cc["frame_hit"] += 1
                all_events.append(f)
                all_loss_avgs.append(one_loss_avg)
            else:
                cc["frame_nohit"] += 1
    zlog(cc)
    arr = np.array(all_loss_avgs)  # [N, Point, NL]
    breakpoint()
    # --

# PYTHONPATH=../../../src/ python3 -m pdb check_loss_records.py input_path:
if __name__ == '__main__':
    main(sys.argv[1:])
