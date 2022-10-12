#

# stat the files for frame, mostly from "fn_stat.py"/"frame_stat.py"

import sys
import os
import re
from typing import Dict, List
from itertools import chain
from collections import Counter
import traceback
import numpy as np
import pandas as pd
from msp2.utils import zlog, default_json_serializer, OtherHelper, Conf, StatRecorder, MyCounter
from msp2.data.inst import Doc, Sent, Mention, set_ee_heads
from msp2.data.vocab import SimpleVocab
from msp2.data.rw import ReaderGetterConf

# --
def do_stat(docs: List[Doc], stat: StatRecorder):
    # --
    def _has_overlap(_f1, _f2):
        start1, end1 = _f1.mention.widx, _f1.mention.wridx
        start2, end2 = _f2.mention.widx, _f2.mention.wridx
        return (_f1.mention.sent.sid==_f2.mention.sent.sid) and not (start1>=end2 or start2>=end1)
    # --
    for doc in docs:
        stat.record_kv("doc", 1)
        for sent in doc.sents:
            stat.record_kv("sent", 1)
            stat.record_kv("tok", len(sent))
            stat.srecord_kv("sent_nframe", len(sent.events))
            cur_pos_list = sent.seq_upos.vals if sent.seq_upos is not None else None
            # frame
            for frame in sent.events:
                widx, wlen = frame.mention.get_span()
                # --
                stat.record_kv("frame", 1)
                # frame target length
                stat.srecord_kv("frame_wlen", wlen)
                # frame target overlap with others?
                stat.record_kv("frame_overlapped", int(any(_has_overlap(frame, f2) for f2 in sent.events if f2 is not frame)))
                # frame type
                stat.srecord_kv("frame_type", frame.type)
                # args
                all_args = Counter()
                stat.srecord_kv("frame_narg", len(frame.args))
                for alink in frame.args:
                    # --
                    stat.record_kv("arg", 1)
                    # arg target length
                    stat.srecord_kv("arg_wlen_m30", min(30, alink.mention.wlen))
                    # arg overlap with others?
                    stat.record_kv("arg_overlapped", int(any(_has_overlap(alink, a2) for a2 in frame.args if a2 is not alink)))
                    # arg role
                    stat.srecord_kv("arg_role", alink.role)
                    # --
                    all_args[alink.role] += 1
                # check repeat
                for rr, cc in all_args.items():
                    stat.srecord_kv("arg_repeat", cc, c=cc)
                    if cc>1:
                        stat.srecord_kv("arg_repeatR", f"{cc}*{rr}")
                # --
            # for prediction over this sentence
            cur_frames = {}  # widx -> [types: List, {arg_widx: List[roles]}]
            for frame in sent.events:
                widx = frame.mention.shead_widx
                if widx is not None:
                    if widx not in cur_frames:
                        cur_frames[widx] = [[], {}]
                    list_type, map_args = cur_frames[widx]
                    list_type.append(frame.type)
                for arg in frame.args:
                    _cross = (arg.arg.sent is not sent)
                    stat.record_kv("arg_cross_sent", int(_cross))
                    if _cross:
                        continue
                    arg_widx = arg.arg.mention.shead_widx
                    if arg_widx is not None:
                        if arg_widx not in map_args:
                            map_args[arg_widx] = []
                        map_args[arg_widx].append(arg.role)
            # stat
            for list_type, map_args in cur_frames.values():
                stat.srecord_kv("m_evt_cc", len(list_type))
                stat.srecord_kv("m_evt_ccR", len(list_type), c=len(list_type))
                stat.srecord_kv("m_evt_tt", '|'.join(sorted(list_type)))
                for list_role in map_args.values():
                    stat.srecord_kv("m_arg_cc", len(list_role))
                    stat.srecord_kv("m_arg_ccR", len(list_role), c=len(list_role))
                    stat.srecord_kv("m_arg_tt", '|'.join(sorted(list_role)))
            # --
    # --

# =====
class MainConf(Conf):
    def __init__(self):
        self.R = ReaderGetterConf()
        # --

def main(*args):
    conf = MainConf()
    conf.update_from_args(args)
    # --
    reader = conf.R.get_reader()
    docs = list(reader)
    try:
        set_ee_heads(docs)  # set head to do head-related stat
    except:
        pass
    stat = StatRecorder()
    do_stat(docs, stat)
    # --
    zlog(f"Stat for {conf.R.input_path}")
    zlog(f"==\n{OtherHelper.printd_str(stat.summary())}")
    special_d = {k: v.summary_str(topk=100) for k,v in stat.special_values.items()}
    zlog(f"==\n"+OtherHelper.printd_str(special_d, sep='\n\n'))
    # --

if __name__ == '__main__':
    main(*sys.argv[1:])

# --
# PYTHONPATH=../../zsp2021/src/ python3 sz_stat.py input_path:??
