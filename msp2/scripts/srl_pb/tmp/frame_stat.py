#

# stat the files for frame, mostly from "fn_stat.py"

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
from msp2.data.inst import Doc, Sent, Mention
from msp2.data.vocab import SimpleVocab
from msp2.data.rw import ReaderGetterConf

# --
def stat_sents(sents: List[Sent], stat: StatRecorder):
    # --
    def _has_overlap(_f1, _f2):
        start1, end1 = _f1.mention.widx, _f1.mention.wridx
        start2, end2 = _f2.mention.widx, _f2.mention.wridx
        return not (start1>=end2 or start2>=end1)
    # --
    for sent in sents:
        stat.record_kv("sent", 1)
        stat.record_kv("tok", len(sent))
        stat.srecord_kv("sent_ntok_d10", len(sent)//10)
        stat.srecord_kv("sent_nframe", len(sent.events))
        cur_pos_list = sent.seq_upos.vals if sent.seq_upos is not None else None
        # frame
        for frame in sent.events:
            widx, wlen = frame.mention.get_span()
            # --
            stat.record_kv("frame", 1)
            # frame target length
            stat.srecord_kv("frame_wlen", wlen)
            # frame trigger upos
            stat.srecord_kv("frame_trigger_pos", ",".join([] if cur_pos_list is None else cur_pos_list[widx:widx+wlen]))
            # frame target overlap with others?
            stat.record_kv("frame_overlapped", int(any(_has_overlap(frame, f2) for f2 in sent.events if f2 is not frame)))
            # frame type
            stat.srecord_kv("frame_type", frame.type)
            stat.srecord_kv("frame_type0", frame.type.split(".")[0])  # in case of PB
            # args
            all_args = Counter()
            stat.srecord_kv("frame_narg", len(frame.args))
            for alink in frame.args:
                rank = alink.info.get("rank", 1)
                # --
                stat.record_kv("arg", 1)
                stat.record_kv(f"arg_R{rank}", 1)
                # arg target length
                stat.srecord_kv("arg_wlen_m30", min(30, alink.mention.wlen))
                # arg overlap with others?
                stat.record_kv("arg_overlapped", int(any(_has_overlap(alink, a2) for a2 in frame.args if a2 is not alink)))
                stat.record_kv(f"arg_overlapped_R{rank}",
                               int(any(_has_overlap(alink, a2) for a2 in frame.args if
                                       a2 is not alink and a2.info.get("rank", 1) == rank)))
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

# --
def stat_docs(docs: List[Doc], stat: StatRecorder):
    for doc in docs:
        stat.record_kv("doc", 1)
        stat.srecord_kv("doc_nsent_d10", len(doc.sents)//10)
        stat_sents(doc.sents, stat)
    # --

# =====
class MainConf(Conf):
    def __init__(self):
        self.R = ReaderGetterConf()
        self.result_key = ""  # by default, use input file name!
        self.result_center = "res.json"  # store key
        # for query
        self.key_re_pattern = ".*"

def main(*args):
    conf = MainConf()
    conf.update_from_args(args)
    # --
    if conf.R.input_path:
        # stat mode
        # --
        reader = conf.R.get_reader()
        insts = list(reader)
        stat = StatRecorder()
        if len(insts) > 0:
            if isinstance(insts[0], Doc):
                stat_docs(insts, stat)
            else:
                stat_sents(insts, stat)
        # --
        key = conf.R.input_path
        res = {}
        for ss in [stat.plain_values, stat.special_values]:
            res.update(ss)
        show_res = [f"{kk}: {str(res[kk])}\n" for kk in sorted(res.keys())]
        zlog(f"# -- Stat Mode, Read from {key} and updating {conf.result_center}:\n{''.join(show_res)}")
        if conf.result_center:
            if os.path.isfile(conf.result_center):
                d0 = default_json_serializer.from_file(conf.result_center)
            else:
                d0 = {}
            d0[key] = res
            default_json_serializer.to_file(d0, conf.result_center)
            # breakpoint()
    else:
        # query mode: query across datasets (key)
        data = default_json_serializer.from_file(conf.result_center)
        pattern = re.compile(conf.key_re_pattern)
        hit_keys = sorted(k for k in data.keys() if re.fullmatch(pattern, k))
        zlog(f"Query for {hit_keys}")
        # breakpoint()
        # --
        while True:
            try:
                code = input(">> ")
            except EOFError:
                break
            except KeyboardInterrupt:
                continue
            code = code.strip()
            if len(code) == 0: continue
            # --
            zlog(f"Eval `{code}':")
            for k in hit_keys:
                d = data[k]
                try:
                    one_res = eval(code)
                except:
                    one_res = traceback.format_exc()
                zlog(f"#--{k}:\n{one_res}")
            # --
        # --

# PYTHONPATH=../src/ python3 frame_stat.py
if __name__ == '__main__':
    main(*sys.argv[1:])

"""
# stat for all
for f in */*.ud.json ../fn/parsed/fn1*_*.*.*; do
PYTHONPATH=../../zsp2021/src/ python3 frame_stat.py input_path:$f
done |& tee _log_stat
"""
