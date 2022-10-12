#

# print specific frames

import sys
from typing import Dict, List
from itertools import chain
from collections import Counter
from msp2.utils import zlog, default_json_serializer, OtherHelper, wrap_color, Conf, init_everything
from msp2.data.inst import Doc, Sent, Frame
from msp2.data.vocab import SimpleVocab
from msp2.data.rw import DataReader, LineStreamer, DataWriter

class PFConf(Conf):
    def __init__(self):
        self.input_path = ''
        self.input_format = 'zjson_doc'
        self.filter_code = 'False'

def filter_frame(f: Frame, _ff, conf: PFConf):
    # arg repeat
    arg_counts = Counter()  # role -> count
    for arg in f.args:
        role = arg.role
        arg_counts[role] += 1
    arg_repeat = any(z>=2 for z in arg_counts.values())
    # frame repeat
    repeat_posi, repeat_type, repeat_frame = False, False, False
    for f2 in f.sent.events:
        if f2 is f: continue
        if f2.mention == f.mention:
            repeat_posi = True
            if f.type == f2.type:
                repeat_type = True
                # further check args
                args1 = sorted([(a.role, a.mention.get_sig()) for a in f.args])
                args2 = sorted([(a.role, a.mention.get_sig()) for a in f2.args])
                if args1 == args2:
                    repeat_frame = True
                    break  # here we can safely break
    # --
    return eval(_ff)

def str_frame(f: Frame):
    toks = list(f.sent.seq_word.vals)
    all_anns = [(f.mention, f.type, "blue")] + [(a.arg.mention, a.role, "red") for a in f.args]
    for one_mention, one_name, one_color in all_anns:
        widx, wridx = one_mention.widx, one_mention.wridx
        toks[widx] = wrap_color("[", bcolor=one_color) + toks[widx]
        toks[wridx-1] = toks[wridx-1] + wrap_color(f"]{one_name}", bcolor=one_color)
    return " ".join(toks)

def main(*args):
    conf = init_everything(PFConf(), args)
    # --
    # read all frames
    reader = DataReader(LineStreamer(conf.input_path), conf.input_format)
    all_sents: List[Sent] = []
    for one in reader:
        if isinstance(one, Doc):
            all_sents.extend(one.sents)
        else:
            assert isinstance(one, Sent)
            all_sents.append(one)
    # --
    _ff = compile(conf.filter_code, "", "eval")
    all_count = 0
    filtered_frames = []
    for s in all_sents:
        for f in s.events:
            all_count += 1
            if filter_frame(f, _ff, conf):
                filtered_frames.append(f)
    zlog(f"Filter {all_count} -> {len(filtered_frames)}")
    for i2, f2 in enumerate(filtered_frames):
        zlog(f"#{i2} {str_frame(f2)}")
        # print(f"#{i2} {str_frame(f2)}")
    # breakpoint()

# PYTHONPATH=../../src/ python3 fn_print_frame.py input_path:fn15/fulltext.json "filter_code:arg_repeat"
if __name__ == '__main__':
    main(*sys.argv[1:])
