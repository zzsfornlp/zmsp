#

# directly perform extra postprocessing and write *.a2 files

import os
import re
import math
import sys
from itertools import chain
from collections import Counter, defaultdict, OrderedDict
from msp2.data.inst import yield_sents, yield_frames, Sent, Doc, Frame, set_ee_heads
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import Conf, zlog, zopen, zwarn, init_everything, default_json_serializer, OtherHelper, zglob, mkdir_p

# --
def _process_evt(evt, cur_eidx):
    p_res = evt.info.get('frames')  # List[Frame: (EID, (label,aid)...)]
    if p_res is None:  # process this one!
        evt_args = defaultdict(list)
        for arg in evt.args:
            evt_args[arg.label].append(arg)
        if 'Theme1' in evt_args:
            evt_args['Theme'] = evt_args['Theme1']
            del evt_args['Theme1']
        elif 'Theme2' in evt_args:
            evt_args['Theme'] = evt_args['Theme2']
            del evt_args['Theme2']
        assert len(evt_args) <= 2
        if 'Theme' not in evt_args:  # must have theme!
            evt.info['frames'] = []
            return
        # --
        cur_frames = [[]]  # List[List(Tuple(label, aid))]
        for _key in list(evt_args.keys()):
            # check current ones
            one_parts = []
            for arg in evt_args[_key]:
                if arg.arg.label == 'Protein':
                    one_parts.append((_key, arg.arg.id))
                else:  # evt!
                    evt2 = arg.arg
                    _process_evt(evt2, cur_eidx)
                    one_parts.extend([(_key, z[0]) for z in evt2.info['frames']])
            # multiplied together
            cur_frames = [a+[b] for a in cur_frames for b in one_parts]
        # --
        # add ids
        for ii in range(len(cur_frames)):
            cur_eidx[0] += 1
            cur_frames[ii] = [f"E{cur_eidx[0]}"] + cur_frames[ii]
        # --
        evt.info['frames'] = cur_frames
    # --

def main(input_file: str, output_dir: str):
    cc = Counter()
    mkdir_p(output_dir, raise_error=True)
    # --
    reader = ReaderGetterConf().get_reader(input_path=input_file)
    for inst in reader:
        cc['doc'] += 1
        doc_id = inst.id
        with zopen(os.path.join(output_dir, f"{doc_id}.a2"), 'w') as fd:
            cur_eidx = [0]  # recording
            ts = {}
            for sent in inst.sents:
                cc['sent'] += 1
                # --
                cols0 = sent.info['orig_words']
                cols1 = [" ".join(z.split("-")) for z in sent.info['char_posi_info']]
                # propagate frames
                for evt in sent.events:
                    cc['evt'] += 1
                    _process_evt(evt, cur_eidx)
                # write out
                for evt in sent.events:
                    widx = evt.mention.shead_widx
                    for frame in evt.info['frames']:
                        cc['frame'] += 1
                        tid = f"T{10000+int(frame[0][1:])}"  # note: this should be enough
                        tsig = f"{evt.label} {cols1[widx]}\t{cols0[widx]}"
                        if tsig in ts:
                            tid = ts[tsig]
                        else:
                            ts[tsig] = tid
                            fd.write(f"{tid}\t{tsig}\n")
                        arg_s = "" if len(frame)==1 else (" " + " ".join([f"{a}:{b}" for a,b in frame[1:]]))
                        fd.write(f"{frame[0]}\t{evt.label}:{tid}{arg_s}\n")
    # --
    zlog(f"Read from {input_file} to {output_dir}: {cc}")
    # --

# --
# python3 -m msp2.tasks.zmtl3.scripts.genia.write_a2 IN OUT
# perl beesl/bioscripts/eval/a2-normalize.pl -v -g beesl/data/corpora/GE11/dev/ -o out_norm OUT/*.a2
# perl beesl/bioscripts/eval/a2-evaluate.pl -g beesl/data/corpora/GE11/dev/ -t1 -sp out_norm/*.a2
# note: ok, slightly worse (-0.5) than beesl's postprocess (53.72 vs 54.24) -> just ~2 points worse for binding
if __name__ == '__main__':
    main(*sys.argv[1:])
