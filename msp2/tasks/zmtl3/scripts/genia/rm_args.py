#

# script to remove args to help debugging

import os
import re
import math
import sys
from itertools import chain
from collections import Counter, defaultdict, OrderedDict
from msp2.data.inst import yield_sents, yield_frames, Sent, Doc, Frame, set_ee_heads
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import Conf, zlog, zopen, zwarn, init_everything, default_json_serializer, OtherHelper, zglob, mkdir_p

def main(input_file: str, output_file: str, remove_all=0):
    cc = Counter()
    remove_all = int(remove_all)
    # --
    reader = ReaderGetterConf().get_reader(input_path=input_file)
    docs = []
    for inst in reader:
        cc['doc'] += 1
        docs.append(inst)
        for sent in inst.sents:
            cc['sent'] += 1
            for ef in sent.entity_fillers:
                cc['ef'] += 1
            for evt in sent.events:
                cc['evt'] += 1
                for arg in list(evt.args):
                    cc['arg'] += 1
                    arg.delete_self()
            if remove_all:
                sent.clear_entity_fillers()
                sent.clear_events()
    # --
    if output_file:
        with WriterGetterConf().get_writer(output_path=output_file) as writer:
            writer.write_insts(docs)
    zlog(f"Read {len(docs)} from {input_file} to {output_file}: {cc}")
    # --

# --
# python3 -m msp2.tasks.zmtl3.scripts.genia.rm_args IN OUT
if __name__ == '__main__':
    main(*sys.argv[1:])
