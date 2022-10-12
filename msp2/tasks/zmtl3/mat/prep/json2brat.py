#

# convert back to brat for visualization

import os
import sys
from collections import OrderedDict, Counter
from msp2.utils import zopen, zlog, mkdir_p
from msp2.data.inst import yield_sents, yield_frames, Sent, Doc, CharIndexer
from msp2.data.rw import ReaderGetterConf, WriterGetterConf


# --
def mention2cspan(mention):
    positions = mention.sent.word_positions
    widx, wlen = mention.get_span()
    cstart, cend = positions[widx][0], sum(positions[widx+wlen-1])
    return cstart, cend

def main(input_file: str, output_prefix: str):
    cc = Counter()
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        mkdir_p(output_dir, raise_error=True)
    reader = ReaderGetterConf().get_reader(input_path=input_file)
    for inst in reader:
        cc['inst'] += 1
        doc_id = inst.id
        source_text = inst.get_text()
        with zopen(output_prefix+f"{doc_id}.txt", 'w') as fd:
            fd.write(source_text)
        # --
        mentions = OrderedDict()
        lines = []
        for sent in yield_sents(inst):
            cc['sent'] += 1
            for evt in sent.events:
                cc['evt'] += 1
                assert id(evt) not in mentions
                _tid = f"T{len(mentions)+1}"
                c0, c1 = mention2cspan(evt.mention)
                lines.append(f"T{len(mentions)+1}\t{evt.label} {c0} {c1}\t{source_text[c0:c1]}")
                mentions[id(evt)] = (_tid, evt, )
        reln = 1
        for one in mentions.values():
            for arg in one[1].args:
                a1, a2 = one[0], mentions[id(arg.arg)][0]
                cc['rel'] += 1
                lines.append(f"R{reln}\t{arg.label} Arg1:{a1} Arg2:{a2}")
                reln += 1
        with zopen(output_prefix+f"{doc_id}.ann", 'w') as fd:
            for line in lines:
                fd.write(line+"\n")
    # --
    zlog(f"Read from {input_file} to {output_prefix}: {cc}")
    # --

# python3 -m msp2.tasks.zmtl3.mat.prep.json2brat ...
if __name__ == '__main__':
    main(*sys.argv[1:])
