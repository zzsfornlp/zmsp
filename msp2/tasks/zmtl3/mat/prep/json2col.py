#

# json to col

import os
import sys
from collections import OrderedDict, Counter
from msp2.utils import zopen, zlog, zwarn
from msp2.data.inst import yield_sents, yield_frames, Sent, Doc, CharIndexer
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.data.vocab import SeqSchemeHelperStr

# --
def main(input_file: str, output_file: str, scheme='BIO', sep='\t'):
    cc = Counter()
    _SEP = sep
    helper = SeqSchemeHelperStr(scheme)
    reader = ReaderGetterConf().get_reader(input_path=input_file)
    output_lines = []
    for inst in reader:
        cc['inst'] += 1
        for sent in yield_sents(inst):
            cc['sent'] += 1
            cc['word'] += len(sent)
            cc['item'] += len(sent.events)
            # --
            spans = [evt.mention.get_span() + (evt.label, ) for evt in sent.events]
            tags0 = helper.spans2tags(spans, len(sent))
            tags = tags0[0][0]  # only first layer!
            if len(tags0) > 1:
                zwarn(f"Get more layers: {tags0[1:]}")
            for ii in range(len(sent)):
                output_lines.append(f"{sent.seq_word.vals[ii]}{_SEP}{tags[ii]}\n")
            output_lines.append("\n")
    # --
    zlog(f"Read from {input_file} to {output_file}: {cc}")
    if output_file:
        with zopen(output_file, 'w') as fd:
            fd.write(''.join(output_lines))
    # --

# --
# python3 -m msp2.tasks.zmtl3.mat.prep.json2col ...
if __name__ == '__main__':
    main(*sys.argv[1:])
