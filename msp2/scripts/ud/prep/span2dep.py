#

# from span-srl to dep-srl with dep tree

from collections import Counter
from msp2.data.inst import yield_sents, HeadFinder
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import zlog

# --
def main(input_file: str, output_file: str):
    cc = Counter()
    all_insts = list(ReaderGetterConf().get_reader(input_path=input_file))
    hf = HeadFinder("NOUN")
    for sent in yield_sents(all_insts):
        cc["sent"] += 1
        for evt in sent.events:
            cc["frame"] += 1
            for arg in list(evt.args):
                cc["arg"] += 1
                m = arg.arg.mention
                widx, wlen = m.get_span()
                hidx = hf.find_shead(m.sent, widx, wlen)
                m.set_span(hidx, 1)
    # --
    with WriterGetterConf().get_writer(output_path=output_file) as writer:
        writer.write_insts(all_insts)
    # --
    zlog(f"Convert from {input_file} to {output_file}: {cc}")
    # --

# --
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# --
# PYTHONPATH=?? python3 span2dep.py IN OUT

