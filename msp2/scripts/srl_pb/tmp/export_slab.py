#

# export the seq-labs for seq model

from msp2.data.inst import yield_sents
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

#
def main():
    insts = list(ReaderGetterConf().get_reader())  # read from stdin
    for sent in yield_sents(insts):
        sorted_evts = sorted(sent.events, key=lambda x: x.mention.get_span())
        for evt in sorted_evts:
            print(" ".join(evt.info["slab"]))

# PYTHONPATH=../src/ python3 export_slab.py <IN >OUT
if __name__ == '__main__':
    main()
