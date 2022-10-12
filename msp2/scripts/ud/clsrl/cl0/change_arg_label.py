#

# simple change between ARG*<->A*

from collections import Counter
from msp2.data.inst import yield_sents
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import Conf, zlog

class MainConf(Conf):
    def __init__(self):
        self.R = ReaderGetterConf()
        self.W = WriterGetterConf()
        self.direction = "short"  # short:ARG->A, long:A->ARG

# --
def long2short(s: str):
    return s.replace("ARG", "A", 1)

def short2long(s: str):
    return s.replace("A", "ARG", 1)
# --

def main(*args):
    conf = MainConf()
    conf.update_from_args(args)
    # --
    ff = {"short": long2short, "long": short2long}[conf.direction]
    cc = Counter()
    # --
    reader, writer = conf.R.get_reader(), conf.W.get_writer()
    for inst in reader:
        for sent in yield_sents(inst):
            for evt in sent.events:
                for arg in evt.args:
                    l1 = arg.label
                    l2 = ff(l1)
                    arg.set_label(l2)
                    cc[f"{l1}=>{l2}"] += 1
        writer.write_inst(inst)
    writer.close()
    zlog(f"Change finish: {cc}")
    # --

# PYTHONPATH=../src/ python3 change_arg_label.py input_path:?? output_path:??
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
