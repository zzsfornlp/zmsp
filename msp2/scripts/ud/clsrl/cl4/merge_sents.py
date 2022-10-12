#

# merge nearby sentences into one!

from collections import Counter
from msp2.data.inst import yield_sents
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import Conf, zlog

class MainConf(Conf):
    def __init__(self):
        self.R = ReaderGetterConf()
        self.W = WriterGetterConf()

def main(*args):
    conf = MainConf()
    conf.update_from_args(args)
    # --
    reader = conf.R.get_reader()
    cc = Counter()
    last_sent = None
    outputs = []
    for inst in reader:
        for sent in yield_sents(inst):
            cc["orig_sent"] += 1
            cc["orig_evt"] += len(sent.events)
            cc["orig_arg"] += sum(len(z.args) for z in sent.events)
            if last_sent is not None and sent.seq_word.vals == last_sent.seq_word.vals:
                # merge evts and args
                for evt in sent.events:
                    widx, wlen = evt.mention.get_span()
                    new_evt = last_sent.make_event(widx, wlen, type=evt.type)
                    for arg in evt.args:
                        assert arg.arg.sent is evt.sent
                        a_widx, a_wlen = arg.mention.get_span()
                        new_ef = last_sent.make_entity_filler(a_widx, a_wlen, type=arg.arg.type)
                        new_evt.add_arg(new_ef, role=arg.role)
            else:
                outputs.append(sent)
                last_sent = sent
    # --
    with conf.W.get_writer() as writer:
        for sent in outputs:
            cc["new_sent"] += 1
            cc["new_evt"] += len(sent.events)
            cc["new_arg"] += sum(len(z.args) for z in sent.events)
            writer.write_inst(sent)
    # --
    zlog(f"Stat: {cc}")

# PYTHONPATH=../src/ python3 merge_sents.py input_path:?? output_path:??
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
