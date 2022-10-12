#

# prepare data from ace (see "msp2/scripts/event")

import sys
from collections import Counter
from msp2.data.inst import yield_sents
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import zlog, zwarn, Conf, init_everything
from msp2.tasks.zmtl3.mod.extract.evt_arg import onto as zonto

# --
class MainConf(Conf):
    def __init__(self):
        self.onto = 'ace'
        self.input_file = ""
        self.output_file = ""
        # --

# --
def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    onto = zonto.Onto.load_onto(conf.onto)
    valid_args = set([r.name for r in onto.roles])
    cc = Counter()
    dargs = Counter()
    reader = ReaderGetterConf().get_reader(input_path=conf.input_file)
    with WriterGetterConf().get_writer(output_path=conf.output_file) as writer:
        for inst in reader:
            cc['inst'] += 1
            for sent in yield_sents(inst):
                cc['sent'] += 1
                for ef in sent.entity_fillers:
                    cc['ef'] += 1
                    hwidx, hwlen = ef.mention.get_span(hspan=True)  # using hwlen!
                    if hwidx is not None:
                        ef.mention.set_span(hwidx, hwlen)
                for evt in sent.events:
                    cc['evt'] += 1
                    evt.set_label(':'.join(evt.label.split(".")))  # '.' to ':'
                    ff = onto.find_frame(evt.label)
                    assert ff is not None
                    for arg in list(evt.args):
                        cc['arg'] += 1
                        if arg.label not in valid_args:
                            dargs[arg.label.split("-")[0]] += 1
                            cc['argD'] += 1
                            arg.delete_self()
                        else:
                            cc['argV'] += 1
            writer.write_inst(inst)
    # --
    zlog(f"Read from {conf.input_file} to {conf.output_file}: {cc}")
    zlog(f"Ignored args: {dargs}")
    # --

# python3 -m msp2.tasks.zmtl3.scripts.data.prep_ace_arzh input_file:?? output_file:??
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
