#

# change labels for frames & args

from collections import Counter
from msp2.data.inst import yield_sents
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import Conf, zlog

class MainConf(Conf):
    def __init__(self):
        self.R = ReaderGetterConf()
        self.W = WriterGetterConf()
        # --
        self.f_ef = "lambda x: x"
        self.f_evt = "lambda x: x"
        self.f_arg = "lambda x: x"
        # --

def do_change(conf: MainConf, **kwargs):
    # --
    cc_ef, cc_evt, cc_arg = Counter(), Counter(), Counter()
    f_ef, f_evt, f_arg = [eval(z) for z in [conf.f_ef, conf.f_evt, conf.f_arg]]
    # --
    def _change(_item, _f, _cc):
        l1 = _item.label
        l2 = _f(l1)
        if l1 != l2:
            _item.set_label(l2)
            _cc[f"{l1}=>{l2}"] += 1
        else:
            _cc["keep"] += 1
    # --
    reader, writer = conf.R.get_reader(), conf.W.get_writer()
    for inst in reader:
        for sent in yield_sents(inst):
            for ef in sent.entity_fillers:
                _change(ef, f_ef, cc_ef)
            for evt in sent.events:
                _change(evt, f_evt, cc_evt)
                for arg in evt.args:
                    _change(arg, f_arg, cc_arg)
        writer.write_inst(inst)
    writer.close()
    zlog(f"Change finish: \n{cc_ef}\n{cc_evt}\n{cc_arg}")
    # --

# --
def main(*args):
    conf = MainConf()
    conf.update_from_args(args)
    do_change(conf)
    # --

# PYTHONPATH=../src/ python3 change_label.py input_path:?? output_path:??
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
