# --

from collections import Counter
from mspx.nn import BK
from mspx.tools.annotate import *
from mspx.utils import Conf, zlog, init_everything, ConfEntryCallback, ZHelper
from mspx.data.rw import ReaderGetterConf, WriterGetterConf

class MainConf(Conf):
    def __init__(self):
        self.R = ReaderGetterConf()
        self.W = WriterGetterConf()
        # --
        self.anns = ConfEntryCallback(lambda s: self.callback_entries(s, T=AnnotatorConf))
        self.ann_batch_size = 1  # read how many instances and then fire?
        self.report_batch_interval = 1000  # how many batches to report
        # --

def ana_main(*args):
    conf = MainConf()
    conf = init_everything(conf, args)
    # --
    annotators = []
    for z in conf.anns:
        ca = getattr(conf, z[0])
        ann = ca.make_node()
        annotators.append(ann)
    zlog(f"Ready to annotate with {annotators}: {conf.R} {conf.W}")
    # --
    reader = conf.R.get_reader()
    with conf.W.get_writer() as writer:
        with BK.no_grad_env():  # note: decoding mode!!
            cc = Counter()
            ccs = [Counter() for ann in annotators]
            for _batch in ZHelper.yield_batches(reader, conf.ann_batch_size):
                for ii, ann in enumerate(annotators):
                    res = ann.annotate(_batch)
                    if res is not None:
                        ccs[ii].update(res)
                writer.write_insts(_batch)
                cc['inst'] += len(_batch)
                if cc['inst'] % conf.report_batch_interval == 0:
                    zlog(f"Annotate progress: {cc} {ccs}", timed=True)
    zlog(f"Annotate Finish: {cc} {ccs}", timed=True)
    # --

# PYTHONPATH=../src/ python3 -mpdb -m mspx.cli.annotate WHAT ...
if __name__ == '__main__':
    import sys
    ana_main(*sys.argv[1:])
