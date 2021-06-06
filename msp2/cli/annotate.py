#

# annotator

import sys
from typing import List
from msp2.utils import Conf, zlog, init_everything
# from msp2.data.stream import BatchArranger
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.tools.annotate import Annotator
from msp2.nn import BK

class MainConf(Conf):
    def __init__(self):
        self.R = ReaderGetterConf()
        self.W = WriterGetterConf()
        # -----
        # ann: to be added
        self.ann_batch_size = 1  # read how many instances and then fire?
        self.report_batch_interval = 1000  # how many batches to report

def main(annotators: str, *args):
    # find annotators
    conf = MainConf()
    all_confs, all_ann_types = [], []
    for ann_ss in annotators.split(","):
        if len(ann_ss) == 0:
            continue  # ignore empty
        t_res, ann_ss_real = Annotator.try_load_and_lookup(ann_ss, ret_name=True)
        one_conf_type, one_ann_type = t_res.conf, t_res.T
        one_conf = one_conf_type()
        all_confs.append(one_conf)
        all_ann_types.append(one_ann_type)
        assert not hasattr(conf, ann_ss_real)
        setattr(conf, ann_ss_real, one_conf)  # add conf
    # --
    conf = init_everything(conf, args)
    zlog(f"Ready to annotate with {annotators}: {conf.R} {conf.W}")
    # init all annotators
    all_anns: List[Annotator] = [at(cc) for cc, at in zip(all_confs, all_ann_types)]
    # --
    reader, writer = conf.R.get_reader(), conf.W.get_writer()
    # =====
    def _process(_batch: List):
        for ann in all_anns:
            ann.annotate(_batch)
        writer.write_insts(_batch)
    # =====
    with BK.no_grad_env():  # note: decoding mode!!
        c, c2 = 0, 0
        cur_insts = []
        for one in reader:
            # input one
            c += 1
            cur_insts.append(one)
            # process?
            if len(cur_insts) >= conf.ann_batch_size:
                _process(cur_insts)
                cur_insts.clear()
                c2 += 1
                if c2 % conf.report_batch_interval == 0:
                    zlog(f"Annotate roughly: inst={c},batch={c2}")
        # remaining ones
        if len(cur_insts) > 0:
            _process(cur_insts)
    zlog(f"Annotate Finish: processed {c} insts.")

# PYTHONPATH=../src/ python3 -m msp2.cli.annotate "[annotators,...]" ...
if __name__ == '__main__':
    main(*sys.argv[1:])
