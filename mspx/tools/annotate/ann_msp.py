#

# annotate with an MSP model

__all__ = [
    "AnnotatorMspConf", "AnnotatorMsp",
]

from typing import List, Union
from collections import Counter
from mspx.data.inst import Doc, Sent
from mspx.utils import zwarn, zglob1, ZObject
from mspx.proc.run import ZTaskCenter, InputBatch, ZDatasetConf, ZDataset
from .annotator import *

@AnnotatorConf.rd('msp')
class AnnotatorMspConf(AnnotatorConf):
    def __init__(self):
        super().__init__()
        self.msp_ds = ZDatasetConf.direct_conf(do_cache_insts=False, _rm_names=["R", "W"])
        self.tc_model_path = ""
        self.tc_kwargs = {}

@AnnotatorMspConf.conf_rd()
class AnnotatorMsp(Annotator):
    def __init__(self, conf: AnnotatorMspConf):
        super().__init__(conf)
        conf: AnnotatorMspConf = self.conf
        # --
        _path = zglob1(conf.tc_model_path)
        self.tc = ZTaskCenter.load(_path, quite=False, tc_kwargs=conf.tc_kwargs)
        self.tc.model.eval()
        # fake a dataset
        self.fake_ds = ZDataset(conf.msp_ds, name='fake')
        if len(self.fake_ds.tasks) == 0:  # by default add all tasks
            self.fake_ds.tasks.update({z:"" for z in self.tc.tasks})
        # --

    def annotate(self, insts: List[Doc]):
        cc = Counter()
        for ibatch in self.fake_ds.yield_batches(external_stream=insts):
            one_res = self.tc.model(ibatch, do_pred=True)
            cc.update(one_res)
        return cc

# --
# b mspx/tools/annotate/ann_msp:42
