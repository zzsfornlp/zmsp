#

# basic ones for the annotators

__all__ = [
    "AnnotatorConf", "Annotator", "AnnotatorCompositeConf", "AnnotatorComposite",
]

from typing import List, Type
from msp2.data.inst import DataInstance, Doc, Sent
from msp2.utils import Conf, Registrable

# -----
class AnnotatorConf(Conf):
    def __init__(self):
        pass

class Annotator(Registrable):
    def __init__(self, conf: AnnotatorConf): self.conf = conf
    # note: inplace modification!
    def annotate(self, insts: List[DataInstance]): raise NotImplementedError()

# -----
class AnnotatorCompositeConf(AnnotatorConf):
    def __init__(self):
        super().__init__()

class AnnotatorComposite(Annotator):
    def __init__(self, conf: AnnotatorCompositeConf,  annotators: List[Annotator] = None):
        super().__init__(conf)
        self.annotators = annotators

    def annotate(self, insts: List[DataInstance]):
        for ann in self.annotators:  # simply annotate in sequence
            ann.annotate(insts)
