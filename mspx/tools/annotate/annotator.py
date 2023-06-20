#

# basic ones for the annotators

__all__ = [
    "AnnotatorConf", "Annotator",
]

from typing import List, Type
from mspx.data.inst import Doc
from mspx.utils import Conf, Registrable, Configurable


@Registrable.rd('ANN')
class AnnotatorConf(Conf):
    @classmethod
    def get_base_conf_type(cls): return AnnotatorConf
    @classmethod
    def get_base_node_type(cls): return Annotator

@Registrable.rd('_ANN')
class Annotator(Configurable):
    def __init__(self, conf: AnnotatorConf, **kwargs):
        super().__init__(conf, **kwargs)

    def get_batcher(self):
        return None

    # note: inplace modification!
    def annotate(self, insts: List[Doc]):
        raise NotImplementedError()
