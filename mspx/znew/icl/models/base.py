#

# base class

__all__ = [
    "NewBaseModelConf", "NewBaseModel"
]

from mspx.utils import Conf, Registrable, Configurable
from mspx.nn import NnConf, NnLayer

class NewBaseModelConf(NnConf):
    pass

class NewBaseModel(NnLayer):
    pass
