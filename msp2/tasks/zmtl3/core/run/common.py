#

# some common helpers

from typing import List, Union
import numpy as np
from msp2.data.inst import Sent, Frame

class DataItem:
    def __init__(self, inst: Union[Sent, Frame]):
        if isinstance(inst, Sent):
            self.sent = inst
            self.frame = None
        elif isinstance(inst, Frame):
            self.sent = inst.sent
            self.frame = inst
        else:
            raise NotImplementedError()
        # --
        self.info = {}

    def __len__(self):  # used for various purpose?
        return len(self.sent)  # sentence length

    def __repr__(self):
        return f"Frame({self.frame})" if self.frame is not None else f"Sent({self.sent})"

class InputBatch:
    def __init__(self, items: List, dataset):
        self.items = [z if isinstance(z, DataItem) else DataItem(z) for z in items]
        self.dataset = dataset  # mainly for extra information!
        self.info = {}
        # --

    def __len__(self):
        return len(self.items)
