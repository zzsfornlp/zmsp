#

from typing import List, Dict

# A model tells all the story
# todo(note): just an interface!
class ZModel:
    # list(mini-batch) of to-decode instances,
    # results are written in-place, return info.
    def predict_on_batch(self, insts: List, **kwargs):
        raise NotImplementedError()

    # list(mini-batch) of annotated instances
    # optional results are written in-place? return info.
    def loss_on_batch(self, annotated_insts: List, loss_factor=1., **kwargs):
        raise NotImplementedError()

    # called before each mini-batch
    def refresh_batch(self, training: bool):
        raise NotImplementedError()

    # called for one step of paramter-update
    def update(self, lrate: float, grad_factor: float):
        raise NotImplementedError()

    # instance preparer(inst -> inst) None means no specific preparer
    def get_inst_preper(self, training: bool, **kwargs):
        return None

    # get all values need to be schedules, like lrate
    def get_scheduled_values(self) -> Dict:
        raise NotImplementedError()

    # load model conf & params
    def load(self, path: str, **kwargs):
        raise NotImplementedError()

    # save model conf & params
    def save(self, path: str, **kwargs):
        raise NotImplementedError()
