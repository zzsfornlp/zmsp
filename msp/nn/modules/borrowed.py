#

# borrowed from BK specific module

from msp.nn import BK
from msp.nn.layers import BasicNode

#
class BorrowedNode(BasicNode):
    def __init__(self, pc: BK.ParamCollection, mod: BK.Module, output_dims=None):
        super().__init__(pc, f"{self.__class__.__name__}:{mod.__class__.__name__}", None)
        # -----
        self.mod = mod
        BK.to_device(self.mod)  # move to target device
        self.output_dims = output_dims
        # collect parameters
        prefix_name = self.pc.nnc_name(self.name, True) + "/"
        named_params = self.pc.param_add_external(prefix_name, mod)
        # add to self.params
        for one_name, one_param in named_params:
            assert one_name not in self.params
            self.params[one_name] = one_param

    def refresh(self, rop=None):
        # todo(note): not following convention here, simply on the model
        super().refresh(rop)
        if self.rop.training:
            self.mod.train()
        else:
            self.mod.eval()

    def get_output_dims(self, *input_dims):
        return self.output_dims

    def __call__(self, *input, **kwargs):
        return self.mod(*input, **kwargs)  # everything to the Module
