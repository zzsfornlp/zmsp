#

# the main module

import torch
import numpy as np
from mspx.utils import Conf, Configurable, Registrable, zwarn
from .mod_base import BaseModConf, load_model_toker

class ModelConf(Conf):
    def __init__(self):
        self.base_model = BaseModConf()

    def make_node(self, *args, **kwargs):
        return MyModel(self, *args, **kwargs)

class MyModel(Configurable, torch.nn.Module):
    def __init__(self, conf: ModelConf, **kwargs):
        super().__init__(conf, **kwargs)
        torch.nn.Module.__init__(self)  # note: extraly init Module!
        conf: ModelConf = self.conf
        # --
        # get base model
        self.model_helper, self.base_model, self.toker = load_model_toker(conf.base_model)

    def my_first_device(self):
        return next(self.base_model.parameters()).device

    def my_first_param(self):
        return next(self.base_model.parameters())

    def __repr__(self):
        return f"MyModel({type(self.base_model)})"

    def get_scheduled_values(self):
        return {}

    @property
    def is_clm(self):
        return self.model_helper.model_type == 'clm'
    @property
    def is_s2s(self):
        return self.model_helper.model_type == 's2s'
    @property
    def is_mlm(self):
        return self.model_helper.model_type == 'mlm'

    def tosave_state_dict(self):
        ret = {}
        # for k, v in self.state_dict(keep_vars=True).items():
        for k, v in self.named_parameters():
            if v.requires_grad:  # note: simply by this flag!!
                ret[k] = v.detach().cpu()
        return ret

    def from_state_dict(self, state_dict, strict=None):
        if strict is not None:
            self.load_state_dict(state_dict, strict=strict)
        else:  # otherwise, first try strict, then relax if there are errors
            try:
                self.load_state_dict(state_dict, strict=True)
            except:
                import traceback
                zwarn(f"#== Error in strict loading:\n{traceback.format_exc()}\n#==")
                self.load_state_dict(state_dict, strict=False)
        # --

    def forward(self, task, **kwargs):
        return task.model_forward(**kwargs)  # note: simply do task-specific ones!

    # --
    # helpers

    @staticmethod
    def batch_2d(inputs, pad_val=None, max_len=None, max_len_mul=None,
                 extras=([], []), left_truncate=False, left_pad=False, dtype=None, warn_truncate=False):
        bs = len(inputs)
        # --
        # decide max length
        _extras_left, _extras_right = extras
        _extras_budget = len(_extras_left) + len(_extras_right)  # extra budgets!
        _data_max_len = max((len(z)+_extras_budget for z in inputs), default=1)
        if max_len is None:  # no truncating!
            max_len = _data_max_len
        else:
            max_len = min(_data_max_len, max_len)
        if max_len_mul:  # make it multiply of MUL
            max_len = max_len_mul * ((max_len + max_len_mul - 1) // max_len_mul)
        # --
        # pad & dtype
        if pad_val is None:  # default padding is 0!
            pad_val = 0
            if len(inputs) > 0 and len(inputs[0]) > 0:
                pad_val = 0. if isinstance(inputs[0][0], float) else 0
        if dtype is None:  # guess dtype
            if isinstance(pad_val, int):
                dtype = np.int64
            elif isinstance(pad_val, float):
                dtype = np.float32
        # --
        arr = np.full([bs, max_len], pad_val, dtype=dtype)
        arr_m = np.zeros([bs, max_len], dtype=np.float32)
        arr_off = np.zeros([bs], dtype=np.int64)  # offsets for re-indexing: new_idx = orig_idx + offset
        _real_budget = max_len - _extras_budget  # real budget for the contents
        for ii, vv in enumerate(inputs):
            _offset = 0
            # truncate
            if len(vv) > _real_budget:
                if warn_truncate:
                    zwarn(f"Truncating sequence {len(vv)} to {_real_budget}")
                if left_truncate:
                    _start = len(vv) - _real_budget
                    _offset -= _start  # todo(note): truncated ones will be lost!
                    vv = vv[_start:]
                else:
                    vv = vv[:_real_budget]
            # add extras
            vv = _extras_left + vv + _extras_right
            _offset += len(_extras_left)
            # padding
            _pad_size = max_len - len(vv)
            assert _pad_size >= 0
            if left_pad:
                arr[ii, _pad_size:] = vv
                arr_m[ii, _pad_size:] = 1.
                _offset += _pad_size
            else:
                arr[ii, :len(vv)] = vv
                arr_m[ii, :len(vv)] = 1.
            arr_off[ii] = _offset
        # --
        return torch.from_numpy(arr), torch.from_numpy(arr_m), torch.from_numpy(arr_off)
        # --
