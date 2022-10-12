#

# using pytorch

from typing import Union, SupportsFloat, List
from collections import OrderedDict
from msp2.utils import zwarn, zlog
import traceback
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import math

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .common import NIConf, OptimConf

# =====
# Part 0: basic ones!

# some vars
Expr = torch.Tensor
Module = torch.nn.Module
CPU_DEVICE = torch.device("cpu")
DEFAULT_DEVICE = CPU_DEVICE
Function = torch.autograd.Function

# types
float32 = torch.float32
float64 = torch.float64
int32 = torch.int32
int64 = torch.int64
long = torch.long
uint8 = torch.uint8

# Basic Conf
class TrNIConf(NIConf):
    def __init__(self):
        super().__init__()
        # --
        # other nn-lib specific settings
        self.dist_backend = "nccl"
        self.dist_rank = 0
        self.dist_world_size = 1  # really activate if >1
        self.dist_find_unused_parameters = False
        # --
        # apex.amp
        self.amp_opt_level = ''


BKNIConf = TrNIConf

# singleton
_global_tr_conf = TrNIConf()
def get_global_conf():
    return _global_tr_conf
def set_gloabl_conf(conf: TrNIConf):
    global _global_tr_conf
    _global_tr_conf = conf

def init(conf: TrNIConf = None):
    if conf is None:
        conf = get_global_conf()
    else:
        set_gloabl_conf(conf)
    # --
    torch.set_num_threads(conf.num_threads)
    torch.manual_seed(conf.random_seed)
    torch.cuda.manual_seed(conf.random_cuda_seed)
    if conf.device >= 0:
        global DEFAULT_DEVICE
        DEFAULT_DEVICE = torch.device(f"cuda:{conf.device}")
    if conf.dist_world_size > 1:
        import os
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12365'
        # initialize the process group
        dist.init_process_group(conf.dist_backend, rank=conf.dist_rank, world_size=conf.dist_world_size)
    # --

# --
# ddp related
def use_ddp():
    return get_global_conf().dist_world_size > 1

def ddp_world_size():
    return get_global_conf().dist_world_size

def ddp_rank():
    return get_global_conf().dist_rank

def is_main_process():
    return get_global_conf().dist_rank <= 0  # rank0 is the main one!!

# --
# from fairseq
class ModuleProxyWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        assert hasattr(module, "module"), \
            "ModuleProxyWrapper expects input to wrap another module"
        self.module = module

    def __getattr__(self, name):
        """Forward missing attributes to twice-wrapped module."""
        try:
            # defer to nn.Module's logic
            return super().__getattr__(name)
        except AttributeError:
            try:
                # forward to the once-wrapped module
                return getattr(self.module, name)
            except AttributeError:
                # forward to the twice-wrapped module
                return getattr(self.module.module, name)

    def state_dict(self, *args, **kwargs):
        """Forward to the twice-wrapped module."""
        return self.module.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """Forward to the twice-wrapped module."""
        return self.module.module.load_state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
# --

def wrap_ddp_model(model):
    conf = get_global_conf()
    if conf.amp_opt_level:
        from apex.parallel import DistributedDataParallel as DDP
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP0
        DDP = lambda model: DDP0(model, device_ids=[conf.device], find_unused_parameters=conf.dist_find_unused_parameters)
    # --
    m1 = DDP(model)
    m2 = ModuleProxyWrapper(m1)
    return m2
# --

def refresh():
    # not doing this, since it makes things slower
    # torch.cuda.empty_cache()
    pass

# todo(note): default init
init()

# =====
# Part 1: still basic ones!

# also useful for ndarray
def get_shape(t: Union[Expr, np.ndarray], dim=None):
    shapes = list(t.shape)
    return shapes if (dim is None) else shapes[dim]

def is_zero_shape(t: Union[Expr, np.ndarray], dim=None):
    shapes = list(t.shape)
    return any(s==0 for s in shapes) if (dim is None) else shapes[dim]==0

# is_tensor
def is_expr(v):
    return isinstance(v, Expr)
is_tensor = is_expr

# todo(note): similar to Torch.Optimizer, but with extra settings
class Optim:
    def __init__(self, conf: OptimConf, lrf_sv: SupportsFloat):
        self.conf = conf
        self.lrf_sv = lrf_sv
        self._optim = None  # Optimizer
        self._cached_lrate = 0.

    @staticmethod
    def _get_optim(conf: OptimConf, params):
        optim_type = conf.optim_type
        if optim_type == "sgd":
            _opt = optim.SGD(params, lr=0., momentum=conf.sgd_momentum, weight_decay=conf.weight_decay)
        elif optim_type == "adagrad":
            _opt = optim.Adagrad(params, lr=0., weight_decay=conf.weight_decay)
        elif optim_type == "adam":
            _opt = optim.Adam(params, lr=0., betas=conf.adam_betas, eps=conf.adam_eps, weight_decay=conf.weight_decay)
        elif optim_type == "adamw":
            _opt = optim.AdamW(params, lr=0., betas=conf.adam_betas, eps=conf.adam_eps, weight_decay=conf.weight_decay)
        elif optim_type == "adamw2":
            from transformers.optimization import AdamW  # another one!
            _opt = AdamW(params, lr=0., betas=conf.adam_betas, eps=conf.adam_eps, weight_decay=conf.weight_decay)
        elif optim_type == "adadelta":
            _opt = optim.Adadelta(params, lr=0., rho=conf.adadelta_rho, eps=conf.adadelta_eps, weight_decay=conf.weight_decay)
        else:
            raise NotImplementedError(f"Unknown optim {optim_type}.")
        return _opt

    def add_params(self, params):
        params = list(params)
        if len(params) == 0:
            return  # no adding!
        if self._optim is None:
            self._optim = Optim._get_optim(self.conf, params)
        else:
            self._optim.add_param_group({'params': params})

    def __repr__(self):
        sizes = [len(param_group['params']) for param_group in self._optim] if self._optim is not None else []
        return f"{self.conf.optim_type}(sizes={sizes},lr={self._cached_lrate})"

    # return List[List[Parameter]]
    def get_params(self):
        if self._optim is None:
            return []
        return [param_group['params'] for param_group in self._optim.param_groups]

    def get_all_params(self):
        return [p for pg in self.get_params() for p in pg]

    def update(self, overall_lrate: float, grad_factor: float):
        conf = self.conf
        if self._optim is None:
            return  # no params to update!
        # change lrate
        cur_lrate = overall_lrate * float(self.lrf_sv)  # get real lrate
        if self._cached_lrate != cur_lrate:
            # schedule lrate, do as lr_scheduler does
            for param_group in self._optim.param_groups:
                param_group['lr'] = cur_lrate
            self._cached_lrate = cur_lrate
        # update
        # todo(+N): careful in distributed cases
        all_params = self.get_all_params()
        if grad_factor != 1.:
            # todo(note): useful for batch-split, div grad by splits
            for p in all_params:
                if p.grad is not None:
                    p.grad.data.mul_(grad_factor)
        if conf.grad_clip > 0. and len([p for p in all_params if p.grad is not None])>0:
            clip_grad_norm_(all_params, conf.grad_clip)
        # step
        self._optim.step()
        self._optim.zero_grad()

# for checking params
class MyParamManager:
    def __init__(self):
        self.all_params = {}  # id -> info

    def record_param(self, p, info):
        p_id = id(p)
        assert p_id not in self.all_params, "Currently cannot record repeated params"
        self.all_params[p_id] = (get_shape(p), info)

    def delete_param(self, p):
        p_id = id(p)
        has_id = p_id in self.all_params
        if has_id:
            self.all_params.remove(p_id)
        return has_id

    def check_params(self, ps):
        remaining_keys = set(self.all_params.keys())
        extra_ones = []
        to_check = list(ps)
        for p in to_check:
            p_id = id(p)
            if p_id in remaining_keys:
                remaining_keys.remove(p_id)
            else:
                extra_ones.append((get_shape(p), ))
        missing_ones = [(k, self.all_params[k]) for k in remaining_keys]
        if len(extra_ones)>0:
            zwarn("Check-params extra:" + '\n'.join([str(x) for x in extra_ones]))
        if len(missing_ones)>0:
            zwarn("Check-params missing:" + '\n'.join([str(x) for x in missing_ones]))

_default_param_manager = MyParamManager()
def get_current_param_manager(): return _default_param_manager

# save and load
def new_param(size, dtype=torch.float32, device=None, requires_grad=True):
    device = DEFAULT_DEVICE if device is None else device
    ret = nn.Parameter(torch.empty(size, dtype=dtype, device=device), requires_grad=requires_grad)
    # add to param-manager for later checking
    get_current_param_manager().record_param(ret, traceback.format_stack())  # record it
    return ret

def init_param(x: nn.Parameter, init: Union[str, np.ndarray], lookup: bool=False, scale: float=1.):
    conf = get_global_conf()
    assert not conf.use_my_init, "now use ones from pytorch for param init"
    # --
    shape = get_shape(x)
    if not isinstance(init, str):
        data = x.new_tensor(init)  # make it to tensor
        with no_grad_env():
            x.set_(data * scale)
        return x
    # -----
    if init == "zero":
        with no_grad_env():
            x.zero_()
            return x
    # -----
    if len(shape) == 1:  # always zero for 1d
        with no_grad_env():
            x.zero_()
            return x
    # -----
    with no_grad_env():
        if lookup:
            _iscale = np.sqrt(3.0 / shape[-1])
            nn.init.uniform_(x, -_iscale, _iscale)
            # todo(+N): again back to previous init method
            # nn.init.normal_(x)
            x *= scale
        elif init == "default" or init == "glorot":
            nn.init.xavier_uniform_(x)
            x *= scale
        elif init == "ortho":
            # todo(note): assume squared matrices
            assert len(shape)==2 and (shape[0]%shape[1]==0 or shape[1]%shape[0]==0), "Invalid shape for ortho init"
            s0, s1 = shape
            if s0>s1:
                for i in range(s0//s1):
                    nn.init.orthogonal_(x[i*s1:(i+1)*s1,:])
            else:
                for i in range(s1//s0):
                    nn.init.orthogonal_(x[:,i*s0:(i+1)*s0])
        else:
            assert False, "Unknown init method for BKTR: " + init
    return x

# --
# special helper
def get_inita_xavier_uniform(shape):
    # from "_calculate_fan_in_and_fan_out"
    d_out, d_in = shape[:2]
    receptive_field_size = 1
    if len(shape) > 2:
        receptive_field_size = np.prod(shape[2:]).item()
    fan_in, fan_out = d_in*receptive_field_size, d_out*receptive_field_size
    # from "xavier_uniform_"
    std = math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return a
# --

def change_state_dict(d, cut_mods, del_mods):
    # -- cut submodule ...
    for cut_mod in cut_mods:
        if cut_mod[-1] != '.':
            cut_mod = cut_mod + "."
        d2 = OrderedDict()
        for k, v in d.items():
            if k.startswith(cut_mod):
                d2[k[len(cut_mod):]] = v
        d = d2
    # -- del submodule ...
    for del_mod in del_mods:
        d2 = OrderedDict()
        for k, v in d.items():
            if not k.startswith(del_mod):
                d2[k] = v
        d = d2
    # --
    return d

def save_model(model: Module, path: str, cut_mods=(), del_mods=(), **kwargs):
    d = model.state_dict()
    d = change_state_dict(d, cut_mods, del_mods)
    torch.save(d, path)

def load_model(model: Module, path: str, strict=None, map_location=None, cut_mods=(), del_mods=(), **kwargs):
    if map_location is None:
        map_location = DEFAULT_DEVICE
    d = torch.load(path, map_location=map_location)
    d = change_state_dict(d, cut_mods, del_mods)
    # load it
    if strict is not None:
        model.load_state_dict(d, strict=strict)
    else:  # otherwise, first try strict, then relax if there are errors
        try:
            model.load_state_dict(d, strict=True)
        except:
            import traceback
            zwarn(f"#== Error in strict loading:\n{traceback.format_exc()}\n#==")
            model.load_state_dict(d, strict=False)
    # --

# =====
# Part 2: functions

# -----
# new

# (inputs: python data type) -> FloatTensor
def input_real(inputs, device=None):
    if is_expr(inputs):
        return inputs
    return torch.tensor(inputs, dtype=torch.float32, device=(DEFAULT_DEVICE if device is None else device))

# (inputs: python data type of indexes) -> LongTensor
def input_idx(inputs, device=None):
    if is_expr(inputs):
        return inputs
    return torch.tensor(inputs, dtype=torch.long, device=(DEFAULT_DEVICE if device is None else device))

# inputs with dtype
def input_tensor(inputs, device=None, dtype=torch.float32):
    if is_expr(inputs):
        return inputs
    return torch.tensor(inputs, dtype=dtype, device=(DEFAULT_DEVICE if device is None else device))

# (shape: ..., value: float) -> FloatTensor
def constants(shape, value=0., dtype=torch.float32, device=None):
    return torch.full(shape, value, dtype=dtype, device=(DEFAULT_DEVICE if device is None else device))

# constants(..., dtype=long)
def constants_idx(shape, value=0, device=None):
    return torch.full(shape, value, dtype=torch.long, device=(DEFAULT_DEVICE if device is None else device))

# return 2D eye matrix
def eye(n, device=None):
    return torch.eye(n, dtype=torch.float32, device=(DEFAULT_DEVICE if device is None else device))

# arange
def arange_idx(*args, device=None):
    return torch.arange(*args, dtype=torch.long, device=(DEFAULT_DEVICE if device is None else device))

# (shape: ..., p: rate of 1., mul: multiply) -> Tensor
def random_bernoulli(shape, p: float, mul: float, device=None):
    x = torch.full(shape, p, dtype=torch.float32, device=(DEFAULT_DEVICE if device is None else device))
    r = torch.bernoulli(x) * mul
    return r

# [0,1)
def rand(shape, dtype=torch.float32, device=None):
    return torch.rand(shape, dtype=dtype, device=(DEFAULT_DEVICE if device is None else device))

def randint(*args, dtype=torch.long, device=None):
    return torch.randint(*args, dtype=dtype, device=(DEFAULT_DEVICE if device is None else device))

# -----
# operations

# (input_list: list of tensors [bias, weight1, input1, ...]) -> Tensor
def affine(input_list: List):
    base, base_idx = input_list[0], 1
    while base_idx < len(input_list):
        base = torch.addmm(base, input_list[base_idx+1], input_list[base_idx].t())
        base_idx += 2
    return base

# allow >2d input, but still 2d weights
def affine2(input_list: List):
    prefidx_dimensions = None
    base, base_idx = input_list[0], 1
    while base_idx < len(input_list):
        cur_input = input_list[base_idx+1]
        cur_dims = get_shape(cur_input)
        if len(cur_dims) > 2:
            prefidx_dimensions = cur_dims[:-1]
        base = torch.addmm(base, cur_input.view([-1, cur_dims[-1]]), input_list[base_idx].t())
        base_idx += 2
    # reshape back
    if prefidx_dimensions is None:
        return base
    else:
        last_dim = get_shape(base, -1)
        prefidx_dimensions.append(last_dim)
        return base.view(prefidx_dimensions)

# allow >2d input and unsqueezed shapes, simply use matmul
def affine3(input_list: List):
    bias, base_idx = input_list[0], 1
    base = 0
    while base_idx < len(input_list):
        base = base + torch.matmul(input_list[base_idx + 1], input_list[base_idx].t())
        base_idx += 2
    return base + bias

# (input_list: list of Tensors, dim: ...) -> Tensor
def concat(input_list: List, dim=-1):
    if len(input_list) == 1:
        return input_list[0]
    return torch.cat(input_list, dim)

# (t: Tensor, idxes: list(batch-size) of idx) -> Tensor
def gather_one_lastdim(t, idxes):
    idx_t = input_idx(idxes)  # [...]
    idx_t2 = torch.unsqueeze(idx_t, -1)  # [..., 1]
    dim = len(t.shape)-1
    return torch.gather(t, dim, idx_t2).squeeze(-1)  # [...]

# (weight: Tensor(Param), inputs: list of int) -> Tensor
def lookup(weight, inputs):
    idxes_t = input_idx(inputs)
    return F.embedding(idxes_t, weight)

# (t: Tensor, idxes: list of ints, dim:...) -> Tensor
def select(t, idxes, dim=0):
    idxes_t = input_idx(idxes)
    return torch.index_select(t, dim=dim, index=idxes_t)

# try to use pytorch's version, otherwise use our own
def simple_repeat_interleave(t, r: int, dim: int):
    try:
        # raise NotImplementedError()
        return torch.repeat_interleave(t, r, dim)
    except:
        dim_p1 = dim + 1
        old_shape = list(t.shape)
        repeats = [1]*dim_p1 + [r] + [1]*(len(old_shape) - dim_p1)
        new_shape = old_shape[:dim] + [-1] + old_shape[dim_p1:]
        t1 = t.unsqueeze(dim_p1).repeat(repeats).view(new_shape)  # new axis at next one
        return t1

# others ...
as_tensor = torch.as_tensor
abs = torch.abs
avg = torch.mean
bilinear = F.bilinear  # (N,*,in1_features), (N,*,in2_features), (out_features, in1_features, in2_features), (out_features,) -> (N,*,out_features)
binary_cross_entropy = F.binary_cross_entropy
binary_cross_entropy_with_logits = F.binary_cross_entropy_with_logits
cdist = torch.cdist
clamp = torch.clamp
chunk = torch.chunk
cumsum = torch.cumsum
diagflat = torch.diagflat
dot = torch.dot
dropout = F.dropout
elu = F.elu
exp = torch.exp
expand = lambda x, *args: x.expand(*args)
gelu = F.gelu
log = torch.log
logsigmoid = F.logsigmoid
log_softmax =F.log_softmax
logsumexp = torch.logsumexp
max = torch.max  # todo(note): with dim, return tuple
max_elem = torch.max  # todo(note): max_elem(a, b)
min = torch.min
min_elem = torch.min
masked_select = torch.masked_select
matmul = torch.matmul
norm = torch.norm
pad = F.pad
relu = F.relu
reshape = torch.reshape
rsqrt = torch.rsqrt
sigmoid = torch.sigmoid
sign = torch.sign
softmax = F.softmax
squeeze = torch.squeeze
split = torch.split
sqrt = torch.sqrt
stack = torch.stack
sum = torch.sum
tanh = torch.tanh
topk = torch.topk  # todo(note): return (val, idx)
transpose = torch.transpose
unsqueeze = torch.unsqueeze
where = torch.where
zeros = lambda shape: constants(shape, value=0.)

# -----
# loss related

# score_expr[gold_idx] -= margin (* should be arith-same as the adding one)
# todo(+4): currently return to_dense values since otherwise will get backward error!
def minus_margin(score_expr, gold_idxes_expr, margin: float):
    score_shape = get_shape(score_expr)  # [..., dim]
    # to 2d
    score_shape_2d = [int(np.prod(score_shape[:-1])), score_shape[-1]]
    size0_2d = score_shape_2d[0]
    # todo(warn): to_dense since otherwise will get backward error!
    sparse_minus = torch.sparse.FloatTensor(stack([arange_idx(size0_2d), gold_idxes_expr.view(-1)], dim=0),
                                            constants([size0_2d], -margin), score_shape_2d).to_dense()
    score_expr_2d = score_expr.view(score_shape_2d) + sparse_minus
    return score_expr_2d.view(score_shape)

# - log softmax(margin(score_expr))[idx]
# [*, C], [*] -> [*]
def loss_nll(score_expr, gold_idxes, margin=0., label_smoothing=0.):
    gold_idxes_t = input_idx(gold_idxes)
    if margin > 0.:
        score_expr = minus_margin(score_expr, gold_idxes_t, margin)
    # no average or sum-reduce for the output
    # output = F.nll_loss(F.log_softmax(score_expr, dim=-1), gold_idxes_t, size_average=False, reduce=False)
    log_softmax_score = F.log_softmax(score_expr, dim=-1)
    if label_smoothing > 0.:
        N = score_expr.size(-1) - 1.
        weight = score_expr.new_ones(score_expr.size()) * (label_smoothing / N)
        weight.scatter_(-1, gold_idxes.unsqueeze(-1), (1. - label_smoothing))
        ret = - (weight * log_softmax_score).sum(dim=-1)  # [*, C]
        # note: substract baseline
        ret += ((1-label_smoothing) * math.log(1-label_smoothing) + label_smoothing * math.log(label_smoothing/N + 1e-10))
    else:
        ret = - log_softmax_score.gather(-1, gold_idxes_t.unsqueeze(-1)).squeeze(-1)  # [*, C]
    return ret

# - (score[idx] - max(score'+margin))
def loss_hinge(score_expr, gold_idxes, margin=0.):
    if margin > 0.:
        score_expr = minus_margin(score_expr, gold_idxes, margin)
    score_max_ones, _ = torch.max(score_expr, -1)
    # 2d input for score
    score_gold_ones0 = gather_one_lastdim(score_expr, gold_idxes)
    score_gold_ones = score_gold_ones0.squeeze(-1)
    output = score_max_ones - score_gold_ones
    return output

# binary cross-entropy loss
# [*], [*](0 or 1) -> [*]
def loss_binary(score_expr, gold_idxes, margin=0., label_smoothing=0.):
    if margin > 0.:
        score_expr = score_expr + (1.-2*gold_idxes) * margin
    # --
    v0 = F.binary_cross_entropy_with_logits(score_expr, gold_idxes, reduction='none')
    if label_smoothing > 0.:  # todo(+N): can we directly put into targets?
        v1 = F.binary_cross_entropy_with_logits(score_expr, 1.-gold_idxes, reduction='none')
        ret = v1 * label_smoothing + v0 * (1.-label_smoothing)
        # note: substract baseline
        ret += ((1.-label_smoothing) * math.log(1.-label_smoothing) + label_smoothing * math.log(label_smoothing))
    else:
        ret = v0
    return ret  # [*]

# -----
# (maybe) non-trackable funtions, used for returning Python values

# sampling, return (idx_tensor, value_tensor)
def multinomial_select(prob, num=1):
    idxes = torch.multinomial(prob, num)
    values = torch.gather(prob, -1, idxes)
    return idxes, values

# sampling 1 with gumble
# argmax(logprob + -log(-log(Unif[0,1]))), return (val, idx)
# todo(note): keepdim for the output
def category_sample(logprob, dim=-1, keep_dim=True):
    G = torch.rand(logprob.shape, dtype=torch.float32, device=DEFAULT_DEVICE)
    X = logprob-(-G.log()).log()
    V, I = X.max(dim)
    if keep_dim:
        V, I = V.unsqueeze(-1), I.unsqueeze(-1)
    return V, I

def multinomial_choice(prob_t, num_samples: int, replacement: bool):
    input_shape = get_shape(prob_t)  # [*, L]
    if len(input_shape) > 2:
        prob_t = prob_t.view([np.prod(input_shape[:-1]), input_shape[-1]])
    ret0 = torch.multinomial(prob_t, num_samples, replacement)
    ret = ret0.view(input_shape[:-1] + [num_samples])
    return ret

def gather(t, idxes, dim=-1):
    idxes_t = input_idx(idxes)
    return torch.gather(t, dim, idxes_t)

# special gather for the first several dims
# t: [s1, s2, ..., sn-1, sn, ...]; idx: [s1, s2, ..., sn-1, k]
def gather_first_dims(t, idxes, dim):
    t_shape = get_shape(t)
    if dim < 0:
        dim = len(t_shape) + dim
    idxes_t = input_idx(idxes)
    idx_shape = get_shape(idxes_t)
    assert t_shape[:dim] == idx_shape[:-1]
    # flatten and index select
    t_shape0, t_shape1 = t_shape[:dim+1], t_shape[dim+1:]
    flatten_t = t.view([-1] + t_shape1)  # [s1*...*sn, ...]
    basis_t = arange_idx(np.prod(t_shape0[:-1])).view(t_shape0[:-1]) * t_shape0[-1]  # [s1, ..., sn-1]
    basis_t = (basis_t.unsqueeze(-1) + idxes_t).view(-1)  # [*]
    output_t0 = torch.index_select(flatten_t, dim=0, index=basis_t)  # [*, ...]
    return output_t0.view(idx_shape + t_shape1)

# =====
# values & backward

# return numpy values
# todo(warn): should never modify on this
def get_value(t):
    return t.detach().cpu().numpy()

def set_value(t, val, resize=False):
    with torch.autograd.no_grad():  # avoid recording grad_fn
        if not resize:
            assert get_shape(t)==get_shape(val)
        # if resize:  # if need to resize
        #     src_shape, trg_shape = get_shape(t), get_shape(val)
        #     if src_shape != trg_shape:
        #         t.resize_(trg_shape)
        t.set_(input_real(val).to(t.device))

def get_cpu_tensor(t):
    return t.detach().cpu()

def backward(loss, loss_factor=1., retain_graph=False):
    if loss_factor != 1.:
        loss = loss * loss_factor
    loss.backward(retain_graph=retain_graph)

# directly setting param
def zero_row(param, row):
    with torch.autograd.no_grad():
        # avoid recording grad_fn
        param[row].fill_(0.)

def no_grad_env():
    return torch.autograd.no_grad()

def has_nan(t):
    return int(torch.isnan(t).sum())

# =====

# todo(0): special one, not API
# from torch.nn.backends.thnn import backend as thnn_backend
try:
    # torch 1.0
    _VF = torch._C._VariableFunctions
    gru_oper = _VF.gru_cell
    lstm_oper = _VF.lstm_cell
except:
    # torch 0.4.1
    from torch.nn.backends.thnn import backend as thnn_backend
    gru_oper = thnn_backend.GRUCell
    lstm_oper = thnn_backend.LSTMCell

# specific one for batched inv
if int(torch.__version__.split(".")[0])>=1:
    # only available torch>=1.0.0
    get_inverse = lambda M, DiagM: M.inverse()
else:
    # otherwise use gesv, which is deprecated in some later version
    get_inverse = lambda M, DiagM: DiagM.gesv(M)[0]

# special routines
# todo(note): for mask->idx: 1) topk+sort, 2) padding with extra 1s; currently using 2) since nonzero is already sorted
# the inputs should be 1. or 0. (float); [*, L] -> [*, max-count]
def mask2idx(mask_t, padding_idx=0):
    mask_shape = get_shape(mask_t)  # [*, L]
    # --
    # judge zero-shape
    if any(z==0 for z in mask_shape):
        zz = zeros(mask_shape[:-1]+[1])  # [*, 1], put an one here!
        return zz.long(), zz.float()
    # --
    counts = mask_t.sum(-1).long()  # [*]
    max_count_t = counts.max(-1, keepdim=True)[0]
    max_count = max_count_t.item()  # int, the max expanding
    padding_counts = max_count_t - counts  # [*]
    max_padding_count = padding_counts.max().item()  # int, the max count of padding
    pad_t = (arange_idx(max_padding_count) < padding_counts.unsqueeze(-1)).float()  # [*, max_pad]
    concat_t = concat([mask_t, pad_t], -1)  # [*, L+max_pad]
    final_shape = mask_shape[:-1] + [max_count]
    ret_idxes = concat_t.nonzero(as_tuple=False)[:, -1].reshape(final_shape)
    # get valid mask and set 0 for invalid ones
    slen = mask_shape[-1]
    valid_mask = (ret_idxes < slen).float()
    ret_idxes[ret_idxes >= slen] = padding_idx
    return ret_idxes, valid_mask

# maxpool 1d at last dim
def max_pool1d(input, kernel):
    orig_shape = get_shape(input)
    # make it 3d
    tmp_res = F.max_pool1d(input.view([-1]+orig_shape[-2:]), kernel)
    real_res = tmp_res.view(orig_shape[:-1] + [-1])
    return real_res

# detach in training
def go_detach(x, scale: float, is_training: bool):
    if is_training:
        if scale == 1.:  # no detach
            return x
        elif scale == 0.:  # full detach
            return x.detach()
        else:  # scale gradient (allow >1 or <0)
            return scale * x + (1-scale) * (x.detach())
    else:
        return x  # no need to detach if not training!
# --

def get_emb_with_initscale(num_embeddings: int, embedding_dim: int, initscale=1., **kwargs):
    ret = nn.Embedding(num_embeddings, embedding_dim, **kwargs)
    if initscale != 1:
        with torch.no_grad():
            ret.weight *= initscale
    return ret
