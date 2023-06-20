#

# using pytorch

from typing import Union, SupportsFloat, List
from collections import OrderedDict
from mspx.utils import zwarn, zlog, WithWrapper, get_sysinfo
import traceback
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
import math
import os

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .common import NIConf, OptimConf

# =====
# Part 0: basic ones!

# note: even fix the seeds, when running parallel with GPU, on different machines (for example N vs A, and probably with different GPUs, or even from where python is called[might be related with transformers, since from_pretarined is the point where "get_rng_state()" changes ...]), large rand might still be different?

# Basic Conf
class TrNIConf(NIConf):
    def __init__(self):
        self.seed0 = 0  # init one!
        self.num_threads = 4  # maximum NUM_THREADS if using cpu
        self.device = -1  # -1: cpu, [0,): gpu
        self.fp16 = False  # whether using fp16
        # --
        # other nn-lib specific settings
        self.dist_backend = "nccl"
        self.dist_rank = 0
        self.dist_world_size = 1  # really activate if >1
        self.dist_addr = 'localhost'
        self.dist_port = 12365
        self.dist_find_unused_parameters = False
        # --
        # apex.amp
        self.amp_opt_level = ''
        # torch.amp
        self.use_torch_amp = False  # use torch's amp rather than apex's?
        # --

    @property
    def use_apex(self):
        return bool(self.amp_opt_level)

BKNIConf = TrNIConf

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
DEFAULT_FLOAT = torch.float32
DEFAULT_INT = torch.long

# singleton
_global_tr_conf = TrNIConf()
def get_global_conf():
    return _global_tr_conf
def set_gloabl_conf(conf: TrNIConf):
    global _global_tr_conf
    _global_tr_conf = conf

# --
# def _find_idle_port(start: int):
#     # --
#     def is_port_in_use(port: int) -> bool:
#         import socket
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             return s.connect_ex(('localhost', port)) == 0
#     # --
#     for z in range(10):
#         curr = start + z*7
#         if not is_port_in_use(curr):
#             return curr
#     return start  # let it err!
# --

def init(conf: TrNIConf = None, **kwargs):
    global DEFAULT_DEVICE
    global DEFAULT_FLOAT
    # --
    if conf is None:
        conf = get_global_conf()
    else:
        set_gloabl_conf(conf)
    conf.direct_update(**kwargs)
    # --
    torch.set_num_threads(min(conf.num_threads, int(os.environ.get('OMP_NUM_THREADS', 100))))
    init_seed(conf.seed0)  # first init a fixed one!
    if conf.device >= 0:
        DEFAULT_DEVICE = torch.device(f"cuda:{conf.device}")
        zlog(f"Init NN with default_device={DEFAULT_DEVICE}")
        zlog(f"GPU-info:\n{get_sysinfo(ret_str=False, get_gpu_info=True)['gpu']}")
    if conf.dist_world_size > 1:
        os.environ['MASTER_ADDR'] = conf.dist_addr
        # os.environ['MASTER_PORT'] = str(_find_idle_port(conf.dist_port))
        os.environ['MASTER_PORT'] = str(conf.dist_port)
        # initialize the process group
        DEFAULT_DEVICE = torch.device(f"cuda:{conf.dist_rank}")  # reset!
        torch.cuda.set_device(conf.dist_rank)
        dist.init_process_group(conf.dist_backend, rank=conf.dist_rank, world_size=conf.dist_world_size)
        zlog(f"Init NN.dist with {conf.dist_addr}/{conf.dist_port} {conf.dist_rank}/{conf.dist_world_size} {DEFAULT_DEVICE}.")
    if conf.fp16:
        DEFAULT_FLOAT = torch.float16
        zlog(f"Init NN with fp16!")
    # --

# note: make it a specific function!
def init_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
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
    return get_global_conf().dist_rank == 0  # rank0 is the main one!!

# --
# from fairseq
class ModuleProxyWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        assert hasattr(module, "module"), "ModuleProxyWrapper expects input to wrap another module"
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
    if conf.use_apex:
        from apex.parallel import DistributedDataParallel as DDP
    else:
        from torch.nn.parallel import DistributedDataParallel as DDP0
        DDP = lambda model: DDP0(model, device_ids=[conf.device])
    # --
    m1 = DDP(model)
    m2 = ModuleProxyWrapper(m1)
    return m2
# --
# amp related
def use_amp():
    conf = get_global_conf()
    return bool(conf.amp_opt_level) or conf.use_torch_amp

def autocast_env(enabled=None):
    if enabled is None:
        conf = get_global_conf()
        enabled = conf.use_torch_amp
    if enabled:
        try:
            ret = torch.cuda.amp.autocast(enabled=enabled)
            return ret
        except:
            zwarn(f"There are no autocast in current version: {torch.__version__}")
    return WithWrapper()

class AmpManager:
    def __init__(self):
        conf = get_global_conf()
        self.amp_opt_level = conf.amp_opt_level
        self.use_apex_amp = conf.use_apex
        self.scaler = None
        if self.use_apex_amp:
            zlog(f"Use Apex amp with {conf.amp_opt_level}!")
        else:
            self.scaler = torch.cuda.amp.GradScaler()
            zlog(f"Use torch.amp!")

    def initialize(self, model, optimizer):
        if self.use_apex_amp:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.amp_opt_level)
        return model, optimizer

    def with_scale(self, loss, optimizer):
        if self.use_apex_amp:
            from apex import amp
            return amp.scale_loss(loss, optimizer)
        else:
            ret = self.scaler.scale(loss)
            return WithWrapper(item=ret)

    def with_autocast(self):
        return autocast_env(enabled=(not self.use_apex_amp))

    def step(self, optimizer):
        if not self.use_apex_amp:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self):
        if not self.use_apex_amp:
            self.scaler.update()

# --
def get_optimizer(conf: OptimConf, params, init_lrate: float):
    if conf.optim_type == 'adam':
        return torch.optim.Adam(params, lr=init_lrate, betas=conf.adam_betas, eps=conf.adam_eps, weight_decay=conf.weight_decay)
    elif conf.optim_type == 'adamw':
        if conf.weight_decay == 0.:
            zwarn("AdamW gets zero weight_decay!")
        return torch.optim.AdamW(params, lr=init_lrate, betas=conf.adam_betas, eps=conf.adam_eps, weight_decay=conf.weight_decay)
    elif conf.optim_type == 'fused_adam':
        from apex.optimizers import FusedAdam
        return FusedAdam(params, lr=init_lrate, betas=conf.adam_betas, eps=conf.adam_eps, weight_decay=conf.weight_decay)
    elif conf.optim_type == 'sgd':
        return torch.optim.SGD(params, lr=init_lrate, momentum=conf.sgd_momentum, weight_decay=conf.weight_decay)
    else:
        raise NotImplementedError(f"UNK optim {conf.optim_type}")

# --

def refresh():
    # not doing this, since it makes things slower
    # torch.cuda.empty_cache()
    pass

# note: default init
init()
# --

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

# --
# save & load

def split_slname(name: str):
    if ',,,' in name:
        name, specs = name.rsplit(',,,', 1)
    else:
        specs = ''
    return name, specs

def change_slname(name: str, add_suffix: str = None, rm_suffixes=None, rm_specs=False):
    name, specs = split_slname(name)
    if rm_suffixes:
        for rr in rm_suffixes:
            if name.endswith(rr):
                name = name[:-len(rr)]
    if add_suffix:
        name = name + add_suffix
    if not rm_specs and specs:
        name = name + ",,," + specs
    return name

def parse_slname(name: str, quite=True):
    cut_mods, del_mods, sub_mods = [], [], []
    name, specs = split_slname(name)
    if specs:
        for spec in specs.split(","):
            if spec[0] == 'C':
                cut_mods.append(spec[1:])
            elif spec[0] == 'D':
                del_mods.append(spec[1:])
            elif spec[0] == 'S':
                sub_mods.append(spec[1:].split("=="))
            else:
                raise NotImplementedError()
        if not quite:
            zlog(f"Parse prefix: {name}###{specs} -> {cut_mods} {del_mods} {sub_mods}")
    ret = {"path": name, "cut_mods": cut_mods, "del_mods": del_mods, "sub_mods": sub_mods}
    return ret

def change_state_dict(d, **kwargs):
    # note: keep the order & follow the order in *mods
    cut_mods, del_mods, sub_mods = [kwargs.get(z, []) for z in ['cut_mods', 'del_mods', 'sub_mods']]
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
    # -- sub names ...
    for sub0, sub1 in sub_mods:
        d2 = OrderedDict()
        for k, v in d.items():
            k2 = k.replace(sub0, sub1)
            if k2:  # another shortcut for del!
                d2[k2] = v
        d = d2
    # --
    return d

def save_mod(mod: Union[Module, OrderedDict], path: str, quite=True, **kwargs):
    d = mod if isinstance(mod, dict) else mod.state_dict()
    kwargs0 = parse_slname(path, quite=quite)
    kwargs0.update(kwargs)
    path = kwargs0.pop('path')
    d = change_state_dict(d, **kwargs0)
    torch.save(d, path)
    if not quite:
        zlog(f"Save {len(mod) if isinstance(mod, dict) else mod} to {path}", func="io")
    # --

def load_mod(mod: Module, path: Union[dict, str], quite=True, strict=None, map_location=None, **kwargs):
    if map_location is None:
        map_location = DEFAULT_DEVICE
    if isinstance(path, str):
        kwargs0 = parse_slname(path, quite=quite)
        path = kwargs0.pop('path')
        d = torch.load(path, map_location=map_location)
    else:
        kwargs0 = kwargs
        d = path
    d = change_state_dict(d, **kwargs0)
    # load it
    if strict is not None:
        mod.load_state_dict(d, strict=strict)
    else:  # otherwise, first try strict, then relax if there are errors
        try:
            mod.load_state_dict(d, strict=True)
        except:
            import traceback
            zwarn(f"#== Error in strict loading:\n{traceback.format_exc()}\n#==")
            mod.load_state_dict(d, strict=False)
    if not quite:
        zlog(f"Load {mod} from {len(path) if isinstance(path, dict) else path}", func="io")
    # --

# =====
# Part 2: functions

# inputs with dtype
def input_tensor(inputs, device=None, dtype=None):
    if is_expr(inputs):
        return inputs.to(device=device, dtype=dtype)
    return torch.tensor(inputs, dtype=dtype, device=(DEFAULT_DEVICE if device is None else device))

# (inputs: python data type) -> FloatTensor
def input_real(inputs, device=None):
    return input_tensor(inputs, device, DEFAULT_FLOAT)

# (inputs: python data type of indexes) -> LongTensor
def input_idx(inputs, device=None):
    return input_tensor(inputs, device, DEFAULT_INT)

# (shape: ..., value: float) -> FloatTensor
def constants(shape, value=0., dtype=DEFAULT_FLOAT, device=None):
    return torch.full(shape, value, dtype=dtype, device=(DEFAULT_DEVICE if device is None else device))

# constants(..., dtype=long)
def constants_idx(shape, value=0, device=None):
    return torch.full(shape, value, dtype=DEFAULT_INT, device=(DEFAULT_DEVICE if device is None else device))

# create a new parameter
def new_param(shape, init=0., scale=1., dtype=DEFAULT_FLOAT, device=None, requires_grad=True):
    t = torch.empty(shape, dtype=dtype, device=(DEFAULT_DEVICE if device is None else device))
    # init
    if init is not None:
        if isinstance(init, str):
            getattr(nn.init, init+"_")(t)
        elif isinstance(init, (float, int)):
            t.fill_(init)
        else:
            set_value(t, init)
    if scale != 1.:
        t *= scale
    # --
    ret = nn.Parameter(t, requires_grad=requires_grad)
    return ret

# return 2D eye matrix
def eye(n, device=None):
    return torch.eye(n, dtype=DEFAULT_FLOAT, device=(DEFAULT_DEVICE if device is None else device))

# arange
def arange_idx(*args, device=None, unsqueeze_num=0):
    ret = torch.arange(*args, dtype=DEFAULT_INT, device=(DEFAULT_DEVICE if device is None else device))
    if unsqueeze_num > 0:
        ret = ret.view(get_shape(ret) + [1] * unsqueeze_num)
    return ret

# (shape: ..., p: rate of 1., mul: multiply) -> Tensor
def random_bernoulli(shape, p: float, mul: float, device=None):
    x = torch.full(shape, p, dtype=DEFAULT_FLOAT, device=(DEFAULT_DEVICE if device is None else device))
    r = torch.bernoulli(x) * mul
    return r

# [0,1)
def rand(shape, dtype=DEFAULT_FLOAT, device=None):
    return torch.rand(shape, dtype=dtype, device=(DEFAULT_DEVICE if device is None else device))

def randint(*args, dtype=DEFAULT_INT, device=None, **kwargs):
    return torch.randint(*args, **kwargs, dtype=dtype, device=(DEFAULT_DEVICE if device is None else device))

# (input_list: list of Tensors, dim: ...) -> Tensor
def concat(input_list: List, dim=-1, do_broadcast_expand=False):
    if len(input_list) == 1:  # note: no need to do anything!
        return input_list[0]
    if do_broadcast_expand:  # do broadcast and expand to make other dims match
        shapes = [get_shape(z) for z in input_list]
        for ss in shapes:
            ss[dim] = 1  # this dim does not count!
        full_shape = list(torch.broadcast_shapes(*shapes))
        new_input_list = []
        for ii, tt in enumerate(input_list):
            ss = get_shape(tt)
            _unsqueeze_times = len(full_shape) - len(ss)
            if _unsqueeze_times > 0:  # make new dims
                ss = [1]*_unsqueeze_times + ss
                tt = tt.view(ss)
            full_shape[dim] = ss[dim]  # temply make it back to original!
            if ss != full_shape:
                tt = tt.expand(full_shape)
            new_input_list.append(tt)
        input_list = new_input_list
    return torch.cat(input_list, dim)

# (t: Tensor, idxes: list(batch-size) of idx, dim: ...) -> Tensor
def gather_one(t, idxes, dim=-1):
    idx_t = input_idx(idxes)  # [...]
    idx_t2 = torch.unsqueeze(idx_t, dim)  # [..., 1, ...]
    return torch.gather(t, dim, idx_t2).squeeze(dim)  # [...]

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
max_d = torch.max  # note: with dim, return tuple
max_elem = torch.max  # note: max_elem(a, b)
min_d = torch.min
min_elem = torch.min
masked_select = torch.masked_select
matmul = torch.matmul
norm = torch.norm
pad = F.pad
relu = F.relu
repeat_interleave = torch.repeat_interleave
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
topk = torch.topk  # note: return (val, idx)
transpose = torch.transpose
unsqueeze = torch.unsqueeze
where = torch.where
zeros = constants  # note: rename!
zeros_like = torch.zeros_like

# - log softmax(margin(score_expr))[idx]
# [..., C, ...], [..., ...] -> [..., ...]
def loss_nll(score_expr, gold_idxes, dim=-1, label_smoothing=0., margin=0.):
    gold_idxes_t1 = input_idx(gold_idxes).unsqueeze(dim)  # [..., 1, ...]
    if margin > 0:
        _m = torch.full_like(score_expr, fill_value=margin)
        _m.scatter_(dim, gold_idxes_t1, 0.)
        score_expr = score_expr + _m
    # note: keep it simple!
    nll_t = - score_expr.log_softmax(dim=dim)  # [..., C, ...]
    ret_t = nll_t.gather(dim, gold_idxes_t1).squeeze(dim)  # [..., ...]
    if label_smoothing > 0.:
        ret_t = (1.-label_smoothing) * ret_t + label_smoothing * nll_t.mean(dim)  # [..., ...]
    return ret_t

# - (score[idx] - max(score'+margin))
# [..., C, ...], [..., ...] -> [..., ...]
def loss_hinge(score_expr, gold_idxes, dim=-1, margin=0.):
    if is_zero_shape(score_expr):  # note: since we need to do maximize
        return score_expr.sum(dim)
    # --
    gold_idxes_t1 = input_idx(gold_idxes).unsqueeze(dim)  # [..., 1, ...]
    if margin > 0.:
        _m = torch.full_like(score_expr, fill_value=margin)
        _m.scatter_(dim, gold_idxes_t1, 0.)
        score_expr_m = score_expr + _m
    else:
        score_expr_m = score_expr
    # --
    gold_score = score_expr.gather(dim, gold_idxes_t1).squeeze(dim)  # [..., ...]
    max_score, _ = score_expr_m.max(dim)  # [..., ...]
    ret = max_score - gold_score
    return ret

# binary cross-entropy loss
# [*], [*](0 or 1) -> [*]
def loss_binary(score_expr, gold_idxes, margin=0., label_smoothing=0.):
    gold_idxes = input_real(gold_idxes)  # make it float!
    if margin > 0.:
        score_expr = score_expr + (1.-2*gold_idxes) * margin
    if label_smoothing > 0.:
        gold_idxes = gold_idxes + (1.-2*gold_idxes) * label_smoothing
    ret = F.binary_cross_entropy_with_logits(score_expr, gold_idxes)
    return ret

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
    flatten_t = t.view([np.prod(t_shape0).item()] + t_shape1)  # [s1*...*sn, ...]
    basis_t = arange_idx(np.prod(t_shape0[:-1])).view(t_shape0[:-1]) * t_shape0[-1]  # [s1, ..., sn-1]
    basis_t = (basis_t.unsqueeze(-1) + idxes_t).view(-1)  # [*]
    output_t0 = torch.index_select(flatten_t, dim=0, index=basis_t)  # [*, ...]
    return output_t0.view(idx_shape + t_shape1)

# =====
# values & backward

# return numpy values
def get_value(t):
    return t.detach().cpu().numpy()

def set_value(t, val, resize=False):
    with torch.autograd.no_grad():  # avoid recording grad_fn
        if not resize:
            assert get_shape(t)==get_shape(val)
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

def no_grad_env(no_grad=True):
    if no_grad:
        return torch.autograd.no_grad()
    else:
        return WithWrapper()

def no_change_rand_env(no_change=True):
    if no_change:
        curr_state = torch.random.get_rng_state()  # recover random state!
        return WithWrapper(f_end=(lambda: torch.random.set_rng_state(curr_state)))
    else:
        return WithWrapper()

def has_nan(t):
    return int(torch.isnan(t).sum())

# special routines
# note: for mask->idx: 1) argsort, 2) pad 1s + nonzero, 3) loop; => v2 is the fastest!
# the inputs should be 1. or 0. (float); [*, L, *] -> [*, max-count, *]
def mask2idx(mask_f, dim=-1, pad=0):
    mask_shape = mask_f.shape  # [*, L, *]
    # --
    # judge zero-shape
    if any(z==0 for z in mask_shape):
        _shape = list(mask_shape)
        _shape[dim] = 1
        zz = torch.zeros(_shape)  # [*, 1, *], put an one here!
        return zz.to(DEFAULT_INT), zz.to(DEFAULT_FLOAT)
    # --
    mask_f = mask_f.to(DEFAULT_FLOAT)  # [*, L, *]
    # get max counts
    counts = mask_f.sum(dim=dim, keepdims=True)  # [*, 1, *]
    max_count = max(1, int(counts.max().item()))  # M
    padding_counts = max_count - counts  # [*, 1, *]
    max_padding_count = int(padding_counts.max().item())  # int, the max count of padding
    # pad and concat
    _arange_idx = arange_idx(max_padding_count)  # [mp]
    to_expand_dim = (-dim-1) if dim<0 else (len(mask_shape)-1-dim)
    pad_t = (_arange_idx.view([max_padding_count]+[1]*to_expand_dim) < padding_counts).to(DEFAULT_FLOAT)  # [*, mp, *]
    concat_t = torch.cat([mask_f, pad_t], dim)  # [*, L+mp, *]
    # nonzero and extract
    final_shape = list(mask_shape)
    final_shape[dim] = max_count
    if dim != -1 or dim != len(mask_shape) - 1:
        final_shape = final_shape[:dim] + final_shape[dim:][1:] + [max_count]
        _p0 = list(range(len(mask_shape)))  # [0, N)
        _p1 = _p0[:dim] + _p0[dim:][1:] + [dim]
        _p2 = _p0[:dim] + [-1] + [z-1 for z in _p0[dim:][1:]]
        ret_idxes = concat_t.permute(_p1).nonzero(as_tuple=False)[:, -1].view(final_shape).permute(_p2)
    else:
        ret_idxes = concat_t.nonzero(as_tuple=False)[:, dim].view(final_shape)  # [*, M, *]
    # get valid mask and set pad for invalid ones
    max_len = mask_shape[dim]  # L
    valid_mask = (ret_idxes < max_len).to(DEFAULT_FLOAT)  # [*, M, *]
    ret_idxes[valid_mask<=0.] = pad
    return ret_idxes, valid_mask

def idx2mask(idxes_t, mask_t, full_len: int, dim=-1):
    input_shape = idxes_t.shape  # [*, N, *]
    # --
    # judge zero-shape
    if any(z == 0 for z in input_shape):
        _shape = list(input_shape)
        _shape[dim] = 1
        zz = torch.zeros(_shape)  # [*, 1, *], put an one here!
        return zz.to(DEFAULT_FLOAT)
    # --
    if full_len is None:  # infer it!
        full_len = idxes_t.max().item()  # overall max!
    output_shape = list(input_shape)
    output_shape[dim] = 1+full_len  # [*, 1+L, *]
    ret0 = torch.zeros(output_shape).to(DEFAULT_FLOAT)
    idxes_t = idxes_t + (idxes_t<0).long() * full_len  # prepare proper index
    idxes_t = idxes_t.clamp(min=0, max=full_len-1)  # clamp!
    ret0.scatter_(dim, (1+idxes_t) * mask_t.long(), 1.)  # +1 to put idx0 as NIL
    ret = ret0.narrow(dim, 1, full_len)
    return ret  # [*, L, *]

# sampling 1 with gumble
# argmax(logprob + -log(-log(Unif[0,1]))), return selected idx
def category_sample(logprob, dim=-1, keepdim=True, eps=1e-10, top_k=0, top_p=0.0):
    filter_value = float('-inf')
    if top_k > 0:
        logprob = logprob.clone()  # clone it to modify inplace
        top_k = min(top_k, logprob.size(dim))  # Safety check
        indices_to_remove = logprob < (logprob.topk(top_k, dim=dim)[0].narrow(dim, -1, 1))
        logprob[indices_to_remove] = filter_value
    if top_p > 0.:
        logprob = logprob.clone()  # clone it to modify inplace
        sorted_logits, sorted_indices = logprob.sort(dim=dim, descending=True)
        cumulative_probs = sorted_logits.softmax(dim).cumsum(dim)
        idx_boundary = (cumulative_probs <= top_p).long().sum(dim, keepdims=True)  # [..., 1, ...]
        idx_boundary.clamp_(max=logprob.size(dim)-1)
        value_boundary = sorted_logits.gather(dim, idx_boundary)
        logprob[logprob<value_boundary] = filter_value
    # --
    G = torch.rand_like(logprob)
    X = logprob - (-(G+eps).log() + eps).log()
    _, I = X.max(dim, keepdim=keepdim)
    return I

# select topk ones
# [*, D, *], [*, 1, *], [*, D, *] => [*, D, *]
def select_topk(score_t, topk_t, mask_t=None, dim=-1, noise=0.):
    # if zero-shape
    if is_zero_shape(score_t):
        return score_t * 0.  # select nothing!
    # --
    # prepare K
    if isinstance(topk_t, int):
        K = topk_t
        tmp_shape = list(score_t.shape)
        tmp_shape[dim] = 1  # set it as 1
        topk_t = constants_idx(tmp_shape, K)  # [*, 1, *]
    else:
        K = topk_t.max().item()
    exact_rank_t = topk_t - 1  # [*, 1, *]
    exact_rank_t.clamp_(min=0, max=K-1)  # make it in valid range!
    # mask values
    if mask_t is not None:
        _extra_score = torch.zeros_like(score_t)
        _extra_score[mask_t<=0.] = float('-inf')
        score_t = score_t + _extra_score
    # add some small noise to break tie
    if noise > 0:
        _extra_noise = torch.rand_like(score_t) * noise
        score_t = score_t + _extra_noise
    # topk
    topk_vals, _ = score_t.topk(K, dim, largest=True, sorted=True)  # [*, K, *]
    # gather score
    sel_thresh = topk_vals.gather(dim, exact_rank_t)  # [*, 1, *]
    # get topk_mask (if K is 0, then select nothing!)
    topk_mask = ((score_t >= sel_thresh) & (topk_t > 0)).to(DEFAULT_FLOAT)  # [*, D, *]
    if mask_t is not None:
        topk_mask *= mask_t
    return topk_mask

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

def get_emb_with_initscale(num_embeddings: int, embedding_dim: int, initscale=1., **kwargs):
    ret = nn.Embedding(num_embeddings, embedding_dim, **kwargs)
    if initscale != 1:
        with torch.no_grad():
            ret.weight *= initscale
    return ret
