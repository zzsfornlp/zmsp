# using pytorch

import torch
from torch import nn, optim
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np
from typing import List

from .common import COMMON_CONFIG, get_unique_name, _my_get_params_init

Expr = torch.Tensor
Module = torch.nn.Module
CPU_DEVICE = torch.device("cpu")
DEFAULT_DEVICE = CPU_DEVICE
T_INIT = torch.nn.init

# types
float32 = torch.float32
float64 = torch.float64
int32 = torch.int32
int64 = torch.int64
uint8 = torch.uint8

def init():
    torch.set_num_threads(COMMON_CONFIG.num_threads)
    torch.manual_seed(COMMON_CONFIG.random_seed)
    torch.cuda.manual_seed(COMMON_CONFIG.random_cuda_seed)
    if COMMON_CONFIG.device >= 0:
        global DEFAULT_DEVICE
        DEFAULT_DEVICE = torch.device("cuda:%d"%COMMON_CONFIG.device)

def refresh():
    # not doing this, since it makes things slower
    # torch.cuda.empty_cache()
    pass

# todo(warn): default init
init()

# also useful for ndarray
def get_shape(t, dim=None):
    shapes = [i for i in t.shape]
    if dim is None:
        return shapes
    else:
        return shapes[dim]

# is_tensor
def is_expr(v):
    return isinstance(v, Expr)

is_tensor = is_expr

# parameter init from BK (similar to common.get_params_init)
# return a tensor here; (out_p4i is the real shape[0] for more reasonable init for some cases)
def get_params_init(shape, init, lookup, out_p4i, scale):
    # if COMMON_CONFIG.use_my_init:
    #     return _my_get_params_init(shape, init, lookup)
    assert not COMMON_CONFIG.use_my_init, "now use ones from pytorch for param init"
    x = torch.empty(*shape, dtype=torch.float32, device=DEFAULT_DEVICE)
    if len(shape) == 1:
        nn.init.zeros_(x)
    else:
        if lookup:
            _iscale = np.sqrt(3.0 / shape[-1])
            nn.init.uniform_(x, -_iscale, _iscale)
            # todo(+N): again back to previous init method
            # nn.init.normal_(x)
            x *= scale
        elif init == "default" or init == "glorot":
            out_size = shape[0]
            assert out_size % out_p4i == 0, "Bad output shape pieces for init value!"
            s0 = out_size//out_p4i
            for i in range(out_p4i):
                nn.init.xavier_uniform_(x[i*s0:(i+1)*s0])
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
            assert False, "Unknown init method for BKTR: "+init
    return x

# todo(note): similar to Torch.Optimizer, but with extra settings
class Optim:
    def __init__(self, optim_type, lrf_sv, oconf, params):
        self.params_ = params
        self.lrf_sv_ = lrf_sv
        if optim_type == "sgd":
            opt_ = optim.SGD(params, lr=0., momentum=oconf.sgd_momentum, weight_decay=oconf.weight_decay)
        elif optim_type == "adagrad":
            opt_ = optim.Adagrad(params, lr=0., weight_decay=oconf.weight_decay)
        elif optim_type == "adam":
            opt_ = optim.Adam(params, lr=0., betas=oconf.adam_betas, eps=oconf.adam_eps, weight_decay=oconf.weight_decay)
        elif optim_type == "adadelta":
            opt_ = optim.Adadelta(params, lr=0., rho=oconf.adadelta_rho, weight_decay=oconf.weight_decay)
        else:
            raise NotImplementedError("Unknown optim %s." % optim_type)
        self.opt_ = opt_
        #
        self.no_step_lrate0_ = oconf.no_step_lrate0
        self.cached_lrate_ = 0.
        self.grad_clip_ = oconf.grad_clip

    def update(self, overall_lrate, grad_factor):
        cur_lrate = overall_lrate * float(self.lrf_sv_)
        if self.cached_lrate_ != cur_lrate:
            # schedule lrate, do as lr_scheduler does
            for param_group in self.opt_.param_groups:
                param_group['lr'] = cur_lrate
            self.cached_lrate_ = cur_lrate
        # check if we need update
        parameters = list(filter(lambda p: p.grad is not None, self.params_))
        if (cur_lrate<=0. and self.no_step_lrate0_) or (len(parameters) == 0):
            # no update
            self.opt_.zero_grad()
        else:
            # todo(warn): useful for batch-split, div grad by splits
            if grad_factor != 1.:
                for p in self.params_:
                    if p.grad is not None:
                        p.grad.data.mul_(grad_factor)
            if self.grad_clip_ > 0.:
                clip_grad_norm_(self.params_, self.grad_clip_)
            self.opt_.step()
            self.opt_.zero_grad()

# todo(warn): here nn.Module simply used for Param Collection
class ParamCollection:
    def __init__(self, new_name_conv=True):
        self.model_ = nn.Module()
        self.optims_ = []
        self.paramid2optid_ = {}  # id -> list
        #
        self.name_dict = {}
        self.new_name_conv = new_name_conv
        self.new_name_conv_stack = []

    # =====
    def nnc_push(self, name):
        self.new_name_conv_stack.append(name)

    def nnc_pop(self, name):
        # todo(note): there can be out-of-order because of wrappers
        ridx = len(self.new_name_conv_stack) - 1 - self.new_name_conv_stack[::-1].index(name)
        x = self.new_name_conv_stack.pop(ridx)
        assert x == name, "Name unmatched!!"

    def nnc_name(self, name, check_stack=True):
        if check_stack:
            assert name == self.new_name_conv_stack[-1]
        if self.new_name_conv:
            return "/".join(self.new_name_conv_stack)
        else:
            return name
    # =====

    def get_unique_name(self, name):
        return get_unique_name(self.name_dict, name)

    # add a torch.nn.Module's parameters
    def param_add_external(self, name, mod: nn.Module):
        ret_pairs = []
        for one_subname, one_param in mod.named_parameters():
            one_subname = "_".join(one_subname.split("."))  # cannot include "."
            self.model_.register_parameter(name+"/"+one_subname, one_param)
            ret_pairs.append((one_subname, one_param))
        return ret_pairs

    # register param
    def param_new(self, name, shape, init_weights, lookup=False):
        # almost all params are float
        p = Parameter(torch.as_tensor(init_weights, dtype=torch.float32, device=DEFAULT_DEVICE))
        assert name not in self.model_._parameters  # no modules in this pc
        self.model_.register_parameter(name, p)
        return p

    # special mode
    # todo(WARN): for changing names while stacking modules; not elegant!
    def param_rename(self, old_name, new_name):
        p = self.model_.__getattr__(old_name)
        self.model_.__delattr__(old_name)
        self.model_.register_parameter(new_name, p)

    # freeze param
    def param_set_trainable(self, p, trainable):
        bool_trainable = bool(trainable)
        if p.requires_grad != bool_trainable:
            p.requires_grad = bool_trainable

    # tconf should have other properties: momentum, grad_clip,
    def optimizer_set(self, optim_type, lrf_sv, oconf, params: List = None, check_repeat=True, check_full=False):
        if params is None:
            params = self.model_.parameters()
        if len(params) > 0:
            optim = Optim(optim_type, lrf_sv, oconf, params)
            cur_optid = len(self.optims_)
            self.optims_.append(optim)
        # track all params
        for p in params:
            paramid = id(p)
            if paramid not in self.paramid2optid_:
                self.paramid2optid_[paramid] = [cur_optid]
            else:
                assert not check_repeat, "Err: repeated params in multiple optimizer!"
                self.paramid2optid_[paramid].append(paramid)
        if check_full:
            for p in self.model_.parameters():
                assert id(p) in self.paramid2optid_, "Err: failed checking full, there is unadded param"

    def optimizer_update(self, overall_lrate, grad_factor):
        for optim in self.optims_:
            optim.update(overall_lrate, grad_factor)

    def save(self, path):
        torch.save(self.model_.state_dict(), path)

    def load(self, path, strict=True):
        model = torch.load(path, map_location=DEFAULT_DEVICE)
        self.model_.load_state_dict(model, strict=strict)

# ===== the functions

# ----- inputs

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

# (shape: ..., value: float) -> FloatTensor
def constants(shape, value=0., dtype=torch.float32, device=None):
    return torch.full(shape, value, dtype=dtype, device=(DEFAULT_DEVICE if device is None else device))

#
def constants_idx(shape, value=0, device=None):
    return torch.full(shape, value, dtype=torch.long, device=(DEFAULT_DEVICE if device is None else device))

# return 2D eye matrix
def eye(n, device=None):
    return torch.eye(n, dtype=torch.float32, device=(DEFAULT_DEVICE if device is None else device))

#
def arange_idx(*args, device=None):
    return torch.arange(*args, dtype=torch.long, device=(DEFAULT_DEVICE if device is None else device))

# (shape: ..., p: rate of 1., mul: multiply) -> Tensor
def random_bernoulli(shape, p, mul, device=None):
    x = torch.full(shape, p, dtype=torch.float32, device=(DEFAULT_DEVICE if device is None else device))
    r = torch.bernoulli(x) * mul
    return r

#
def rand(shape, dtype=torch.float32, device=None):
    return torch.rand(shape, dtype=dtype, device=(DEFAULT_DEVICE if device is None else device))

# move to other device
def to_device(x, device=None):
    return x.to(device=(DEFAULT_DEVICE if device is None else device))

#
def copy(x, device=None):
    # return torch.tensor(x, dtype=x.dtype, device=(x.device if device is None else device))
    return x.clone().detach()

# ----- ops

# (input_list: list of tensors [bias, weight1, input1, ...]) -> Tensor
def affine(input_list):
    base, base_idx = input_list[0], 1
    while base_idx < len(input_list):
        base = torch.addmm(base, input_list[base_idx+1], input_list[base_idx].t())
        base_idx += 2
    return base

# allow >2d input, but still 2d weights
def affine2(input_list):
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
def affine3(input_list):
    bias, base_idx = input_list[0], 1
    base = 0
    while base_idx < len(input_list):
        base = base + torch.matmul(input_list[base_idx + 1], input_list[base_idx].t())
        base_idx += 2
    return base + bias

# (t: Tensor, size: how many pieces, dim: ...) -> list of Tensors
def chunk(t, size, dim=-1):
    return torch.chunk(t, size, dim)

# (x: Tensor, y: Tensor) -> Tensor (Elem-Multiply)
def cmult(x, y):
    return x*y
    # return torch.addcmul(torch.tensor(0.), 1, x, y)

# (input_list: list of Tensors, dim: ...) -> Tensor
def concat(input_list, dim=-1):
    if len(input_list) == 1:
        return input_list[0]
    return torch.cat(input_list, dim)

# (t: Tensor, p: float) -> Tensor
def dropout(t, p):
    # here always assume training
    return F.dropout(t, p, training=True)

# (t: Tensor, idxes: list(batch-size) of idx) -> Tensor
def gather_one_lastdim(t, idxes):
    idx_t = input_idx(idxes)
    idx_t2 = torch.unsqueeze(idx_t, -1)
    dim = len(t.shape)-1
    return torch.gather(t, dim, idx_t2)

# (t: Tensor, aixs: ...) -> Tensor
def log_softmax(t, dim=-1):
    return F.log_softmax(t, dim=dim)

# (weight: Tensor(Param), inputs: list of int) -> Tensor
def lookup(weight, inputs):
    idxes_t = input_idx(inputs)
    return F.embedding(idxes_t, weight)

# (t: Tensor, idxes: list of ints, dim:...) -> Tensor
def select(t, idxes, dim=0):
    idxes_t = input_idx(idxes)
    return torch.index_select(t, dim=dim, index=idxes_t)

abs = torch.abs
avg = torch.mean
bilinear = F.bilinear       # (N,*,in1_features), (N,*,in2_features), (out_features, in1_features, in2_features), (out_features,) -> (N,*,out_features)
binary_cross_entropy = F.binary_cross_entropy
binary_cross_entropy_with_logits = F.binary_cross_entropy_with_logits
clamp = torch.clamp
diagflat = torch.diagflat
elu = F.elu
exp = torch.exp
expand = lambda x, *args: x.expand(*args)
gelu = getattr(F, "gelu", None)  # todo(warn): on older versions, this does not exist
log = torch.log
logsigmoid = F.logsigmoid
logsumexp = torch.logsumexp
max = torch.max         # todo(warn): with dim, return tuple
max_elem = torch.max    # todo(warn): max_elem(a, b)
min = torch.min
min_elem = torch.min
masked_select = torch.masked_select
matmul = torch.matmul
pad = F.pad
relu = F.relu
reshape = torch.reshape
sigmoid = torch.sigmoid
softmax = F.softmax
squeeze = torch.squeeze
split = torch.split
sqrt = torch.sqrt
stack = torch.stack
sum = torch.sum
tanh = torch.tanh
topk = torch.topk       # todo(warn): return (val, idx)
transpose = torch.transpose
unsqueeze = torch.unsqueeze
where = torch.where
zeros = lambda shape: constants(shape, value=0.)

# =====
# loss functions (mainly 2d input raw scores)

# score_expr[gold_idx] -= margin (* should be arith-same as the adding one)
# todo(+4): currently return to_dense values since otherwise will get backward error!
def _minus_margin(score_expr, gold_idxes_expr, margin):
    score_shape = get_shape(score_expr)
    # to 2d
    score_shape_2d = [int(np.prod(score_shape[:-1])), score_shape[-1]]
    size0_2d = score_shape_2d[0]
    # todo(warn): to_dense since otherwise will get backward error!
    sparse_minus = torch.sparse.FloatTensor(stack([arange_idx(size0_2d), gold_idxes_expr.view(-1)], dim=0),
                                            constants([size0_2d], -margin), score_shape_2d).to_dense()
    score_expr_2d = score_expr.view(score_shape_2d) + sparse_minus
    return score_expr_2d.view(score_shape)

# used for global scoring
minus_margin = _minus_margin

# - log softmax(margin(score_expr))[idx]
# [*, C], [*] -> [*]
def loss_nll(score_expr, gold_idxes, margin=0.):
    gold_idxes_t = input_idx(gold_idxes)
    if margin > 0.:
        score_expr = _minus_margin(score_expr, gold_idxes_t, margin)
    # no average or sum-reduce for the output
    # output = F.nll_loss(F.log_softmax(score_expr, dim=-1), gold_idxes_t, size_average=False, reduce=False)
    log_softmax_score = F.log_softmax(score_expr, dim=-1)
    picked_vals = gather_one_lastdim(log_softmax_score, gold_idxes_t)
    return - picked_vals.squeeze(-1)

# - (score[idx] - max(score'+margin))
def loss_hinge(score_expr, gold_idxes, margin=0.):
    if margin > 0.:
        score_expr = _minus_margin(score_expr, gold_idxes, margin)
    score_max_ones, _ = torch.max(score_expr, -1)
    # 2d input for score
    score_gold_ones0 = gather_one_lastdim(score_expr, gold_idxes)
    score_gold_ones = score_gold_ones0.squeeze(-1)
    output = score_max_ones - score_gold_ones
    return output

# =====
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

def set_value(t, val):
    with torch.autograd.no_grad():
        # avoid recording grad_fn
        t.set_(input_real(val))

def get_cpu_tensor(t):
    return t.detach().cpu()

def backward(loss, loss_factor: float):
    if loss_factor != 1.:
        loss = loss * loss_factor
    loss.backward()

# directly setting param
def zero_row(param, row):
    with torch.autograd.no_grad():
        # avoid recording grad_fn
        param[row].fill_(0.)

def no_grad_env():
    return torch.autograd.no_grad()

#
def has_nan(t):
    return int(torch.isnan(t).sum())

# =====

# nn.Conv1d
# parameters already added here
class CnnOper:
    def __init__(self, node, n_input, n_output, n_window):
        # todo(warn): still belong to node
        self.w = node.add_param("w", (n_output, n_input, n_window))
        self.b = node.add_param("b", (n_output,))
        self.conv_padding = n_window//2
        self.node = node
        self.n_output = n_output

    def conv(self, input_expr):
        cur_dims = get_shape(input_expr)
        expr1 = input_expr.view([-1, cur_dims[-2], cur_dims[-1]]).transpose(-2, -1)
        val0 = F.conv1d(expr1, self.w, self.b, padding=self.conv_padding)
        #
        reshape_dims = cur_dims
        reshape_dims[-1] = -1
        reshape_dims[-2] = self.n_output
        val1 = val0.view(reshape_dims).transpose(-2, -1)
        return val1

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
# todo(note): for mask->idx: 1) topk+sort, 2) padding with extra 1s; currently using 2)
# the inputs should be 1. or 0. (float); [*, L] -> [*, max-count]
def mask2idx(mask_t, padding_idx=0):
    mask_shape = get_shape(mask_t)  # [*, L]
    counts = mask_t.sum(-1).long()  # [*]
    max_count_t = counts.max(-1, keepdim=True)[0]
    max_count = max_count_t.item()  # int, the max expanding
    padding_counts = max_count_t - counts  # [*]
    max_padding_count = padding_counts.max().item()  # int, the max count of padding
    pad_t = (arange_idx(max_padding_count) < padding_counts.unsqueeze(-1)).float()  # [*, max_pad]
    concat_t = concat([mask_t, pad_t], -1)  # [*, L+max_pad]
    final_shape = mask_shape[:-1] + [max_count]
    ret_idxes = concat_t.nonzero()[:, -1].reshape(final_shape)
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
