# using pytorch

import torch
from torch import nn, optim
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
import numpy as np

from .common import COMMON_CONFIG, get_unique_name, _my_get_params_init
from msp.utils import zwarn

Expr = torch.Tensor
DEFAULT_DEVICE = torch.device("cpu")

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
# return a tensor here
def get_params_init(shape, init, lookup):
    if COMMON_CONFIG.use_my_init:
        return _my_get_params_init(shape, init, lookup)
    x = torch.empty(*shape, dtype=torch.float32, device=DEFAULT_DEVICE)
    if len(shape) == 1:
        nn.init.zeros_(x)
    else:
        if lookup:
            scale = np.sqrt(3.0 / shape[-1])
            nn.init.uniform_(x, -scale, scale)
        elif init == "default" or init == "glorot":
            nn.init.xavier_uniform_(x)
        elif init == "ortho":
            nn.init.orthogonal_(x)
        else:
            assert False, "Unknown init method for BKTR: "+init
    return x

# todo(warn): here nn.Module simply used for Param Collection
class ParamCollection(object):
    def __init__(self):
        self.model_ = nn.Module()
        self.optim_ = None
        self.prev_lrate_ = None
        self.grad_clip_ = None
        #
        self.name_dict = {}

    def get_unique_name(self, name):
        return get_unique_name(self.name_dict, name)

    # register param
    def param_new(self, name, shape, init_weights, lookup=False):
        # almost all params are float
        p = Parameter(torch.as_tensor(init_weights, dtype=torch.float32, device=DEFAULT_DEVICE))
        self.model_.register_parameter(name, p)
        return p

    # freeze param
    def param_set_trainable(self, p, trainable):
        bool_trainable = bool(trainable)
        if p.requires_grad != bool_trainable:
            p.requires_grad = bool_trainable

    # tconf should have other properties: momentum, grad_clip,
    def optimizer_set(self, optim_type, init_lrate, tconf):
        if optim_type == "sgd":
            self.optim_ = optim.SGD(self.model_.parameters(), lr=init_lrate, momentum=tconf.sgd_momentum)
        elif optim_type == "adagrad":
            self.optim_ = optim.Adagrad(self.model_.parameters(), lr=init_lrate)
        elif optim_type == "adam":
            self.optim_ = optim.Adam(self.model_.parameters(), lr=init_lrate, betas=tconf.adam_betas, eps=tconf.adam_eps)
        else:
            raise NotImplementedError("Unknown optim %s." % optim_type)
        self.prev_lrate_ = init_lrate
        self.grad_clip_ = tconf.grad_clip

    def optimizer_update(self, lrate):
        if self.prev_lrate_ != lrate:
            # schedule lrate, do as lr_scheduler does
            for param_group in self.optim_.param_groups:
                param_group['lr'] = lrate
            self.prev_lrate_ = lrate
        if self.grad_clip_ > 0.:
            clip_grad_norm_(self.model_.parameters(), self.grad_clip_)
        self.optim_.step()
        self.model_.zero_grad()

    def save(self, path):
        torch.save(self.model_.state_dict(), path)

    def load(self, path):
        self.model_.load_state_dict(torch.load(path))

# ===== the functions

# ----- inputs

# (inputs: python data type) -> FloatTensor
def input_real(inputs):
    if is_expr(inputs):
        return inputs
    return torch.tensor(inputs, dtype=torch.float32, device=DEFAULT_DEVICE)

# (inputs: python data type of indexes) -> LongTensor
def input_idx(inputs):
    if is_expr(inputs):
        return inputs
    return torch.tensor(inputs, dtype=torch.long, device=DEFAULT_DEVICE)

# (shape: ..., value: float) -> FloatTensor
def constants(shape, value=0.):
    return torch.full(shape, value, dtype=torch.float32, device=DEFAULT_DEVICE)

#
def constants_idx(shape, value=0):
    return torch.full(shape, value, dtype=torch.long, device=DEFAULT_DEVICE)

# return 2D eye matrix
def eye(n):
    return torch.eye(n, dtype=torch.float32, device=DEFAULT_DEVICE)

#
def arange_idx(*args):
    return torch.arange(*args, dtype=torch.long, device=DEFAULT_DEVICE)

# (shape: ..., p: rate of 1., mul: multiply) -> Tensor
def random_bernoulli(shape, p, mul):
    x = torch.full(shape, p, dtype=torch.float32, device=DEFAULT_DEVICE)
    r = torch.bernoulli(x) * mul
    return r

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
binary_cross_entropy_with_logits = F.binary_cross_entropy_with_logits
clamp = torch.clamp
diagflat = torch.diagflat
elu = F.elu
exp = torch.exp
expand = lambda x, *args: x.expand(*args)
log = torch.log
logsigmoid = F.logsigmoid
logsumexp = torch.logsumexp
max = torch.max         # todo(warn): with dim, return tuple
max_elem = torch.max    # todo(warn): max_elem(a, b)
masked_select = torch.masked_select
matmul = torch.matmul
pad = F.pad
relu = F.relu
reshape = torch.reshape
sigmoid = torch.sigmoid
softmax = F.softmax
squeeze = torch.squeeze
split = torch.split
stack = torch.stack
sum = torch.sum
tanh = torch.tanh
topk = torch.topk
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

def gather(t, idxes, dim=-1):
    idxes_t = input_idx(idxes)
    return torch.gather(t, dim, idxes_t)

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

def backward(loss):
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
class CnnOper(object):
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
