#

# special ones

__all__ = [
    "TeeGPConf", "TeeGPNode", "TeeFunction",
    "PCGradHelperConf", "PCGradHelperNode", "OptimSparseHelperConf", "OptimSparseHelperNode",
]

from typing import Dict, List
import numpy as np
from ..backends import BK
from .base import *
from msp2.utils import Random, zlog

# --
class TeeGPConf(BasicConf):
    def __init__(self):
        super().__init__()
        self._isize = -1
        # --
        # GP: turned off if both 0.
        self.change_rate0 = 0.  # how much to change for grad0, by default always change grad0 to grad1
        self.change_rate1 = 0.
        self.batch_dim = 0  # how many dims to regard as batch
        # split range: >0 for [:dim*s], <0 for [dim*-s:]
        self.split0 = 1.
        self.split1 = 1.

# --
class TeeFunction(BK.Function):
    @staticmethod
    def forward(ctx, x, conf: TeeGPConf):
        # simply return two pieces
        ctx._saved_conf = conf
        return x.clone(), x.clone()

    @staticmethod
    def backward(ctx, grad0, grad1):
        conf: TeeGPConf = ctx._saved_conf
        _change_rate0, _change_rate1, _batch_dim = conf.change_rate0, conf.change_rate1, conf.batch_dim
        # --
        # breakpoint()
        # --
        shape = BK.get_shape(grad0)
        size0 = int(np.prod(shape[:_batch_dim]))
        size1 = int(np.prod(shape[_batch_dim:]))
        r_shape = [size0, size1]
        r_grad0, r_grad1 = grad0.view(r_shape), grad1.view(r_shape)  # [s0, s1]
        _dot = (r_grad0 * r_grad1).sum(-1, keepdim=True)  # [s0, 1]
        # --
        final_grads = []
        # --
        for curr, other, rate in zip([r_grad0, r_grad1], [r_grad1, r_grad0], [_change_rate0, _change_rate1]):
            if rate > 0.:
                _other_s2 = (other * other).sum(-1, keepdims=True)  # [s0, 1]
                _offset = (_dot / _other_s2) * other  # [s0, s1] => (gi.gj)/(gj.gj) * gj
                # _full_trg = curr - (_dot<0).float() * _offset  # [s0, s1]
                # r_final = rate * _full_trg + (1.-rate) * curr  # [s0, s1], simply linear combination
                r_final = curr - rate * ((_dot<0).float() * _offset)  # [s0, s1]
            else:
                r_final = curr
            final_grads.append(r_final.view(shape))
        # --
        # breakpoint()
        ret = final_grads[0] + final_grads[1]
        return ret, None
# --

@node_reg(TeeGPConf)
class TeeGPNode(BasicNode):
    def __init__(self, conf: TeeGPConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: TeeGPConf = self.conf
        # --
        self.apply_gp = (conf.change_rate0>0) or (conf.change_rate1>0)
        # split mask
        self._mask0 = self._get_split_mask(conf._isize, conf.split0)
        self._mask1 = self._get_split_mask(conf._isize, conf.split1)

    def _get_split_mask(self, s: int, r: float):
        if r>=1. or r<=-1.:
            return 1.
        elif r>=0:
            return (BK.arange_idx(s) < int(s*r)).float()
        else:
            return (BK.arange_idx(s) >= int(s*(1+r))).float()

    def forward(self, x):
        # gp
        if self.apply_gp:
            a, b = TeeFunction.apply(x, self.conf)  # one2two!
        else:
            a, b = x, x
        # split
        ret0, ret1 = a*self._mask0, b*self._mask1
        return ret0, ret1
    # --

# ==
# PCGradHelper
# adapted from: https://github.com/WeiChengTseng/Pytorch-PCGrad/blob/master/pcgrad.py and https://github.com/tianheyu927/PCGrad/blob/master/PCGrad_tf.py and https://github.com/wgchang/PCGrad-pytorch-example/blob/master/pcgrad-example.py

class PCGradHelperConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self.conflicting_loss_names = []  # the names that need to especially resolve grads, others simply add!
        self.conflicting_change_rates = []  # N*N: [curr, other], for example, [0,1,0.1,0] will mean loss0->(1)loss1, loss1->(0.1)loss0
        self.shuffle_losses = False  # whether shuffle?
        self.exclude_emb = False
        # grad drop?
        self.graddrop_whole = []
        self.graddrop_partial = []

@node_reg(PCGradHelperConf)
class PCGradHelperNode(BasicNode):
    def __init__(self, conf: PCGradHelperConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: PCGradHelperConf = self.conf
        # --
        self.conflicting_loss_name_set = set(conf.conflicting_loss_names)
        _num_closs = len(conf.conflicting_loss_names)
        assert _num_closs >= 2, "Not enough losses!"
        _tmp_rates = [float(z) for z in conf.conflicting_change_rates]
        self.conflicting_change_rates = [_tmp_rates[i*_num_closs:(i+1)*_num_closs] for i in range(_num_closs)]
        # grad drop?
        self.graddrop_whole = [float(z) for z in conf.graddrop_whole] + [0.] * _num_closs  # pad 0.!
        self.graddrop_partial = [float(z) for z in conf.graddrop_partial] + [0.] * _num_closs  # pad 0.!
        # --

    def _zero_grads(self, params):
        for p in params:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        # --

    def _get_grads(self, params, flatten: bool, drop_whole=0., drop_partial=0.):
        grads = [p.grad.detach().clone() if p.grad is not None else BK.zeros(p.shape) for p in params]
        if drop_whole > 0.:
            _gen = Random.get_generator('loss')
            _mask = (_gen.random(len(grads)) < drop_whole)
            grads = [(g*0. if m else g) for g,m in zip(grads, _mask)]
        if drop_partial > 0.:
            grads = [g*(BK.rand(g.shape)<drop_partial).float() for g in grads]
        if flatten:
            return BK.concat([z.flatten() for z in grads], 0)
        else:
            return grads
        # --

    def _add_grads(self, trg_list, src, flatten: bool):
        if flatten:
            cur_idx = 0
            for t in trg_list:
                _numel = t.numel()
                t.add_(src[cur_idx:cur_idx+_numel].view(t.shape))
                cur_idx += _numel
            assert cur_idx == len(src)
        else:
            assert len(trg_list) == len(src)
            for t, s in zip(trg_list, src):
                t.add_(s)
        # --

    def _proj_grads(self, flattened_grads):
        _shuffle = self.conf.shuffle_losses
        if _shuffle:
            _gen = Random.get_generator('loss')
        _rates = self.conflicting_change_rates
        # --
        all_g = []
        for i, cur_g in enumerate(flattened_grads):
            new_g = cur_g.clone()
            other_idxes = list(range(len(flattened_grads)))
            if _shuffle:
                _gen.shuffle(other_idxes)
            for j in other_idxes:
                other_g = flattened_grads[j]
                rate = _rates[i][j]
                if rate>0.:
                    _dot = (new_g * other_g).sum()
                    _other_s2 = (other_g * other_g).sum()
                    _offset = (_dot / _other_s2) * other_g
                    new_g.sub_(rate * ((_dot < 0).float() * _offset))
                    # -- just checking!
                    if BK.get_value(_dot).item() < 0:
                        zlog(f"Here! _dot<0 as _dot={_dot}, _off={_dot / _other_s2}")
                    # --
            all_g.append(new_g)
        ret = BK.stack(all_g, 0).sum(0)  # [*]
        return ret

    def do_backward(self, parameters, loss_dict: Dict, loss_factor: float):
        conf: PCGradHelperConf = self.conf
        # --
        # check conflicting losses
        conflicting_losses = [loss_dict[k] for k in conf.conflicting_loss_names]  # gather the special ones
        remaining_losses = [v for k,v in loss_dict.items() if k not in self.conflicting_loss_name_set]
        remaining_loss = BK.stack(remaining_losses).sum() if len(remaining_losses)>0 else None  # remaining ones
        # --
        # store the original grads (for example, if we are accumulating grads)
        param_list = list(p for p in parameters if p.requires_grad)  # keep the ones that requires grads
        if conf.exclude_emb:
            _max_numel = max(p.numel() for p in param_list)
            param_list = [p for p in param_list if p.numel()<_max_numel]  # note: usually the biggest one is emb!!
        all_final_grads = self._get_grads(param_list, False)
        # --
        # backward the losses
        retain_graph_count = len(conflicting_losses) + (1 if remaining_loss is not None else 0)  # how many backwards needed?
        flattened_grads = []  # flattened ones!
        for one_ii, one_loss in enumerate(conflicting_losses):
            # zero grad and backward
            self._zero_grads(param_list)
            BK.backward(one_loss, loss_factor=loss_factor, retain_graph=(retain_graph_count>1))
            retain_graph_count -= 1
            # collect grad
            one_flat_grads = self._get_grads(
                param_list, True, drop_whole=self.graddrop_whole[one_ii], drop_partial=self.graddrop_partial[one_ii])  # [*]
            flattened_grads.append(one_flat_grads)
        # do grad projection and add them
        final_proj_grad = self._proj_grads(flattened_grads)
        self._add_grads(all_final_grads, final_proj_grad, True)
        # the final extra one, if there are
        if remaining_loss is not None:
            self._zero_grads(param_list)
            BK.backward(remaining_loss, loss_factor=loss_factor, retain_graph=False)
            one_grads = self._get_grads(param_list, False)
            self._add_grads(all_final_grads, one_grads, False)  # directly add it!
        # finally assign grads back
        for p, g in zip(param_list, all_final_grads):
            p.grad = g
        return

# =====
class OptimSparseHelperConf(BasicConf):
    def __init__(self):
        super().__init__()
        # --
        self.gs_lambdas = []  # gs-lambda for each layer
        self.gs_group_dim = 0  # ...
        self.es_lambdas = []  # es-lambda for each layer
        self.es_group_dim = 0  # sum(group) over which dim? [0=Out, 1=In]

@node_reg(OptimSparseHelperConf)
class OptimSparseHelperNode(BasicNode):
    def __init__(self, conf: OptimSparseHelperConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: OptimSparseHelperConf = self.conf
        # --
        # note: by default 0. for these lambdas!
        self.gs_lambdas = [float(z) for z in conf.gs_lambdas] + [0.]*100
        self.es_lambdas = [float(z) for z in conf.es_lambdas] + [0.]*100

    def do_update(self, layered_parameters: List, lrate: float):
        with BK.no_grad_env():
            self._do_update(layered_parameters, lrate)

    def _do_update(self, layered_parameters: List, lrate: float):
        _gs_group_dim = self.conf.gs_group_dim
        _es_group_dim = self.conf.es_group_dim
        # --
        # adopted from https://github.com/jaehong-yoon93/CGES/blob/master/main.py
        for lidx, params in enumerate(layered_parameters):
            gs_lambda = self.gs_lambdas[lidx]
            es_lambda = self.es_lambdas[lidx]
            for W in params:  # [Out, In]
                if len(BK.get_shape(W)) != 2:
                    continue  # only update Ws
                # GS
                if gs_lambda>0.:
                    W_sum = (W*W).sum(dim=_gs_group_dim, keepdims=True)
                    W_rsqrt = BK.rsqrt(W_sum + 1e-10)
                    gl_plus = BK.relu(1. - (gs_lambda * lrate) * W_rsqrt)
                    W.set_(gl_plus * W)
                # ES
                if es_lambda>0.:
                    W_sign = BK.sign(W)
                    W_abs = W.abs()
                    W_sum = W_abs.sum(dim=_es_group_dim, keepdims=True)
                    el_plus = BK.relu(W_abs - (es_lambda * lrate) * W_sum)
                    W.set_(W_sign * el_plus)
        # --

# --
# b msp2/nn/layers/special.py:152
