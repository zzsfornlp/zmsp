#

# algorithms for dpar

__all__ = [
    "DparHelper", "DparHelperConf",
]

from mspx.utils import Conf, Configurable, ZHelper, Constants, WithWrapper
from mspx.nn import BK, split_at_dim, log_sum_exp

class DparHelperConf(Conf):
    def __init__(self):
        super().__init__()
        # --
        self.mode = 'nproj'  # nproj, proj, greedy

@DparHelperConf.conf_rd()
class DparHelper(Configurable):
    def __init__(self, conf: DparHelperConf = None, **kwargs):
        super().__init__(conf, **kwargs)
        conf: DparHelperConf = self.conf
        self.set_mode(conf.mode)

    def set_mode(self, mode):
        self.curr_mode = mode

    # temporally changing mode
    def switch_mode_env(self, mode=''):
        if not mode:
            return WithWrapper()
        else:
            old_mode = str(self.curr_mode)
            return WithWrapper((lambda: self.set_mode(mode)), (lambda: self.set_mode(old_mode)))

    def mask2len(self, mask_t: BK.Expr):
        len_t = mask_t.sum(-1).to(BK.DEFAULT_INT)  # [*]
        # check valid seq-mask (starting & continuous)
        mask2 = mask_t * (BK.arange_idx(BK.get_shape(mask_t, -1)) < len_t.unsqueeze(-1)).to(BK.DEFAULT_FLOAT)  # [*, L]
        assert (mask_t != mask2).sum().item() == 0
        return len_t

    # [*, L] -> [*, L, L]
    def make_dpar_mask(self, mask_t: BK.Expr):
        valid_t = mask_t.unsqueeze(-1) * mask_t.unsqueeze(-2)  # [*, L, L]
        valid_t[..., 0, :] = 0.
        valid_t *= (1. - BK.eye(BK.get_shape(valid_t, -1)))
        return valid_t

    # note: multiple root case!
    def _nproj_Lh(self, scores_t: BK.Expr, mask_t: BK.Expr):
        _NEG = Constants.REAL_PRAC_MIN
        # get unlabeled scores and mask invalid values
        _ut = BK.logsumexp(scores_t, -1)  # [*, m, h]
        _shape = BK.get_shape(_ut)
        diag1_m = BK.eye(_shape[-1])  # [m, h]
        valid_t = self.make_dpar_mask(mask_t)
        # _ut = _ut + _NEG * (1. - valid_t)
        _ut = _ut.masked_fill((valid_t == 0), _NEG)
        # minus max-val & exp
        _exp = (_ut - _ut.max(-1, keepdims=True)[0].detach()).exp()  # [*, m, h], note: remember to detach!
        A = _exp * (1. - diag1_m)  # 0 if m==h
        # --
        A_sum = A.sum(-1, keepdim=True) + 1e-6  # [*, m, 1] note: ensure inversable!
        D = A_sum * diag1_m  # [*, m, h]
        L = D - A  # [*, m, h], note: r0 already added to D!
        Lh = L[..., 1:, 1:]
        return Lh

    # --
    # APIs: S[*, L(m), L(h), V], M[*, L]

    # -> [*]
    def get_partition(self, scores_t: BK.Expr, mask_t: BK.Expr):
        _mode = self.curr_mode
        if _mode == 'nproj':
            Lh = self._nproj_Lh(scores_t, mask_t)
            ret = Lh.logdet()
            return ret
        elif _mode == 'proj':
            raise NotImplementedError("TODO")
        elif _mode == 'greedy':  # head selection!
            _emask = self.make_dpar_mask(mask_t).unsqueeze(-1)  # [*, L, L, 1]
            ss = scores_t + (1.-_emask) * Constants.REAL_PRAC_MIN
            _shape = BK.get_shape(ss)
            ret = (log_sum_exp(ss.view(_shape[:2] + [-1]), -1) * mask_t).sum(-1)
            return ret

    # -> [*, L, L, V]
    def get_marginals(self, scores_t: BK.Expr, mask_t: BK.Expr, mode=''):
        _mode = mode if mode else self.curr_mode
        _emask = self.make_dpar_mask(mask_t).unsqueeze(-1)  # [*, L, L, 1]
        # --
        if self.curr_mode == 'greedy':  # do local normalize!
            ss = scores_t + (1. - _emask) * Constants.REAL_PRAC_MIN
            _shape = BK.get_shape(ss)
            pp = ss.view(_shape[:2] + [-1]).log_softmax(-1)
            scores_t = pp.view(_shape)
        # --
        if _mode == 'nproj':
            from mspx.tools.algo.mst import nmarginal_unproj
            ret = nmarginal_unproj(scores_t, mask_t, None)
        elif _mode == 'proj':
            from mspx.tools.algo.mst import nmarginal_proj
            len_t = self.mask2len(mask_t).int()
            ret = nmarginal_proj(scores_t, None, BK.get_value(len_t))
        elif _mode == 'greedy':  # head selection!
            _shape = BK.get_shape(ss)
            pp = ss.view(_shape[:2] + [-1]).softmax(-1)
            ret = pp.view(_shape)
        # --
        ret = ret * _emask
        return ret

    # -> head[*, L], label[*, L]
    def get_argmax(self, scores_t: BK.Expr, mask_t: BK.Expr, mode=''):
        _mode = mode if mode else self.curr_mode
        # --
        if self.curr_mode == 'greedy':  # do local normalize!
            _emask = self.make_dpar_mask(mask_t).unsqueeze(-1)  # [*, L, L, 1]
            ss = scores_t + (1. - _emask) * Constants.REAL_PRAC_MIN
            _shape = BK.get_shape(ss)
            pp = ss.view(_shape[:2] + [-1]).log_softmax(-1)
            scores_t = pp.view(_shape)
        # --
        if _mode == 'nproj':
            from mspx.tools.algo.mst import nmst_unproj
            len_t = self.mask2len(mask_t).int()
            return nmst_unproj(scores_t, None, BK.get_value(len_t))[:2]
        elif _mode == 'proj':
            from mspx.tools.algo.mst import nmst_proj
            len_t = self.mask2len(mask_t).int()
            return nmst_proj(scores_t, None, BK.get_value(len_t))[:2]
        elif _mode == 'greedy':
            from mspx.tools.algo.mst import nmst_greedy
            return nmst_greedy(scores_t, None, None)[:2]
