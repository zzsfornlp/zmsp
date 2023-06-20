#

# some algorithms that can implemented by directly manipulating tensors (possibly with GPUs)
# all accept and return Tensors: BK.is_expr=True.; unless otherwise provide return_arr=True.
# (no need to converted to np.array and calculate with CPU)
# todo(warn): the arguments are different than the CPU versions!

import math
from mspx.nn import BK
from mspx.utils import Constants, zwarn

# todo(warn): specially differentiate nmst and mst, accepting and returning different things (tensor vs. arr)
try:
    from .cmst import cmst_unproj as mst_unproj
    from .cmst import cmst_proj as mst_proj
    from .cmst import cmst_greedy as mst_greedy
    from .cmst import cmarginal_unproj as marginal_unproj
    from .cmst import cmarginal_proj as marginal_proj
    from .cmst import cmarginal_greedy as marginal_greedy
except:
    zwarn("cython version of MST has not been compiled, use python version instead!")
    from .mst import mst_unproj, mst_proj, mst_greedy
    from .mst import marginal_unproj, marginal_proj, marginal_greedy

# =====
# algorithm wrappers

# todo(+1): simple for unlabeled situation
def _common_nmst(CPU_f, scores_expr, mask_expr, lengths_arr, labeled, ret_arr):
    assert labeled
    with BK.no_grad_env():
        # argmax-label: [BS, m, h]
        scores_unlabeled_max, labels_argmax = scores_expr.max(-1)
        #
        scores_unlabeled_max_arr = BK.get_value(scores_unlabeled_max)
        mst_heads_arr, _, mst_scores_arr = CPU_f(scores_unlabeled_max_arr, lengths_arr, labeled=False)
        # [BS, m]
        mst_heads_expr = BK.input_idx(mst_heads_arr)
        # mst_labels_expr = BK.gather_one_lastdim(labels_argmax, mst_heads_expr).squeeze(-1)
        mst_labels_expr = labels_argmax.gather(-1, mst_heads_expr.unsqueeze(-1)).squeeze(-1)
        # prepare for the outputs
        if ret_arr:
            return mst_heads_arr, BK.get_value(mst_labels_expr), mst_scores_arr
        else:
            return mst_heads_expr, mst_labels_expr, BK.input_real(mst_scores_arr)

#
def nmst_unproj(scores_expr, mask_expr, lengths_arr, labeled=True, ret_arr=False):
    return _common_nmst(mst_unproj, scores_expr, mask_expr, lengths_arr, labeled, ret_arr)

def nmst_proj(scores_expr, mask_expr, lengths_arr, labeled=True, ret_arr=False):
    return _common_nmst(mst_proj, scores_expr, mask_expr, lengths_arr, labeled, ret_arr)

# [BS, Len, Len, L], [BS, Len] -> [BS, Len]
# todo(warn): assume the inputs' unmasked entries have already been masked with small values
# todo(+1): simple for unlabeled situation
def nmst_greedy(scores_expr, mask_expr, lengths_arr, labeled=True, ret_arr=False):
    assert labeled
    with BK.no_grad_env():
        scores_shape = BK.get_shape(scores_expr)
        maxlen = scores_shape[1]
        # mask out diag
        scores_expr += BK.diagflat(BK.constants([maxlen], Constants.REAL_PRAC_MIN)).unsqueeze(-1)
        # combined last two dimension and Max over them
        combined_scores_expr = scores_expr.view(scores_shape[:-2] + [-1])
        combine_max_scores, combined_max_idxes = BK.max_d(combined_scores_expr, dim=-1)
        # back to real idxes
        last_size = scores_shape[-1]
        greedy_heads = combined_max_idxes // last_size
        greedy_labels = combined_max_idxes % last_size
        if ret_arr:
            mst_heads_arr, mst_labels_arr, mst_scores_arr = [BK.get_value(z) for z in (greedy_heads, greedy_labels, combine_max_scores)]
            return mst_heads_arr, mst_labels_arr, mst_scores_arr
        else:
            return greedy_heads, greedy_labels, combine_max_scores

#
def nmarginal_greedy(scores_expr, mask_expr, lengths_arr, labeled=True):
    raise NotImplementedError("No implementation (no usage) for this one!")

# ==
# todo(warn): for debugging
last_lm00 = None
last_marginals = None
# ==

# float[BS, Len, Len, L]
# todo(warn): ensure norm at each m, in case of numerical issues or the situation of no solutions
def _ensure_margins_norm(marginals_expr):
    full_shape = BK.get_shape(marginals_expr)
    combined_marginals_expr = marginals_expr.view(full_shape[:-2] + [-1])       # [BS, Len, Len*L]
    # should be 1., but for no-solution situation there can be small values (+1 for all in this case, norm later)
    # make 0./0. = 0.
    combined_marginals_sum = combined_marginals_expr.sum(dim=-1, keepdim=True)
    combined_marginals_sum += (combined_marginals_sum < 1e-5).float() * 1e-5
    # then norm
    combined_marginals_expr /= combined_marginals_sum
    return combined_marginals_expr.view(full_shape)

# [BS, Len, Len, L], [BS, Len] -> [BS, Len, Len, L]
# Matrix-Tree Theorem: elegant algorithm, only remaining un-cycled ones and canceling cycled ones
# For each subgraph-config, #cycle=0: C00 = 1; #cycle>0: Cn0 - Cn1 + Cn2 ... Cnn = 0
# todo(warn): assume the inputs' unmasked entries have already been masked with small values (exp(z)=0.)
# todo(warn): also the order of dimension is slightly different than the original algorithm, here we have [m, h]
# todo(+1): simple for unlabeled situation
# todo(warn): be careful about Numerical Unstability when the matrix is not inversable, which will make it 0/0!!
# todo(note): mask out non-valid values (diag, padding, root-mod), need to be careful about this?
def nmarginal_unproj(scores_expr, mask_expr, lengths_arr, labeled=True):
    assert labeled
    with BK.no_grad_env():
        scores_shape = BK.get_shape(scores_expr)
        maxlen = scores_shape[1]
        # todo(warn): it seems that float32 is not enough for inverse when the model gets better (scores gets more diversed)
        diag1_m = BK.eye(maxlen).double()        # [m, h]
        scores_expr_d = scores_expr.double()
        mask_expr_d = mask_expr.double()
        invalid_pad_expr_d = 1.-mask_expr_d
        # [*, m, h]
        full_invalid_d = (diag1_m + invalid_pad_expr_d.unsqueeze(-1) + invalid_pad_expr_d.unsqueeze(-2)).clamp(0., 1.)
        full_invalid_d[:, 0] = 1.
        #
        # first make it unlabeled by sum-exp
        scores_unlabeled = BK.logsumexp(scores_expr_d, dim=-1)    # [BS, m, h]
        # force small values at diag entries and padded ones
        scores_unlabeled_diag_neg = scores_unlabeled + Constants.REAL_PRAC_MIN * full_invalid_d
        # # minus the MaxElement to make it more stable with larger values, to make it numerically stable.
        # [BS, m, h]
        # todo(+N): since one and only one Head is selected, thus minus by Max will make it the same?
        #  I think it will be canceled out since this is like left-mul A by a diag Q
        scores_unlabeled_max = (scores_unlabeled_diag_neg.max(-1)[0] * mask_expr_d).unsqueeze(-1)   # [BS, m, 1]
        scores_exp_unlabeled = BK.exp(scores_unlabeled_diag_neg - scores_unlabeled_max)
        # # todo(0): co-work with minus-max, force too small values to be 0 (serve as pruning, the gap is ?*ln(10)).
        # scores_exp_unlabeled *= (1 - (scores_exp_unlabeled<1e-10)).double()
        # force 0 at diag entries (again)
        scores_exp_unlabeled *= (1. - diag1_m)
        # assign non-zero values (does not matter) to (0, invalid) to make the matrix inversable
        scores_exp_unlabeled[:, :, 0] += (1. - mask_expr_d)      # the value does not matter?
        # construct L(or K) Matrix: L=D-A
        A = scores_exp_unlabeled
        A_sum = A.sum(dim=-1, keepdim=True)                 # [BS, m, 1]
        # # =====
        # todo(0): can this avoid singular matrix: feels like adding aug-values to h==0(COL0) to-root scores.
        # todo(+N): there are cases that the original matrix is not inversable (no solutions for trees)!!
        A_sum += 1e-6
        # A_sum += A_sum * 1e-4 + 1e-6
        #
        D = A_sum.expand(scores_shape[:-1])*diag1_m         # [BS, m, h]
        L = D - A                                           # [BS, m, h]
        # get the minor00 matrix
        LM00 = L[:, 1:, 1:]          # [BS, m-1, h-1]
        # # Debug1
        # try:
        #     # for idx in range(scores_shape[0]):
        #     #         one_det = float(LM00[idx].det())
        #     #         assert not math.isnan(one_det)
        #     #     LM00_CPU = LM00.cpu()
        #     #     LM00_CPU_inv = LM00_CPU.inverse()
        #     scores_exp_unlabeled_CPU = scores_exp_unlabeled.cpu()
        #     LM00_CPU = LM00.cpu()
        #     assert BK.has_nan(LM00_CPU) == 0
        # except:
        #     assert False, "Problem here"
        #
        # det and inverse; using LU decomposition to hit two birds with one stone.
        diag1_m00 = BK.eye(maxlen-1).double()
        # deprecated operation
        # LM00_inv, LM00_lu = diag1_m00.gesv(LM00)                # [BS, m-1, h-1]
        # # todo(warn): lacking P here, but the partition should always be non-negative!
        # LM00_det = BK.abs((LM00_lu*diag1_m00).sum(-1).prod(-1))         # [BS, ]
        # d(logZ)/d(LM00) = (LM00^-1)^T
        # # directly inverse (need pytorch >= 1.0)
        LM00_inv = LM00.inverse()
        # LM00_inv = BK.get_inverse(LM00, diag1_m00)
        LM00_grad = LM00_inv.transpose(-1, -2)              # [BS, m-1, h-1]
        # marginal(m,h) = d(logZ)/d(score(m,h)) = d(logZ)/d(LM00) * d(LM00)/d(score(m,h)) = INV_mm - INV_mh
        # padding and minus
        LM00_grad_pad = BK.pad(LM00_grad, [1,0,1,0], 'constant', 0.)                    # [BS, m, h]
        LM00_grad_pad_sum = (LM00_grad_pad * diag1_m).sum(dim=-1, keepdim=True)     # [BS, m, 1]
        marginals_unlabeled = A * (LM00_grad_pad_sum - LM00_grad_pad)                         # [BS, m, h]
        # make sure each row sum to 1.
        marginals_unlabeled[:, 0, 0] = 1.
        # finally, get labeled results
        marginals_labeled = marginals_unlabeled.unsqueeze(-1) * BK.exp(scores_expr_d - scores_unlabeled.unsqueeze(-1))
        #
        # # Debug2
        # try:
        #     # for idx in range(scores_shape[0]):
        #     #         one_det = float(LM00[idx].det())
        #     #         assert not math.isnan(one_det)
        #     #     LM00_CPU = LM00.cpu()
        #     #     LM00_CPU_inv = LM00_CPU.inverse()
        #     scores_exp_unlabeled_CPU = scores_exp_unlabeled.cpu()
        #     LM00_CPU = LM00.cpu()
        #     marginals_unlabeled_CPU = marginals_unlabeled.cpu()
        #     assert BK.has_nan(marginals_unlabeled_CPU) == 0
        #     #
        #     global last_lm00, last_marginals
        #     last_lm00 = LM00_CPU
        #     last_marginals = marginals_unlabeled_CPU
        # except:
        #     assert False, "Problem here"
        #
        # back to plain float32
        masked_marginals_labeled = marginals_labeled * (1.-full_invalid_d).unsqueeze(-1)
        ret = masked_marginals_labeled.float()
        return _ensure_margins_norm(ret)

# b tasks/zdpar/algo/nmst:102

# [BS, Len, Len, L], [BS, Len] -> [BS, Len, Len, L]
# use CPU's UNLABELED dynamic-programming inside-outside algorithm
# todo(+1): simple for unlabeled situation
# todo(warn): outside is similar to unproj, but do not need that much masks here,
#  since most are handled well in the CPU algorithm
def nmarginal_proj(scores_expr, mask_expr, lengths_arr, labeled=True):
    assert labeled
    with BK.no_grad_env():
        # first make it unlabeled by sum-exp
        scores_unlabeled = BK.logsumexp(scores_expr, dim=-1)  # [BS, m, h]
        # marginal for unlabeled
        scores_unlabeled_arr = BK.get_value(scores_unlabeled)
        marginals_unlabeled_arr = marginal_proj(scores_unlabeled_arr, lengths_arr, False)
        # back to labeled values
        marginals_unlabeled_expr = BK.input_real(marginals_unlabeled_arr)
        marginals_labeled_expr = marginals_unlabeled_expr.unsqueeze(-1) * BK.exp(scores_expr - scores_unlabeled.unsqueeze(-1))
        # [BS, m, h, L]
        return _ensure_margins_norm(marginals_labeled_expr)
