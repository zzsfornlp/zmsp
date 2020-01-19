#

# high-order parsing

from msp.utils import zwarn, zfatal

# todo(note): which version of TurboParser and AD3?
# Updated: now directly use AD3's example and TurboParser
# https://github.com/andre-martins/AD3/commit/22131c7457614dd159546500cd1a0fd8cdf2d282
# https://github.com/andre-martins/TurboParser/commit/a87b8e45694c18b826bb3c42e8344bd32928007d
#
import numpy as np
try:
    from .parser2 import parse2
except:
    zwarn("Cannot find high-order parsing lib, please compile them if needed")
    def parse2(*args, **kwargs):
        raise NotImplementedError("Compile the C++ codes for this one!")

# TODO(WARN): be careful about when there are no trees in the current pruning mask! (especially for proj)
#  and here only do unlabeled parsing, since labeled ones will cost more, handling labels at the outside
# high order parsing decode
# for the *_pack, is an list/tuple of related indexes and scores, None means no such features
def hop_decode(slen: int, projective: bool, o1_masks, o1_scores, o2sib_pack, o2g_pack, o3gsib_pack):
    # dummy arguments, use None will get argument error
    dummy_int_arr = np.array([], dtype=np.int32)
    dummy_double_arr = dummy_int_arr.astype(np.double)
    # prepare inputs
    projective = int(projective)
    assert o1_masks is not None, "Must provide masks"
    if o1_scores is None:  # dummy o1 scores
        o1_scores = np.full([slen, slen], 0., dtype=np.double)
    use_o2sib = int(o2sib_pack is not None)
    use_o2g = int(o2g_pack is not None)
    use_o3gsib = int(o3gsib_pack is not None)
    # o2sib: m, h, s, scores
    if not use_o2sib:
        o2sib_pack = [dummy_int_arr, dummy_int_arr, dummy_int_arr, dummy_double_arr]
    # o2g: m, h, g, scores
    if not use_o2g:
        o2g_pack = [dummy_int_arr, dummy_int_arr, dummy_int_arr, dummy_double_arr]
    # o3gsib: m, h, s, g, scores
    if not use_o3gsib:
        o3gsib_pack = [dummy_int_arr, dummy_int_arr, dummy_int_arr, dummy_int_arr, dummy_double_arr]
    # =====
    # return is vector<int> which becomes a list in py
    ret = parse2(slen, projective, True, o1_masks.reshape(-1), o1_scores.reshape(-1),
                 use_o2sib, *o2sib_pack, use_o2g, *o2g_pack, use_o3gsib, *o3gsib_pack)
    ret[0] = 0  # set ROOT's parent as self
    return ret
