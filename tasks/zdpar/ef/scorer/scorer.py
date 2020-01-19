#

from msp.utils import Conf, Helper
from msp.nn import BK
from msp.nn.layers import BasicNode, Affine, BiAffineScorer, AttConf

# -----
# scorer
# the scorer only cares about giving (raw) scores for the head-mod pairs
# all the neural model parameters belongs here!

# -----
# confs

# the head-mod pair scorer
class ScorerConf(Conf):
    def __init__(self):
        self._input_dim = -1  # enc's (input) last dimension
        self._num_label = -1  # number of labels
        # space transferring
        self.arc_space = 512  # arc_space==0 means no_arc_score
        self.lab_space = 128
        # -----
        # final biaffine scoring
        self.transform_act = "elu"
        self.ff_hid_size = 0
        self.ff_hid_layer = 0
        self.use_biaffine = True
        self.use_ff = True
        self.use_ff2 = False
        self.biaffine_div = 1.
        self.biaffine_init_ortho = False
        # distance clip?
        self.arc_dist_clip = -1
        self.arc_use_neg = False

    def get_dist_aconf(self):
        return AttConf().init_from_kwargs(clip_dist=self.arc_dist_clip, use_neg_dist=self.arc_use_neg)

# todo(WARN): about no_arc_score (deprecated and deleted)
"""
Actually, after further thinking, arc+label score is a reasonable choice, since if there are only label-score, then there seem to be no param-sharing things in the final scoring. Actually, on PTB, NoPOS/arc=0 is hard to converge with default hp (starting to be reasonable after Epoch-70+), although +POS it converges.  Therefore, when parameterizing NN, think about the intuition and make it easier to train with proper param-sharing.
-> My guess for the non-convergence is that it is hard to learn localness from purely lexicalized input; arc-scorer may be easier to learn localness since most arcs are local, POS can also provide helpful info on this aspect, maybe.
"""

# -----
# actual model components

# scoring for pairs of head-mod on structured enc repr
# orig-enc -> s-enc -> h/m-cache
class Scorer(BasicNode):
    def __init__(self, pc: BK.ParamCollection, sconf: ScorerConf):
        super().__init__(pc, None, None)
        # options
        input_dim = sconf._input_dim
        arc_space = sconf.arc_space
        lab_space = sconf.lab_space
        ff_hid_size = sconf.ff_hid_size
        ff_hid_layer = sconf.ff_hid_layer
        use_biaffine = sconf.use_biaffine
        use_ff = sconf.use_ff
        use_ff2 = sconf.use_ff2
        biaffine_div = sconf.biaffine_div
        biaffine_init_ortho = sconf.biaffine_init_ortho
        transform_act = sconf.transform_act
        #
        self.input_dim = input_dim
        self.num_label = sconf._num_label
        # attach/arc
        self.arc_m = self.add_sub_node("am", Affine(pc, input_dim, arc_space, act=transform_act))
        self.arc_h = self.add_sub_node("ah", Affine(pc, input_dim, arc_space, act=transform_act))
        self.arc_scorer = self.add_sub_node(
            "as", BiAffineScorer(pc, arc_space, arc_space, 1, ff_hid_size, ff_hid_layer=ff_hid_layer,
                                 use_biaffine=use_biaffine, use_ff=use_ff, use_ff2=use_ff2,
                                 biaffine_div=biaffine_div, biaffine_init_ortho=biaffine_init_ortho))
        # only add distance for arc
        if sconf.arc_dist_clip > 0:
            # todo(+N): how to include dist feature?
            # self.dist_helper = self.add_sub_node("dh", AttDistHelper(pc, sconf.get_dist_aconf(), arc_space))
            self.dist_helper = None
            raise NotImplemented("TODO")
        else:
            self.dist_helper = None
        # labeling
        self.lab_m = self.add_sub_node("lm", Affine(pc, input_dim, lab_space, act=transform_act))
        self.lab_h = self.add_sub_node("lh", Affine(pc, input_dim, lab_space, act=transform_act))
        self.lab_scorer = self.add_sub_node(
            "ls", BiAffineScorer(pc, lab_space, lab_space, self.num_label, ff_hid_size, ff_hid_layer=ff_hid_layer,
                                 use_biaffine=use_biaffine, use_ff=use_ff, use_ff2=use_ff2,
                                 biaffine_div=biaffine_div, biaffine_init_ortho=biaffine_init_ortho))

    # get separate params of MLP and scorer
    def get_split_params(self):
        params0 = Helper.join_list(z.get_parameters() for z in [self.arc_m, self.arc_h, self.lab_m, self.lab_h])
        params1 = Helper.join_list(z.get_parameters() for z in [self.arc_scorer, self.lab_scorer])
        return params0, params1

    # transform to specific space (+pre-computation for mod) (currently arc/label * m/h)
    # [*, input_dim] -> *[*, space_dim]

    def transform_space_arc(self, senc_expr, calc_h: bool=True, calc_m: bool=True):
        # for head
        if calc_h:
            ah_expr = self.arc_h(senc_expr)
        else:
            ah_expr = None
        # for mod
        if calc_m:
            am_expr = self.arc_m(senc_expr)
            am_pack = self.arc_scorer.precompute_input0(am_expr)
        else:
            am_pack = None
        return ah_expr, am_pack

    def transform_space_label(self, senc_expr, calc_h: bool=True, calc_m: bool=True):
        # for arc
        if calc_h:
            lh_expr = self.lab_h(senc_expr)
        else:
            lh_expr = None
        # for mod
        if calc_m:
            lm_expr = self.lab_m(senc_expr)
            lm_pack = self.lab_scorer.precompute_input0(lm_expr)
        else:
            lm_pack = None
        return lh_expr, lm_pack

    # =====
    # separate scores

    def score_arc(self, am_pack, ah_expr, mask_m=None, mask_h=None):
        return self.arc_scorer.postcompute_input1(am_pack, ah_expr, mask0=mask_m, mask1=mask_h)

    def score_label(self, lm_pack, lh_expr, mask_m=None, mask_h=None):
        return self.lab_scorer.postcompute_input1(lm_pack, lh_expr, mask0=mask_m, mask1=mask_h)

    # =====
    # special mode for first order full score

    def transform_and_arc_score(self, senc_expr, mask_expr=None):
        ah_expr = self.arc_h(senc_expr)
        am_expr = self.arc_m(senc_expr)
        arc_full_score = self.arc_scorer.paired_score(am_expr, ah_expr, mask_expr, mask_expr)
        return arc_full_score

    def transform_and_label_score(self, senc_expr, mask_expr=None):
        lh_expr = self.lab_h(senc_expr)
        lm_expr = self.lab_m(senc_expr)
        lab_full_score = self.lab_scorer.paired_score(lm_expr, lh_expr, mask_expr, mask_expr)
        return lab_full_score

    # =====
    # plain mode for scoring

    def transform_and_arc_score_plain(self, mod_srepr, head_srepr, mask_expr=None):
        ah_expr = self.arc_h(head_srepr)
        am_expr = self.arc_m(mod_srepr)
        arc_full_score = self.arc_scorer.plain_score(am_expr, ah_expr, mask_expr, mask_expr)
        return arc_full_score

    def transform_and_label_score_plain(self, mod_srepr, head_srepr, mask_expr=None):
        lh_expr = self.lab_h(head_srepr)
        lm_expr = self.lab_m(mod_srepr)
        lab_full_score = self.lab_scorer.plain_score(lm_expr, lh_expr, mask_expr, mask_expr)
        return lab_full_score
