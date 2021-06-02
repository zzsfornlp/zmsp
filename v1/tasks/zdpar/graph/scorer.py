#

# graph decoder

from msp.utils import Conf, zcheck
from msp.nn import BK
from msp.nn.layers import BasicNode, Affine, BiAffineScorer, AttConf, AttDistHelper

#
class GraphScorerConf(Conf):
    def __init__(self):
        self._input_dim = -1        # enc's last dimension
        self._num_label = -1
        # details
        self.arc_space = 512
        self.lab_space = 128
        # scorer
        self.ff_hid_size = 0
        self.ff_hid_layer = 0
        self.use_biaffine = True
        self.use_ff = True
        self.use_ff2 = False
        self.biaffine_div = 1.
        #
        self.transform_act = "elu"
        self.biaffine_init_ortho = False
        # distance clip?
        self.arc_dist_clip = -1
        self.arc_use_neg = False

    def get_dist_aconf(self):
        return AttConf().init_from_kwargs(clip_dist=self.arc_dist_clip, use_neg_dist=self.arc_use_neg)

# [*, len, D] -> attach-score [*, m-len, h-len], label-score [*, m-len, h-len, L]
class GraphScorer(BasicNode):
    def __init__(self, pc: BK.ParamCollection, sconf: GraphScorerConf):
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
        self.arc_dist_clip = sconf.arc_dist_clip
        # scorer
        biaffine_div = sconf.biaffine_div
        biaffine_init_ortho = sconf.biaffine_init_ortho
        transform_act = sconf.transform_act
        # attach/arc
        self.arc_m = self.add_sub_node("am", Affine(pc, input_dim, arc_space, act=transform_act))
        self.arc_h = self.add_sub_node("ah", Affine(pc, input_dim, arc_space, act=transform_act))
        self.arc_scorer = self.add_sub_node("as", BiAffineScorer(pc, arc_space, arc_space, 1, ff_hid_size, ff_hid_layer=ff_hid_layer, use_biaffine=use_biaffine, use_ff=use_ff, use_ff2=use_ff2, biaffine_div=biaffine_div, biaffine_init_ortho=biaffine_init_ortho))
        # only add distance for arc
        if sconf.arc_dist_clip > 0:
            self.dist_helper = self.add_sub_node("dh", AttDistHelper(pc, sconf.get_dist_aconf(), arc_space))
        else:
            self.dist_helper = None
        # labeling
        self.lab_m = self.add_sub_node("lm", Affine(pc, input_dim, lab_space, act=transform_act))
        self.lab_h = self.add_sub_node("lh", Affine(pc, input_dim, lab_space, act=transform_act))
        self.lab_scorer = self.add_sub_node("ls", BiAffineScorer(pc, lab_space, lab_space, sconf._num_label, ff_hid_size, ff_hid_layer=ff_hid_layer, use_biaffine=use_biaffine, use_ff=use_ff, use_ff2=use_ff2, biaffine_div=biaffine_div, biaffine_init_ortho=biaffine_init_ortho))

    # transform to specific space (currently arc/label * m/h)
    # [*, len, input_dim] -> *[*, len, space_dim]
    def transform_space(self, enc_expr):
        am_expr = self.arc_m(enc_expr)
        ah_expr = self.arc_h(enc_expr)
        lm_expr = self.lab_m(enc_expr)
        lh_expr = self.lab_h(enc_expr)
        return am_expr, ah_expr, lm_expr, lh_expr

    # ===== scorers
    # score all pairs: [*, len1, D1], [*, len2, D2], [*, len1], [*, len2] -> [*, len1, len2]
    def score_arc_all(self, am_expr, ah_expr, m_mask_expr, h_mask_expr):
        if self.dist_helper:
            lenm, lenh = BK.get_shape(am_expr, -2), BK.get_shape(ah_expr, -2)
            ah_rel1, _ = self.dist_helper(lenm, lenh)
        else:
            ah_rel1 = None
        arc_scores = self.arc_scorer.paired_score(am_expr, ah_expr, m_mask_expr, h_mask_expr, rel1_t=ah_rel1)
        ret = arc_scores.squeeze(-1)        # squeeze the last one
        return ret

    # score all pairs and all labels: ... -> [*, len1, len2, N]
    def score_label_all(self, lm_expr, lh_expr, m_mask_expr, h_mask_expr):
        lab_scores = self.lab_scorer.paired_score(lm_expr, lh_expr, m_mask_expr, h_mask_expr)
        return lab_scores

    # # score selected paired inputs: [*, len, D1], [*, len, D2], [*, len] -> [*, len]
    # def score_arc_select(self, am_expr_sel, ah_expr_sel, mask):
    #     arc_scores = self.arc_scorer.plain_score(am_expr_sel, ah_expr_sel, mask, None)
    #     ret = arc_scores.squeeze(-1)        # squeeze the last one
    #     return ret

    # score selected paris' all labels: ... -> [*, len, N]
    def score_label_select(self, am_expr_sel, ah_expr_sel, mask):
        lab_scores = self.lab_scorer.plain_score(am_expr_sel, ah_expr_sel, mask, None)
        return lab_scores
