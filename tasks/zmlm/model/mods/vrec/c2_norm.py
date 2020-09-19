#

# Component 2 for MAtt: normer
# (scores [*, len_q, len_k, head], accu_attns) => normalized-scores [*, len_q, len_k, head]

from typing import List, Union, Dict, Iterable
from msp.utils import Constants, Conf, zwarn, Helper
from msp.nn import BK
from msp.nn.layers import BasicNode, ActivationHelper, Affine, NoDropRop, NoFixRop, Dropout, PosiEmbedding2, LayerNorm
from .base import ConcreteNodeConf, ConcreteNode

# -----

# conf
class MAttNormerConf(Conf):
    def __init__(self):
        # special
        # self.param_init_scale = 1.  # actually no params in this node
        # noop compare score (fixed)
        self.use_noop = False  # attend to nothing
        self.noop_fixed_val = 0.  # fixed noop score
        # norm
        # todo(+N): here no longer norm multiple times since that seems complicated
        self.norm_mode = 'cand'  # flatten/head/cand/head_cand/binary
        self.norm_prune = 0.  # prune final normalized scores to 0. if <=this
        # concrete
        self.cconf = ConcreteNodeConf()
        # dropout directly on the normalized scores
        self.attn_dropout = 0.

# for getting normalized scores (like probs)
class MAttNormerNode(BasicNode):
    def __init__(self, pc: BK.ParamCollection, head_count, conf: MAttNormerConf):
        super().__init__(pc, None, None)
        self.conf = conf
        self.head_count = head_count
        # -----
        self._norm_f = getattr(self, "_norm_"+conf.norm_mode)  # shortcut
        self._norm_dims = {'flatten': [-1], 'head': [-1], 'cand': [-2], 'head_cand': [-1, -2], 'binary': [-1]}[conf.norm_mode]
        self.norm_prune = conf.norm_prune
        # cnode: special attention
        self.cnode = self.add_sub_node("cn", ConcreteNode(pc, conf.cconf))
        # attention dropout: # no-fix attention dropout, not elegant here
        rr = NoFixRop()
        self.adrop = self.add_sub_node("adrop", Dropout(pc, (), init_rop=rr, fix_rate=conf.attn_dropout))

    # =====
    # different strategies to calculate prob matrix

    @property
    def norm_dims(self):
        return self._norm_dims

    # helper function for noop
    def _normalize(self, cnode: ConcreteNode, orig_scores, use_noop: bool, noop_fixed_val: float, temperature: float, dim: int):
        cur_shape = BK.get_shape(orig_scores)  # original
        orig_that_dim = cur_shape[dim]
        cur_shape[dim] = 1
        if use_noop:
            noop_scores = BK.constants(cur_shape, value=noop_fixed_val)  # [*, 1, *]
            to_norm_scores = BK.concat([orig_scores, noop_scores], dim=dim)  # [*, D+1, *]
        else:
            to_norm_scores = orig_scores  # [*, D, *]
        # normalize
        prob_full = cnode(to_norm_scores, temperature=temperature, dim=dim)  # [*, ?, *]
        if use_noop:
            prob_valid, prob_noop = BK.split(prob_full, [orig_that_dim, 1], dim)  # [*, D|1, *]
        else:
            prob_valid, prob_noop = prob_full, None
        return prob_valid, prob_noop, prob_full

    # 1. direct normalize over flattened [len_k, head]
    def _norm_flatten(self, scores, temperature):
        conf = self.conf
        cnode = self.cnode
        use_noop, noop_fixed_val = conf.use_noop, conf.noop_fixed_val
        # -----
        orig_shape = BK.get_shape(scores)
        prob_valid, prob_noop, prob_full = self._normalize(
            cnode, scores.view(orig_shape[:-2]+[-1]), use_noop, noop_fixed_val, temperature, -1)
        attn = prob_valid.view(orig_shape)  # back to 2d shape
        # [*, len_q, len_k, head], [*, len_q, len_k*head], [*, len_q, ?], [*, len_q, len_k*head+?]
        return attn, [prob_valid], [prob_noop], [prob_full], [-1]

    # 2. norm on head-dim for each cand
    def _norm_head(self, scores, temperature):
        conf = self.conf
        cnode = self.cnode
        use_noop, noop_fixed_val = conf.use_noop, conf.noop_fixed_val
        # -----
        prob_valid, prob_noop, prob_full = self._normalize(cnode, scores, use_noop, noop_fixed_val, temperature, -1)
        # [*, len_q, len_k, head], [*, len_q, len_k, head], [*, len_q, len_k, ?], [*, len_q, len_k, head+?]
        return prob_valid, [prob_valid], [prob_noop], [prob_full], [-1]

    # 3. norm on cand-dim for each head
    def _norm_cand(self, scores, temperature):
        conf = self.conf
        cnode = self.cnode
        use_noop, noop_fixed_val = conf.use_noop, conf.noop_fixed_val
        # -----
        prob_valid, prob_noop, prob_full = self._normalize(cnode, scores, use_noop, noop_fixed_val, temperature, -2)
        # [*, len_q, len_k, head], [*, len_q, len_k, head], [*, len_q, ?, head], [*, len_q, len_k+?, head]
        return prob_valid, [prob_valid], [prob_noop], [prob_full], [-2]

    # 4. multiplying (2) and (3)
    def _norm_head_cand(self, scores, temperature):
        conf = self.conf
        cnode = self.cnode
        use_noop, noop_fixed_val = conf.use_noop, conf.noop_fixed_val
        # -----
        prob_valid1, prob_noop1, prob_full1 = self._normalize(cnode, scores, use_noop, noop_fixed_val, temperature, -1)
        prob_valid2, prob_noop2, prob_full2 = self._normalize(cnode, scores, use_noop, noop_fixed_val, temperature, -2)
        # [*, len_q, len_k, head], [*, len_q, len_k, head], ...
        return prob_valid1*prob_valid2, [prob_valid1, prob_valid2], [prob_noop1, prob_noop2], [prob_full1, prob_full2], [-1, -2]

    # 5. binary for each individual one
    def _norm_binary(self, scores, temperature):
        conf = self.conf
        cnode = self.cnode
        use_noop, noop_fixed_val = conf.use_noop, conf.noop_fixed_val
        # -----
        prob_valid, prob_noop, prob_full = self._normalize(cnode, scores.unsqueeze(-1), use_noop, noop_fixed_val, temperature, -1)
        # [*, len_q, len_k, head], [*, len_q, len_k, head], [*, len_q, len_k, head, ?], [*, len_q, len_k, head, 1+?]
        attn = prob_valid.squeeze(-1)
        return attn, [prob_valid], [prob_noop], [prob_full], [-1]

    # input score is [*, len_q, len_k, head], output normalized ones and some extras
    def __call__(self, scores, temperature):
        attn, list_prob_valid, list_prob_noop, list_prob_full, list_dims = self._norm_f(scores, temperature)
        # prune
        if self.norm_prune > 0.:
            attn = attn * (attn > self.norm_prune).float()
        # dropout
        attn = self.adrop(attn)
        return attn, list_prob_valid, list_prob_noop, list_prob_full, list_dims
