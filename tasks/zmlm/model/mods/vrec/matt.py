#

# multihead attention node
# -- include the components of: c{1,2,3}_*
# -- (Q, K, V, accu_attns, rel_dist) => (results_value, attn, attn_info)

from typing import List, Union, Dict, Iterable
from msp.utils import Constants, Conf, zwarn, Helper
from msp.nn import BK
from msp.nn.layers import BasicNode
from .c1_score import MAttScorerConf, MAttScorerNode
from .c2_norm import MAttNormerConf, MAttNormerNode
from .c3_collect import MAttCollectorConf, MAttCollectorNode

# -----

# conf
class MAttConf(Conf):
    def __init__(self):
        self.head_count = 8
        self.scorer_conf = MAttScorerConf()
        self.normer_conf = MAttNormerConf()
        self.collector_conf = MAttCollectorConf()

# node
class MAttNode(BasicNode):
    def __init__(self, pc, dim_q, dim_k, dim_v, conf: MAttConf):
        super().__init__(pc, None, None)
        self.conf = conf
        # -----
        self.scorer = self.add_sub_node("c1", MAttScorerNode(pc, dim_q, dim_k, conf.head_count, conf.scorer_conf))
        self.normer = self.add_sub_node("c2", MAttNormerNode(pc, conf.head_count, conf.normer_conf))
        self.collector = self.add_sub_node("c3", MAttCollectorNode(pc, dim_q, dim_v, conf.head_count, conf.collector_conf))

    def get_output_dims(self, *input_dims):
        return self.collector.get_output_dims(*input_dims)

    # input *[*, len?, Din]
    def __call__(self, query, key, value, accu_attn, mask_k=None, mask_qk=None, rel_dist=None, temperature=1., forced_attn=None):
        if forced_attn is None:  # need to calculate the attns
            scores = self.scorer(query, key, accu_attn, mask_k=mask_k, mask_qk=mask_qk, rel_dist=rel_dist)
            normer_ret = self.normer(scores, temperature)
        else:  # skip the structured part
            scores = None
            normer_ret = (forced_attn, [], [], [], [])
        result_value = self.collector(query, value, normer_ret[0], accu_attn)
        # return the output of the three compoents:
        # [*, len_q, len_k, head], ([*, len_q, len_k, head], ...), [*, len_q, Dv]
        return scores, normer_ret, result_value
