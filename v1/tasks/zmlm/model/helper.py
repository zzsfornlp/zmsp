#

# some helper nodes

from typing import List, Dict
import numpy as np
from msp.utils import zwarn, Random, Helper, Conf
from msp.data import VocabPackage
from msp.nn import BK
from msp.nn.layers import BasicNode, PosiEmbedding2
from .mods.vrec import VRecEncoderConf, VRecEncoder, VRecConf, VRecNode
from .base import BaseModuleConf, BaseModule

# =====
# the representation preparation node
# (forward a vrec multiple times with fixed attns)

class RPrepConf(Conf):
    def __init__(self):
        # outside
        self.rprep_num_update = 0  # calling times
        self.rprep_lambda_emb = 0.  # emb_t*L+ent_t*(1-L)
        self.rprep_lambda_accu = 0.  # accu_attn*L+last_attn(1-L)
        # the hided vrec: maybe overall make it simple and less parameterized
        self._rprep_vr_conf = VRecConf()  # the fixed-attn vrec
        # -- only open certain options
        self.rprep_d_v = 128
        self.rprep_v_act = "elu"
        self.rprep_v_drop = 0.1
        self.rprep_collect_mode = "ch"
        self.rprep_collect_reducer_mode = "max"

    def do_validate(self):
        # make fewer options; all parameters include: combiner(lstm), collector(v-affine)
        self._rprep_vr_conf.comb_mode = "lstm"  # just make it lstm
        self._rprep_vr_conf.matt_conf.collector_conf.d_v = self.rprep_d_v
        self._rprep_vr_conf.matt_conf.collector_conf.v_act = self.rprep_v_act
        self._rprep_vr_conf.matt_conf.collector_conf.v_drop = self.rprep_v_drop
        self._rprep_vr_conf.matt_conf.collector_conf.collect_mode = self.rprep_collect_mode
        self._rprep_vr_conf.matt_conf.collector_conf.collect_reducer_mode = self.rprep_collect_reducer_mode

class RPrepNode(BasicNode):
    def __init__(self, pc, dim: int, conf: RPrepConf):
        super().__init__(pc, None, None)
        self.conf = conf
        # -----
        self.dim = dim
        if conf.rprep_num_update > 0:
            self.vrec_node = self.add_sub_node("v", VRecNode(pc, dim, conf._rprep_vr_conf))
        else:
            self.vrec_node = None

    def get_output_dims(self, *input_dims):
        return (self.dim, )

    @property
    def active(self):
        return self.conf.rprep_num_update>0

    # input from emb(very-first input), enc(encoder's output) and cache(vrec-enc's cache)
    def __call__(self, emb_t, enc_t, cache):
        if not self.active:
            return enc_t
        conf = self.conf
        rprep_lambda_emb, rprep_lambda_accu= conf.rprep_lambda_emb, conf.rprep_lambda_accu
        # input_t: [*, len_q, D]
        if rprep_lambda_emb == 0.:
            input_t = enc_t  # by default, only enc_t
        else:
            input_t = emb_t * rprep_lambda_emb + enc_t * (1. - rprep_lambda_emb)
        cur_hidden = input_t
        # another encoder
        if conf.rprep_num_update > 0:
            # attn_t: [*, len_q, len_k, head]
            attn_t = cache.list_accu_attn[-1] * rprep_lambda_accu + cache.list_attn[-1] * rprep_lambda_accu
            # call it
            cur_cache = self.vrec_node.init_call(input_t)
            for _ in range(conf.rprep_num_update):
                cur_hidden = self.vrec_node.update_call(cur_cache, forced_attn=attn_t)
        return cur_hidden

# =====
# various entropy

class EntropyHelper:
    @staticmethod
    def cross_entropy(p, q, dim=-1):
        return - (p * (q + 1e-10).log()).sum(dim)

    @staticmethod
    def kl_divergence(p, q, dim=-1):
        return EntropyHelper.cross_entropy(p, q, dim) - EntropyHelper.cross_entropy(p, p, dim)

    @staticmethod
    def kl_divergence_bd(p, q, dim=-1):  # for both directions
        return 0.5 * (EntropyHelper.kl_divergence(p, q, dim) + EntropyHelper.kl_divergence(q, p, dim))

    @staticmethod
    def js_divergence(p, q, dim=-1):
        m = 0.5 * (p+q)
        return 0.5 * (EntropyHelper.kl_divergence(p, m, dim) + EntropyHelper.kl_divergence(q, m, dim))

    @staticmethod
    def get_method(m):
        return {"cross": EntropyHelper.cross_entropy, "kl": EntropyHelper.kl_divergence, "js": EntropyHelper.js_divergence,
                "kl2": EntropyHelper.kl_divergence_bd, "klbd": EntropyHelper.kl_divergence_bd}[m]
