#

# data-preper

__all__ = [
    "DataPreperConf", "DataPreper", "DataSamplerConf", "DataSampler",
    "yield_insts_with_idxes", "yield_aggr_sents",
]

from typing import List
import math
from mspx.data.inst import yield_sents, Sent
from mspx.utils import Conf, Registrable, Configurable, Random, zlog, ZHelper

# --
# base

@Registrable.rd('DP')
class DataPreperConf(Conf):
    def __init__(self):
        super().__init__()
        self.seed = 0

    @classmethod
    def get_base_conf_type(cls): return DataPreperConf
    @classmethod
    def get_base_node_type(cls): return DataPreper

@Registrable.rd('_DP')
class DataPreper(Configurable):
    def get_rng_rands(self, rng):  # check rng signature
        return rng.randint(100,size=7).tolist()

    def get_gen(self):
        _seed = self.conf.seed
        gen = Random.get_np_generator(_seed)  # make sure to create the same data!
        zlog(f"Start rng with {self}(seed={_seed}): {self.get_rng_rands(gen)}")
        return gen

    # yield data insts with input inst stream!
    def prep_insts(self, stream):
        # yield from stream
        raise NotImplementedError()

# --
# sampler

@DataPreperConf.rd('sampler')
class DataSamplerConf(DataPreperConf):
    def __init__(self):
        super().__init__()
        # --
        self.sample_s = 1.0  # (>1=N,<1=Rate) random sample how much at the very beginning, as pre-processing for convenience!
        self.sample_shuffle = True  # whether shuffle in presample?
        self.sample_reverse = False  # from back to start (for convenience)
        self.size_f = '1'  # weight for each inst: or len(x)

@DataSamplerConf.conf_rd()
class DataSampler(DataPreper):
    def __init__(self, conf: DataSamplerConf, **kwargs):
        super().__init__(conf, **kwargs)
        self.size_f = ZHelper.eval_ff(conf.size_f, default_args='x')

    @staticmethod
    def do_presample(insts: List, s: float, size_f=None):
        assert s > 0
        # --
        if size_f is None:  # 1 for each!
            all_sizes = [1.] * len(insts)
        else:
            all_sizes = [size_f(z) for z in insts]
        trg = (sum(all_sizes) * s) if s <= 1. else s
        # --
        idx, curr = 0, 0.
        while curr < trg and idx < len(all_sizes):
            curr += all_sizes[idx]
            idx += 1
        ret = insts[:idx]  # make it >= target!
        # breakpoint()
        zlog(f"do_presample: trg={trg}, curr={curr}, sample={idx}/{len(insts)}")
        return ret

    def prep_insts(self, stream):
        conf: DataSamplerConf = self.conf
        # --
        if (not conf.sample_shuffle) and (not conf.sample_reverse) and conf.sample_s == 1.:
            yield from stream  # the simplest case!
        # else need to first get all!
        _insts = list(stream)
        _len0 = len(_insts)
        _gen = self.get_gen()
        if conf.sample_reverse:
            _insts.reverse()
        if conf.sample_shuffle:
            _gen.shuffle(_insts)
        if conf.sample_s != 1.:
            _insts = DataSampler.do_presample(_insts, conf.sample_s, self.size_f)
        zlog(f"DataSampler(s={conf.sample_s}): {_len0}=>{len(_insts)} instances.")
        yield from _insts

# --
# helpers
def yield_insts_with_idxes(stream, idxes, follow_idxes=True):
    if follow_idxes:
        # yield insts by the order of idxes
        all_insts = list(stream)
        for ii in idxes:
            yield all_insts[ii]
    else:
        idxes_set = set(idxes)
        for ii, ss in enumerate(stream):
            if ii in idxes_set:
                idxes_set.remove(ii)
                yield ss
            if len(idxes_set) == 0:
                break
    # --

# concatenate nearby sentences
# -> trying to make the lengths within [len0, len1]
def yield_aggr_sents(stream, len0: int, len1: int, len_f=None):
    if len_f is None:
        len_f = lambda s: len(s)
    len0, len1 = int(len0), int(len1)
    # --
    for doc in stream:
        cur_sents = []
        cur_len = 0
        for sent in doc.sents:
            one_len = len_f(sent)
            while cur_len + one_len > len1:  # until this one can fit
                if cur_len >= len0:  # yield the current ones; note: could be >len1!
                    yield Sent.combine_sents(cur_sents)
                    cur_sents = []
                    cur_len = 0
                elif len(cur_sents) > 0:  # remove the first one and retry
                    cur_len -= len_f(cur_sents[0])
                    cur_sents = cur_sents[1:]
                else:  # one_len already > len1, let it go ...
                    break
            # --
            cur_sents.append(sent)
            cur_len += one_len
            # --
        # --
        if len(cur_sents) > 0 and cur_len >= len0:
            yield Sent.combine_sents(cur_sents)  # the remaining ones; note: could be >len1!
        # --
    # --
