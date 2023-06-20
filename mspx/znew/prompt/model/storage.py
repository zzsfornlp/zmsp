#

# cache storage

import torch
import numpy as np
from mspx.utils import Conf, Configurable, zwarn, Random, zlog
from collections import Counter

# --
# storage

class StorageConf(Conf):
    def __init__(self):
        # knn helper
        self.sim_f = 'ndist'  # similarity function
        self.dist_p = 2.  # L?
        self.tau = 1.

class MyStorage(Configurable):
    def __init__(self, conf: StorageConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: StorageConf = self.conf
        # --
        self.key = None
        self.value = {}  # value-item -> Tensor or List
        # --
        self.selector = None

    # [Lq, R], [Lk, R]
    def get_sim(self, tq, tk):
        conf: StorageConf = self.conf
        # --
        # obtain similarity of [L0, L1]
        if conf.sim_f == 'ndist':
            _sim = - ((tq.unsqueeze(1) - tk.unsqueeze(0)).abs() ** conf.dist_p).mean(-1)
        elif conf.sim_f == 'kl':
            _tq, _tk = tq.log_softmax(-1).unsqueeze(1), tk.log_softmax(-1).unsqueeze(0)
            _sim = - (_tq.exp() * (_tq - _tk)).sum(-1)
        else:
            raise NotImplementedError()
        _ret = _sim / conf.tau
        # breakpoint()
        return _ret

    # add key together with extra values
    def add(self, t_key, **t_values):
        conf: StorageConf = self.conf
        # --
        if self.key is None:
            self.key = t_key
            self.value.update(t_values)
        else:
            self.key = torch.cat([self.key, t_key], 0)
            for kk in list(self.value.keys()):
                vv = self.value[kk]
                vv2 = t_values[kk]
                if isinstance(vv, list):
                    vv.extend(vv2)
                else:
                    self.value[kk] = torch.cat([vv, vv2], 0)
        # --

    # returning topk ones
    def search(self, t_query, k: int):
        conf: StorageConf = self.conf
        # --
        t_key = self.key
        _sim = self.get_sim(t_query, t_key)  # [Lq, Lk]
        k = min(k, _sim.shape[-1])
        ret_sims, ret_idxes = _sim.topk(k, dim=-1)  # [Lq, K]
        return ret_sims, ret_idxes

    # search and obtain target distribution
    def search_and_distr(self, t_query, k: int, target):
        _sims, _idxes = self.search(t_query, k)  # [Lq, K]
        t_target = self.value[target].squeeze(-1)  # [Lk]
        _trg = t_target[_idxes]  # [Lq, K]
        # note: simply use max-target as the number of labels
        _ret0 = torch.zeros([t_query.shape[0], t_target.max().item()+1]).to(t_query)  # [Lq, L]
        _probs = _sims.softmax(-1)  # [Lq, K]
        # breakpoint()
        _ret = _ret0.scatter_add(-1, _trg, _probs)  # [Lq, L]
        return _ret


# --
# selector

class SelectorConf(Conf):
    def __init__(self):
        self.sel_k = 0  # enabled if >0
        self.sel_stra = "fixed"  # fixed-random, random, ...
        self.sel_seed = 100
        self.sel_balance_label = False  # whether balance label
        self.random_delta = 0.
        self.final_shuffle = False

class MySelector(Configurable):
    def __init__(self, conf: SelectorConf, insts, **kwargs):
        super().__init__(conf, **kwargs)
        conf: SelectorConf = self.conf
        # --
        self.pool = list(insts)
        self._gen = Random.get_np_generator(conf.sel_seed)
        self._gen.shuffle(self.pool)
        self.label_counts = Counter([self.get_label(z) for z in self.pool])
        # --
        self._build_pool()
        zlog(f"Build pool {len(self.pool)} with label_counts: {self.label_counts}")
        # --

    def get_label(self, inst):
        return inst['_map']['label']

    def get_input(self, inst):
        return inst['_map']['input']

    def _build_pool(self):
        conf: SelectorConf = self.conf
        _stra = conf.sel_stra
        # --
        if _stra == 'fixed':
            pass
        elif _stra == 'bm25':
            # pip install rank-bm25
            from rank_bm25 import BM25Okapi
            from nltk.tokenize import word_tokenize
            self.cache_bm25 = BM25Okapi([word_tokenize(self.get_input(z)) for z in self.pool])
        elif _stra == 'sbert':
            # pip install sentence-transformers
            from sentence_transformers import SentenceTransformer
            _sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
            embeddings = _sbert_model.encode([self.get_input(z) for z in self.pool])
            self.sbert_model = _sbert_model
            self.cache_sbert = embeddings  # [N, D]
        else:
            raise NotImplementedError()
        return None

    def select(self, query_insts):
        conf: SelectorConf = self.conf
        _stra = conf.sel_stra
        # --
        ret = []
        _pool = self.pool
        for q in query_insts:
            # --
            # score
            _plen = len(_pool)
            _final_arr_score = np.zeros(_plen)
            if _stra == 'fixed':  # simply use the random-fixed ones!
                arr_score = np.arange(_plen)
            elif _stra == 'bm25':
                from nltk.tokenize import word_tokenize
                arr_score = - np.asarray(self.cache_bm25.get_scores(word_tokenize(self.get_input(q))))
            elif _stra == 'sbert':
                from sentence_transformers import SentenceTransformer, util
                embeddings = self.sbert_model.encode([self.get_input(q)])
                arr_score = - np.asarray(util.cos_sim(embeddings, self.cache_sbert)[0])  # [Ntest, Npool]
            else:
                raise NotImplementedError()
            _final_arr_score += arr_score
            if conf.random_delta > 0:
                _final_arr_score = _final_arr_score + self._gen.random(_plen) * conf.random_delta
            # --
            # select
            selected_ones = []
            _K = conf.sel_k
            _label_cc = Counter()
            _label_budget = (_K / len(self.label_counts)) if conf.sel_balance_label else _K
            rank_idxes = np.argsort(_final_arr_score)
            for one_di in rank_idxes:
                if len(selected_ones) >= _K:
                    break
                one_dp = _pool[one_di]
                if one_dp is q:
                    # breakpoint()
                    continue  # not self!
                _lab = self.get_label(one_dp)
                if _label_cc[_lab] >= _label_budget:
                    continue
                selected_ones.append(one_dp)
                _label_cc[_lab] += 1
            # --
            # final shuffle
            if conf.final_shuffle:
                self._gen.shuffle(ret)
            # --
            ret.append(selected_ones)
            # breakpoint()
        # --
        return ret

# --
# b mspx/znew/prompt/model/storage:??
