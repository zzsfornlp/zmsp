#

# demonstration selection

import numpy as np
from collections import Counter
from mspx.utils import Conf, zglob1, default_json_serializer, zlog, Random, ZHelper
from .helper import *

# --
# helper for sentence repr formatting
class InstFormator:
    def __init__(self, rs):
        import nltk
        from nltk.corpus import stopwords
        self.stopwords = set(stopwords.words('english'))
        self.rs = rs

    def _spell_spath(self, spath):
        _m = {'nsubj': 'nominal subject', 'obj': 'object', 'obl': 'oblique nominal', 'nmod': 'nominal modifier'}
        new_spath = []
        for z in spath:
            if z.startswith('^'):
                s = "up with "
                z = z[1:]
            else:
                s = "down with "
            s = s + _m.get(z, z)
            new_spath.append(s)
        ret = ", ".join(new_spath)
        return ret

    def _extend_spine(self, inst, add_stop_words, add_all, do_simple=False):
        idxes0 = sum(inst['spines'], [])
        if do_simple:
            idxes0 = [idxes0[0], idxes0[-1]]  # simply keep these two nodes!
        idxes = sorted(idxes0)
        if add_all:
            words = inst['tokens'][min(idxes):max(idxes)+1]
        elif add_stop_words:
            hit_idxes = set(idxes)
            words = []
            for ii in range(min(idxes), max(idxes)+1):
                if ii in hit_idxes or str.lower(inst['tokens'][ii]) in self.stopwords:
                    words.append(inst['tokens'][ii])
        else:
            words = [inst['tokens'][z] for z in idxes]
        ret = ' '.join(words)
        return ret

    def _extend_spine_v2(self, inst):
        from mspx.data.inst import Sent, DepTree
        spine_evt, spine_top, spine_ef = inst['spines']
        sent = Sent(inst['tokens'])
        sent.build_dep_tree(inst['syntax'][0], inst['syntax'][1])
        # --
        inc_set = {'aux', 'cop', 'mark', 'det', 'clf', 'case', 'fixed', 'flat', 'compound'}
        idxes0 = set(sum(inst['spines'], []))
        extra_idxes = set()
        _chs_lists = sent.tree_dep.chs_lists
        _heads = sent.tree_dep.seq_head.vals
        _labs = sent.tree_dep.get_labels(level=1)
        for ii in spine_evt + spine_ef:
            for cc in _chs_lists[ii+1]:
                if _labs[cc] in inc_set:
                    extra_idxes.add(cc)
                if _labs[cc] == 'cc':  # conj
                    if _labs[ii] == 'conj' and (_heads[ii]-1) in idxes0:
                        extra_idxes.add(cc)
        # --
        idxes0.update(extra_idxes)
        final_idxes = sorted(idxes0)
        words = [inst['tokens'][z] for z in final_idxes]
        ret = ' '.join(words)
        return ret

    def form_sent_repr(self, inst):
        # mention specific sent repr
        reprs = {
            'pair': lambda inst0: f"""In the event "{inst0['evt']}", the entity "{inst0['ent']}" plays what role?""",
            'sent': lambda inst0: f"{inst0['sent']}",
            'spath1': lambda inst0: f"Syntactic path is {', '.join(inst0['spath'])}.",
            'spath2': lambda inst0: f"Syntactic path is {self._spell_spath(inst0['spath'])}.",
            'spine': lambda inst0: self._extend_spine(inst0, False, False),
            'spineS': lambda inst0: self._extend_spine(inst0, True, False),
            'spineA': lambda inst0: self._extend_spine(inst0, False, True),
            'spine0': lambda inst0: self._extend_spine(inst0, False, False, do_simple=True),
            'spine0A': lambda inst0: self._extend_spine(inst0, False, True, do_simple=True),
            'spinev2': lambda inst0: self._extend_spine_v2(inst0),
        }
        _rfs = [reprs[z] for z in self.rs]
        rets = [ff(inst) for ff in _rfs]
        ret = " ".join(rets)
        # breakpoint()
        return ret
# --

class IclSelectConf(Conf):
    def __init__(self):
        # phase 0: instance-agnostic selection
        self.sel0 = IclSelStrategyConf.direct_conf(k=8, final_shuffle=True)
        # phase 1: instance-specific selection (by default using the overall ones)
        self.sel1 = IclSelStrategyConf.direct_conf(k=-1)

class IclSelStrategyConf(Conf):
    def __init__(self):
        self.k = 8  # number of instance to select; no selection if <0
        self.filter_f = '1'  # lambda x,t: bool
        self.score_stra = ['random']  # score method
        self.score_w = ["1."]  # (conditional) score weights
        self.score_flatratio = [-1]  # flatten out topk (k*this) scores for each round?
        self.final_shuffle = False  # do final shuffling?
        self.random_delta = 0.  # add random noise?
        # signature budgets
        self.sig_f = "()"  # signatures for inst
        self.sig_budgets = []  # budget per signature
        # how to form sent-repr
        self.repr_sent = ['pair', 'sent']  # spath1, spath2, ...

class IclSelStrategy:
    def __init__(self, conf: IclSelStrategyConf):
        self.conf = conf
        # --
        self._gen = Random.get_generator('sel_stra')
        self._filter_f = ZHelper.eval_ff(conf.filter_f, "x,t", locals=locals(), globals=globals())
        self._sig_f = ZHelper.eval_ff(conf.sig_f, "x,t", locals=locals(), globals=globals())
        self._sig_budgets = [int(z) for z in conf.sig_budgets]
        self._fw = [ZHelper.eval_ff(z, "x", locals=locals(), globals=globals()) for z in conf.score_w]
        self._repr_former = InstFormator(conf.repr_sent)
        # --
        # pre-building
        self._data_pool = None

    def prebuild_pool(self, data_pool):
        data_pool = [z for z in data_pool if self._filter_f(z, None)]
        self._data_pool = data_pool
        # --
        conf: IclSelStrategyConf = self.conf
        for _stra in conf.score_stra:
            if _stra == 'bm25':
                # pip install rank-bm25
                from rank_bm25 import BM25Okapi
                from nltk.tokenize import word_tokenize
                self.cache_bm25 = BM25Okapi([word_tokenize(self._repr_former.form_sent_repr(z)) for z in data_pool])
            elif _stra == 'sbert':
                # pip install sentence-transformers
                from sentence_transformers import SentenceTransformer
                _sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                embeddings = _sbert_model.encode([self._repr_former.form_sent_repr(z) for z in data_pool])
                self.cache_sbert = embeddings  # [N, D]

    def prebuild_test(self, data_test):
        conf: IclSelStrategyConf = self.conf
        for _stra in conf.score_stra:
            if _stra == 'bm25':
                # pip install rank-bm25
                from rank_bm25 import BM25Okapi
                from nltk.tokenize import word_tokenize
                self.cache_score_bm25 = [self.cache_bm25.get_scores(word_tokenize(self._repr_former.form_sent_repr(z))) for z in data_test]
            elif _stra == 'sbert':
                # pip install sentence-transformers
                from sentence_transformers import SentenceTransformer, util
                _sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
                embeddings = _sbert_model.encode([self._repr_former.form_sent_repr(z) for z in data_test])
                self.cache_score_sbert = util.cos_sim(embeddings, self.cache_sbert)  # [Ntest, Npool]
                # breakpoint()

    def _get_vv(self, ll, ii):
        return ll[ii] if ii<len(ll) else ll[-1]

    def do_sel(self, data_pool, inst_test=None, quiet=True, one_ii=None):
        conf: IclSelStrategyConf = self.conf
        # filter
        if self._data_pool is not None:  # use cached one
            data_pool = self._data_pool
        else:
            data_pool = [z for z in data_pool if self._filter_f(z, inst_test)]
        if conf.k < 0:  # directly pass through (sel turned off!)
            ret = data_pool
            if not quiet:
                zlog(f"Passing through with {len(ret)} instances.")
        else:
            # score
            _plen = len(data_pool)
            _final_arr_score = np.zeros(_plen)
            for _sii, _strategy in enumerate(conf.score_stra):
                if _strategy == 'random':
                    arr_score = self._gen.random(_plen)
                elif _strategy == 'bm25':
                    arr_score = -np.asarray(self.cache_score_bm25[one_ii])
                elif _strategy == 'sbert':
                    arr_score = -np.asarray(self.cache_score_sbert[one_ii])
                elif _strategy == 'spath':
                    arr_score = -np.asarray([score_spath(inst_test, z) for z in data_pool])
                else:
                    raise NotImplementedError(f"UNK score strategy: {_strategy}")
                _wii = self._get_vv(self._fw, _sii)(inst_test)
                arr_score *= _wii
                _frii = self._get_vv(conf.score_flatratio, _sii)
                if _frii > 0:
                    _top_idxes = np.argsort(arr_score)[:int(_frii*conf.k)]
                    arr_score = np.zeros(_plen)
                    arr_score[_top_idxes] = -100.  # note: this should be enough, remember lower better!
                _final_arr_score += arr_score
            if conf.random_delta > 0:
                _final_arr_score = _final_arr_score + self._gen.random(_plen) * conf.random_delta
            # rank and obtain
            _K = conf.k
            ret = []
            _sig_budgets = self._sig_budgets
            sig_ccs = [Counter() for _ in _sig_budgets]
            rank_idxes = np.argsort(_final_arr_score)
            for one_di in rank_idxes:
                if len(ret) >= _K:
                    break
                one_dp = data_pool[one_di]
                one_sigs = self._sig_f(one_dp, inst_test)
                assert isinstance(one_sigs, tuple)
                if any(sig_ccs[sig_ii][sig_vv]>=_sig_budgets[sig_ii] for sig_ii, sig_vv in enumerate(one_sigs)):
                    continue  # out of budget
                for sig_ii, sig_vv in enumerate(one_sigs):
                    sig_ccs[sig_ii][sig_vv] += 1
                ret.append(one_dp)
            if not quiet:
                zlog(f"Select {len(ret)} instances with sig: {sig_ccs}")
            # final shuffle
            if conf.final_shuffle:
                self._gen.shuffle(ret)
        return ret

def select_demonstrations(data_test, data_pool, conf: IclSelectConf, task_helper=None, **kwargs):
    conf: IclSelectConf = IclSelectConf.direct_conf(conf, **kwargs)
    sel0, sel1 = IclSelStrategy(conf.sel0), IclSelStrategy(conf.sel1)
    # --
    zlog("Start to select data ...")
    ret = []
    data_pool_selected = sel0.do_sel(data_pool, None, quiet=False)  # agnostic of the test instance
    sel1.prebuild_pool(data_pool_selected)  # prebuild
    sel1.prebuild_test(data_test)  # prebuild
    for one_ii, one_data in enumerate(data_test):
        selected_dps = sel1.do_sel(None, one_data, one_ii=one_ii)  # instance specific
        ret.append((selected_dps, one_data))
    return ret

# --
# b mspx/znew/icl/select:133
