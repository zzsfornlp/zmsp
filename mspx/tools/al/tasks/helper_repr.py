#

__all__ = [
    "QueryReprHelperConf", "QueryReprHelper"
]

import numpy as np

from mspx.utils import zlog, zwarn, Conf
from mspx.nn import SimConf, BK

class QueryReprHelperConf(Conf):
    def __init__(self):
        self.mode = 'cluster'
        # iterative mode
        self.repr_weights = [0.]  # (1-w)*score-unc + w*score-rep
        self.repr_specs = [1., 1., 1.]  # score-repr = sU*Unn - sB*annB - sP*annP
        self.repr_hid_key = 'repr_hid'
        self.repr_hid_minusavg = True
        # self.repr_aggr_hid = 'avg'  # average the hidden repr
        # self.repr_aggr_sim = 'avg'  # average the metrics
        self.repr_sim = SimConf().direct_update(func='cos')  # what similarity to consider?
        self.repr_sb_incp = False  # include previous ones into sb
        # cluster mode
        self.repr_cluster_k = 10

class QueryReprHelper:
    def __init__(self, sents_qA, sents_qU, weight_repr: float, conf: QueryReprHelperConf):
        self.conf = conf
        # --
        self.weight_repr = weight_repr
        self.sim = conf.repr_sim.make_node()
        self.sim.eval()
        self._sim_matrix = None
        # --
        self.sents_qA, self.sents_qU = sents_qA, sents_qU
        self.dsids2idx = {ss.dsids: ii for ii,ss in enumerate(sents_qA+sents_qU)}
        # --
        zlog(f"Setup {self}")

    def __repr__(self):
        return f"QueryReprHelper: ALL={len(self.sents_qA)+len(self.sents_qU)} | qA={len(self.sents_qA)}, qU={len(self.sents_qU)}"

    @property
    def sim_matrix(self):
        if self._sim_matrix is None:
            conf: QueryReprHelperConf = self.conf
            _rkey = conf.repr_hid_key
            _sents = self.sents_qA + self.sents_qU
            _reprs0 = BK.input_real(np.asarray([s.arrs[_rkey].mean(-2) for s in _sents]))  # [N, D]
            if conf.repr_hid_minusavg:
                _reprs = _reprs0 - _reprs0.mean(0, keepdims=True)
            else:
                _reprs = _reprs0
            _res = self.sim(_reprs, _reprs)  # [N, N]
            self._sim_matrix = _res
        return self._sim_matrix

    def repr_selection(self, sents, unc_scores, budgets, target_budget):
        conf: QueryReprHelperConf = self.conf
        if conf.mode == 'iterative':
            return self.iterative_selection(sents, unc_scores, budgets, target_budget)
        elif conf.mode == 'cluster':
            return self.cluster_selction(sents, unc_scores, budgets, target_budget)
        else:
            raise NotImplementedError(f"UNK mode {conf.mode}")

    def cluster_selction(self, sents, unc_scores, budgets, target_budget):
        conf: QueryReprHelperConf = self.conf
        _rkey = conf.repr_hid_key
        _arr = np.asarray([s.arrs[_rkey].mean(-2) for s in sents])  # [N, D]
        # --
        # kmeans
        zlog(f"Doing kmeans on {_arr.shape}")
        clu_k = min(conf.repr_cluster_k, len(sents))
        from sklearn.cluster import KMeans
        res = KMeans(n_clusters=clu_k, random_state=0).fit(_arr)
        final_cands = [[] for _ in range(clu_k)]
        for ii, ll in enumerate(res.labels_):  # put cands
            final_cands[ll].append(ii)
        # select
        for vs in final_cands:
            vs.sort(key=lambda x: unc_scores[x])  # pop from last
        remaining_budget = target_budget
        rets = []
        next_ii = 0
        while remaining_budget >= 0 and any(len(z)>0 for z in final_cands):
            if len(final_cands[next_ii]) > 0:
                one = final_cands[next_ii].pop()
                rets.append(one)
                remaining_budget -= budgets[one]
            next_ii = (next_ii + 1) % len(final_cands)
        return rets

    def iterative_selection(self, sents, unc_scores, budgets, target_budget):
        conf: QueryReprHelperConf = self.conf
        wr = self.weight_repr
        sU, sB, sP = conf.repr_specs
        zlog(f"Select [{len(sents)}] with {self}")
        t_sim = self.sim_matrix
        zlog(f"Obtain sim_matrix: {t_sim.shape}")
        rets = []
        # --
        t_sb_mask = BK.input_real([0] * len(t_sim))  # inner-idx to bool
        _lenP = len(self.sents_qA)
        if conf.repr_sb_incp:
            t_sb_mask[:_lenP] = 1.
        t_unc = BK.zeros([len(sents)]) if unc_scores is None else BK.input_real(unc_scores)
        sent_idxes = [self.dsids2idx[ss.dsids] for ss in sents]  # input-idx to inner-idx
        t_sent_idxes = BK.input_idx(sent_idxes)
        t_sim1 = t_sim[t_sent_idxes]
        t_su_mask = BK.input_real([0] * len(t_sim))  # inner-idx to bool
        t_su_mask[t_sent_idxes] = 1.
        t_sel_mask = BK.input_real([0] * len(t_sent_idxes))  # input-idx to bool
        # first sP is not changing
        curr_sp = 0. if _lenP <= 0 else t_sim1[:, :_lenP].mean(-1)
        # --
        NEG_INF = -10000.
        remaining_budget = target_budget
        while remaining_budget > 0:
            curr_sb = t_sim1[:, t_sb_mask>0].mean(-1) if (t_sb_mask>0).any() else 0.
            curr_su = t_sim1[:, t_su_mask>0].mean(-1) if (t_su_mask>0).any() else 0.
            t_scores = (1-wr) * t_unc + wr * (sU * curr_su - sB * curr_sb - sP * curr_sp) + t_sel_mask * NEG_INF
            curr_best = t_scores.argmax().item()
            if curr_best in rets:
                break  # no one remaining
            rets.append(curr_best)
            t_sb_mask[t_sent_idxes[curr_best]] = 1.
            t_su_mask[t_sent_idxes[curr_best]] = 0.
            t_sel_mask[curr_best] = 1.
            remaining_budget -= budgets[curr_best]
        return rets

# --
# repr_weights:1. query_emb_bname:roberta-base 'specs_qemb:conf_sbase2:task_name:enc0 bert_lidx:6'
# b mspx/tools/al/tasks/helper_repr:
