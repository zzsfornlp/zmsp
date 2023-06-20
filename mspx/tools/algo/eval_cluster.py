#

# cluster evaluation
# https://aclanthology.org/P14-2006.pdf

import numpy as np

def _safe_div(x, y):
    return x if y == 0 else x / y

def _get_f1(P, R):
    return _safe_div(2 * P * R, P + R)

def _align(base_clusters, match_clusters):
    import numpy as np
    # calculate overlap
    _overlap = np.zeros([len(base_clusters), len(match_clusters)])
    _score = np.zeros([len(base_clusters), len(match_clusters)])
    for b_idx, b_items in enumerate(base_clusters):
        b_item_set = set(b_items)
        for m_idx, m_items in enumerate(match_clusters):
            n_overlap = sum(z in b_item_set for z in m_items)
            _overlap[b_idx, m_idx] = n_overlap
            _score[b_idx, m_idx] = _safe_div(2*n_overlap, len(b_items)+len(m_items))
    # align
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(_score, True)
    b_map, m_map = {int(a):int(b) for a,b in zip(row_ind, col_ind)}, \
        {int(b):int(a) for a,b in zip(row_ind, col_ind)}
    return b_map, m_map, _overlap

class _EvalHelper:
    @staticmethod
    def _eval_muc(base_clusters, match_clusters, base_map, match_map):
        nom, denom = 0, 0
        for b_idx, b_items in enumerate(base_clusters):
            nom += len(b_items)
            denom += len(b_items) - 1
            hit_group = set()
            for vv in b_items:
                if vv not in match_map:
                    nom -= 1  # not found!
                else:
                    m_idx = match_map[vv]
                    if m_idx not in hit_group:
                        hit_group.add(m_idx)
                        nom -= 1  # extra one!
        ret = _safe_div(nom, denom)
        return ret

    @staticmethod
    def _eval_bcube(base_clusters, match_clusters, base_map, match_map):
        nom, denom = 0, 0
        for b_idx, b_items in enumerate(base_clusters):
            _len = len(b_items)
            denom += _len
            b_item_set = set(b_items)
            for m_items in match_clusters:
                n_overlap = sum(z in b_item_set for z in m_items)
                nom += _safe_div(n_overlap**2, _len)
        ret = _safe_div(nom, denom)
        return ret

    @staticmethod
    def _eval_ceafm(base_clusters, match_clusters, base_map, match_map):
        b_map, m_map, _overlap = _align(base_clusters, match_clusters)
        nom, denom = 0, 0
        for b_idx, b_items in enumerate(base_clusters):
            denom += len(b_items)
            if b_idx in b_map:
                nom += _overlap[b_idx, b_map[b_idx]]
        ret = _safe_div(nom, denom)
        return ret

    @staticmethod
    def _eval_ceafe(base_clusters, match_clusters, base_map, match_map):
        b_map, m_map, _overlap = _align(base_clusters, match_clusters)
        nom, denom = 0, 0
        for b_idx, b_items in enumerate(base_clusters):
            denom += 1
            if b_idx in b_map:
                m_idx = b_map[b_idx]
                nom += _safe_div(2 * _overlap[b_idx, m_idx], len(b_items)+len(match_clusters[m_idx]))
        ret = _safe_div(nom, denom)
        return ret

    # note: special eval for blanc
    @staticmethod
    def _special_eval_blanc(base_clusters, match_clusters):
        # --
        def _yield_pairs(_list):
            _len = len(_list)
            for _i0 in range(_len):
                for _i1 in range(_i0+1, _len):
                    yield (_list[_i0], _list[_i1])
        # --
        def _calc(_base_set, _match_set):
            _overlap = len(_base_set.intersection(_match_set))
            return _safe_div(_overlap, len(_base_set))
        # --
        # collect info
        infos = [(set(), set()), (set(), set())]
        for one_ii, one_clusters in enumerate([base_clusters, match_clusters]):
            _link, _nlink = infos[one_ii]
            for one_items in one_clusters:
                for _sig_link in _yield_pairs(one_items):
                    _link.add(_sig_link)
            for items0, items1 in _yield_pairs(one_clusters):
                for a in items0:
                    for b in items1:
                        _nlink.add((a, b))
        # --
        # calculate
        (b_link, b_nlink), (m_link, m_nlink) = infos
        p_link, r_link = _calc(m_link, b_link), _calc(b_link, m_link)
        f_link = _get_f1(p_link, r_link)
        p_nlink, r_nlink = _calc(m_nlink, b_nlink), _calc(b_nlink, m_nlink)
        f_nlink = _get_f1(p_nlink, r_nlink)
        res = {k:v for k,v in locals().items() if k[0] in 'prf' and k.endswith('link')}
        res['f1'] = (f_link+f_nlink) / 2
        return res

def eval_cluster(pred_clusters, gold_clusters, methods=None):
    # build reverse maps
    pred_map, gold_map = {}, {}
    for ii, vv in enumerate(pred_clusters):
        assert len(vv) > 0
        for vv2 in vv:
            assert vv2 not in pred_map
            pred_map[vv2] = ii
    for ii, vv in enumerate(gold_clusters):
        assert len(vv) > 0
        for vv2 in vv:
            assert vv2 not in gold_map
            gold_map[vv2] = ii
    # eval them
    res = {}
    if methods is None:
        # methods = ["muc", "bcube", "ceafm", "ceafe", "blanc"]  # all of them
        methods = ["muc", "bcube", "ceafe"]
    for method in methods:
        if method == 'blanc':  # special one!
            res[method] = _EvalHelper._special_eval_blanc(gold_clusters, pred_clusters)
        else:
            _ff = getattr(_EvalHelper, f"_eval_{method}")
            P = _ff(pred_clusters, gold_clusters, pred_map, gold_map)
            R = _ff(gold_clusters, pred_clusters, gold_map, pred_map)
            f1 = _get_f1(P, R)
            res[method] = {'f1': f1, 'P': P, 'R': R}
    if all(z in methods for z in ["muc", "bcube", "ceafe"]):
        res['conll'] = (res['muc']['f1'] + res['bcube']['f1'] + res['ceafe']['f1']) / 3
    # --
    return res
