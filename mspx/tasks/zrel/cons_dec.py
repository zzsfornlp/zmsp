#

# more complex decoding!

__all__ = [
    "ConsDecoder",
]

import numpy as np
from collections import OrderedDict

class ConsDecoder:
    def __init__(self, cons_name, ztask):
        self.LAB_NIL = '_NIL_'
        self.SEP = '___'  # note: special!
        # prepare vocabs
        name_maps = []  # name -> v-name
        idx_maps = []  # v-name -> idx
        for vv in ztask.vpack:
            _m = {}
            _m2 = {}
            for ii, nn in enumerate(vv.full_i2w):
                _m2[nn] = ii
                if self.SEP in nn:
                    nn1 = nn.split(self.SEP, 1)[-1]
                else:
                    nn1 = nn
                _m[nn] = nn
                _m[nn1] = nn
            _m['_NIL_'] = self.LAB_NIL  # note: specific for idx=0!
            _m2[self.LAB_NIL] = 0
            name_maps.append(_m)
            idx_maps.append(_m2)
        # compile constraints
        vocR, vocH, vocT = ztask.vpack
        nameR, nameH, nameT = name_maps
        idxR, idxH, idxT = idx_maps
        arr_fs = np.zeros([len(vocH), len(vocT), len(vocR)]) - 1000.  # [H, T, R]
        from .cons_table import CONS_SET
        allowed_triples = CONS_SET[cons_name]
        for h, r, t in allowed_triples:
            h, r, t = nameH[h], nameR[r], nameT[t]  # voc-name
            arr_fs[idxH[h], idxT[t], idxR[r]] = 0.
        _IDX_NIL = 0
        arr_fs[:, :, _IDX_NIL] = 0.  # allow NIL
        # --
        self.vpack = ztask.vpack
        self.name_maps = name_maps
        self.idx_maps = idx_maps
        self.arr_fs = arr_fs
        self.labelR = self.vpack[0].full_i2w.copy()
        self.labelR[_IDX_NIL] = self.LAB_NIL
        # breakpoint()
        # --

    # [Lh], [Lm], [Lh, Lm]
    # note: in the local models, there are cases where unlikely HT types are picked since high arg prob,
    #  this might be the limitation of local modeling ...
    def cons_decode(self, arrH, arrT, arr_prob, arr_pmask, existing_map=None):
        nameR, nameH, nameT = self.name_maps
        idxR, idxH, idxT = self.idx_maps
        _labelR = self.labelR
        arr_fs = self.arr_fs
        # --
        from ad3 import factor_graph as fg
        graph = fg.PFactorGraph()
        variables = OrderedDict()
        for items, nmap, imap in zip([arrH, arrT], [nameH, nameT], [idxH, idxT]):
            for item in items:
                if item is not None and id(item) not in variables:
                    # read topk scores
                    labels, scores = [], []
                    for k, v in item.info['topk'].items():
                        if v > 0 and k in nmap:
                            labels.append(nmap[k])
                            scores.append(np.log(v))
                    var = graph.create_multi_variable(len(labels))
                    variables[id(item)] = (var, item, labels, np.asarray([imap[z] for z in labels]))
                    var.set_log_potentials(np.asarray(scores).astype(np.double))
        for hidx, one_h in enumerate(arrH):
            if one_h is None: continue
            for tidx, one_t in enumerate(arrT):
                if one_t is None: continue
                if arr_pmask[hidx, tidx] <= 0: continue
                _key = (id(one_h), id(one_t))
                existing_alink = None if existing_map is None else existing_map.get(_key)
                var = graph.create_multi_variable(len(_labelR))
                variables[_key] = (var, (one_h, one_t, existing_alink), None, None)
                var.set_log_potentials(np.log(arr_prob[hidx, tidx].clip(min=1e-12)).astype(np.double))
                # factor
                dense_vals = arr_fs[variables[id(one_h)][-1][:,None], variables[id(one_t)][-1]]
                graph.create_factor_dense([variables[id(one_h)][0], variables[id(one_t)][0], var], dense_vals.ravel())
        _, assignments, _, _ = graph.solve()
        cur_idx = 0
        for entries in variables.values():
            _, item, labels, _ = entries
            if labels is None:  # alink
                one_h, one_t, existing_alink = item
                _res = assignments[cur_idx:cur_idx+len(_labelR)]
                cur_idx += len(_labelR)
                _best_label = _labelR[np.argmax(_res)]
                if _best_label != self.LAB_NIL:
                    if existing_alink is None:
                        existing_alink = one_h.add_arg(one_t, None)
                        existing_alink.info['is_pred'] = True
                    existing_alink.set_label(_best_label)  # rewrite label!
                elif existing_alink is not None:
                    existing_alink.set_label(self.LAB_NIL)  # rewrite label!
            else:
                _res = assignments[cur_idx:cur_idx+len(labels)]
                cur_idx += len(labels)
                _best_label = labels[np.argmax(_res)].split(self.SEP, 1)[-1]
                if _best_label != self.LAB_NIL and _best_label != item.label:
                    item.set_label(_best_label)  # reset label!
        assert cur_idx == len(assignments)
        # breakpoint()
        # --

# --
# b mspx/tasks/zrel/cons_dec:
