#

# helper for data augmentation

__all__ = [
    "ArgAugConf", "ArgAugHelper",
]

from msp2.nn import BK
from msp2.utils import Conf, zlog, Random
import numpy as np

# --
class ArgAugConf(Conf):
    def __init__(self):
        self.aug_mode = "rand"  # mode: dep(syntax-chunk), arg(arg-chunk), rand/rand2(random-chunk) arg2(arg-only)
        self.aug_shuf_noise = 10000.  # add & sort
        self.aug_mask_rates = [1.]  # num of masks to insert: num -> rate
        self.aug_mask_budget = 256  # at most this
        self.aug_combs = [1,1]  # combine chunk pieces? [a, b)
        # --
        # specific for dep
        self.aug_types = ['pred','conj','nmod']  # what types to search
        self.aug_dists = [2,1]  # what distances to search: down,up
        # --

class ArgAugHelper:
    def __init__(self, conf: ArgAugConf, arg_mod):
        self.conf = conf
        self.arg_mod = arg_mod
        # --
        _pre_types = {  # some prebuilt ones as shortcuts
            'pred': ['nsubj', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp', 'obl', 'advcl', 'acl'],
            'conj': ['conj', 'parataxis'],
        }
        self.aug_types = set(sum([_pre_types.get(z, [z]) for z in conf.aug_types], []))
        zlog(f"ArgAugHelper will search for {self.aug_types}")
        self.aug_mask_rates = [z/sum(conf.aug_mask_rates) for z in conf.aug_mask_rates]
        # --

    def do_aug(self, item, content, mask_id, no_cache=False):
        aug_mode = self.conf.aug_mode
        if aug_mode == 'rand':
            return self.do_aug_rand(item, content, mask_id, no_cache)
        elif aug_mode in ['rand2', 'dep', 'arg']:
            return self.do_aug_chunk(item, content, mask_id, aug_mode, no_cache)
        elif aug_mode == 'arg2':
            return self.do_aug_arg2(item, content, mask_id, no_cache)
        else:
            raise NotImplementedError(f"UNK aug_mode: {aug_mode}]")
        # --

    def _search_nodes(self, start_widx: int, tree_dep):
        ret = set()
        dist_down, dist_up = self.conf.aug_dists
        aug_types = self.aug_types
        dep_heads, dep_labels = tree_dep.seq_head.vals, [z.split(":")[0] for z in tree_dep.seq_label.vals]
        dep_chs = tree_dep.chs_lists  # note: remember +1
        # --
        def _search_down(_widx, _budget):
            if _widx in ret: return
            ret.add(_widx)
            if _budget > 0:
                for _ch in dep_chs[_widx+1]:
                    if dep_labels[_ch] in aug_types:
                        _search_down(_ch, _budget-1)
        # --
        cur_widx = start_widx
        for _ in range(dist_up+1):
            _search_down(cur_widx, dist_down)
            # --
            if dep_labels[cur_widx] in aug_types and dep_heads[cur_widx] > 0:
                cur_widx = dep_heads[cur_widx] - 1  # remember -1
            else:
                break
            # --
        # --
        return ret

    def _chunk_sent(self, widxes, tree_dep):
        _len = len(tree_dep)
        _split_points = []
        for ii, widx in enumerate(widxes):
            _r0, _r1 = tree_dep.ranges[widx]  # simply add the splitting points
            _split_points.append(_r0)
            _split_points.append(_r1+1)
        _split_points = sorted(set(_split_points))
        return _split_points

    def do_aug_chunk(self, item, content, mask_id, aug_mode, no_cache=False):
        conf = self.conf
        evt = item.frame
        center_widx = evt.mention.shead_widx
        _gen = Random.get_generator("aug")
        # --
        # get vals
        _key = "_aug_key"
        _cache = item.info.get(_key)  # [slen]
        if _cache is None or no_cache:
            # search node and do chunk
            if aug_mode == 'arg':  # simply split by mentions!
                _split_points = []
                for item in [evt] + evt.args:
                    _span = item.mention.get_span()
                    _split_points.append(_span[0])
                    _split_points.append(_span[0]+_span[1])
                _split_points = sorted(set(_split_points))
                _cache = _split_points
            elif aug_mode == 'rand2':
                _split_points = list(range(len(evt.sent)+1))  # simply allow all
                _cache = _split_points
            else:
                widxes = self._search_nodes(center_widx, evt.sent.tree_dep)
                _cache = self._chunk_sent(widxes, evt.sent.tree_dep)
            if not no_cache:
                item.info[_key] = _cache
        _split_points = _cache
        # --
        # comb?
        a, b = [max(1,z) for z in conf.aug_combs]
        _sp_len = len(_split_points)
        if a<b and b>1 and _sp_len>0:  # do combs
            _cs = _gen.randint(a, b, size=_sp_len).tolist()
            _new_split_points = []
            sii = 0  # add the first one
            while sii < _sp_len:
                _new_split_points.append(sii)
                sii += _cs[sii]  # jump by how much
            if _new_split_points[-1] != _split_points[-1]:  # add the last one!
                _new_split_points.append(_split_points[-1])
            _split_points = _new_split_points
        # --
        # shuffle and mask
        _c_len = len(_split_points) - 1  # number of chunks to shuffle
        _c_v = np.arange(_c_len) + _gen.random_sample(_c_len) * conf.aug_shuf_noise
        _v_neg, _v_pos = -1., _c_v.max().item() + 1.
        _lastp = _split_points[0]
        _s_vals = [_v_neg] * _lastp
        for _pp, _vv in zip(_split_points[1:], _c_v):
            _s_vals.extend([float(_vv)] * (_pp - _lastp))
            _lastp = _pp
        _s_vals.extend([_v_pos] * (len(evt.sent) - _lastp))
        _c_nmask = _gen.choice(len(self.aug_mask_rates), size=_c_len+1, p=self.aug_mask_rates)  # [clen+1]
        _s_nmasks = {a:int(b) for a,b in zip(_split_points, _c_nmask)}
        # --
        # put vals to current seq
        # todo(+N): currently only consider one sent!
        seq_vals = []  # stable sort by adding idxes as the second one
        _DEFAULT = _v_neg
        for ii, tok in enumerate(content[2]):
            if tok is None or tok.sent is not evt.sent:
                seq_vals.append((_DEFAULT, ii, 0))
            else:
                seq_vals.append((_s_vals[tok.widx], ii, _s_nmasks.get(tok.widx, 0)))
                if tok.widx == center_widx:
                    _DEFAULT = _v_pos
        seq_vals.sort()  # shuffle it!
        seq_idxes = []
        for _, ii, nn in seq_vals:
            seq_idxes.extend([-1] * nn)  # add mask
            seq_idxes.append(ii)
        # --
        # make aug ones
        # shuffle and add mask
        mask_pads = [[mask_id], 1, None, 0., 0, 0]
        _tmp_content = [a+[b] for a,b in zip(content, mask_pads)]  # make pad idx -1
        ret = [[z[i] for i in seq_idxes] for z in _tmp_content]  # final reindex!
        return tuple(ret)

    def _get_arg_pieces(self, evt):
        # get all the spans, note: use full spans here!
        tok2piece = {}  # tok-id -> piece-id
        items = [evt] + evt.args
        items.sort(key=lambda x: ((0 if x.mention.sent.sid is None else x.mention.sent.sid), ) + x.mention.get_span())
        for ii, item in enumerate(items):
            toks = item.mention.get_tokens()
            for t in toks:
                _k = t.get_indoc_id(False)
                if _k not in tok2piece:
                    tok2piece[_k] = ii
        # --
        return len(items), tok2piece

    def do_aug_arg2(self, item, content, mask_id, no_cache=False):
        conf = self.conf
        evt = item.frame
        center_token = evt.mention.shead_token
        _gen = Random.get_generator("aug")
        # --
        # get vals
        _key = "_aug_key2"
        _cache = item.info.get(_key)
        if _cache is None or no_cache:
            _cache = self._get_arg_pieces(item.frame)
            if not no_cache:
                item.info[_key] = _cache
        n_piece, tok2piece = _cache
        # --
        # shuffle and mask
        _v_pieces = np.arange(n_piece) + _gen.random_sample(n_piece) * conf.aug_shuf_noise
        _v_neg, _v_pos = -1., _v_pieces.max().item() + 1.
        _v_pieces = _v_pieces.tolist()
        # --
        seq_vals = []
        _DEFAULT = _v_neg
        for ii, tok in enumerate(content[2]):
            if tok is center_token:
                _DEFAULT = _v_pos  # todo(+N): currently special ones are mainly just [SEP]
            # --
            if tok is None:  # note: still keep None since that could mean special tokens!
                seq_vals.append((_DEFAULT, ii, -1))
            else:
                _k = tok.get_indoc_id(False)
                if _k not in tok2piece:
                    continue  # note: remove the tokens at outside!
                pii = tok2piece[_k]
                seq_vals.append((_v_pieces[pii], ii, pii))
        seq_vals.sort()  # shuffle it!
        seq_idxes = []
        last_pii = -1
        for _, ii, pii in seq_vals:
            if pii != last_pii:
                nn = int(_gen.choice(len(self.aug_mask_rates), p=self.aug_mask_rates))
                seq_idxes.extend([-1] * nn)  # add mask
            seq_idxes.append(ii)
            last_pii = pii
        # --
        # make aug ones
        # shuffle and add mask
        mask_pads = [[mask_id], 1, None, 0., 0, 0]
        _tmp_content = [a + [b] for a, b in zip(content, mask_pads)]  # make pad idx -1
        ret = [[z[i] for i in seq_idxes] for z in _tmp_content]  # final reindex!
        return tuple(ret)

    def do_aug_rand(self, item, content, mask_id, no_cache=False):
        conf = self.conf
        _gen = Random.get_generator("aug")
        # --
        # decide range
        cur_toks = content[2]
        rand_r0, rand_r1 = 0, len(cur_toks)  # shuffle ranges: [left, right)
        while rand_r0<rand_r1 and cur_toks[rand_r0] is None:
            rand_r0 += 1
        while rand_r0<rand_r1 and cur_toks[rand_r1-1] is None:
            rand_r1 -= 1
        # --
        # split the idxes
        a, b = [max(1, z) for z in conf.aug_combs]
        if b<=1:  # uni-gram
            r_idxes = [[z] for z in range(rand_r0, rand_r1)]
        else:
            _cs = _gen.randint(a, b, size=len(cur_toks)).tolist()
            r_idxes = []
            sii = rand_r0
            while sii < rand_r1:
                _end = min(sii+_cs[sii], rand_r1)
                r_idxes.append(list(range(sii, _end)))
                sii = _end
        # --
        # shuffle
        _c_len = len(r_idxes)
        _c_v = np.arange(_c_len) + _gen.random_sample(_c_len) * conf.aug_shuf_noise
        _c_argsort = np.argsort(_c_v)
        if len(self.aug_mask_rates) == 1:  # no need to add masks
            seq_idxes = list(range(rand_r0)) + sum([r_idxes[z] for z in _c_argsort], []) + list(range(rand_r1, len(cur_toks)))
        else:  # insert masks
            mask_bud = max(0, conf.aug_mask_budget - len(cur_toks))
            seq_idxes = list(range(rand_r0))
            _c_nmask = _gen.choice(len(self.aug_mask_rates), size=_c_len+1, p=self.aug_mask_rates)  # [clen+1]
            while _c_nmask.sum() > mask_bud:
                _ciip = (_c_nmask>0) / (_c_nmask>0).sum()
                _cii = _gen.choice(len(_ciip), p=_ciip)
                _c_nmask[_cii] = 0  # remove this piece
            for _ii, _rrs in enumerate([r_idxes[z] for z in _c_argsort] + [list(range(rand_r1, len(cur_toks)))]):
                seq_idxes.extend([-1] * _c_nmask[_ii])
                seq_idxes.extend(_rrs)
        # --
        # return
        mask_pads = [[mask_id], 1, None, 0., 0, 0]
        _tmp_content = [a + [b] for a, b in zip(content, mask_pads)]  # make pad idx -1
        ret = [[z[i] for i in seq_idxes] for z in _tmp_content]  # final reindex!
        return tuple(ret)

# --
# b msp2/tasks/zmtl3/mod/extract/evt_arg/h_aug:
