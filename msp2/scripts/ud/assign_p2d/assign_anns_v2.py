#

# another version of "assign_anns"
# note: mainly for conll09 (en & zh & cs & ca & es)

import sys
import re
from collections import Counter
import string
from typing import List
from collections import defaultdict
from msp2.utils import Conf, zlog, init_everything, zwarn, Constants, AlgoHelper
from msp2.data.inst import yield_sents, Sent
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

# --
class MainConf2(Conf):
    def __init__(self):
        self.input = ReaderGetterConf()
        self.aux = ReaderGetterConf()
        self.output = WriterGetterConf()
        self.output_sent_and_discard_nonhit = False
        # --
        # indexer options
        self.key_dels = "-_"  # delete these chars for key
        self.search_topk = 1  # search for topK?

class SimpleTfIdfSearcher:
    def __init__(self, corpus):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus)
        self.vectorizer = vectorizer
        self.X = X  # [n_sample, n_feat]<Sparse>
        zlog(f"Build SimpleTfIdfSearcher of {X.shape}")

    def search(self, documents, K: int):
        import numpy as np
        Q = self.vectorizer.transform(documents)  # [n_input, n_feat]<Sparse>
        results = np.dot(Q, self.X.T).toarray()  # [n_input, n_sample]
        topk_indexes = results.argpartition(-K)[:,-K:]  # [n_input, K]
        topk_vals = np.take_along_axis(results, topk_indexes, -1)  # [n_input, K]
        # final sort
        _tmp_idxes = topk_vals.argsort(-1)[:,::-1]  # [n_input, K]
        ret_indexes = np.take_along_axis(topk_indexes, _tmp_idxes, -1)  # [n_input, K]<sorted>
        return ret_indexes

class MyIndexer2:
    def __init__(self, conf: MainConf2):
        self.conf = conf
        # --
        self.delete_char_set = set(conf.key_dels)
        self.all_items = []
        self.key_cands = defaultdict(list)  # cands by key
        self.simple_searcher = None
        # --

    def __len__(self):
        return len(self.all_items)

    def _get_key(self, sent):
        _dels = self.delete_char_set
        # note: delete certain chars!
        ss = ''.join(sent.seq_word.vals)
        # ss = ''.join(sent.seq_word.vals).lower()  # note: no lower!
        ss = ''.join([c for c in ss if c not in _dels])
        return ss

    def put(self, sent):
        assert self.simple_searcher is None, "Cannot grow it dynamically!"
        _key = self._get_key(sent)
        # --
        _cands = self.key_cands[_key]
        if not any(s.seq_word.vals==sent.seq_word.vals for s in _cands):
            # no exact matching!
            self.key_cands[_key].append(sent)
            self.all_items.append(sent)
            if len(self.key_cands[_key]) > 1:
                zwarn(f"Sents with same keys: {[z.seq_word for z in self.key_cands[_key]]}")
        # --

    def query(self, sent):
        # --
        if self.simple_searcher is None:
            self.simple_searcher = SimpleTfIdfSearcher([' '.join(s.seq_word.vals) for s in self.all_items])
        # --
        _key = self._get_key(sent)
        _cands = self.key_cands[_key]  # by key
        match_status = ""
        # --
        # get candidates
        if len(_cands) <= 0:  # no match?
            _idxes = self.simple_searcher.search([' '.join(sent.seq_word.vals)], self.conf.search_topk)[0]
            _cands = [self.all_items[z] for z in _idxes]
            match_status += "Fuzzy_"
            # breakpoint()
        else:
            match_status += "Key_"
        # --
        # iter all the cands
        _cand_packs = []
        for cand in _cands:
            _res = AlgoHelper.align_seqs(sent.seq_word.vals, cand.seq_word.vals, prefer_cont=True)
            _dist = sum(1 for z in _res[0] if z is None) + sum(1 for z in _res[1] if z is None)
            _cand_packs.append((cand, _res, _dist))
        best_cand, best_cand_res, _ = min(_cand_packs, key=(lambda x: x[-1]))
        res = self._align_sents(sent, best_cand, best_cand_res)
        # --
        return res, match_status

    # get head word from a span
    def _get_head(self, idxes: List, tree):
        _depths = tree.depths
        _heads = tree.seq_head.vals
        _labels = tree.seq_label.vals
        # --
        _best_idx, _best_d = idxes[0], _depths[idxes[0]]
        for ii in idxes[1:]:
            if _depths[ii] < _best_d:  # note: here prefer the first ones!
                _best_idx, _best_d = ii, _depths[ii]
        # --
        # not good phrase!
        if (_heads[_best_idx]-1 in idxes) or any(ii!=_best_idx and (_heads[ii]-1 not in idxes) for ii in idxes):
            _words = tree.sent.seq_word.vals
            _ss = " / ".join([f"({ii})[W={_words[ii]}][H={_heads[ii]-1}][L={_labels[ii]}]" for ii in idxes])
            zwarn(f"Strange phrase head=[{_best_idx}]: "+_ss)
        # --
        return _best_idx

    # sub align (align on chars)
    def _sub_align_toks(self, toks1: List[str], toks2: List[str]):
        # info of end of toks
        end_map1, end_map2 = {}, {}  # char-end-idx -> word-idx
        for _toks, _map in zip([toks1, toks2], [end_map1, end_map2]):
            c = 0
            # note: if tok is "", then let the later ones overwrite previous
            for _i, _t in enumerate(_toks):
                c += len(_t)
                _map[c-1] = _i  # -1 for that actual end-char
        # align the char seq
        align_res = AlgoHelper.align_seqs(''.join(toks1), ''.join(toks2), prefer_cont=True)
        matched_pairs = [(a, b) for a, b in zip(align_res[0], align_res[1]) if (a is not None and b is not None)]
        matched_tok_pairs = []
        ret_groups = []  # [([idxes1], [idxes2]), ([...], [...]), ...]
        _last_s = -1, -1
        for p1, p2 in matched_pairs:  # aligned at end
            s1, s2 = end_map1.get(p1), end_map2.get(p2)
            if s1 is not None and s2 is not None:
                matched_tok_pairs.append((s1, s2))
                assert s1>_last_s[0] and s2>_last_s[1], "Err: not moving forward!"
                ret_groups.append((list(range(_last_s[0]+1, s1+1)), list(range(_last_s[1]+1, s2+1))))
                _last_s = s1, s2
        # --
        if _last_s[0]+1<len(toks1) and _last_s[1]+1<len(toks2):
            ret_groups.append((list(range(_last_s[0]+1, len(toks1))), list(range(_last_s[1]+1, len(toks2)))))
        return ret_groups

    # delete any match that is not connected with before or after, to avoid problems of "de" in ca/es
    # -> let later sub_align to handle!
    def _delete_single_match(self, matched_pairs):
        _dels = []
        for ii in range(1, len(matched_pairs)-1):
            m0, mc, m1 = [matched_pairs[ii+z] for z in [-1,0,1]]
            if (mc[0]==m0[0]+1 and mc[1]==m0[1]+1) or (mc[0]==m1[0]-1 and mc[1]==m1[1]-1):
                pass
            else:
                _dels.append(ii)
        _survives = sorted(set(range(len(matched_pairs))) - set(_dels))
        ret = [matched_pairs[z] for z in _survives]
        return ret

    # check no cycle in trees
    def _check_no_cycle(self, heads):
        for m in range(len(heads)):
            cur = m
            _p = set()
            while cur >= 0:
                # --
                # check loop
                assert cur not in _p
                _p.add(cur)
                cur = heads[cur]-1
                # --
        return

    # --
    _BACKOFF_LABMAP = {
        # todo(+N): some actually cannot be mapped!
        "NAME": "flat", "HYPH": "punct", "HMOD": "compound", "COORD": "cc", "CONJ": "conj",
        "NMOD": "nmod", "AMOD": "advmod", "DEP": "dep", "P": "punct",  # en
        "suj": "nsubj",  # ca/es
    }
    # --

    # align cand and put anns to sent
    def _align_sents(self, sent, cand, align_res):
        _dels = self.delete_char_set
        matched_pairs0 = [(a, b) for a, b in zip(align_res[0], align_res[1]) if (a is not None and b is not None)]
        matched_pairs = self._delete_single_match(matched_pairs0)
        # --
        map1to2, map2to1 = {}, {}  # word idx maps
        words1, words2 = sent.seq_word.vals, cand.seq_word.vals
        tree1, tree2 = sent.tree_dep, cand.tree_dep
        lp1, lp2 = -1, -1
        for p1, p2 in matched_pairs + [(len(align_res[2]), len(align_res[3]))]:  # aligned at end
            # check the mismatched one
            idxes1, idxes2 = list(range(lp1+1, p1)), list(range(lp2+1, p2))
            if len(idxes1)>0 or len(idxes2)>0:
                _toks1 = ["".join([c for c in words1[z] if c not in _dels]) for z in idxes1]
                _toks2 = ["".join([c for c in words2[z] if c not in _dels]) for z in idxes2]
                if ''.join(_toks1) != ''.join(_toks2):
                    zwarn(f"Piece mismatched: {_toks1} vs {_toks2}")
                    # breakpoint()
                # sub align
                _subaligns = self._sub_align_toks(_toks1, _toks2)
                for _iis1, _iis2 in _subaligns:
                    _cur_idxes1, _cur_idxes2 = [idxes1[z] for z in _iis1], [idxes2[z] for z in _iis2]
                    if len(_cur_idxes1)>0 and len(_cur_idxes2)>0:  # only possible to align if both have words
                        h1, h2 = self._get_head(_cur_idxes1, tree1), self._get_head(_cur_idxes2, tree2)
                        assert h1 not in map1to2 and h2 not in map2to1
                        map1to2[h1] = h2
                        # note: specifically map more from 2 to 1
                        for _hh in _cur_idxes2:
                            if h2 in tree2.get_spine(_hh):
                                map2to1[_hh] = h1
            # add the matched one
            assert p1 not in map1to2 and p2 not in map2to1
            map1to2[p1] = p2
            map2to1[p2] = p1
            # next
            lp1, lp2 = p1, p2
        # --
        # assign deps
        _backoff_labmap = MyIndexer2._BACKOFF_LABMAP
        _res_heads, _res_labs, _res_poses = [], [], []
        for i1 in range(len(words1)):
            mapped_i2 = map1to2.get(i1)
            # upos
            if mapped_i2 is None:
                _res_poses.append("X")  # todo(+N): simply put an "X" here!
            else:
                _res_poses.append(cand.seq_upos.vals[mapped_i2])
            # dep
            mapped_i2_hidx = tree2.seq_head.vals[mapped_i2]-1 if mapped_i2 is not None else None
            back_i1_hidx = -1 if mapped_i2_hidx==-1 else map2to1.get(mapped_i2_hidx)
            # --
            if back_i1_hidx is None:  # no map: directly put original ones!
                _res_heads.append(tree1.seq_head.vals[i1])
                _old_lab = tree1.seq_label.vals[i1]
                if _old_lab not in _backoff_labmap:
                    zwarn(f"Unknown old label: {_old_lab}")
                    # breakpoint()
                _res_labs.append(_backoff_labmap.get(_old_lab, "dep"))  # by default "dep"
            else:
                _res_heads.append(back_i1_hidx+1)  # note: remember +1
                _res_labs.append(tree2.seq_label.vals[mapped_i2])
        # --
        # get a new sent!
        res = Sent.create(words1)
        res.build_uposes(_res_poses)
        res.build_dep_tree(_res_heads, _res_labs)
        self._check_no_cycle(_res_heads)
        return res

# --
def fix_words(sent):
    # some simple fix!
    _MAPS = [
        ("-LRB-", "("), ("-RRB-", ")"),
        ("-LSB-", "["), ("-RSB-", "]"),
        ("-LCB-", "{"), ("-RCB-", "}"),
        (r"\\/", "/"), (r"\\\*", "*"),
    ]
    # note: directly change!!
    word_vals = sent.seq_word.vals
    for ii in range(len(word_vals)):
        ww = word_vals[ii]
        for a,b in _MAPS:
            ww = re.sub(a, b, ww)
        word_vals[ii] = ww
    # --

def main(*args):
    conf: MainConf2 = init_everything(MainConf2(), args)
    # --
    # first read aux ones
    aux_insts = list(conf.aux.get_reader())
    aux_index = MyIndexer2(conf)
    num_aux_sent = 0
    for sent in yield_sents(aux_insts):
        num_aux_sent += 1
        fix_words(sent)
        aux_index.put(sent)
    zlog(f"Read from {conf.aux.input_path}: insts={len(aux_insts)}, sents={num_aux_sent}, len(index)={len(aux_index)}")
    # --
    # then read input
    input_insts = list(conf.input.get_reader())
    output_sents = []
    num_input_sent = 0
    num_reset_sent = 0
    num_hit_sent = 0
    cc_status = Counter()
    for sent in yield_sents(input_insts):
        num_input_sent += 1
        fix_words(sent)
        # --
        trg_sent, trg_status = aux_index.query(sent)
        cc_status[trg_status] += 1
        if trg_sent is not None:
            num_hit_sent += 1
            # note: currently we replace upos & tree_dep
            upos_vals, head_vals, deplab_vals = \
                trg_sent.seq_upos.vals, trg_sent.tree_dep.seq_head.vals, trg_sent.tree_dep.seq_label.vals
            sent.build_uposes(upos_vals)
            sent.build_dep_tree(head_vals, deplab_vals)
            output_sents.append(sent)
        else:
            zlog(f"Miss sent: {sent.seq_word}")
            if not conf.output_sent_and_discard_nonhit:
                output_sents.append(sent)
    # --
    zlog(f"Read from {conf.input.input_path}: insts={len(input_insts)}, sents={num_input_sent}, (out-sent-{len(output_sents)})"
         f"reset={num_reset_sent}({num_reset_sent/num_input_sent:.4f}) hit={num_hit_sent}({num_hit_sent/num_input_sent:.4f})")
    zlog(f"Query status: {cc_status}")
    # write
    with conf.output.get_writer() as writer:
        if conf.output_sent_and_discard_nonhit:
            writer.write_insts(output_sents)
        else:  # write the original insts
            writer.write_insts(input_insts)
    # --

# --
# python3 assign_anns_v2.py input.input_path:?? aux.input_path:?? output.output_path:??
if __name__ == '__main__':
    main(*sys.argv[1:])
