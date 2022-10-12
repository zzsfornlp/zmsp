#

# perform the annotation projection with
# -- src-json, src-txt, trg-txt, align-result, (optional)align-params

from typing import List, Iterable
from collections import Counter
import math
import os
import numpy as np
from msp2.nn import BK
from msp2.utils import Conf, zlog, init_everything, zopen, OtherHelper
from msp2.data.inst import yield_sents, Sent, Frame, Mention
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

class MainConf(Conf):
    def __init__(self):
        # paths
        self.input = ReaderGetterConf()
        self.output = WriterGetterConf()
        self.src_txt = ""
        self.trg_txt = ""
        self.align_txt = ""
        self.fa_prefix = ""
        # detailed proj options
        self.ignore_evts = ["do.01", "be.03", "have.01", "become.03"]  # ignore aux verbs!
        self.err_notfound = True
        self.proj_thresh = 0.01
        self.cons_map0 = True  # use fa's output bidirectional alignments as pre-filtering
        self.exclude_evt_posi = False  # exclude evt posi when mapping args
        self.delete_evt_nofull = True  # delete evt if no full match
        self.delete_sent_nofull = False  # delete sent if no full match
        self.delete_sent_noevts = True  # remove empty sents without evts
        # --

def read_aligns(src_txt: str, trg_txt: str, align_txt: str):
    with zopen(src_txt) as fd:
        src_toks = [line.split() for line in fd]
    with zopen(trg_txt) as fd:
        trg_toks = [line.split() for line in fd]
    with zopen(align_txt) as fd:
        aligns = []
        for line in fd:
            one_align = {}
            for field in line.split():
                i, j = [int(z) for z in field.split('-')]
                if i in one_align:
                    one_align[i].append(j)
                else:
                    one_align[i] = [j]
            aligns.append(one_align)
    assert len(src_toks) == len(trg_toks) and len(src_toks) == len(aligns)
    ret = {}
    for s, t, aa in zip(src_toks, trg_toks, aligns):
        ret[" ".join(s)] = {"src": s, "trg": t, "align": aa}
    zlog(f"Read from {src_txt} {trg_txt} {align_txt}: {len(src_toks)} => {len(ret)}")
    return ret

class FastAlignParams:
    def __init__(self, prefix: str):
        self.T, self.m = FastAlignParams.read_err(prefix+"err")
        self.table = FastAlignParams.read_params(prefix+"params")
        zlog(f"Load from {prefix}: {self}")
        # --

    def __repr__(self):
        return f"FastAlignParams(T={self.T},m={self.m},entries={len(self.table)})"

    @staticmethod
    def read_err(err: str):
        (T, m) = (None, None)
        with zopen(err) as fd:
            for line in fd:
                # expected target length = source length * N
                if 'expected target length' in line:
                    m = float(line.split()[-1])
                # final tension: N
                elif 'final tension' in line:
                    T = float(line.split()[-1])
        return (T, m)

    @staticmethod
    def read_params(t: str):
        ret = {}
        with zopen(t) as fd:
            for line in fd:
                s, t, logprob = line.split()
                logprob = float(logprob)
                ret[(s,t)] = math.exp(logprob)
        return ret

    # --
    def safe_prob(self, s, t):
        return self.table.get((s,t), 1e-9)

    def UnnormalizedProb(self, i: int, j: int, m: int, n: int, alpha: float):
        feat = - abs(j/n - i/m)
        return math.exp(feat * alpha)

    def ComputeZ(self, i: int, m: int, n: int, alpha: float):
        _split = i*n/m
        _floor = int(_split)
        _ceil = _floor+1
        _ratio = math.exp(-alpha / n)
        _num_top = n - _floor
        ezt = ezb = 0
        if _num_top:
            ezt = self.UnnormalizedProb(i, _ceil, m, n, alpha) * (1.0 - pow(_ratio, _num_top)) / (1.0 - _ratio)
        if _floor:
            ezb = self.UnnormalizedProb(i, _floor, m, n, alpha) * (1.0 - pow(_ratio, _floor)) / (1.0 - _ratio)
        return ezb + ezt
    # --

    # --
    def get_probs(self, src_toks: List[str], trg_toks: List[str]):
        # adopted from https://github.com/clab/fast_align/blob/master/src/fast_align.cc
        # note: should use options as the recommended ones
        # --
        diagonal_tension = self.T
        mean_srclen_multiplier = self.m
        prob_align_null = 0.08
        # --
        _len_src, _len_trg = len(src_toks), len(trg_toks)
        # log_prob = Md::log_poisson(_len_src, 0.05 + _len_trg * mean_srclen_multiplier)
        arr_prob = np.zeros((1+_len_src, _len_trg))  # [NULL+src, trg]
        for j in range(_len_trg):  # for each target token
            wj = trg_toks[j]
            arr_prob[0, j] = self.safe_prob('<eps>', wj) * prob_align_null
            az = self.ComputeZ(j+1, _len_trg, _len_src, diagonal_tension) / (1.-prob_align_null)
            for i in range(_len_src):  # if aligning to this soruce token
                wi = src_toks[i]
                prob_a_i = self.UnnormalizedProb(j+1, i, _len_trg, _len_src, diagonal_tension) / az
                arr_prob[i+1, j] = self.safe_prob(wi, wj) * prob_a_i
        return arr_prob  # [src+1, trg]

# --
def map_mention(m: Mention, align_map: dict):
    _widx, _wlen = m.get_span()
    _trg_idxes = [align_map[i] for i in range(_widx, _widx+_wlen) if i in align_map]
    _trg_idxes = sorted(set(_trg_idxes))
    if len(_trg_idxes) == 0:
        return "NoMap", None
    else:
        # note: simply return the full span
        return "Mapped", (_trg_idxes[0], _trg_idxes[-1]-_trg_idxes[0]+1)

# helper
def print_info(info, align_map=None):
    if align_map is None:
        align_map = info['align']
    for k in sorted(align_map.keys()):
        v = align_map[k]
        if not isinstance(v, Iterable):
            v = [v]
        zlog(f"{k}-{v}: {info['src'][k]}|{[info['trg'][z] for z in v]}")

# [1+src, trg]
def get_align_map(align_probs: np.ndarray, thresh: float, cons_map: dict):
    _len_srcp1, _len_trg = align_probs.shape
    # collect all possible ones!
    cands = []
    for t in range(_len_trg):
        for best_s in reversed(np.argsort(align_probs[:,t])):
            _prob = align_probs[best_s, t].item()
            if best_s == 0 or _prob<thresh:
                break
            if cons_map is None or t in cons_map.get(best_s-1, []):  # note: remember to minus one for NULL
                cands.append((best_s, t, _prob))
    # simply greedy
    cands.sort(key=(lambda x: x[-1]), reverse=True)
    src2trg = {}
    src_hit, trg_hit = set(), set()
    for s, t, _prob in cands:
        if s not in src_hit and t not in trg_hit:
            src_hit.add(s)
            trg_hit.add(t)
            src2trg[s-1] = t  # note: remember to minus one for NULL
    return src2trg

# --
def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    # read src/trg/align
    info_align = read_aligns(conf.src_txt, conf.trg_txt, conf.align_txt)
    info_fa = FastAlignParams(conf.fa_prefix)
    # --
    _ignore_evts = set(conf.ignore_evts)
    cc = Counter()
    reader = conf.input.get_reader()
    with conf.output.get_writer() as writer:
        for sent in yield_sents(reader):
            cc["sent"] += 1
            _key = " ".join(sent.seq_word.vals)
            _info = info_align.get(_key)
            if conf.err_notfound:
                assert _info is not None
            if _info is None:
                cc["sent_notfound"] += 1
            # --
            _align_map0 = _info['align']
            _align_probs = info_fa.get_probs(_info["src"], _info["trg"])  # [1+src, trg]
            _align_probs = _align_probs / _align_probs.sum(0, keepdims=True)  # posterior over alignments
            # get new align map (all 1-to-1)
            _align_map = get_align_map(_align_probs, conf.proj_thresh, _align_map0 if conf.cons_map0 else None)
            # --
            assert len(sent) == len(_info["src"])
            trg_sent = Sent.create(_info["trg"])  # create target sent
            # --
            _sent_full = True
            for evt in sent.events:
                # --
                if evt.label in _ignore_evts:
                    cc["_ignore_evt"] += 1
                    continue  # ignore these!
                # --
                _evt_full = True
                cc["evt"] += 1
                cc["arg"] += len(evt.args)
                # map predicate
                status, posi = map_mention(evt.mention, _align_map)
                if posi is None:
                    cc["arg_noevt"] += len(evt.args)
                    trg_evt = None
                    _evt_full = False
                else:
                    trg_evt = trg_sent.make_event(posi[0], posi[1], type=evt.type)
                    # map args
                    _exclude_set = set(range(posi[0], posi[0]+posi[1])) if conf.exclude_evt_posi else set()
                    for arg in evt.args:
                        status, posi = map_mention(arg.mention, _align_map)
                        cc[f"arg_{status}"] += 1
                        if posi is not None:
                            trg_ef = trg_sent.make_entity_filler(posi[0], posi[1], type=arg.arg.type)
                            trg_evt.add_arg(trg_ef, role=arg.role)
                        else:
                            _evt_full = False
                    # --
                # --
                _sent_full = _sent_full and _evt_full
                if not _evt_full and conf.delete_evt_nofull:
                    if trg_evt is not None:
                        trg_sent.delete_frame(trg_evt, 'evt')
                        status = "nofull"
                cc[f"evt_{status}"] += 1
            # --
            if not _sent_full and conf.delete_sent_nofull:
                cc["sent_nofull"] += 1
            elif conf.delete_sent_noevts and len(trg_sent.events)==0:
                cc["sent_empty"] += 1
            else:
                # --
                # if len(trg_sent.events) > len(set(e.mention.shead_widx for e in trg_sent.events)):
                #     breakpoint()
                # --
                writer.write_inst(trg_sent)
                cc["sent_ok"] += 1
    # --
    zlog(f"Finish projection with: \n{OtherHelper.printd_str(cc)}")

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# --
# python3 proj_anns.py input_path:? output_path:? src_txt:? trg_txt:? align_txt:? fa_prefix:?
