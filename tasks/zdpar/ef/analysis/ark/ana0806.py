#

# this one is the overall analysis, maybe need another one for one-vs-one compare/record/annotation

# analyze on the conllu files of 1) gold 2) system-outputs, 3) system2-outputs
# -- sentence-wise, token/step-wise

import sys
import json
import numpy as np
import pandas as pd
from typing import List, Iterable
from collections import defaultdict, Counter
from msp.zext.dpar.conllu_reader import ConlluReader
from msp.utils import zopen, zlog, StatRecorder, Helper, Conf, PickleRW, Constants

NEG_INF = Constants.REAL_PRAC_MIN

# =====
def yield_ones(file):
    r = ConlluReader()
    with zopen(file) as fd:
        for p in r.yield_ones(fd):
            # add extra info
            for line in p.headlines:
                try:
                    one = json.loads(line)
                    p.misc.update(one)
                except:
                    continue
            # add extra info to tokens
            cur_length = len(p)
            if "ef_score" in p.misc:
                ef_order, ef_score = p.misc["ef_order"], p.misc["ef_score"]
                assert len(ef_order) == cur_length and len(ef_score) == cur_length
                for cur_step in range(cur_length):
                    cur_idx, cur_score = ef_order[cur_step], ef_score[cur_step]
                    assert cur_idx > 0
                    cur_t = p.get_tok(cur_idx)
                    cur_t.misc["efi"] = cur_step
                    cur_t.misc["efs"] = cur_score
            if "g1_score" in p.misc:
                g1_score = p.misc["g1_score"][1:]
                assert len(g1_score) == cur_length
                sorted_idxes = list(reversed(np.argsort(g1_score)))
                for cur_step in range(cur_length):
                    cur_idx = sorted_idxes[cur_step] + 1  # offset by root
                    cur_score = g1_score[cur_idx-1]
                    assert cur_idx > 0
                    cur_t = p.get_tok(cur_idx)
                    cur_t.misc["g1i"] = cur_step
                    cur_t.misc["g1s"] = cur_score
            yield p

#
class AnalysisConf(Conf):
    def __init__(self, args):
        self.gold = ""  # gold parse file
        self.f1s = []  # main set of system files
        self.f2s = []  # second set of system files for comparison
        self.cache_pkl = ""  # cache stat to file
        self.force_refresh_cache = False  # ignore cache
        self.output = ""
        # analyze commands
        self.loop = False  # whether start a loop for interactive mode
        self.grain = "tokens"  # sents, tokens, steps
        self.fcode = "True"  # filter code
        self.mcode = "1"  # map code
        self.rcode = ""  # reduce code
        # =====
        self.update_from_args(args)
        self.validate()

def bool2int(x):
    return 1 if x else 0

def norm_pos(x):
    if x in {"NOUN", "PROPN", "PRON"}:
        return "N"
    else:
        return x

# =====
# helpers

class ZObject(object):
    def __init__(self, m=None):
        if m is not None:
            self.update(m)

    def update(self, m):
        for k, v in m.items():
            setattr(self, k, v)

def _get_token_object(gp_token, s1_tokens: List, s2_tokens: List):
    # first put basic nodes
    res = {"g": gp_token, "s1": s1_tokens, "s2": s2_tokens, "s1a": None, "s2a": None, "cr": None}
    for idx, node in enumerate(s1_tokens):
        res["s1"+str(idx)] = node
    for idx, node in enumerate(s2_tokens):
        res["s2"+str(idx)] = node
    # next get correctness info
    g_head, g_label = gp_token.head, gp_token.label
    for prefix in ["s1", "s2"]:
        cur_tokens = res[prefix]
        # results
        cur_heads = [z.head for z in cur_tokens]
        cur_labels = [z.label for z in cur_tokens]
        # correctness
        ucorrs = [int(z==g_head) for z in cur_heads]
        lcorrs = [int(u and z==g_label) for u,z in zip(ucorrs, cur_labels)]
        # set correctness
        for tok, ucorr, lcorr in zip(cur_tokens, ucorrs, lcorrs):
            tok.misc["ucorr"] = ucorr
            tok.misc["lcorr"] = lcorr
        # aggregate: agree only >= n//2+1!
        agr_all_num = len(cur_tokens)
        agr_target = (agr_all_num//2)+1
        u_counter = Counter(cur_heads)
        agr_head, unum = u_counter.most_common(1)[0]
        if unum<agr_target:
            agr_head = prefix  # todo(warn): unique incorrect tag!
        l_counter = Counter((h, l) for h, l in zip(cur_heads, cur_labels))
        agr_hl, lnum = l_counter.most_common(1)[0]
        agr_label = prefix if lnum<agr_target else agr_hl[1]
        # *agr: whether there is an agreement, *num: max matched number, *corr: agreement match gold
        aggr_info = {"num": agr_all_num, "ucorr_avg": sum(ucorrs)/agr_all_num, "lcorr_avg": sum(lcorrs)/agr_all_num,
                     "uagr": (agr_head is not None), "lagr": (agr_label is not None),
                     "head_agr": agr_head, "label_agr": agr_label, "uagr_num": unum, "lagr_num": lnum,
                     "ucorr_agr": int(agr_head==g_head), "lcorr_agr": int(agr_head==g_head and agr_label==g_label),}
        res[prefix+"a"] = ZObject(aggr_info)
    # get cross info on aggr info
    # *m2: both agree and match, *m3: *m2 + also match gold
    s1a, s2a = res["s1a"], res["s2a"]
    s12_um2 = s1a.head_agr == s2a.head_agr
    s12_lm2 = s12_um2 and (s1a.label_agr == s2a.label_agr)
    s12_um3 = (s1a.ucorr_agr and s2a.ucorr_agr)
    s12_lm3 = (s12_um3 and s1a.lcorr_agr and s2a.lcorr_agr)
    cross_info = {"um2": int(s12_um2), "lm2": int(s12_lm2), "um3": int(s12_um3), "lm3": int(s12_lm3),}
    res["cr"] = ZObject(cross_info)
    #
    ret = ZObject(res)
    return ret

def _get_sent_object(cur_token_objs):
    res = {
        # overall
        "len": len(cur_token_objs),
        # separate aggr correctness
        "s1_ucorr_agr": sum(z.s1a.ucorr_agr for z in cur_token_objs), "s1_lcorr_agr": sum(z.s1a.lcorr_agr for z in cur_token_objs),
        "s2_ucorr_agr": sum(z.s2a.ucorr_agr for z in cur_token_objs), "s2_lcorr_agr": sum(z.s2a.lcorr_agr for z in cur_token_objs),
        # separate avg correctness
        "s1_ucorr_avg": sum(z.s1a.ucorr_avg for z in cur_token_objs), "s1_lcorr_avg": sum(z.s1a.lcorr_avg for z in cur_token_objs),
        "s2_ucorr_avg": sum(z.s2a.ucorr_avg for z in cur_token_objs), "s2_lcorr_avg": sum(z.s2a.lcorr_avg for z in cur_token_objs),
        # cross
        "um2": sum(z.cr.um2 for z in cur_token_objs), "lm2": sum(z.cr.lm2 for z in cur_token_objs),
        "um3": sum(z.cr.um3 for z in cur_token_objs), "lm3": sum(z.cr.lm3 for z in cur_token_objs),
    }
    sobj = ZObject(res)
    for t in cur_token_objs:  # additional link
        t.sobj = sobj
    return sobj

#
def main(args):
    conf = AnalysisConf(args)
    # read them
    zlog("Read them all ...")
    gold_parses = list(yield_ones(conf.gold))
    s1_parses = [list(yield_ones(f)) for f in conf.f1s]
    s2_parses = [list(yield_ones(f)) for f in conf.f2s]
    # analyze them
    if conf.force_refresh_cache or conf.cache_pkl=="":
        zlog("Re-stat them all ...")
        assert all(len(gold_parses)==len(z) for z in s1_parses) and all(len(gold_parses)==len(z) for z in s2_parses), \
            "Number of sent mismatched"
        all_sents, all_tokens = [], []
        for sent_idx in range(len(gold_parses)):
            gp_tokens = gold_parses[sent_idx].tokens
            s1ps_tokens = [z[sent_idx].tokens for z in s1_parses]
            s2ps_tokens = [z[sent_idx].tokens for z in s2_parses]
            assert all([z.word for z in gp_tokens]==[z.word for z in zz] for zz in s1ps_tokens)
            assert all([z.word for z in gp_tokens]==[z.word for z in zz] for zz in s2ps_tokens)
            # =====
            # pre-computation
            cur_tokens = []
            # for each valid token
            for token_idx in range(1, len(gp_tokens)):
                now_gp_token = gp_tokens[token_idx]
                now_s1_tokens = [z[token_idx] for z in s1ps_tokens]
                now_s2_tokens = [z[token_idx] for z in s2ps_tokens]
                cur_t_obj = _get_token_object(now_gp_token, now_s1_tokens, now_s2_tokens)
                cur_tokens.append(cur_t_obj)
            # for the whole sentence
            cur_sent = _get_sent_object(cur_tokens)
            all_sents.append(cur_sent)
            all_tokens.extend(cur_tokens)
            # =====
            if conf.cache_pkl:
                zlog(f"Write cache to {conf.cache_pkl}")
                PickleRW.to_file([all_sents, all_tokens], conf.cache_pkl)
    else:
        zlog(f"Reload cache from {conf.cache_pkl}")
        all_sents, all_tokens = PickleRW.from_file(conf.cache_pkl)
    # =====
    # real analyzing
    zlog("Analyzing them all ...")
    fout = open(conf.output, "w") if conf.output else None
    while True:
        if conf.rcode != "":
            ana_insts = {"sents": all_sents, "tokens": all_tokens, "steps": all_tokens}[conf.grain]
            ff_f, ff_m, ff_r = [compile(z, "", "eval") for z in [conf.fcode, conf.mcode, conf.rcode]]
            zlog(f"# compute with {conf.grain}: f={conf.fcode}, m={conf.mcode}, r={conf.rcode}")
            count_all, count_hit = len(ana_insts), 0
            # todo(warn): use local names: 'z', 'cs'
            cs = []
            for z in ana_insts:
                if eval(ff_f):
                    count_hit += 1
                    cs.append(eval(ff_m))
            r = eval(ff_r)
            zlog(f"Rate = {count_hit} / {count_all} = {count_hit/count_all}")
            zlog(f"Res = {r}")
            if fout is not None:
                fout.write(str(r)+"\n")
        #
        if conf.loop:
            cmd = ""
            while len(cmd)==0:
                try:
                    cmd = input(">> ").strip()
                    conf.update_from_args([z.strip() for z in cmd.split("\"") if z.strip()!=""])
                except KeyboardInterrupt:
                    continue  # ignore Ctrl-C
        else:
            break
    if fout is not None:
        fout.close()

# =====
# helper functions

# helper for div
def hdiv(a, b):
    if b == 0:
        return (a, b, 0.)
    else:
        return (a, b, a/b)

# helper for plain counter & distribution
def hdistr(cs, key="count", ret_format="", min_count=0):
    r = Counter(cs)
    c = len(cs)
    get_keys_f = {
        "count": lambda r: sorted(r.keys(), key=(lambda x: -r[x])),
        "name": lambda r: sorted(r.keys()),
    }[key]
    thems = [(k, r[k], r[k]/c) for k in get_keys_f(r)]
    #
    ret_f = {
        "": lambda x: x,
        "s": lambda x: "\n".join([str(z) for z in x]),
    }[ret_format]
    thems = [z for z in thems if z[1]>=min_count]
    return ret_f(thems)

# helper for multi-level counter & distribution
def hmdistr(cs, fks=(), fss=(), ret_format="", min_count=0, show_level=Constants.INT_PRAC_MAX):
    # =====
    # collect
    max_level = 0
    record = [0, None, {}]  # str -> (count, value, next)
    for one in cs:
        max_level = max(len(one), max_level)
        cur_r = record
        cur_r[0] += 1  # overall count
        for i, x in enumerate(one):
            if x not in cur_r[-1]:
                cur_r[-1][x] = [0, None, {}]
            cur_r = cur_r[-1][x]
            cur_r[0] += 1
    # =====
    # sorting methods: Dict[Record] -> List[str]
    if not isinstance(fks, (list, tuple)):
        fks = [fks] * (max_level+1)
    elif len(fks) < max_level+1:
        fks = list(fks) + [""] * (max_level+1)
    MAP_F_KEYS = {
        "count": lambda r: sorted(r.keys(), key=(lambda x: -r[x][0])),  # sort by count
        "value": lambda r: sorted(r.keys(), key=(lambda x: -r[x][1])),  # sort by (summary) value
        "name": lambda r: sorted(r.keys()),
    }
    MAP_F_KEYS[""] = MAP_F_KEYS["count"]
    f_keys = [MAP_F_KEYS[k] for k in fks]
    # summary methods: Record(count, _, Dict[Record]) -> value
    if not isinstance(fss, (list, tuple)):
        fss = [fss] * (max_level+1)
    elif len(fss) < max_level+1:
        fss = list(fss) + [""] * (max_level+1)
    MAP_F_SUMS = {
        "zero": lambda r: 0.,
        "avg": lambda r: sum(k*v[0]/r[0] for k,v in r[-1].items()),
    }
    MAP_F_SUMS[""] = MAP_F_SUMS["zero"]
    f_sums = [MAP_F_SUMS[k] for k in fss]
    # =====
    # recursively visit
    stack_counts = []
    stack_keys = []
    def _visit(cur_r, cur_key_fs=None, cur_sum_fs=None, outputs=None):
        level = len(stack_counts)
        cur_count, _, cur_next = cur_r
        stack_counts.append(cur_count)
        visiting_keys = cur_next.keys() if (cur_key_fs is None) else (cur_key_fs[level](cur_next))
        for k in visiting_keys:
            stack_keys.append(k)
            # stat
            next_r = cur_next[k]
            _visit(next_r, cur_key_fs, cur_sum_fs, outputs)
            next_count = next_r[0]
            next_val = next_r[1]
            if outputs is not None and level<=show_level:
                outputs.append((level, stack_keys.copy(), next_count, next_val, [next_count/z for z in stack_counts]))
            stack_keys.pop()
        # summarize
        if cur_sum_fs is not None:
            cur_r[1] = cur_sum_fs[level](cur_r)
        stack_counts.pop()
    _visit(record, cur_sum_fs=f_sums)
    thems = []
    _visit(record, cur_key_fs=f_keys, outputs=thems)
    #
    ret_f = {
        "": lambda x: x,
        "s": lambda x: "\n".join([str(z) for z in x]),
    }[ret_format]
    thems = [z for z in thems if z[2] >= min_count]
    return ret_f(thems)

#
def hgroup(k, length, N, is_step=True):
    step = k
    all_steps = (2*length-2) if is_step else length-1
    g = int(step/all_steps*N)
    return g

#
def hbin(k, length, N):
    return int(k/length*N)

#
def hcmp_err(a, b):
    return {(0,0):"both_good", (1,1):"both_bad", (0,1):"sys_good", (1,0):"sys_bad"}[(a,b)]

def hclip(x, a, b):
    return min(max((x,a)),b)

def halb0(x):
    return x.split(":")[0]

# =====
# helper functions version 2: dumpers

# for mcode
def hdm_err_break(key, z):
    # unlabeled
    s1_corr = 1-z.s1p_uerr
    s2_corr = 1-z.s2p_uerr
    both_corr = s1_corr * s2_corr
    s1_err = z.s1p_uerr
    s1_first_err = z.s1p_real_uerr
    # labeled
    s1_corr_L = 1-z.s1p_lerr
    s2_corr_L = 1-z.s2p_lerr
    both_corr_L = s1_corr_L*s2_corr_L
    #
    s12_diff, s12_diff_L = s1_corr-s2_corr, s1_corr_L-s2_corr_L
    return (key, [both_corr, s1_corr, s2_corr, s12_diff, both_corr_L, s1_corr_L, s2_corr_L, s12_diff_L, s1_err, s1_first_err])

# for rcode
def hdr_err_break(cs):
    r = {}
    for key, arr in cs:
        if key not in r:
            r[key] = []
        r[key].append(arr)
    ret = {k: np.array(arrs).mean(axis=0).tolist() + [len(arrs)+0., len(arrs)/len(cs)] for k, arrs in r.items()}
    return json.dumps(ret)

# formatting for printing
def hdformat_err_break(line):
    import json
    ret = json.loads(line)
    for key, res in ret.items():
        both_corr, s1_corr, s2_corr, s12_diff, both_corr_L, s1_corr_L, s2_corr_L, s12_diff_L, s1_err, s1_first_err, *_ = res
        print(f"{key} & {s1_corr*100:.2f}/{s1_corr_L*100:.2f} & {s2_corr*100:.2f}/{s2_corr_L*100:.2f} & {s12_diff*100:.2f}/{s12_diff_L*100:.2f} \\\\")

# questions (interactive tryings)
"""
PYTHONPATH=../src/ python3 ana.py gold:en_dev.conllu f1s:out.ef f2s:out.g1 loop:1
#
# 1. UAS/LAS/Error-rate?
"mcode:z.s1a.ucorr_avg" "rcode:hdiv(sum(cs), len(cs))"
"mcode:z.s1a.lcorr_avg" "rcode:hdiv(sum(cs), len(cs))"
# 2. which steps are first? 
# step-bin vs. UAS
"mcode:(hbin(z.s10.efi, z.sobj.len, 10), z.s10.ucorr)" "rcode:hmdistr(cs, fks=['name'], ret_format='s', min_count=10)"
# step-bin vs. label
"mcode:(hbin(z.s10.efi, z.sobj.len, 10), halb0(z.s10.label))" "rcode:hmdistr(cs, fks=['name'], ret_format='s', min_count=10)"
# step-bin vs. ddist
"mcode:(hbin(z.s10.efi, z.sobj.len, 10), z.s10.ddist)" "rcode:hmdistr(cs, fks=['name'], ret_format='s', min_count=10)"
# avg percentage of label
"mcode:(halb0(z.s10.label), z.s10.efi/z.sobj.len)" "rcode:hmdistr(cs, fks=['value'], fss=['', 'avg'], ret_format='s', min_count=10, show_level=0)"
# avg percentage of ddist
"mcode:(hclip(z.s10.ddist, -10, 10), z.s10.efi/z.sobj.len)" "rcode:hmdistr(cs, fks=['value'], fss=['', 'avg'], ret_format='s', min_count=10, show_level=0)"
# 3. cmp
#
# 4. error breakdown
"""

if __name__ == '__main__':
    main(sys.argv[1:])
