#

# analyze on the conllu files of 1) gold 2) system-output, 3) system2-output
# -- sentence-wise, token-wise, TD step-wise

import sys
import json
import numpy as np
from collections import defaultdict, Counter
from msp.zext.dpar.conllu_reader import ConlluReader
from msp.utils import zopen, zlog, StatRecorder, Helper, Conf, PickleRW

# from tasks.zdpar.transition.topdown.decoder import ParseInstance, TdState

class ZObject(object):
    def __init__(self, m=None):
        if m is not None:
            self.update(m)

    def update(self, m):
        for k, v in m.items():
            setattr(self, k, v)

# =====
def yield_ones(file):
    r = ConlluReader()
    with zopen(file) as fd:
        for p in r.yield_ones(fd):
            # add root info
            corder_valid_flag = False
            for line in p.headlines:
                try:
                    misc0 = json.loads(line)["ROOT-MISC"]
                    for s in misc0.split("|"):
                        k, v = s.split("=")
                        p.tokens[0].misc[k] = v
                    corder_valid_flag = True
                    break
                except:
                    continue
            # add children order
            if corder_valid_flag:
                for t in p.tokens:
                    corder_str = t.misc.get("CORDER")
                    if corder_str is not None:
                        t.corder = [int(x) for x in corder_str.split(",")]
                    else:
                        t.corder = []
            yield p

#
class AnalysisConf(Conf):
    def __init__(self, args):
        self.gold = ""  # gold parse file
        self.f1 = ""  # main system file
        self.f2 = ""  # system file 2 for comparison
        self.cache_pkl = ""  # cache stat to file
        self.force_refresh_cache = False  # ignore cache
        self.output = ""
        # analyze commands
        self.loop = False  # whether start a loop for interactive mode
        self.grain = "steps"  # sents, tokens, steps
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

#
def main(args):
    conf = AnalysisConf(args)
    # read them
    zlog("Read them all ...")
    gold_parses = list(yield_ones(conf.gold))
    s1_parses = list(yield_ones(conf.f1))
    s2_parses = list(yield_ones(conf.f2))
    # analyze them
    if conf.force_refresh_cache or conf.cache_pkl=="":
        zlog("Re-stat them all ...")
        ana_sents, ana_tokens, ana_steps = [], [], []
        assert len(gold_parses) == len(s1_parses) and len(gold_parses) == len(s2_parses)
        for gp, s1p, s2p in zip(gold_parses, s1_parses, s2_parses):
            gp_tokens = gp.tokens
            s1p_tokens = s1p.tokens
            s2p_tokens = s2p.tokens
            this_length = len(gp_tokens)
            assert [z.word for z in gp_tokens] == [z.word for z in s1p_tokens]
            assert [z.word for z in gp_tokens] == [z.word for z in s2p_tokens]
            # =====
            # token wise
            this_tokens = []
            for n_idx in range(this_length):
                is_root = n_idx==0
                n_s1p_node = s1p_tokens[n_idx]
                n_gp_node = gp_tokens[n_idx]
                n_s2p_node = s2p_tokens[n_idx]
                # -----
                gp_head, gp_label, gp_children_set = n_gp_node.head, n_gp_node.label0, set(n_gp_node.childs)
                s1p_head, s1p_label, s1p_children_set = n_s1p_node.head, n_s1p_node.label0, set(n_s1p_node.childs)
                s2p_head, s2p_label, s2p_children_set = n_s2p_node.head, n_s2p_node.label0, set(n_s2p_node.childs)
                s12_ueq = s1p_head==s2p_head
                s12_leq = s12_ueq and (s1p_label==s2p_label)
                props_tok = {
                    "gp": n_gp_node, "s1p": n_s1p_node, "s2p": n_s2p_node,
                    # gold label
                    "pos": norm_pos(n_gp_node.upos), "hpos": norm_pos("R0" if is_root else n_gp_node.get_head().upos),
                    "head": 0 if is_root else gp_head, "label": "R0" if is_root else gp_label,
                    # predictions
                    "s1p_label": "R0" if is_root else s1p_label, "s2p_label": "R0" if is_root else s2p_label,
                    "s1p_ucorr": bool2int(s1p_head==gp_head), "s1p_lcorr": bool2int(s1p_head==gp_head and s1p_label==gp_label),
                    "s2p_ucorr": bool2int(s2p_head==gp_head), "s2p_lcorr": bool2int(s2p_head==gp_head and s2p_label==gp_label),
                    "s12_ueq": bool2int(s12_ueq), "s12_leq": bool2int(s12_leq),
                }
                this_tokens.append(ZObject(props_tok))
            # =====
            # step level
            this_steps = []
            # ===== go recursive
            this_reduced = [False] * this_length
            this_arc = [-1] * this_length
            this_numbers = [0, 0]  # (attach-num, reduce-num)
            def _visit(n_s1p_node, stack):
                n_idx = n_s1p_node.idx
                stack.append(n_s1p_node)
                res_h = this_tokens[n_idx]
                # attach
                prev_c_num = 0
                prev_c_uerr = 0
                prev_c_lerr = 0
                h_uerr = 1 - res_h.s1p_ucorr
                for c_ord, c in enumerate(n_s1p_node.corder):
                    res_c = this_tokens[c]
                    s1p_uerr, s1p_lerr = 1-bool2int(res_c.s1p_ucorr), 1-bool2int(res_c.s1p_lcorr)
                    s2p_uerr, s2p_lerr = 1-bool2int(res_c.s2p_ucorr), 1-bool2int(res_c.s2p_lcorr)
                    real_h_reduced = bool2int(this_reduced[res_c.head])  # possible error because of reduced real head
                    #
                    if s1p_uerr:
                        err_type = "err_reduce" if real_h_reduced else "err_attach"
                    else:
                        err_type = "err_none"
                    #
                    props_step = {
                        "len": this_length, "step": len(this_steps), "depth": len(stack), "attach_num": this_numbers[0],
                        "type": "attach", "h": res_h, "c": res_c, "c_ord": c_ord, "h_uerr": h_uerr,
                        # int (count)
                        "prev_c_num": prev_c_num, "prev_c_uerr": prev_c_uerr, "prev_c_lerr": prev_c_lerr,
                        # bool -> int
                        "s1p_uerr": s1p_uerr, "s1p_lerr": s1p_lerr, "s2p_uerr": s2p_uerr, "s2p_lerr": s2p_lerr,
                        "real_h_reduced": real_h_reduced, "s1p_real_uerr": s1p_uerr*(1-real_h_reduced),
                        # which kind of error
                        "err": err_type,
                    }
                    prev_c_num += 1
                    prev_c_uerr += s1p_uerr
                    prev_c_lerr += s1p_lerr
                    this_arc[c] = n_idx
                    this_numbers[0] += 1
                    this_steps.append(ZObject(props_step))
                    # recursive
                    _visit(s1p_tokens[c], stack)
                # reduce non-root
                if n_idx>0:
                    real_c_num = len(res_h.gp.childs)
                    miss_c_num = real_c_num - (prev_c_num - prev_c_uerr)
                    s2p_c_num, s2p_c_uerr, s2p_c_lerr, s2p_c_miss = 0, 0, 0, 0
                    for s2p_c in res_h.s2p.childs:
                        s2p_c_num += 1
                        res_s2p_c = this_tokens[s2p_c]
                        s2p_c_uerr += 1-bool2int(res_s2p_c.s2p_ucorr)
                        s2p_c_lerr += 1-bool2int(res_s2p_c.s2p_lcorr)
                    s2p_c_miss = real_c_num - (s2p_c_num - s2p_c_uerr)
                    props_step = {
                        "len": this_length, "step": len(this_steps), "depth": len(stack), "reduce_num": this_numbers[1],
                        "type": "reduce", "h": res_h, "h_uerr": h_uerr,
                        # int (count)
                        "real_c_num": real_c_num,
                        "prev_c_num": prev_c_num, "prev_c_uerr": prev_c_uerr, "prev_c_lerr": prev_c_lerr,
                        "miss_c_num": miss_c_num, "miss_real_c_num": len([1 for zz in res_h.gp.childs if this_arc[zz]<0]),
                        "s2p_c_num": s2p_c_num, "s2p_c_uerr": s2p_c_uerr, "s2p_c_lerr": s2p_c_lerr, "s2p_c_miss": s2p_c_miss,
                    }
                    this_steps.append(ZObject(props_step))
                    this_reduced[n_idx] = True
                    this_numbers[1] += 1
                stack.pop()
            # =====
            cur_root = s1p_tokens[0]
            cur_stack = []
            _visit(cur_root, cur_stack)
            # sentence level
            assert len(this_steps) == 2*(len(this_tokens)-1)
            assert len([z for z in this_steps if z.type=="attach"]) == len(this_tokens)-1
            this_sent = {
                "tokens": this_tokens, "steps": this_steps
            }
            #
            ana_steps.extend(this_steps)
            ana_tokens.extend(this_tokens)
            ana_sents.append(this_sent)
            if conf.cache_pkl:
                zlog(f"Write cache to {conf.cache_pkl}")
                PickleRW.to_file([ana_sents, ana_tokens, ana_steps], conf.cache_pkl)
    else:
        zlog(f"Reload cache from {conf.cache_pkl}")
        ana_sents, ana_tokens, ana_steps = PickleRW.from_file(conf.cache_pkl)
    # =====
    # real analyzing
    zlog("Analyzing them all ...")
    fout = open(conf.output, "w") if conf.output else None
    while True:
        if conf.rcode != "":
            ana_insts = {"sents": ana_sents, "tokens": ana_tokens, "steps": ana_steps}[conf.grain]
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
                cmd = input(">> ").strip()
                conf.update_from_args([z.strip() for z in cmd.split("\"") if z.strip()!=""])
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
def hmdistr(cs, key="count", ret_format="", min_count=0):
    record = [0, {}]  # str -> (count, next)
    # collect
    for one in cs:
        cur_r = record
        cur_r[0] += 1  # overall count
        for i, x in enumerate(one):
            if x not in cur_r[1]:
                cur_r[1][x] = [0, {}]
            cur_r = cur_r[1][x]
            cur_r[0] += 1
    # recursive visit
    get_keys_f = {
        "count": lambda r: sorted(r.keys(), key=(lambda x: -r[x][0])),  # sort by count
        "name": lambda r: sorted(r.keys()),
    }[key]
    #
    stack_counts = []
    stack_keys = []
    thems = []
    def _visit(cur_r):
        level = len(stack_counts)
        cur_count, cur_next = cur_r
        stack_counts.append(cur_count)
        for k in get_keys_f(cur_next):
            stack_keys.append(k)
            # stat
            next_r = cur_next[k]
            next_count = next_r[0]
            thems.append((level, stack_keys.copy(), next_count, [next_count/z for z in stack_counts]))
            _visit(next_r)
            stack_keys.pop()
        stack_counts.pop()
    _visit(record)
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
def hcmp_err(a, b):
    return {(0,0):"both_good", (1,1):"both_bad", (0,1):"sys_good", (1,0):"sys_bad"}[(a,b)]

def hclip(x, a, b):
    return min(max((x,a)),b)

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
PYTHONPATH=../src/ python3 ana0.py gold:../data/ud22/en_test.conllu f1:en.td1.out f2:en.g.out loop:1
#
# 1. UAS/LAS/Error-rate?
"grain:steps" "fcode:z.type=='attach'" "mcode:z.s1p_uerr" "rcode:hdiv(sum(cs), len(cs))"
"grain:steps" "fcode:z.type=='attach'" "mcode:z.s1p_lerr" "rcode:hdiv(sum(cs), len(cs))"
# real attach/reduce error
"grain:steps" "fcode:z.type=='attach'" "mcode:z.s1p_real_uerr" "rcode:hdiv(sum(cs), len(cs))"
"grain:steps" "fcode:z.type=='reduce'" "mcode:z.miss_real_c_num" "rcode:hdiv(sum(cs), len(cs))"
#
# 2. children order
# 2.1 stat of gold children
"grain:tokens" "fcode:z.hpos!='R0'" "mcode:(z.hpos, z.label)" "rcode:hmdistr(cs, ret_format='s', min_count=10)"
# 2.2 stat of pred chidren
"grain:steps" "fcode:z.type=='attach'" "mcode:(z.h.pos, z.c.s1p_label, z.c_ord)" "rcode:hmdistr(cs, ret_format='s', min_count=10)"
#
# 3. comparison
# 3.1 overall
"grain:steps" "fcode:z.type=='attach'" "mcode:(z.s1p_uerr, z.s2p_uerr, z.s1p_real_uerr, z.step)" "rcode:hmdistr(cs, ret_format='s', min_count=10)"
# 3.2 wrong ones
"grain:steps" "fcode:z.type=='attach' and z.s1p_uerr and not z.s2p_uerr" "mcode:(z.s1p_real_uerr, z.c.s1p.ddist)" "rcode:hmdistr(cs, ret_format='s', min_count=10)"
#
# 4. error breakdown
# what to breakdown? -> z.step, z.c.s1p.rdist, z.c.gp.rdist, z.c.s1p.ddist, z.c.gp.ddist, z.c.label, z.h.pos, z.c.pos,
# hgroup(z.step, z.len, 5), hgroup(z.attach_num, z.len, 5, False), hclip(z.c.s1p.ddist, -20, 20), hclip(z.c.gp.rdist, 0, 7)
# hcmp_err(z.s1p_uerr, z.s2p_uerr), z.s1p_real_uerr
"grain:steps" "fcode:z.type=='attach'" "mcode:(z.c.gp.rdist, hcmp_err(z.s1p_lerr, z.s2p_lerr),)" "rcode:hmdistr(cs, ret_format='s', min_count=0, key='name')"
"grain:steps" "fcode:z.type=='attach'" "mcode:(z.c.gp.rdist, z.s1p_uerr, z.s1p_lerr,)" "rcode:hmdistr(cs, ret_format='s', min_count=0, key='name')"
"grain:steps" "fcode:z.type=='attach'" "mcode:(z.c.gp.rdist, z.s2p_uerr, z.s2p_lerr,)" "rcode:hmdistr(cs, ret_format='s', min_count=0, key='name')"
#
"grain:steps" "fcode:z.type=='attach'" "mcode:hdm_err_break(hgroup(z.step, z.len, 5), z)" "rcode:hdr_err_break(cs)"
"""

# run in batch mode
"""
# -- 
# file: cmd.txt
"grain:steps" "fcode:z.type=='attach'" "mcode:hdm_err_break(hgroup(z.step, z.len, 5), z)" "rcode:hdr_err_break(cs)"
"grain:steps" "fcode:z.type=='attach'" "mcode:hdm_err_break(hgroup(z.attach_num, z.len, 5, False), z)" "rcode:hdr_err_break(cs)"
"grain:steps" "fcode:z.type=='attach'" "mcode:hdm_err_break(hclip(z.c.gp.rdist, 0, 7), z)" "rcode:hdr_err_break(cs)"
"grain:steps" "fcode:z.type=='attach'" "mcode:hdm_err_break(hclip(z.c.s1p.rdist, 0, 7), z)" "rcode:hdr_err_break(cs)"
"grain:steps" "fcode:z.type=='attach'" "mcode:hdm_err_break(hclip(z.c.gp.ddist, -10, 10), z)" "rcode:hdr_err_break(cs)"
"grain:steps" "fcode:z.type=='attach'" "mcode:hdm_err_break(hclip(z.c.s1p.ddist, -10, 10), z)" "rcode:hdr_err_break(cs)"
"grain:steps" "fcode:z.type=='attach'" "mcode:hdm_err_break(z.c.label, z)" "rcode:hdr_err_break(cs)"
"grain:steps" "fcode:z.type=='attach'" "mcode:hdm_err_break(z.c.pos, z)" "rcode:hdr_err_break(cs)"
"grain:steps" "fcode:z.type=='attach'" "mcode:hdm_err_break(z.c.hpos, z)" "rcode:hdr_err_break(cs)"
"grain:steps" "fcode:z.type=='attach'" "mcode:hdm_err_break(z.c.s1p_label, z)" "rcode:hdr_err_break(cs)"
"grain:steps" "fcode:z.type=='attach'" "mcode:hdm_err_break(z.h.pos, z)" "rcode:hdr_err_break(cs)"
#
for wset in test train; do
for cl in ar bg ca hr cs da nl en et "fi" fr de he hi id it ko la lv no pl pt ro ru sk sl es sv uk; do
echo "RUN ${wset} ${cl}"
PYTHONPATH=../src/ python3 ana0.py gold:../data/ud22/${cl}_${wset}.conllu f1:${wset}/${cl}.td.out f2:${wset}/${cl}.g.out loop:1 output:${wset}/${cl}.td.json <cmd.txt
done; done |& tee logana
"""

# combining results from different languages
def combine_results():
    ALL_LANGS = "ar bg ca hr cs da nl en et fi fr de he hi id it ko la lv no pl pt ro ru sk sl es sv uk".split()
    TARGET_LANGS = [z for z in ALL_LANGS if z!="en"]
    MAIN_DIR = "./test/"
    import json
    import numpy as np
    from pprint import pprint
    # read all
    results = {}
    for cl in TARGET_LANGS:
        with open(f"{MAIN_DIR}/{cl}.td.json") as fd:
            for idx, line in enumerate(fd):
                if idx not in results:
                    results[idx] = {}
                line = line.strip()
                if len(line) > 0:
                    one_dict = json.loads(line)
                    for k,v in one_dict.items():
                        if k not in results[idx]:
                            results[idx][k] = []
                        results[idx][k].append(v)
    # average
    # final_results = {k:{} for k in results}
    final_results = [{} for i in range(len(results))]
    for k1 in results:
        for k2 in results[k1]:
            final_results[k1][k2] = np.array(results[k1][k2]).mean(0).tolist()
    with open(f"./{MAIN_DIR}/avg.json", "w") as fd:
        for one in final_results:
            fd.write(json.dumps(one)+"\n")
    print(final_results)
    #
    pp = (lambda x: pprint([(z, x[z]) for z in sorted(x.keys(), key=lambda k: x[k][1]-x[k][2])]))
    return final_results

if __name__ == '__main__':
    main(sys.argv[1:])
