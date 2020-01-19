#

# manually build analysis: see and stat and group all the fixes

try:
    from .ann import *
except:
    from ann import *
import sys
import json
import numpy as np
import pdb

# =====
LANGS = "en ar eu zh fi he hi it ja ko ru sv tr".split()
SYSTEMS = ["b0", "b1"]
RUNS = [1, 2, 3, 4, 5]

# =====
# supervised results
class FileGetterMono:
    def get_gold(self, cur_lang, cur_sys, cur_run):
        return f"../data/UD_RUN/ud24/{cur_lang}_test.conllu"

    def get_pred(self, cur_lang, cur_sys, cur_run):
        return f"../finals24/mg1_{cur_sys}_{cur_lang}_{cur_run}/output.test.txt"

# crossed results from English as source
class FileGetterCross:
    def get_gold(self, cur_lang, cur_sys, cur_run):
        return f"../data/UD_RUN/ud24/{cur_lang}_test.conllu"

    def get_pred(self, cur_lang, cur_sys, cur_run):
        return f"../cl_runs_from_tc/final_g1/cg1_{cur_sys}_en_{cur_run}/crout_{cur_lang}.out"

# =====
# stat for one pair of results
def get_stats(all_sents, all_tokens):
    _stats = [defaultdict(int), defaultdict(int)]  # sys1, sys2
    for one_sent in all_sents:
        # first stat for each system
        cur_len = one_sent.len
        toks, rtoks = one_sent.toks, one_sent.rtoks
        for sys_idx in range(len(SYSTEMS)):
            other_sys_idx = 1 - sys_idx  # the other system idx
            _cur_stat = _stats[sys_idx]
            # basic info
            _cur_stat["sent"] += 1
            _cur_stat["token"] += cur_len
            _cur_stat["ucorr"] += len([z for z in rtoks if z.ss[sys_idx].ucorr])
            _cur_stat["lcorr"] += len([z for z in rtoks if z.ss[sys_idx].lcorr])
            _cur_stat["uerr"] += len([z for z in rtoks if not z.ss[sys_idx].ucorr])
            _cur_stat["lerr"] += len([z for z in rtoks if not z.ss[sys_idx].lcorr])
            # for each fixes
            cur_fixes = one_sent.ptrees[sys_idx].fixes
            for one_fix in cur_fixes:
                one_fix_type = one_fix.type
                _cur_stat[f"fix_{one_fix_type}"] += 1
                _cur_stat[(one_fix_type, one_fix.category[0])] += 1  # specific fix type+category!
                _cur_stat[f"fix_{one_fix_type}_corrs"] += len(one_fix.corrections)
                if one_fix_type == "heading":
                    # then especially for co-fixes (unlabeled errors)
                    # the fix is mainly correct the gold edge of (gold_m, gold_h)
                    # gold_m is sure to be right, but gold_h can still be not-fixed
                    gold_m, gold_h = one_fix.changes[1].m, one_fix.changes[0].m
                    assert one_fix.changes[0].old_h == gold_m
                    assert not toks[gold_m].ss[sys_idx].ucorr and not toks[gold_h].ss[sys_idx].ucorr
                    #
                    hfix, hfix_cofix, hfix_cofix2 = 1, 0, 0
                    hit_center_m_flag = False
                    for z in one_fix.corrections:
                        if z == gold_m:
                            hit_center_m_flag = True
                        else:
                            hfix_cofix += 1
                            if z != gold_h:
                                hfix_cofix2 += 1
                    assert hit_center_m_flag
                    #
                    _cur_stat[f"hfix"] += hfix
                    _cur_stat[f"hfix_cofix"] += hfix_cofix
                    _cur_stat[f"hfix_cofix2"] += hfix_cofix2
                    if toks[gold_m].ss[other_sys_idx].ucorr:  # if in the other system, center-m is correct
                        _cur_stat[f"hfix_other"] += hfix
                        _cur_stat[f"hfix_other_cofix"] += hfix_cofix
                        _cur_stat[f"hfix_other_cofix2"] += hfix_cofix2
                        count_other_cofix_right = 0
                        count_other_cofix_right2 = 0
                        for z in one_fix.corrections:
                            if toks[z].ss[other_sys_idx].ucorr:
                                if z != gold_m:
                                    count_other_cofix_right += 1
                                    if z != gold_h:
                                        count_other_cofix_right2 += 1
                        _cur_stat[f"hfix_other_cofix_right"] += count_other_cofix_right
                        _cur_stat[f"hfix_other_cofix_right2"] += count_other_cofix_right2
    # =====
    # then pre-calculate sth
    for _cur_stat in _stats:
        sent, token = _cur_stat["sent"], _cur_stat["token"]
        _cur_stat.update({
            # uas/las
            "uas": _cur_stat["ucorr"]/token, "las": _cur_stat["lcorr"]/token,
            # number of corrections per fix
            "cpf_head": _cur_stat["fix_heading_corrs"] / _cur_stat["fix_heading"],
            "cpf_attach": _cur_stat["fix_attaching_corrs"] / _cur_stat["fix_attaching"],
            # cofix correct rate
            "cofix_rate": _cur_stat[f"hfix_other_cofix_right"] / (_cur_stat[f"hfix_other_cofix"] + 1e-3),})
    return _stats

#
class AveragedValue:
    def __init__(self, values):
        self.mean = np.mean(values)
        self.dev = np.std(values)
        self.values = values

    def __float__(self):
        return float(self.mean)

    def __repr__(self):
        return f"{self.mean:.4f}({self.dev:.4f})"

#
def show_results(final_results, keys=None):
    if keys is None:
        # default keys
        keys = ["uas", "las", "fix_attaching", "fix_heading", "cpf_head", "cofix_rate"]
    # keys = [("heading", k) for k in ['Others', 'Root', 'Punct', 'Mwe', 'Conj', 'Fun', 'N.Core', 'N.Other', 'C.Core', 'C.Other', 'Mod']]
    # keys = [("attaching", k) for k in ['Others', 'Root', 'Punct', 'Mwe', 'Conj', 'Fun', 'N.Core', 'N.Other', 'C.Core', 'C.Other', 'Mod']]
    lines = []
    res0 = AveragedValue([0.])
    for cur_lang, cur_res in final_results.items():
        lines.append([cur_lang, 0] + [str(cur_res[0].get(k, "--")) for k in keys])
        lines.append([cur_lang, 1] + [str(cur_res[1].get(k, "--")) for k in keys])
        lines.append([cur_lang, -1])
        for k in keys:
            r1, r2 = cur_res[0].get(k, res0), cur_res[1].get(k, res0)
            lines[-1].append(f"{(float(r2)-float(r1)) / (float(r1)+1e-3):.4f}")
    x = pd.DataFrame(lines, columns=["lang", "sys"] + keys)
    return x

# detailed results
# keys = [("heading", k) for k in ['Mwe', 'Conj', 'Fun', 'N.Core', 'N.Other', 'C.Core', 'C.Other', 'Mod']]
# keys = [("attaching", k) for k in ['Others', 'Root', 'Punct', 'Mwe', 'Conj', 'Fun', 'N.Core', 'N.Other', 'C.Core', 'C.Other', 'Mod']]
# todo(note): again, assume comparing two systems
def detailed_result(final_results, extra_keys=()):
    # =====
    # pretty-print: number / number-per-sentence / percentage
    def _pp(num, divs=()):
        num = float(num)
        ss = f"{num:.4f}"
        for one_div in divs:
            one_div = float(one_div)
            ss += f"/{num/one_div:.4f}"
        return ss
    # =====
    # for actual-number / number-per-sent:
    # uas las Num-UErr Num-FAtt(perc) Num-FHead FHead-CPF Num-FHead*CPF(perc) cofix-rate
    # heading: for each: Num Perc/Rank xCPF-Perc/Rank; attaching for each: Num Perc/Rank
    lines = []
    res0 = AveragedValue([0.])
    keys = ["uas", "las", "uerr", "Attach", "Head", "CPF", "Head*CPF", "Cofix"] + list(extra_keys)
    for cur_lang, cur_res in final_results.items():
        results = []
        for i in range(len(cur_res)):
            one_res = cur_res[i]
            sent = one_res["sent"]
            uas, las, uerr = one_res['uas'], one_res['las'], one_res['uerr']
            #
            n_fix_attaching = one_res['fix_attaching']
            n_fix_heading, n_cpf_head = one_res['fix_heading'], one_res['cpf_head']
            mul_head_cpf = n_float(float(n_fix_heading) * float(n_cpf_head))
            cofix_rate = one_res['cofix_rate']
            #
            results.append([uas, las, uerr, n_fix_attaching, n_fix_heading, n_cpf_head, mul_head_cpf, cofix_rate])
            lines.append([cur_lang, i] + [_pp(uas), _pp(las), _pp(uerr), _pp(n_fix_attaching, [sent, uerr]), _pp(n_fix_heading, [sent]), _pp(n_cpf_head), _pp(mul_head_cpf, [sent, uerr]), _pp(cofix_rate)])
            # extra keys
            for one_extra_key in extra_keys:
                cur_div = {"heading": n_fix_heading, "attaching": n_fix_attaching}[one_extra_key[0]]
                cur_fix = one_res.get(one_extra_key, res0)
                results[-1].append(cur_fix)
                # lines[-1].append(_pp(cur_fix, [sent, cur_div]))
                lines[-1].append(_pp(cur_fix, [cur_div]))
        # diff comparing results[-1] on results[-2]
        cur_diff = [cur_lang, -1]
        for a, b in zip(results[-2], results[-1]):
            try:
                delta = (float(b)-float(a)) / (float(a)+1e-3)
                delta = f"{delta*100:.2f}%"
            except:
                delta = "--"
            cur_diff.append(delta)
        lines.append(cur_diff)
    x = pd.DataFrame(lines, columns=["lang", "sys"] + keys)
    return x

# =====
def main():
    fgetter = FileGetterMono()
    # fgetter = FileGetterCross()
    #
    final_results = {}
    for cur_lang in LANGS:
        gold_parses = list(yield_ones(fgetter.get_gold(cur_lang, None, None)))
        tmp_lang_results = []
        all_keys = set()
        for cur_run in RUNS:
            zlog(f"Starting on {cur_lang} {cur_run}")
            # read them
            assert len(SYSTEMS) == 2, "Currently doing pairwise comparison!"
            sys_parses = [list(yield_ones(fgetter.get_pred(cur_lang, z, cur_run))) for z in SYSTEMS]
            # stat them
            all_sents, all_tokens = get_objs(gold_parses, sys_parses, GetterConf())
            #
            one_stats = get_stats(all_sents, all_tokens)
            for x in one_stats:
                all_keys.update(x.keys())
            tmp_lang_results.append(one_stats)
        # get the averaged results
        avg_lang_results = []
        for idx in range(len(SYSTEMS)):
            _one_sys_avg_result = {k:AveragedValue([tmp_lang_results[r][idx][k] for r in range(len(RUNS))]) for k in all_keys}
            avg_lang_results.append(_one_sys_avg_result)
        final_results[cur_lang] = avg_lang_results
    # =====
    # print the averaged things
    PickleRW.to_file(final_results, "_avg_res.pkl")
    show_results(final_results)
    pdb.set_trace()

if __name__ == '__main__':
    main()

"""
PYTHONPATH=../src/ python3 ~
from s2 import *; import pickle; x = pickle.load(open("_avg_mono.0919.pkl", 'rb')); detailed_result(x)
keys1 = [("heading", k) for k in ['Mwe', 'Conj', 'Fun', 'N.Core', 'N.Other', 'C.Core', 'C.Other', 'Mod']]
keys2 = [("attaching", k) for k in ['Others', 'Root', 'Punct', 'Mwe', 'Conj', 'Fun', 'N.Core', 'N.Other', 'C.Core', 'C.Other', 'Mod']]
z1=detailed_result(x, keys1)
z2=detailed_result(x, keys2)
z1.loc[:, ['lang'] + keys1]
z2.loc[:, ['lang'] + keys2]
z1.iloc[:, [0]+list(range(-8,0))]
z2.iloc[:, [0]+list(range(-8,0))]
"""
