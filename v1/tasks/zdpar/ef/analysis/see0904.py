#

# see the fixes

import sys
import json
try:
    from .ann import *
except:
    from ann import *

# write results to std-out
def output_one(d):
    s = json.dumps(d)
    print(s)

# =====
# compare one pair of parses
def main(args):
    conf = AnalysisConf(args)
    # =====
    if conf.load_name == "":
        # recalculate them
        # read them
        zlog("Read them all ...")
        gold_parses = list(yield_ones(conf.gold))
        sys_parses = [list(yield_ones(z)) for z in conf.fs]
        # =====
        # stat them
        zlog("Stat them all ...")
        all_sents, all_tokens = get_objs(gold_parses, sys_parses, conf.getter)
        analyzer = ParsingAnalyzer(conf.ana, all_sents, all_tokens, conf.labeled)
        if conf.save_name != "":
            analyzer.do_save(conf.save_name)
    else:
        analyzer = ParsingAnalyzer(conf.ana, None, None, conf.labeled)
        analyzer.do_load(conf.load_name)
        all_sents, all_tokens = analyzer.vars.sents, analyzer.vars.tokens
        # =====
    # =====
    # collect the tokens according to the fixes
    # one token (and its edge) can be classified into 4/5 classes: reverse-fix, co-fix, attach-fix, label-fix / no-fix?
    # -----
    # basic: analyze for both mono- and cross- parsing for multiple languages, any relations between?
    # question 0: 1) how many errors there? how many fixes are there, especially co-fix? 2) if reverse-fix get fixed, will the co-fixes be also fixed? (s1 as baseline, s2 as better model)
    # question 1: what is s2 better at? reverse or attach? any specific type?
    # question 1.5: extra surprise: confidence?
    # question 2: will extra link stats help parsing and get similar error reduction (for example, for simpler patterns like N-attach)?
    #
    # the layering is: sent -> fix -> token
    num_sys = len(conf.fs)
    # analyze pairs
    assert len(conf.fs) == 2, "Here we only analyzing pairwise info!"
    q1_stats = [defaultdict(int), defaultdict(int)]  # stats for sys1 & sys2
    for one_sent in all_sents:
        # first stat for each system
        cur_len = one_sent.len
        toks, rtoks = one_sent.toks, one_sent.rtoks
        for sys_idx in range(num_sys):
            other_sys_idx = 1-sys_idx  # the other system idx
            _cur_stat = q1_stats[sys_idx]
            # basic info
            _cur_stat["sent"] += 1
            _cur_stat["token"] += cur_len
            _cur_stat["uerr"] += len([z for z in rtoks if not z.ss[sys_idx].ucorr])
            _cur_stat["lerr"] += len([z for z in rtoks if not z.ss[sys_idx].lcorr])
            # for each fixes
            cur_fixes = one_sent.ptrees[sys_idx].fixes
            for one_fix in cur_fixes:
                one_fix_type = one_fix.type
                _cur_stat[f"fix_{one_fix_type}"] += 1
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
    # after collection, do some calculations
    q1_res = []
    for _cur_stat in q1_stats:
        sent, token = _cur_stat["sent"], _cur_stat["token"]
        _cur_res = {"sent": sent, "token": token, "uas": 1.-_cur_stat["uerr"]/token, "las": 1.-_cur_stat["lerr"]/token,
                    "uerr": _cur_stat["uerr"]/token, "lerr": _cur_stat["lerr"]/token,
                    # number of fix per sent
                    "fps_head": _cur_stat["fix_heading"]/sent,
                    "fps_attach": _cur_stat["fix_attaching"]/sent,
                    # number of corrections per sentence
                    "cps_head": _cur_stat["fix_heading_corrs"]/sent,
                    "cps_attach": _cur_stat["fix_attaching_corrs"]/sent,
                    # number of corrections per fix
                    "cpf_head": _cur_stat["fix_heading_corrs"]/_cur_stat["fix_heading"],
                    "cpf_attach": _cur_stat["fix_attaching_corrs"]/_cur_stat["fix_attaching"],
                    # cofix correct rate
                    "cofix_rate": _cur_stat[f"hfix_other_cofix_right"]/(_cur_stat[f"hfix_other_cofix"]+1e-3),
                    }
        q1_res.append(_cur_res)
        Helper.printd(_cur_res, " || ")
    _cmp_res = {k: (q1_res[0][k]-q1_res[1][k])/q1_res[0][k] for k in q1_res[0]}
    Helper.printd(_cmp_res, " || ")

    # todo(note): hypothesis, bert features help fix attachment errors more?
    #  <- near and co-occur words have higher chance to have grammatical relationships
    #  <- frequent specific patterns have higher chance to ...
    # +1 bag of in-between word as weak context?
    # both overall-attachment analysis and heading/attachment analysis
    # bert knows the co-location of links but not the head?

    # back-edge #num-gap for cross-lingual parsing
    # three classes: reverse-edge, fix-by-reverse attachment, remaining attachment

    # calculate stats and breakdown and print examples-ids

if __name__ == '__main__':
    main(sys.argv[1:])

# example seeing
# for zcl in en ar eu zh 'fi' he hi it ja ko ru sv tr; do echo ${zcl}; PYTHONPATH=../src/ python3 see.py gold:../data/UD_RUN/ud24/${zcl}_test.conllu fs:../finals24/mg1_b0_${zcl}_1/output.test.txt,../finals24/mg1_b1_${zcl}_1/output.test.txt; done
# for zcl in en ar eu zh 'fi' he hi it ja ko ru sv tr; do echo ${zcl}; PYTHONPATH=../src/ python3 see.py gold:../data/UD_RUN/ud24/${zcl}_test.conllu fs:../cl_runs_from_tc/final_g1/cg1_b0_en_1/crout_${zcl}.out,../cl_runs_from_tc/final_g1/cg1_b1_en_1/crout_${zcl}.out; done
