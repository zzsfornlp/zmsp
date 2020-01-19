#

# common procedures for analysis

import json
import numpy as np
import pandas as pd
from typing import List, Iterable
from collections import defaultdict

from msp.zext.dpar.conllu_reader import ConlluReader
from msp.utils import zopen, zlog, Random, Conf, PickleRW, Constants, ZObject, wrap_color, Conf

try:
    from .fixer import *
except:
    from fixer import *

# =====
# reading
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
            if "g1_marg" in p.misc:
                g1_marg = p.misc["g1_marg"][1:]
                assert len(g1_marg) == cur_length
                sorted_idxes = list(reversed(np.argsort(g1_marg)))
                for cur_step in range(cur_length):
                    cur_idx = sorted_idxes[cur_step] + 1  # offset by root
                    cur_score = g1_marg[cur_idx - 1]
                    assert cur_idx > 0
                    cur_t = p.get_tok(cur_idx)
                    cur_t.misc["gmi"] = cur_step
                    cur_t.misc["gms"] = cur_score
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

# =====
# object and printing

class GetterConf(Conf):
    def __init__(self):
        self.build_trees = True
        self.build_back_edges = True
        self.build_fixes = True

    def do_validate(self):
        if self.build_back_edges or self.build_fixes:
            assert self.build_trees

def get_token_object(gp_token, sys_tokens: List, tid: int, conf: GetterConf):
    # =====
    def _calc_agr(objs: List):
        x = objs[0]
        return all(x==z for z in objs)
    # =====
    res = {"g": gp_token, "ss": sys_tokens, "id": tid, "nsys": len(sys_tokens)}
    # correctness info
    g_head, g_label = gp_token.head, gp_token.label
    all_heads, all_labels = [g_head], [g_label]
    for cur_idx, cur_token in enumerate(sys_tokens, 1):
        prefix = "s" + str(cur_idx)  # s1, s2, ...
        res[prefix] = cur_token
        cur_head, cur_label = cur_token.head, cur_token.label
        all_heads.append(cur_head)
        all_labels.append(cur_label)
        ucorr = (cur_head == g_head)
        lcorr = (ucorr and (cur_label == g_label))
        cur_token.misc["ucorr"] = int(ucorr)
        cur_token.misc["lcorr"] = int(lcorr)
    # agreement info
    uagr2, uagr3 = _calc_agr(all_heads[1:]), _calc_agr(all_heads)
    lagr2, lagr3 = _calc_agr(all_labels[1:]) and uagr2, _calc_agr(all_labels) and uagr3
    agr_info = {"uagr2": int(uagr2), "uagr3": int(uagr3), "lagr2": int(lagr2), "lagr3": int(lagr3)}
    res.update(agr_info)
    #
    ret = ZObject(res)
    return ret

def get_sent_object(cur_token_objs, sid: int, conf: GetterConf):
    res = {
        "id": sid,
        "len": len(cur_token_objs),
        "toks": [None] + cur_token_objs,  # padding for the artificial root to adjust the idxes
        "rtoks": cur_token_objs,  # real tokens
    }
    # build tree?
    if conf.build_trees:
        nsys = cur_token_objs[0].nsys
        gold_heads = [0] + [z.g.head for z in cur_token_objs]
        gold_labels = ["_"] + [z.g.label0 for z in cur_token_objs]
        all_pred_heads = [[0] + [z.ss[i].head for z in cur_token_objs] for i in range(nsys)]
        # todo(note): use level0 labels
        all_pred_labels = [["_"] + [z.ss[i].label0 for z in cur_token_objs] for i in range(nsys)]
        gold_tree = DepTree(gold_heads, gold_labels, True)
        all_pred_trees = [DepTree(a, b, False) for a,b in zip(all_pred_heads, all_pred_labels)]
        res["gtree"] = gold_tree
        res["ptrees"] = all_pred_trees
        # build back_edge?
        if conf.build_back_edges:
            for i in range(nsys):
                edges = all_pred_trees[i].characterize_edges(gold_tree)[1:]
                for t, e in zip(cur_token_objs, edges):
                    assert e is not None
                    t.ss[i].misc["edge"] = e
        # build fixes?
        fixer = GoldRefFixer(None)
        if conf.build_fixes:
            fix_cates = []  # List[set of categories]
            for i in range(nsys):
                fixer.fix(gold_tree, all_pred_trees[i])
                fix_cates.append(set(z.category for z in all_pred_trees[i].fixes))
            res["fix_cates"] = fix_cates
    # build reversed links for token nodes
    sobj = ZObject(res)
    for t in cur_token_objs:  # additional link
        t.sent = sobj
        t.sid = sid
    return sobj

#
def get_objs(gold_parses: List, sys_parses: List[List], getter: GetterConf):
    assert all(len(gold_parses) == len(z) for z in sys_parses)
    all_sents, all_tokens = [], []
    for sent_idx in range(len(gold_parses)):
        gp_tokens = gold_parses[sent_idx].tokens
        sys_tokens = []
        for one_sys_parse in sys_parses:
            sys_tokens.append(one_sys_parse[sent_idx].tokens)
            assert [z.word for z in gp_tokens] == [z.word for z in sys_tokens[-1]]
        # =====
        # pre-computation
        cur_tokens = []
        # for each valid token
        for token_idx in range(1, len(gp_tokens)):
            now_gp_token = gp_tokens[token_idx]
            now_sys_tokens = [z[token_idx] for z in sys_tokens]
            cur_t_obj = get_token_object(now_gp_token, now_sys_tokens, token_idx, getter)
            cur_tokens.append(cur_t_obj)
        # for the whole sentence
        cur_sent = get_sent_object(cur_tokens, sent_idx, getter)
        all_sents.append(cur_sent)
        all_tokens.extend(cur_tokens)
    return all_sents, all_tokens

def print_obj(obj: ZObject, ann_focus: int=-1, ann_length: int=-1) -> pd.DataFrame:
    if hasattr(obj, "toks"):
        sent_obj = obj
        highlight_tid = -1
    else:
        sent_obj = obj.sent
        highlight_tid = obj.id
    # collect sentence info
    cur_len = sent_obj.len
    cur_tidx = 0
    all_fields = []
    err_head, err_label = defaultdict(list), defaultdict(list)
    nerr_head, nerr_label = defaultdict(int), defaultdict(int)
    for cur_tok in sent_obj.toks[1:]:
        if cur_tidx == highlight_tid:
            CUR_ERR_BACK = "RED"
        else:
            CUR_ERR_BACK = "BLUE"
        #
        cur_tidx += 1
        cur_fields = []
        cur_g = cur_tok.g
        head, label, word, upos = cur_g.head, cur_g.label, cur_g.word, cur_g.upos
        noerr_head, noerr_label = True, True
        for sys_idx in range(1, cur_tok.nsys + 1):
            prefix = "s" + str(sys_idx)
            cur_s = getattr(cur_tok, prefix)
            shead, slabel = cur_s.head, cur_s.label
            # get ranking and confidence
            efi = getattr(cur_s, "efi", None)
            gmi = getattr(cur_s, "gmi", None)
            if efi is not None:
                confi_descr = f"{efi}/{cur_len}"
            elif gmi is not None:
                gms = getattr(cur_s, "gms")
                confi_descr = f"{gmi}/{gms:.4f}"
            else:
                confi_descr = "--"
            #
            if shead != head:
                err_head[prefix].append(cur_tidx)
                nerr_head[prefix] += 1
                noerr_head = False
            if slabel != label:
                err_label[prefix].append(cur_tidx)
                nerr_label[prefix] += 1
                noerr_label = False
            cur_fields.append(wrap_color(str(shead), bcolor=("black" if shead == head else CUR_ERR_BACK)))
            cur_fields.append(wrap_color(slabel, bcolor=("black" if slabel == label else CUR_ERR_BACK)))
            cur_fields.append(confi_descr)
        cur_fields = [wrap_color(str(head), bcolor=("black" if noerr_head else CUR_ERR_BACK)),
                      wrap_color(label, bcolor=("black" if noerr_label else CUR_ERR_BACK))] + cur_fields
        cur_fields.extend([word, upos])
        all_fields.append(cur_fields)
    x = pd.DataFrame(all_fields, index=[f"({z})" for z in range(1, cur_len + 1)])
    # print the table and summarize
    zlog(f"Current instance\n{x.to_string()}\nSid={sent_obj.id}({ann_focus}/{ann_length}), Wid={highlight_tid},"
         f" ErrHead={nerr_head};{err_head}, ErrLabel={nerr_label};{err_label}")
    # print fix info if accessible
    ptrees = getattr(sent_obj, "ptrees", None)
    if ptrees is not None:
        for sys_idx, one_ptree in enumerate(ptrees, 1):
            zlog(f"Print fixes for system {sys_idx}")
            for fid, one_fix in enumerate(one_ptree.fixes):
                zlog(f"#{fid}: {str(one_fix)}")
    return x


# =====
# helper functions
def val2bin(x, bins):
    i = 0
    for b in bins:
        if x<=b:
            return i
        i += 1
    return i
