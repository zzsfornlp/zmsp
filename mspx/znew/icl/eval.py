#

import numpy as np
from collections import Counter
from mspx.utils import Conf, zglob1, default_json_serializer, zlog, Random, ZHelper, F1EvalEntry
from mspx.proc.eval import MatchedPair
from .helper import *

class IclEvalConf(Conf):
    def __init__(self):
        self.eval_bds = ["len_spath(x)", "sel_spath(x)", "x['label']"]
        self.nil_label = ''  # NIL label to calculate F1

def eval_results(gold_dps, pred_dps, conf: IclEvalConf, task_helper=None, **kwargs):
    conf: IclEvalConf = IclEvalConf.direct_conf(conf, **kwargs)
    # --
    _f_bds = []
    for one_bd in conf.eval_bds:
        _f_bds.append(ZHelper.eval_ff(one_bd, 'x', locals=locals(), globals=globals()))
    _f_corr = (lambda x,y: x['label']==y['label'])
    _nil_label = conf.nil_label
    assert len(gold_dps) == len(pred_dps)
    pairs = []
    for gg, pp in zip(gold_dps, pred_dps):
        one_pair = MatchedPair((None if pp['label']==_nil_label else pp), (None if gg['label']==_nil_label else gg))
        pairs.append(one_pair)
    df0, res0 = MatchedPair.get_breakdown(pairs, (lambda x: 1), (lambda x: 1), _f_corr)
    zlog(f"Overall results:\n{df0.to_string()}")
    ret = {'zres': res0[1].res, '_main': res0[1].details}
    for _f_bd in _f_bds:
        df, res = MatchedPair.get_breakdown(pairs, _f_bd, _f_bd, _f_corr, do_macro=True)
        zlog(f"Breakdown results:\n{df.to_string()}")
        ret.update({k: v.details for k, v in res.items()})
    # --
    return ret
