#

import numpy as np
import evaluate
from collections import OrderedDict, defaultdict
from mspx.utils import Conf, Configurable, ZResult, F1EvalEntry, zlog, ZObject

class EvalConf(Conf):
    def __init__(self):
        self.metrics = []
        self.final_weights = [1.]  # by default all 1

class Evaler(Configurable):
    def __init__(self, conf: EvalConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: EvalConf = self.conf
        # --
        self.metrics = OrderedDict([(z, MyMetric.get_metric(z)) for z in conf.metrics])

    def add(self, pred, gold):
        for m in self.metrics.values():
            m.add(pred, gold)

    def get_res(self):
        conf: EvalConf = self.conf
        # --
        _weights = conf.final_weights
        if len(_weights) < len(self.metrics):
            _weights = _weights + [_weights[-1]] * (len(self.metrics) - len(_weights))
        aggr = ResultAggregator()
        ii = 0
        for nn, mm in self.metrics.items():
            one_res = mm.get_res()
            aggr.add(nn, one_res, _weights[ii])
            ii += 1
        ret = aggr.aggr()
        zlog(f"###\nOne-eval-result: {ret}\n###")
        return ret

# --
class ResultAggregator:
    def __init__(self):
        self.all_numbers = []
        self.all_weights = []
        self.all_results = OrderedDict()

    def add(self, key: str, res: ZResult, weight: float):
        self.all_numbers.append(float(res))
        self.all_weights.append(weight)
        self.all_results[key] = res

    def aggr(self):
        if len(self.all_numbers) == 0:
            final_score = 0.
        else:  # weighted sum!!
            final_score = (np.asarray(self.all_numbers) * np.asarray(self.all_weights)).sum() / sum(self.all_weights)
            final_score = final_score.item()
        return ZResult(self.all_results, res=final_score)

# --
class MyMetric:
    @staticmethod
    def get_metric(name):
        my_metrics = {'z_f1': MyMetricF1, 'z_bleu': MyBleuWrapper}
        if name in my_metrics:
            ret = my_metrics[name]()  # todo(+N): allow args?
        else:
            ret = MyMetricWrapper(evaluate.load(name))
        return ret

    def add(self, pred, gold): pass
    def get_res(self): pass

class MyMetricWrapper(MyMetric):
    def __init__(self, eval_metric):
        self.eval_metric = eval_metric

    def add(self, pred, gold):
        self.eval_metric.add(prediction=pred, reference=gold)

    def get_res(self):
        res = self.eval_metric.compute()
        ret = ZResult(res, res=list(res.values())[0])  # simply the first one!
        return ret

class MyBleuWrapper(MyMetric):
    def __init__(self):
        self.pred = []
        self.gold = []

    def add(self, pred, gold):
        self.pred.append(pred)
        if not isinstance(gold, (list, tuple)):
            gold = [gold]
        self.gold.append(gold)

    def get_res(self):
        e = evaluate.load('bleu')
        res = e.compute(predictions=self.pred, references=self.gold)
        ret = ZResult(res, res=res['bleu'])
        return ret

class MyMetricF1(MyMetric):
    def __init__(self, nil_label=None):
        self.nil_label = nil_label
        self.f1 = F1EvalEntry()
        self.pgs = []  # List[(pred, gold)]

    def add(self, pred, gold):
        valid_gold, valid_pred = int(gold != self.nil_label), int(pred != self.nil_label)
        corr = int(gold == pred)
        if valid_pred:
            self.f1.record_p(corr)
        if valid_gold:
            self.f1.record_r(corr)
        self.pgs.append([pred, gold])
        # --

    def get_res(self):
        P, R, F1 = self.f1.prf
        # simple breakdowns
        df, res = get_breakdown(self.pgs)
        zlog(f"#-- Eval res:\n{df.to_string()}\n#--")
        # --
        ret = ZResult({'P': P, 'R': R, 'F1': F1}, res=F1)
        return ret

# --
# utils
def get_breakdown(pairs, pcode=(lambda x: x), gcode=(lambda y: y), corr_code=(lambda x,y: x==y), sort_key=-3, do_micro=True, do_macro=True):
    import pandas as pd
    # --
    # get functions
    functions = [pcode, gcode, corr_code]
    for ii in range(len(functions)):
        if isinstance(functions[ii], str):
            functions[ii] = eval(functions[ii])
    _fp, _fg, _fc = functions
    # --
    res = defaultdict(F1EvalEntry)
    _micro_entry = F1EvalEntry()
    for _pred, _gold in pairs:
        corr = 0
        if _pred is not None and _gold is not None:
            corr = int(_fc(_pred, _gold))
        if _pred is not None:
            key_p = _fp(_pred)
            res[key_p].record_p(corr)
            _micro_entry.record_p(corr)
        if _gold is not None:
            key_g = _fg(_gold)
            res[key_g].record_r(corr)
            _micro_entry.record_r(corr)
    # --
    # final
    details = [(k,)+v.details for k,v in res.items()]
    details = sorted(details, key=(lambda x: x[sort_key]), reverse=True)  # by default, sort by gold count
    if do_macro:
        macro_line = ['Macro_'] + ([0.] * 7)
        macro_line[3] = np.mean([z[3] for z in details])
        macro_line[6] = np.mean([z[6] for z in details])
        macro_line[7] = np.mean([z[7] for z in details])
        details.append(tuple(macro_line))
        res["Macro_"] = ZObject(res=macro_line[-1], prf=[macro_line[z] for z in [3,6,7]], details=macro_line[1:])
    if do_micro:
        details.append(('Micro_',) + _micro_entry.details)
        res["Micro_"] = _micro_entry
    df = pd.DataFrame(details, columns=['T', 'Pc', 'Pa', 'P', 'Rc', 'Ra', 'R', 'F1'])
    return df, res
