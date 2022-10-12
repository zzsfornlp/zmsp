#

# specific eval

from typing import List, Union
from collections import Counter
from msp2.utils import system, zopen, Conf, init_everything, zlog, F1EvalEntry
from msp2.data.inst import Doc, Sent, Mention, Frame, yield_frames, yield_sents
from msp2.proc.eval import EvalConf, EvalResult, Evaluator

# --

class EeEvalConf(EvalConf):
    def __init__(self):
        super().__init__()
        # --
        self.arg_with_ftype = True
        # --

@Evaluator.reg_decorator("ee", conf=EeEvalConf)
class EeEvaler(Evaluator):
    def __init__(self, conf: EeEvalConf):
        super().__init__(conf)
        conf: EeEvalConf = self.conf
        # --

    def match(self, golds: List, preds: List):
        gold_set, pred_set = set(golds), set(preds)
        # --
        posi_gold = Counter([z[0] for z in golds])
        posi_pred = Counter([z[0] for z in preds])
        lab_gold = Counter([z[0]+z[1] for z in golds])
        lab_pred = Counter([z[0]+z[1] for z in preds])
        # --
        posi_gold2 = Counter([z[0] for z in gold_set])
        posi_pred2 = Counter([z[0] for z in pred_set])
        lab_gold2 = Counter([z[0]+z[1] for z in gold_set])
        lab_pred2 = Counter([z[0]+z[1] for z in pred_set])
        # --
        def _calc(_cg, _cp):
            _entry = F1EvalEntry()
            for _k in set(list(_cg.keys()) + list(_cp.keys())):
                _common = min(_cp[_k], _cg[_k])
                _entry.record_r(_common, all=_cg[_k])
                _entry.record_p(_common, all=_cp[_k])
            return [round(z,4) for z in _entry.details]
        def _calc2(_cg, _cp):
            # note: cacl2 still slightly different than oneie's eval since there it overwrites repeated roles
            #  (between a pair of event and entity), nevertheless this happens very rare (<0.1%)
            _entry = F1EvalEntry()
            hit_pred = 0
            for _k, _v in _cp.items():  # enumerate all predictions
                if _cg[_k] > 0:
                    hit_pred += _v
            _entry.record_p(hit_pred, sum(_cp.values()))
            _entry.record_r(hit_pred, sum(_cg.values()))
            return [round(z,4) for z in _entry.details]
        # --
        return {
            "posi": _calc(posi_gold, posi_pred), "lab": _calc(lab_gold, lab_pred),
            "posi2": _calc2(posi_gold2, posi_pred2), "lab2": _calc2(lab_gold2, lab_pred2),
        }

    def eval(self, gold_insts: List[Union[Doc, Sent]], pred_insts: List[Union[Doc, Sent]]):
        _arg_with_ftype = self.conf.arg_with_ftype
        # --
        assert len(gold_insts) == len(pred_insts)
        global_idx = 0
        all_g_evts, all_p_evts = [], []
        all_g_args, all_p_args = [], []
        for g_inst, p_inst in zip(gold_insts, pred_insts):
            for _inst, _evts, _args in ([g_inst, all_g_evts, all_g_args], [p_inst, all_p_evts, all_p_args]):
                for sent in yield_sents(_inst):
                    _evts.append([])
                    _args.append([])
                    for evt in sent.events:
                        _sig_evt = (global_idx, evt.mention.sent.sid, evt.mention.get_span()), (evt.label, )
                        _evts[-1].append(_sig_evt)  # posi-key, label
                        for arg in evt.args:
                            _sig_arg = (global_idx, arg.mention.sent.sid, arg.mention.get_span()) + \
                                       ((evt.label, ) if _arg_with_ftype else ()), (arg.label, )
                            _args[-1].append(_sig_arg)
            global_idx += 1
        # --
        # match them
        res_evt = self.match(sum(all_g_evts, []), sum(all_p_evts, []))
        res_arg = self.match(sum(all_g_args, []), sum(all_p_args, []))
        return EeEvalResult(evt=res_evt, arg=res_arg)

class EeEvalResult(EvalResult):
    def __init__(self, **results):
        self.results = results

    def get_result(self) -> float: return 0.
    def get_brief_str(self) -> str: return ""
    def get_detailed_str(self) -> str: return "\n".join([f"{k}: {v}" for k,v in self.results.items()])
    def get_summary(self) -> dict: return self.results

if __name__ == '__main__':
    from msp2.cli.evaluate import main
    import sys
    main('ee', *sys.argv[1:])

# --
# PYTHONPATH=../src/ python3 eeval.py gold.input_path:?? pred.input_path:??
# python3 -m msp2.scripts.event.eeval gold.input_path:?? pred.input_path:??
