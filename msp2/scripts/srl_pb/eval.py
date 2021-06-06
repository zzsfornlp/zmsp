#

# srl_eval for pb with official script
# wget http://www.lsi.upc.es/~srlconll/srl-eval.pl

import os
import sys
from typing import List, Union
from msp2.utils import system, zopen, MathHelper
from msp2.data.inst import yield_sents
from msp2.data.rw import ConllHelper
from msp2.proc.eval import EvalConf, EvalResult, Evaluator

# --
# use cached gz file
def get_eval_script():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    srl_eval_script = os.path.join(this_dir, "srl-eval.pl")
    if not os.path.exists(srl_eval_script):
        srl_eval_gz_file = os.path.join(this_dir, "srl-eval.pl.gz")
        if os.path.isfile(srl_eval_gz_file):
            # decompress
            system(f"gzip -c -d {srl_eval_gz_file} >{srl_eval_script}", ass=True)
        else:
            raise RuntimeError("Cannot find srl_eval!!")
    # --
    return srl_eval_script

# transform List[Sent,Doc] into props format
def insts2props(insts: List):
    lines = []
    for sent in yield_sents(insts):
        slen = len(sent)
        words = sent.seq_word.vals
        # --
        cur_preds, _, cur_frames = ConllHelper.put_preds(sent.events, slen, lambda f: (words[f.mention.widx], None))
        all_cols = [cur_preds]
        for one_frame in cur_frames:
            if one_frame is None: continue
            one_args = ConllHelper.put_args([a for a in one_frame.args if a.arg.sent is sent], slen)
            all_cols.append(one_args)
        # put into lines
        for ii in range(slen):
            lines.append(" ".join([z[ii] for z in all_cols]) + "\n")  # here, sep by ' '!!
        lines.append("\n")
    return "".join(lines)

# -----

class PbEvalConf(EvalConf):
    def __init__(self):
        super().__init__()
        # --
        self.extra_flags = ""
        self.tmp_file_prefix = "_pb_eval.tmp"  # + ".gold.props"/".pred.props"

@Evaluator.reg_decorator("pb_srl", conf=PbEvalConf)
class PbEvaler(Evaluator):
    def __init__(self, conf: PbEvalConf):
        super().__init__(conf)
        conf: PbEvalConf = self.conf
        # --
        eval_script = get_eval_script()
        self.cmd = f"perl {eval_script} {conf.extra_flags}"  # perl srl-eval.pl <gold> <pred>

    def eval(self, gold_insts: List, pred_insts: List):
        # --
        conf: PbEvalConf = self.conf
        tmp_gold, tmp_pred = conf.tmp_file_prefix+".gold.props", conf.tmp_file_prefix+".pred.props"
        with zopen(tmp_gold, 'w') as fd:
            fd.write(insts2props(gold_insts))
        with zopen(tmp_pred, 'w') as fd:
            fd.write(insts2props(pred_insts))
        # --
        precision_output = system(f"{self.cmd} {tmp_pred} {tmp_gold} 2>/dev/null", ass=True, popen=True)
        recall_output = system(f"{self.cmd} {tmp_gold} {tmp_pred} 2>/dev/null", ass=True, popen=True)
        # parse output
        res = PbEvalResult(precision_output, recall_output)
        return res

class PbEvalResult(EvalResult):
    def __init__(self, precision_output: str, recall_output: str):
        self.details = f"# == Precision ==\n{precision_output}# == Recall ==\n{recall_output}"
        # --
        self.precision, self.precision_n, self.precision_d = self._read_one(precision_output)
        self.recall, self.recall_n, self.recall_d = self._read_one(recall_output)
        P, R = self.precision, self.recall
        self.f1 = MathHelper.safe_div(2*P*R, P+R)
        self.res_line = f"P: {self.precision_n}/{self.precision_d}={self.precision:.4f} " \
                        f"R: {self.recall_n}/{self.recall_d}={self.recall:.4f} F: {self.f1:.4f}"

    def _read_one(self, result: str):
        all_lines = [line.strip() for line in result.strip().split("\n")]
        res_line = [line for line in all_lines if line.startswith("Overall")]
        assert len(res_line) == 1
        res_line = res_line[0]
        # --
        n_corr, n_excess, n_missed = [int(z) for z in res_line.split()[1:4]]
        ret_n, ret_d = n_corr, n_corr+n_missed  # only get recall here!!
        return MathHelper.safe_div(ret_n, ret_d), ret_n, ret_d

    def get_result(self) -> float: return self.f1
    def get_brief_str(self) -> str: return self.res_line
    def get_detailed_str(self) -> str: return self.details
    def get_summary(self) -> dict: return {"recall": self.recall, "precision": self.precision, "f1": self.f1}

if __name__ == '__main__':
    from msp2.cli.evaluate import main
    main('pb_srl', *sys.argv[1:])

"""
# use this one
PYTHONPATH=../src/ python3 -m msp2.scripts.srl_pb.eval gold.input_format:conll12 pred.input_format:conll12 gold.use_multiline:1 pred.use_multiline:1 gold.input_path:dev.conll pred.input_path:pred.conll
# use my evaler
PYTHONPATH=../src/ python3 -m msp2.cli.evaluate pb gold.input_format:conll12 pred.input_format:conll12 gold.use_multiline:1 pred.use_multiline:1 gold.input_path:dev.conll pred.input_path:pred.conll
# --
PYTHONPATH=../src/ python3 -m msp2.scripts.srl_pb.eval gold.input_path:../pb/conll05/test.wsj.conll.ud.json pred.input_path:zout.json
"""
