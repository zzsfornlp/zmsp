#

# fn_eval with semafor's script

from typing import List, Union
import os
import sys
import re
import time
# import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape as xml_escape
from msp2.utils import system, zopen, Conf, init_everything, zlog
from msp2.data.rw import DataFormator, get_reader, ReaderGetterConf
from msp2.data.inst import Doc, Sent, Mention, Frame
from msp2.proc.eval import EvalConf, EvalResult, Evaluator

# how to get the script
"""
wget -np -r -e robots=off -R "index.html*" https://www.cs.cmu.edu/~ark/SEMAFOR/eval
mv www.cs.cmu.edu/~ark/SEMAFOR/eval/ fn_eval
rm -rf www.cs.cmu.edu
"""
# what are the options and how does it eval?
"""
-l: by default utilized, use XML instead of semXML
-e: by default utilized, exact match (no partial)
-n: by default utilized, no NE
-v: by default utilized, verbose
-t: target only
# --
Seems: 1.0*target+1.0*core+0.5*noncore, must match target+arg for arg
"""

# use cached tgz file
def get_eval_dir():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    fn_eval_dir = os.path.join(this_dir, "fn_eval")
    if not os.path.exists(fn_eval_dir):
        fn_eval_tgz_file = os.path.join(this_dir, "fn_eval.tar.gz")
        if os.path.isfile(fn_eval_tgz_file):
            # decompress
            system(f"tar -zxvf {fn_eval_tgz_file} -C {this_dir}", ass=True)
        else:
            raise RuntimeError("Cannot fin fn_eval!!")
    # --
    return fn_eval_dir

# transform List[Sent,Doc] into XML format
def insts2xml(insts: List, unlabeled: bool):
    # --
    def _str_label(_f: Frame, _idxes: List, _id: int, _name: str):
        _m = _f.mention
        _start, _end = _idxes[_m.widx][0], _idxes[_m.wridx-1][1]-1  # in fn-xml, it uses [start, end]
        return f'<label ID="{_id}" end="{_end}" name="{_name}" start="{_start}"/>'
    # --
    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append(f'<corpus ID="100" XMLCreated="{time.ctime()}" name="ONE">')
    lines.append(f'<documents>')
    # iter all insts
    doc_count, sent_count, others_count = 0, 0, 0
    for inst in insts:
        doc_count += 1
        if isinstance(inst, Doc):
            doc_id = inst.id
            all_sents = inst.sents
        else:
            assert isinstance(inst, Sent)
            doc_id = f"DOC{doc_count}"
            all_sents = [inst]
        # --
        # simply make one paragraph
        lines.append(f'<document ID="{doc_count+1}" description="{doc_id}">\n<paragraphs>\n<paragraph ID="1" documentOrder="1">\n<sentences>')
        # iter all sents
        for sent in all_sents:
            sent_count += 1
            lines.append(f'<sentence ID="{sent_count}">')
            # text and idxes
            cur_toks: List[str] = sent.seq_word.vals
            cur_text = xml_escape(" ".join(cur_toks))
            cur_idxes = []  # [start, end)
            for one_tok in cur_toks:
                if len(cur_idxes) == 0:
                    cur_idxes.append((0, len(one_tok)))
                else:  # +1 for the space in between
                    cur_idxes.append((cur_idxes[-1][-1]+1, cur_idxes[-1][-1]+1+len(one_tok)))
            # --
            lines.append(f'<text>{cur_text}</text>')
            if len(sent.events) == 0:
                lines.append(f'<annotationSets/>')
            else:
                lines.append(f'<annotationSets>')
                for evt in sent.events:
                    # this annotationSet
                    others_count += 1
                    evt_type = 'Event' if unlabeled else evt.type
                    lines.append(f'<annotationSet ID="{others_count}" frameName="{evt_type}">\n<layers>')
                    # target layer
                    others_count += 1
                    lines.append(f'<layer ID="{others_count}" name="Target">\n<labels>\n{_str_label(evt, cur_idxes, others_count+1, "Target")}\n</labels>\n</layer>')
                    others_count += 1  # need two here!
                    # fe layer
                    others_count += 1
                    lines.append(f'<layer ID="{others_count}" name="FE">\n<labels>')
                    for alink in evt.args:
                        if alink.info.get("rank", 1) != 1:
                            continue  # note: ignore rank!=1 ones
                        others_count += 1
                        arg_role = 'Event' if unlabeled else alink.role
                        lines.append(_str_label(alink.arg, cur_idxes, others_count, arg_role))
                    lines.append(f'</labels>\n</layer>')
                    # --
                    lines.append(f'</layers>\n</annotationSet>')
                lines.append(f'</annotationSets>')
            lines.append(f'</sentence>')
        # --
        lines.append(f'</sentences>\n</paragraph>\n</paragraphs>\n</document>')
    lines.append(f'</documents>')
    lines.append(f'</corpus>')
    return '\n'.join(lines)

# -----

class FnEvalConf(EvalConf):
    def __init__(self):
        super().__init__()
        # --
        self.target_only = False  # "-t" for fn_eval, only eval targets
        # todo(note): actually unlabeled mode is not good for arg since 'fnSemScore' will join same-role args!
        self.fn_unlabeled = False  # replace all types with dummy ones; note: in this way no diff core vs non-core
        self.extra_flags = ""
        self.tmp_file_prefix = "_fn_eval.tmp"  # + ".gold.xml"/".pred.xml"

@Evaluator.reg_decorator("semafor", conf=FnEvalConf)
class FnEvaler(Evaluator):
    def __init__(self, conf: FnEvalConf):
        super().__init__(conf)
        conf: FnEvalConf = self.conf
        # --
        eval_dir = get_eval_dir()
        eval_script = os.path.join(eval_dir, "scoring", "fnSemScore_modified.pl")
        self.cmd = f"perl {eval_script} -l -e -n -v {'-t' if conf.target_only else ''} {conf.extra_flags} {os.path.join(eval_dir, 'framesSingleFile.xml')} {os.path.join(eval_dir, 'frRelationModified.xml')}"

    def eval(self, gold_insts: List[Union[Doc, Sent]], pred_insts: List[Union[Doc, Sent]]):
        # --
        conf: FnEvalConf = self.conf
        tmp_gold, tmp_pred = conf.tmp_file_prefix+".gold.xml", conf.tmp_file_prefix+".pred.xml"
        with zopen(tmp_gold, 'w') as fd:
            fd.write(insts2xml(gold_insts, conf.fn_unlabeled))
        with zopen(tmp_pred, 'w') as fd:
            fd.write(insts2xml(pred_insts, conf.fn_unlabeled))
        # --
        final_cmd = f"{self.cmd} {tmp_gold} {tmp_pred}"
        outputs = system(final_cmd, ass=True, popen=True)
        # parse outputs
        final_line = outputs.strip().split("\n")[-1]
        res = FnEvalResult(final_line, outputs)
        return res

class FnEvalResult(EvalResult):
    def __init__(self, res_line: str, details: str):
        self.res_line = res_line
        d_str = r"(\d+\.\d+)"
        pat = re.match(rf'.*Recall={d_str} \({d_str}/{d_str}\)  Precision={d_str} \({d_str}/{d_str}\)  Fscore={d_str}', res_line)
        self.recall, self.recall_n, self.recall_d, self.precision, self.precision_n, self.precision_d, self.f1 = [float(z) for z in pat.groups()]
        self.details = details

    def get_result(self) -> float: return self.f1
    def get_brief_str(self) -> str: return self.res_line
    def get_detailed_str(self) -> str: return self.details
    def get_summary(self) -> dict: return {"recall": self.recall, "precision": self.precision, "f1": self.f1}

if __name__ == '__main__':
    from msp2.cli.evaluate import main
    main('semafor', *sys.argv[1:])

"""
# example runs
PYTHONPATH=../src/ python3 -m msp2.scripts.srl_fn.eval gold.input_path:? pred.input_path:?
# use this one
PYTHONPATH=../src/ python3 -m msp2.cli.evaluate msp2.scripts.srl_fn.eval/semafor gold.input_path:fn15_fulltext.dev.json pred.input_path:fn15_out.dev.json target_only:0 fn_unlabeled:0 result_file:
# use my evaler
PYTHONPATH=../src/ python3 -m msp2.cli.evaluate fn gold.input_path:fn15_fulltext.dev.json pred.input_path:fn15_out.dev.json print_details:1
"""
