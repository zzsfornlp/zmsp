#

# simply compare two system outputs and one gold file and possibly support instance viewing and annotation

import sys
import json
import traceback
import numpy as np
import pandas as pd
from typing import List, Iterable
from collections import defaultdict, Counter
from msp.zext.dpar.conllu_reader import ConlluReader
from msp.data import Vocab
from msp.utils import zopen, zlog, Random, Conf, PickleRW, Constants, ZObject, wrap_color, Helper

from msp.zext.ana import AnalyzerConf, Analyzer, ZRecNode, ZNodeVisitor, AnnotationTask

try:
    from .common import *
except:
    from common import *

# truncat/norm float numbers
def n_float(x, digit=4):
    mul = 10**digit
    return int(x*mul)/mul

class AnalysisConf(Conf):
    def __init__(self, args):
        super().__init__()
        self.gold = ""  # gold parse file
        self.fs = []  # list of system files
        self.vocab = ""  # (optional) vocab file
        self.ana = AnalyzerConf()  # basic analyzer conf
        self.getter = GetterConf()
        self.labeled = False
        # save and load
        self.save_name = ""
        self.load_name = ""
        #
        # =====
        self.update_from_args(args)
        self.validate()

#
class AccVisitor(ZNodeVisitor):
    def visit(self, node: ZRecNode, values: List):
        # calculate UAS and LAS
        zc_count = len(node.objs)
        assert zc_count == node.count
        zc_uas, zc_las = [], []
        if len(node.objs)>0:
            for i in range(node.objs[0].nsys):
                one_uas = sum(z.ss[i].ucorr for z in node.objs) / (zc_count + 1e-5)
                one_las = sum(z.ss[i].lcorr for z in node.objs) / (zc_count + 1e-5)
                zc_uas.append(n_float(one_uas))
                zc_las.append(n_float(one_las))
        node.props.update({"acc": f" u={zc_uas} Du={[n_float(a-b) for a,b in zip(zc_uas[:-1], zc_uas[1:])]} "
                                  f"l={zc_las} Dl={[n_float(a-b) for a,b in zip(zc_las[:-1], zc_las[1:])]}"})
        return None

    def show(self, node: ZRecNode):
        return f"{node.props.get('acc', 'NOPE')}"

#
# general protocol: sents, tokens, vs
class ParsingAnalyzer(Analyzer):
    def __init__(self, conf, all_sents: List, all_tokens: List, labeled: bool=False, vocab: Vocab=None):
        super().__init__(conf)
        #
        self.labeled = labeled
        self.set_var("sents", all_sents, explanation="init", history_idx=-1)
        self.set_var("tokens", all_tokens, explanation="init", history_idx=-1)
        self.set_var("voc", vocab, explanation="init", history_idx=-1)
        #
        self.cur_ann_task: PerrAnnotationTask = None

    # =====
    # commands

    # protocol: target: instances, gcode: d for each instances; return one node
    def do_group(self, insts_target: str, gcode: str, sum_key="count") -> ZRecNode:
        return self._do_group(insts_target, gcode, sum_key, AccVisitor())

    # protocol: target: znode, scode: show code
    def do_show(self, inst_node: str, scode: str):
        # todo(note): maybe using pd is better!?
        pass

    # print for an obj
    def do_print(self, ocode):
        vs = self.vars
        _ff = compile(ocode, "", "eval")
        obj = eval(_ff)
        print_obj(obj)

    # -----
    # looking/annotating at specific instances. protocol: target

    # start new annotation task or TODO(+N) recover the previous one
    def do_ann_start(self, insts_target: str) -> AnnotationTask:
        assert self.cur_cmd_target is not None, "Should assign this to a var to avoid accidental loss!"
        vs = self.vars
        insts = self.get_and_check_type(insts_target, list)
        new_task = PerrAnnotationTask(insts)
        # todo(note): here need auto save?
        if self.cur_ann_task is not None and self.cur_ann_task.remaining>0:
            zlog("Warn: the previous annotation task has not been finished yet!")
        self.cur_ann_task = new_task
        return new_task

    # create one annotation
    def do_ac(self, *ann):
        pass

    # jump to and print a new instance, 0 means printing current instance
    def do_aj(self, offset=0):
        offset = int(offset)
        assert self.cur_ann_task is not None, "Now attached ann task yet!"
        self.cur_ann_task.set_focus(offset=offset)
        self.cur_ann_task.show_current()

# parsing error annotation
class PerrAnnotationTask(AnnotationTask):
    def __init__(self, objs: List):
        super().__init__(objs)
        assert len(objs)>0
        # todo(warn): specific method for telling sent-mode or tok-mod
        self.is_sent = hasattr(objs[0], "toks")

    def show_current(self):
        obj = self.get_obj()
        print_obj(obj, self.focus, self.length)
