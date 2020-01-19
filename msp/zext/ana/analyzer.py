#

# the general analyzer

import numpy as np
import pandas as pd
import traceback
from typing import List, Iterable
from shlex import split as sh_split
from pdb import set_trace

from msp.utils import zopen, zlog, Conf, PickleRW, ZObject, Helper

from .node2 import ZRecNode, ZNodeVisitor

class AnalyzerConf(Conf):
    def __init__(self):
        self.auto_save_name = "./_tmp_analysis.pkl"

# analyzer: need to conform to each commands' local naming protocol
class Analyzer:
    def __init__(self, conf: AnalyzerConf):
        self.conf = conf
        self.history = []
        # naming convention: those start with "_" will not be saved
        self.vars = ZObject()  # vars mapping
        self.traces = {}  # var_name -> set-history-idx
        # auto save
        self.last_save_point = 0  # last save before this idx
        # current cmd info
        self.cur_cmd_line = None
        self.cur_cmd_target = None
        self.cur_cmd_args = None

    def __del__(self):
        if self.last_save_point < len(self.history):
            auto_save_name = self.conf.auto_save_name
            if len(auto_save_name) > 0:
                self.do_save(self.conf.auto_save_name)

    # =====
    # some helper functions
    def set_var(self, target, v, explanation=None, history_idx=None):
        if hasattr(self.vars, target):
            zlog(f"Overwriting the existing var `{target}`")
        if target not in self.traces:
            self.traces[target] = []
        # (explanation, history-idx)
        history_idx = len(self.history) if history_idx is None else history_idx
        self.traces[target].append((explanation, history_idx))  # the current cmd will be recorded into history
        setattr(self.vars, target, v)

    def get_var(self, target):
        if hasattr(self.vars, target):
            return getattr(self.vars, target)
        else:
            assert False, f"Cannot find var `{target}`"

    def str_cmd(self, one):
        target, args = one
        return f"{target} = {args}"

    # =====
    # some general commands
    def do_history(self, num=-10):
        num = int(num)
        cur_len = len(self.history)
        if num<0:  # listing recent mode
            num = min(int(-num), cur_len)
            zlog(f"Listing histories: all = {cur_len}")
            # back to front
            for i in range(1, num+1):
                real_idx = cur_len - i
                zlog(f"[#{i}|{real_idx}] {self.str_cmd(self.history[real_idx])}")
        else:
            zlog(f"Listing histories idx = {num}: {self.str_cmd(self.history[num])}")
        return None

    def do_trace(self, target):
        v = self.get_var(target)
        for explanation, history_idx in self.traces[target]:
            zlog(f"Var `{target}`: ({explanation}, {self.str_cmd(self.history[history_idx]) if history_idx>=0 else ''})")

    # most general runner
    def do_eval(self, code):
        vs = self.vars  # convenient local variable
        _ff = compile(code, "", "eval")
        ret = eval(_ff)
        return ret

    def do_pdb(self):
        set_trace()
        return None

    def do_load(self, file):
        zlog(f"Try loading vars from {file}")
        x = PickleRW.from_file(file)
        self.vars.update(x)

    def do_save(self, file):
        zlog(f"Try saving vars to {file}")
        self.last_save_point = len(self.history) + 1  # plus one for the current one
        PickleRW.to_file(self.vars, file)

    # =====
    # some useful ones

    # protocol: target: instances, fcode: d for each instances; return instances
    def do_filter(self, insts_target: str, fcode: str) -> List:
        vs = self.vars
        _ff = compile(fcode, "", "eval")
        insts = self.get_and_check_type(insts_target, list)
        ret = [d for d in insts if eval(_ff)]
        zlog(f"Filter by {fcode}: from {len(insts)} to {len(ret)}, {len(ret)/(len(insts)+1e-7)}")
        return ret

    # protocol: target of instances, jcode return a list for each of them
    def do_join(self, insts_target: str, jcode: str) -> List:
        vs = self.vars
        _ff = compile(jcode, "", "eval")
        insts = self.get_and_check_type(insts_target, list)
        ret = [eval(_ff) for d in insts]
        ret = Helper.join_list(ret)
        zlog(f"Join-list by {jcode}: from {len(insts)} to {len(ret)}")
        return ret

    # protocol: target: instances, kcode: (key) d for inst: return sorted list
    def do_sort(self, insts_target: str, kcode: str) -> List:
        vs = self.vars
        _ff = compile(kcode, "", "eval")
        insts = self.get_and_check_type(insts_target, list)
        tmp_tuples = [(d, eval(_ff)) for d in insts]
        tmp_tuples.sort(key=lambda x: x[1])
        ret = [x[0] for x in tmp_tuples]
        zlog(f"Sort by key={kcode}: len = {len(ret)}")
        return ret

    # protocol: target: instances, gcode: d for each instances; return one node
    def do_group(self, insts_target: str, gcode: str, sum_key="count") -> ZRecNode:
        return self._do_group(insts_target, gcode, sum_key, None)

    # protocol: target: instances, gcode: d for each instances; return one node
    def _do_group(self, insts_target: str, gcode: str, sum_key: str, visitor: ZNodeVisitor) -> ZRecNode:
        vs = self.vars
        _ff = compile(gcode, "", "eval")
        insts = self.get_and_check_type(insts_target, list)
        ret = ZRecNode(None, [])
        for d in insts:
            ret.add_seq(eval(_ff), obj=d)
        # visitor
        if visitor is not None:
            try:
                ret.rec_visit(visitor)
            except:
                zlog(traceback.format_exc())
                zlog("Error of visitor.")
            _show_node_f = lambda x: visitor.show(x)
        else:
            _show_node_f = lambda x: "--"
        # some slight summaries here
        all_count = len(insts)
        all_nodes = ret.get_descendants(key=sum_key)
        ss = []
        for z in all_nodes:
            all_parents = z.get_antecedents()
            if len(all_parents) > 0:
                assert all_parents[0].count == all_count
            perc_info = ', '.join([f"{z.count/(zp.count+1e-6):.4f}" for zp in all_parents])
            ss.append(['=='*len(z.path), str(z.path), f"{z.count}({perc_info})" f"{_show_node_f(z)}"])
        # sstr = "\n".join(ss)
        # sstr = ""
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 1000)
        pdf = pd.DataFrame(ss)
        zlog(f"Group {len(insts)} instances by {gcode}, all {len(ss)} nodes:\n{pdf.to_string()}")
        return ret

    # similar to group, but using pd instead
    def do_get_pd(self, insts_target: str, gcode: str) -> pd.DataFrame:
        vs = self.vars
        _ff = compile(gcode, "", "eval")
        insts = self.get_and_check_type(insts_target, list)
        #
        fields = [eval(_ff) for d in insts]
        ret = pd.DataFrame(fields)
        zlog(f"Group {len(insts)} instances by {gcode} to pd.DataFrame shape={ret.shape}.")
        return ret

    # calculation and manipulation on pd, also assuming the local name is d
    def do_cal_pd(self, inst_pd: str, scode: str):
        vs = self.vars
        _ff = compile(scode, "", "eval")
        d = self.get_and_check_type(inst_pd, pd.DataFrame)
        #
        ret = eval(_ff)
        zlog(f"Calculation on pd.DataFrame by {scode}, and get another one as: {str(ret)}")
        return ret

    # =====
    # helpers
    def get_and_check_type(self, target, dtype):
        ret = self.get_var(target)
        assert isinstance(ret, dtype), f"Wrong typed target, {type(ret)} instead of {dtype}"
        return ret

    # =====
    #
    def process(self, args: List[str]):
        assert len(args) > 0, "Empty command"
        cmd_name, real_args = args[0], args[1:]
        # similar to the cmd package
        method_name = "do_" + cmd_name
        assert hasattr(self, method_name), f"Unknown command {cmd_name}"
        zlog(f"Performing command: {args}")
        return getattr(self, method_name)(*real_args)

    # format is "[target-name=]args..."
    def get_cmd(self):
        target, args = None, []
        line = input(">> ")
        # first check assignment target
        cmd = line.strip()
        tmp_fields = cmd.split("=", 1)
        if len(tmp_fields) > 1:
            tmp_target, remainings = [x.strip() for x in tmp_fields]
            if str.isidentifier(tmp_target):
                target = tmp_target
                cmd = remainings
        # then split into args
        args = sh_split(cmd)
        return target, args, line

    def loop(self):
        target, args = None, []
        stop = False
        while True:
            # read one command
            while True:
                try:
                    target, args, line = self.get_cmd()
                except KeyboardInterrupt:
                    continue
                except EOFError:
                    stop = True
                    break
                if len(args) > 0:  # get valid commands
                    break
            if stop:
                break
            # process it
            try:
                self.cur_cmd_line = line
                self.cur_cmd_target = target
                self.cur_cmd_args = args
                v = self.process(args)
            except AssertionError as e:
                zlog(f"Checking Error: " + str(e))
                continue
            except:
                zlog(f"Command error: " + str(traceback.format_exc()))
                continue
            cmd_count = len(self.history)
            if v is not None:  # does not store None
                if target is not None:
                    self.set_var(target, v)
            self.history.append((target, args))
            zlog(f"Finish command #{cmd_count}: {self.str_cmd(self.history[-1])}")


# requiring that objs have "id" field
class AnnotationTask:
    def __init__(self, objs: List):
        self.pool = objs
        self.focus = 0
        self.length = len(self.pool)
        self.remaining = len(self.pool)
        self.annotations = {z.id: None for z in objs}  # id -> ann-obj

    # to be overridden
    def obj_finished(self, obj):
        return self.annotations[obj.id] is not None

    # get current focus
    def get_obj(self):
        idx = self.focus
        obj = self.pool[idx]
        return obj

    #
    def annotate_obj(self):
        raise NotImplementedError("to be implemented")

    # setting focus and return the new focus
    def set_focus(self, idx=None, offset=0):
        if idx is not None:
            new_focus = idx
        else:
            new_focus = self.focus + offset
        if new_focus>=0 and new_focus<self.length:
            self.focus = new_focus
        else:
            zlog(f"Focus setting failed: at the boundary: {self.focus}, {self.length}")
        return self.focus

    def rearrange(self):
        finised_pool = []
        unfinished_pool = []
        for obj in self.pool:
            if self.obj_finished(obj):
                finised_pool.append(obj)
            else:
                unfinished_pool.append(obj)
        new_pool = finised_pool + unfinished_pool
        self.pool = new_pool
        self.focus = len(finised_pool)
        self.remaining = len(unfinished_pool)
