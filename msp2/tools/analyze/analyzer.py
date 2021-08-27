#

# analyzer

__all__ = [
    "AnalyzerConf", "AnalyzerCommand", "Analyzer", "AnnotationTask",
]

from typing import List, Type, Dict
from itertools import chain
import traceback
import pandas as pd
import numpy as np
from msp2.utils import Conf, zlog, zfatal, ZObject, default_pickle_serializer, OtherHelper, Registrable, AccEvalEntry, F1EvalEntry
from .helper import CmdLineConf, CmdLineParser, RecordNode, RecordNodeVisitor

# --
class AnalyzerConf(Conf):
    def __init__(self):
        self.auto_save_name = "./_tmp_analysis.pkl"
        self.raise_error = False  # useful for debugging
        self.cmd_conf = CmdLineConf()
        self.last_var = "_lastvar"  # special var

class AnalyzerCommand:
    def __init__(self, cmd: str, target: str, args: List, kwargs: Dict):
        self.cmd = cmd
        self.target = target
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f"[{self.target}] {self.args} {self.kwargs}"

class Analyzer(Registrable):
    def __init__(self, conf: AnalyzerConf):
        self.conf = conf
        self.history: List[AnalyzerCommand] = []
        # naming convention: those start with "_" will not be saved
        self.vars = ZObject()  # vars mapping
        self.traces = {}  # var_name -> (explanation, history-idx)
        # tmp ones
        self._cur_cmd: AnalyzerCommand = None
        self._cur_ann_task: AnnotationTask = None

    @classmethod
    def get_ann_type(cls): raise NotImplementedError()

    @property
    def cur_ann_task(self):
        if self._cur_ann_task is None:
            try:
                cur_ann_var_name = self.get_var("_cur_ann_var_name")
                self._cur_ann_task = self.get_var(cur_ann_var_name)
            except:
                return None
        return self._cur_ann_task

    def __del__(self):
        auto_save_name = self.conf.auto_save_name
        if len(auto_save_name) > 0:
            self.do_save(self.conf.auto_save_name)

    # =====
    # some helper functions
    def set_var(self, target: str, v: object, explanation=None):
        if hasattr(self.vars, target):
            zlog(f"Overwriting the existing var `{target}'")
        if target not in self.traces:
            self.traces[target] = []
        # (explanation, history-idx)
        self.traces[target].append((explanation, len(self.history)))
        setattr(self.vars, target, v)

    def get_var(self, target: str):
        if hasattr(self.vars, target):
            return getattr(self.vars, target)
        else:
            assert False, f"Cannot find var `{target}`"

    def get_history(self, n: int):
        if n>=0 and n<len(self.history):
            return self.history[n]
        else:
            return None

    def get_and_check_type(self, target: str, dtype: Type):
        ret = self.get_var(target)
        assert isinstance(ret, dtype), f"Wrong typed target, {type(ret)} instead of {dtype}"
        return ret

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
                zlog(f"[#{i}|{real_idx}] {self.get_history(real_idx)}")
        else:
            zlog(f"Listing histories idx = {num}: {self.get_history(num)}")
        return None

    def do_trace(self, target: str):
        v = self.get_var(target)  # check existence
        for explanation, history_idx in self.traces[target]:
            zlog(f"Var `{target}`: ({explanation}, [{history_idx}]{self.get_history(history_idx)})")

    # a general runner
    def do_eval(self, code: str, mname: str=""):
        s, m, vs = self, OtherHelper.get_module(self), self.vars  # convenient local variable
        if mname:
            import importlib
            m2 = importlib.import_module(mname)
        ret = eval(code)
        return ret

    def do_pdb(self):
        from pdb import set_trace
        set_trace()
        return None

    # note: no load & save history related ones since hard to maintain if loading other's state?
    def do_load(self, file: str):
        zlog(f"Try loading vars from {file}")
        x = default_pickle_serializer.from_file(file)
        self.vars.update(x)  # note: update rather than replace!!

    def do_save(self, file: str):
        zlog(f"Try saving vars to {file}")
        default_pickle_serializer.to_file(self.vars, file)

    # =====
    # some useful ones

    # protocol: target: instances, fcode: d for each instances; return instances
    def do_filter(self, insts_target: str, fcode: str) -> List:
        s, m, vs = self, OtherHelper.get_module(self), self.vars
        _ff = compile(fcode, "", "eval")
        insts = self.get_and_check_type(insts_target, list)
        ret = []
        for d in insts:
            if eval(_ff):
                ret.append(d)
        # ret = [d for d in insts if eval(_ff)]
        zlog(f"Filter by {fcode}: from {len(insts)} to {len(ret)}, {len(ret)/(len(insts)+1e-7)}")
        return ret

    # protocol: target of instances, jcode return a list for each of them
    def do_join(self, insts_target: str, jcode: str) -> List:
        s, m, vs = self, OtherHelper.get_module(self), self.vars
        _ff = compile(jcode, "", "eval")
        insts = self.get_and_check_type(insts_target, list)
        ret = []
        for d in insts:
            ret0 = eval(_ff)
            ret.extend(ret0)
        # ret0 = [eval(_ff) for d in insts]
        # ret = list(chain.from_iterable(ret0))
        zlog(f"Join-list by {jcode}: from {len(insts)} to {len(ret)}")
        return ret

    # protocol: target: instances, kcode: (key) d for inst: return sorted list
    def do_sort(self, insts_target: str, kcode: str) -> List:
        s, m, vs = self, OtherHelper.get_module(self), self.vars
        _ff = compile(kcode, "", "eval")
        insts = self.get_and_check_type(insts_target, list)
        tmp_tuples = [(d, eval(_ff)) for d in insts]
        tmp_tuples.sort(key=lambda x: x[1])
        ret = [x[0] for x in tmp_tuples]
        zlog(f"Sort by key={kcode}: len = {len(ret)}")
        return ret

    # protocol: target: instances, gcode: d for each instances; return one node
    def do_group(self, insts_target: str, gcode: str, sum_key="count") -> RecordNode:
        return self._do_group(insts_target, gcode, sum_key, None)

    # protocol: target: instances, gcode: d for each instances; return one node
    def _do_group(self, insts_target: str, gcode: str, sum_key: str, visitor: RecordNodeVisitor) -> RecordNode:
        s, m, vs = self, OtherHelper.get_module(self), self.vars
        _ff = compile(gcode, "", "eval")
        insts = self.get_and_check_type(insts_target, list)
        # collect all seqs
        ret = RecordNode.new_root()
        for d in insts:
            ret.record_seq(eval(_ff), obj=d)
        # visitor
        if visitor is not None:
            try:
                ret.rec_visit(visitor)
            except:
                zlog(traceback.format_exc())
                zlog("Error of visitor.")
        # some slight summaries here
        all_count = len(insts)
        if not str.isidentifier(sum_key):
            sum_key = eval(sum_key)  # eval the lambda expression
        all_nodes = ret.get_descendants(key=sum_key)
        ss = []
        for z in all_nodes:
            all_parents = z.get_antecedents()
            if len(all_parents) > 0:
                assert all_parents[0].count == all_count
            perc_info = ', '.join([f"{z.count/(zp.count+1e-6):.4f}" for zp in all_parents])
            ss.append(['=='*len(z.path), str(z.path), f"{z.count}({perc_info})", z.get_content()])
        # sstr = "\n".join(ss)
        # sstr = ""
        # pd.set_option('display.width', 1000)
        # pd.set_option('display.max_colwidth', 1000)
        pdf = pd.DataFrame(ss)
        pdf_str = pdf.to_string()
        zlog(f"Group {len(insts)} instances by {gcode}, all {len(ss)} nodes:\n{pdf_str}")
        return ret

    # filter + group
    def do_fg(self, insts_target: str, fcode: str, gcode: str, **g_kwargs):
        f_res = self.do_filter(insts_target, fcode)
        self.set_var(self.conf.last_var, f_res)  # store tmp
        g_res = self.do_group(self.conf.last_var, gcode, **g_kwargs)
        return g_res

    # correlation
    def do_corr(self, insts_target: str, acode: str, bcode: str):
        s, m, vs = self, OtherHelper.get_module(self), self.vars
        _ffa = compile(acode, "", "eval")
        _ffb = compile(bcode, "", "eval")
        insts = self.get_and_check_type(insts_target, list)
        a_vals, b_vals = [], []
        for d in insts:
            a_vals.append(eval(_ffa))
            b_vals.append(eval(_ffb))
        # --
        from scipy.stats import pearsonr, spearmanr
        zlog(f"Pearson={pearsonr(a_vals,b_vals)}")
        zlog(f"Spearman={spearmanr(a_vals,b_vals)}")
        return None

    # similar to group, but using pd instead
    def do_get_pd(self, insts_target: str, gcode: str) -> pd.DataFrame:
        s, m, vs = self, OtherHelper.get_module(self), self.vars
        _ff = compile(gcode, "", "eval")
        insts = self.get_and_check_type(insts_target, list)
        # --
        fields = [eval(_ff) for d in insts]
        ret = pd.DataFrame(fields)
        zlog(f"Group {len(insts)} instances by {gcode} to pd.DataFrame shape={ret.shape}.")
        return ret

    # calculation and manipulation on pd, also assuming the local name is d
    def do_cal_pd(self, inst_pd: str, scode: str):
        s, m, vs = self, OtherHelper.get_module(self), self.vars
        _ff = compile(scode, "", "eval")
        d = self.get_and_check_type(inst_pd, pd.DataFrame)
        # --
        ret = eval(_ff)
        zlog(f"Calculation on pd.DataFrame by {scode}, and get another one as: {str(ret)}")
        return ret

    # --
    # do breakdown and eval

    # shortcut!
    def do_break_eval2(self, insts_target: str, pcode: str, gcode: str,
                       corr_code="d.pred.label == d.gold.label", pint=0, **kwargs):
        pcode2 = pcode.replace("d.pred", f"d.preds[{pint}]")
        corr_code2 = corr_code.replace("d.pred", f"d.preds[{pint}]")
        return self.do_break_eval(insts_target, pcode2, gcode, corr_code2, **kwargs)

    # real go!
    def do_break_eval(self, insts_target: str, pcode: str, gcode: str, corr_code="d.pred.label == d.gold.label",
                      sort_key='-1', truncate_items=100, pdb=False):
        s, m, vs = self, OtherHelper.get_module(self), self.vars
        sort_key = int(sort_key)
        _fp, _fg = compile(pcode, "", "eval"), compile(gcode, "", "eval")
        _fcorr = compile(corr_code, "", "eval")
        insts = self.get_and_check_type(insts_target, list)
        # --
        res = {}
        for d in insts:
            corr = 0
            # --
            no_pred = False
            try:  # use try/except to set this!
                key_p = eval(_fp)
            except:
                no_pred = True
            # --
            if not no_pred and d.gold is not None:
                corr = eval(_fcorr)
            if not no_pred:
                key_p = eval(_fp)
                if key_p not in res:
                    res[key_p] = F1EvalEntry()
                res[key_p].record_p(int(corr))
            if d.gold is not None:
                key_g = eval(_fg)
                if key_g not in res:
                    res[key_g] = F1EvalEntry()
                res[key_g].record_r(int(corr))
        # final
        details = [(k,)+v.details for k,v in res.items()]
        details = sorted(details, key=(lambda x: x[sort_key]), reverse=True)
        # --
        pdf = pd.DataFrame(details)
        pdf_str = pdf[:int(truncate_items)].to_string()
        zlog(f"Break-eval {len(insts)} instances by {pcode}/{gcode}:\n{pdf_str}")
        if pdb:
            breakpoint()
        return res

    # --
    # ann related
    def do_ann_attach(self, name: str):
        s, m, vs = self, OtherHelper.get_module(self), self.vars
        # --
        # todo(note): keep this special name for this special purpose
        if name == "_detach":
            self._cur_ann_task = None
            self.set_var("_cur_ann_var_name", None)
            return
        # --
        z = self.get_and_check_type(name, AnnotationTask)
        zlog(f"Attach ann_task: from {self.cur_ann_task} to {z}")
        self.set_var("_cur_ann_var_name", z)  # set special name!!

    def do_ann_new(self, insts_target: str, fcode: str=None, try_attach=1):
        s, m, vs = self, OtherHelper.get_module(self), self.vars
        # --
        assert self._cur_cmd.target is not None, "Should assign this to a var to avoid accidental loss!"
        vs = self.vars
        insts = self.get_and_check_type(insts_target, list)
        if fcode is None:
            new_task = self.__class__.get_ann_type()(insts)
        else:
            new_task = eval(fcode)(insts)
        # todo(note): here need auto save?
        try_attach = bool(int(try_attach))
        if try_attach:
            if self.cur_ann_task is not None:
                zlog("Detach current task and try attach the new one!!")
                self._cur_ann_task = None
            # note: directly set name, which will be assigned later
            # todo(+N): maybe source of certain bugs?
            self.set_var("_cur_ann_var_name", self._cur_cmd.target)  # set special name!!
            zlog("New ann task, and ann_var_name set!")
        return new_task

    # =====
    # main ones

    def process(self, cmd: AnalyzerCommand):
        args = cmd.args
        cmd_name, real_args = args[0], args[1:]
        # similar to the cmd package
        method_name = "do_" + cmd_name
        # first check ann then check self!
        cur_ann_task = self.cur_ann_task
        if cur_ann_task is not None and hasattr(cur_ann_task, method_name):
            zlog(f"Performing annotator's command: {cmd}")
            return getattr(cur_ann_task, method_name)(*real_args, **cmd.kwargs)
        else:
            assert hasattr(self, method_name), f"Unknown command {cmd_name}"
            zlog(f"Performing command: {cmd}")
            return getattr(self, method_name)(*real_args, **cmd.kwargs)

    def loop(self, file: str = None):
        if file is None:  # use pre-set
            parser = CmdLineParser(self.conf.cmd_conf)
        else:
            parser = CmdLineParser(self.conf.cmd_conf, cmd_input=file)
        # loop
        for res in parser:
            cmd = AnalyzerCommand(*res)
            self._cur_cmd = cmd
            # process it
            if self.conf.raise_error:
                v = self.process(cmd)
            else:
                try:
                    v = self.process(cmd)
                except AssertionError as e:
                    zlog(f"Checking Error: " + str(e))
                    continue
                except:
                    zlog(f"Command error: " + str(traceback.format_exc()))
                    continue
            self._cur_cmd = None  # reset
            cmd_count = len(self.history)
            if v is not None:  # does not store None
                if cmd.target is not None:
                    self.set_var(cmd.target, v)
                # also store special VAR!
                self.set_var(self.conf.last_var, v)
            self.history.append(cmd)
            zlog(f"Finish command #{cmd_count}: {cmd}")

    def main(self, *args, **kwargs):
        self.loop()

# --
class AnnotationTask:
    def __init__(self, objs: List):
        self.objs = objs
        self.cur_idx = 0
        self.length = len(objs)

    @property
    def cur_obj(self):
        return self.objs[self.cur_idx]

    @property
    def cur_status(self):
        return f"# == Current ANN status: {self.cur_idx}/{self.length}"

    def obj_info(self, obj, **kwargs) -> str:
        return None  # if None, then no info!

    # setting cur_idx and return the new cur_idx
    def set_cur(self, offset=0, abs_idx=-1, no_jump_over=False):
        if abs_idx>=0:
            new_focus = abs_idx
        else:
            new_focus = self.cur_idx + offset
        if no_jump_over:
            if new_focus>=0 and new_focus<self.length:
                self.cur_idx = new_focus
            else:
                zlog(f"Cur_idx setting failed: at the boundary: {self.cur_idx}, {self.length}")
        else:
            self.cur_idx = max(0, min(new_focus, self.length-1))
        return self.cur_idx

    # =====
    # common commands

    def do_aj(self, offset=0, abs_idx=-1, no_jump_over=False):
        offset, abs_idx, no_jump_over = int(offset), int(abs_idx), bool(int(no_jump_over))
        # --
        old_idx = self.cur_idx
        self.set_cur(offset, abs_idx, no_jump_over)
        zlog(f"JUMP from {old_idx} to {self.cur_idx}")
        self.do_ap()
        return

    def do_ap(self, **kwargs):
        cur_obj_info = self.obj_info(self.cur_obj, **kwargs)
        ret = ""
        if cur_obj_info is not None:
            ret += cur_obj_info + "\n"
        ret += self.cur_status
        zlog(ret)
