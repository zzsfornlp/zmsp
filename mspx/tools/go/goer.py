#

__all__ = [
    "GoerConf", "Goer"
]

# main runner

import re
from typing import Dict
from mspx.utils import Conf, Configurable, zlog, zopen, Random, ZHelper
from .task import *
from .engine import *

class GoerConf(Conf):
    def __init__(self):
        self.engine = MyEngineConf.direct_conf(_rm_names=['name'])
        # --
        self.name = ""  # run name (upmost level name)
        self.shuffle = False
        self.task_use_tidx = True  # use tidx as task's id
        # tasks sels
        self.task_pats = [""]  # selected running patterns
        self.task_sels = []  # only run sel ids
        self.task_snum = 1  # split number
        self.task_sidx = [0]  # split id
        # read from input table
        self.input_table_file = ""
        self.var = {}  # extra fill-in vars
        # read from args
        self.inputs = []
        self.task = MyTaskConf()
        self.i2o = {}  # I2O: [[KEY]]: lambda x: ...
        # --

class Goer(Configurable):
    def __init__(self, conf: GoerConf = None, **kwargs):
        super().__init__(conf, **kwargs)
        conf: GoerConf = self.conf
        # --
        self.i2o = \
            {k: eval(v) if v.startswith('lambda') else eval("lambda x: " + v) for k,v in conf.i2o.items()}
        self.task_pats = [re.compile(z) for z in conf.task_pats]
        # --

    @staticmethod
    def extend_choices(init_id: str, choices: Dict):
        # --
        def _extend2(_item):  # level 2 extend
            if isinstance(_item, list):
                _pat = f"%0{len(str(max(0, len(_item)-1)))}d"
                return [((_pat % _ii), _zz) for _ii, _zz in enumerate(_item)]
            elif isinstance(_item, dict):
                return list(_item.items())
            else:
                raise NotImplementedError()
        # --
        def _extend1(_item):  # level 1 extend
            if isinstance(_item, list):
                _them = [(chr(ord('A')+ii), _extend2(zz)) for ii, zz in enumerate(_item)]
            elif isinstance(_item, dict):
                _them = [(_k, _extend2(_v)) for _k, _v in _item.items()]
            else:
                raise NotImplementedError()
            # flatten
            _rets = [("", "")]
            for _choice_name, _choices in _them:
                _rets = sum([[
                    ((a if len(_choices)==1 else a+_choice_name+a2), f"{b} {b2}") for a2,b2 in _choices] for a,b in _rets], [])
            return _rets
        # --
        def _repl(_s, _d):
            for _k, _v in _d.items():
                _s = _s.replace(f"[[{_k}]]", _v)
            return _s
        # --
        currs = [{"ID": init_id}]  # note: first make ID as the same as k
        # --
        for k2, v2 in choices.items():  # for each [[VAR]]
            if isinstance(v2, str):
                for one in currs:
                    one[k2] = _repl(v2, one)   # note: update inplace
            else:
                curr_choices = _extend1(v2)
                new_currs = []
                for one in currs:
                    for _k, _v in curr_choices:
                        new_one = one.copy()
                        if _k:  # note: update id
                            new_one["ID"] = new_one["ID"] + "_" + _k
                        new_one[k2] = _repl(_v, one)
                        new_currs.append(new_one)
                currs = new_currs
        # --
        return currs
        # --

    @staticmethod
    def make_task(base_task: Dict, choice: Dict):
        dd = base_task.copy()
        for k in list(dd.keys()):
            if isinstance(dd[k], str):
                for k2, v2 in choice.items():
                    k2 = f"[[{k2}]]"
                    if k2 in dd[k]:
                        dd[k] = dd[k].replace(k2, v2)
                dd[k] = re.sub(r"\[\[[A-Z]+]]", "", dd[k])  # remove all others
        return MyTask.make_task(dd)

    def select_names(self, names):
        sel_indexes = []
        for ii, nn in enumerate(names):
            if any((z.search(nn) is not None) for z in self.task_pats):
                sel_indexes.append(ii)
        return sel_indexes

    def main(self):
        conf: GoerConf = self.conf
        engine = MyEngine(conf.engine, name=conf.name)
        # first decide tasks
        sel_idxes = None
        if conf.input_table_file:
            with zopen(conf.input_table_file) as fd:
                s = fd.read()
                exec(s, globals(), locals())  # must make a "table_tasks"
                _base_task, _choices0 = locals()['table_tasks'][conf.name]  # read the table from file
                _choices = self.extend_choices(conf.name, _choices0)
                _pad_tids = ZHelper.pad_strings(list(range(len(_choices))), '0')
                for _i, _c in enumerate(_choices):  # update with externals
                    _c.update(conf.var)
                    _c['TIDX'] = str(_i)  # overall task idx
                    if conf.task_use_tidx:
                        _c['ID'] = f"{conf.name}_{_pad_tids[_i]}"
                zlog(f"Extend choices[{len(_choices)}] with {_base_task}")
                tasks = [self.make_task(_base_task, z) for z in _choices]
                sel_idxes = self.select_names([z['ID'] for z in _choices])
        else:
            tasks = []
            for one_ii, one_in in enumerate(conf.inputs):
                repl_dict = {'[[IN]]': one_in, "[[TIDX]]": str(one_ii)}
                repl_dict.update({f'[[{k}]]': ff(one_in) for k,ff in self.i2o.items()})
                t = MyTask(conf.task.new_with_repl(repl_dict))
                tasks.append(t)
                sel_idxes = self.select_names(conf.inputs)
        # --
        if conf.task_sels:  # select by idx
            _sel_idxes = set([int(z) for z in conf.task_sels])
            sel_idxes = [z for z in sel_idxes if z in _sel_idxes]
        # --
        if len(sel_idxes) < len(tasks):
            old_tasks = tasks
            tasks = [old_tasks[ii] for ii in sel_idxes]
            zlog(f"Select by {conf.task_pats} & {conf.task_sels}: [{len(tasks)}]/[{len(old_tasks)}]")
        # --
        # further split
        if conf.task_snum > 1:
            assert all(z < conf.task_snum for z in conf.task_sidx)
            old_tasks = tasks
            tasks = [z for i, z in enumerate(old_tasks) if (i%conf.task_snum) in conf.task_sidx]
            zlog(f"Select piece {conf.task_sidx}/{conf.task_snum}: {len(tasks)}/{len(old_tasks)}")
        # --
        if conf.shuffle:
            _gen = Random.get_generator('go')
            _gen.shuffle(tasks)
        # --
        zlog(f"Run {len(tasks)} tasks with {engine}")
        engine.main(tasks)
        # --
