#

# core component

"""
# -- dir structure (assuming at ROOT-DIR)
al.conf  # store conf
al.log  # store logs
al.record.json  # store records
al.ref.json  # reference data (for oracle in simulation)
iter${I}/*  # files for iter${I}
-- data.init.json  # data available (L+U) at the start of iter-I
-- data.query.json  # data(subset) for query in iter-I
-- data.ann.json  # data(subset) annotated in Iter-I
-- data.comb.json  # combined data for training
-- tz  # final dir for trained model
# -- al procedure (at iter-I)
0. setup: external if I==0 else use "iter-(I-1)/comb" as init
1. query: "init"->"query" with last model (if I>0) or nothing (if I==0)
2. annotate: "query"->"ann" with simul+ref or external-ann
3. combine: *->"comb", put data together for training
4. train: train model
"""

import os
from typing import Union
import shutil
from collections import OrderedDict, Counter, defaultdict
from mspx.data.rw import ReaderGetterConf, WriterGetterConf
from mspx.utils import zlog, zcheck, Conf, Serializable, Configurable, Logger, default_json_serializer, Constants, Timer, mkdir_p, ZHelper, wrap_color, zglob1, resymlink, zwarn
from .tasks import *

# conf
class ALConf(Conf):
    def __init__(self):
        # file names
        self.file_conf = 'al.conf.json'  # load if there are!
        self.no_load_conf = False  # no loading conf!
        self.file_log = Logger.MAGIC_APP_CODE + 'al.log'  # append by default!
        self.file_record = 'al.record.json'
        self.file_ref = 'al.ref.json'
        self.file_dev = 'al.dev.json'
        self.dir_prefix = "iter"  # iter dir
        # procedure
        self.qann_times = 1  # whether doing multiple qann inside one iter: details specific to tasks
        self.start = ''  # which step to run from: 'Iter.substep' or 'Iter(.0)' or auto-detect('')
        self.stop = ''  # which step to stop: 'Iter.substep' or 'Iter(.-1)' or ALL-annotated('') or one-step('~')
        self.special = ''   # special task rather than running
        # R & W
        self.R = ReaderGetterConf().direct_update(input_allow_std=False)
        self.W = WriterGetterConf()
        # specific task helper
        self.al_task: ALTaskConf = ALTaskConf.get_entries()
        self.simul = True  # do simul mode (delete annotations and make a ref at setup & auto-ann)
        # 0: setup
        self.setup_output = "data.init.json"
        self.setup_dataL = []  # input labeled data
        self.setup_dataU = []  # input unlabeled data (labeled and goes to ref if simul)
        self.setup_dataD = []  # input dev data
        # 1: query
        self.query_output = "data.query.json"
        self.query2_output = "data.query2.json"
        self.query_i0_model = ""  # an extra model for query in iter0
        self.debug_query_use_refii = -1  # ii>=this special mode by feeding ref for querying
        # 2: ann
        self.ann_output = "data.ann.json"
        self.ann2_output = "data.ann2.json"
        # 3: comb
        self.comb_output = "data.comb.json"
        self.combF_output = "data.combF.json"  # full combined file (ann+pred)
        # 4: train
        self.train_output_dir = "tz"  # output-dir

    @property
    def train_output(self):
        return os.path.join(self.train_output_dir, 'zmodel.best.m')  # fix this name!

    @property
    def query_iter0_model(self):
        return zglob1(self.query_i0_model)

# project
class ALProject(Configurable):
    # AL steps in one loop
    @staticmethod
    def get_procedure(qann_times: int):
        ret = OrderedDict()
        ret['setup'] = {}
        for ii in range(qann_times):
            suffix = '' if ii==0 else str(ii+1)
            ret['query' + suffix] = {}
            ret['ann' + suffix] = {}
        ret['comb'] = {}
        ret['train'] = {}
        return ret

    def __init__(self, conf: ALConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ALConf = self.conf
        # --
        self.PROCEDURE = self.get_procedure(conf.qann_times)
        # read record
        if os.path.isfile(conf.file_record):
            records = list(default_json_serializer.yield_iter(conf.file_record))
        else:
            records = []
        self.records = records
        # start and stop
        self.step_curr = self.normalize_step_name(conf.start, 0) if conf.start else self.auto_judge_step()
        if conf.stop == '~':  # just run one step!
            self.step_stop = self.step_curr
        else:
            self.step_stop = self.normalize_step_name(conf.stop, -1) if conf.stop else (Constants.INT_PRAC_MAX, 0)
        # task
        self.task: ALTaskHelper = conf.al_task.make_node()
        self.task.setup_iter(self.step_curr[0])  # setup AL iter!
        # --
        zlog(f"Init: {self}")
        # --

    def __repr__(self):
        return f"ALProject(@{os.path.abspath(os.curdir)})[{self.task}]: {self.step_curr} => {self.step_stop}"

    def name2step(self, step_name: str): return list(self.PROCEDURE.keys()).index(step_name)
    def step2name(self, step_ss: int): return list(self.PROCEDURE.keys())[step_ss]

    def checking(self, flag: bool, s: str):
        zcheck(flag, s, err_act='err')

    def get_dir_name(self, ii: int = None):
        if ii is None:
            ii = self.step_curr[0]
        return f"{self.conf.dir_prefix}{ii:02d}"  # make it easier to blob

    def get_output_name(self, ii: int = None, ss: Union[int, str] = None, offset=0):
        dir_name = self.get_dir_name(ii)
        if ss is None:
            ss = self.step_curr[1]
        if isinstance(ss, str):
            assert ss in self.PROCEDURE
            ss = list(self.PROCEDURE.keys()).index(ss)  # must be there!
        step_name = self.step2name(ss + offset)
        file_name = getattr(self.conf, step_name + "_output")
        return os.path.join(dir_name, file_name)

    def get_last_model(self, ii: int = None):
        if ii is None:
            ii = self.step_curr[0]
        if ii > 0:
            last_model = self.get_output_name(ii-1, 'train')
        else:  # allow an external model!
            conf: ALConf = self.conf
            last_model = conf.query_iter0_model if conf.query_iter0_model else None
        return last_model

    # automatically judge current step by files available
    def auto_judge_step(self):
        conf: ALConf = self.conf
        # check iter
        ii = 0
        while os.path.isdir(self.get_dir_name(ii)):
            ii += 1
        ii -= 1
        if ii < 0:  # nothing there
            ii = 0
        # check step
        ss = 0
        for step_name in self.PROCEDURE:
            if os.path.isfile(self.get_output_name(ii, ss)):
                ss += 1
                continue
        if ss >= len(self.PROCEDURE):
            ii += 1
            ss = 0
        return (ii, ss)  # iter, step

    # normalizing names
    def normalize_step_name(self, step_name: str, default_step):
        if '.' not in step_name:
            step_name = step_name + "." + str(default_step)
        ii, ss = step_name.split(".")
        ii = int(ii)
        try:
            ss = int(ss)
            if ss < 0:
                ss = (ss + len(self.PROCEDURE))
        except:
            ss = self.name2step(ss)
        return (ii, ss)

    # --
    def run(self):
        conf: ALConf = self.conf
        # --
        if conf.special:
            with Timer(f"RUN special.{conf.special}"):  # note: will throw err if problems
                result = getattr(self, f"run_{conf.special}")()
                if isinstance(result, dict):
                    result = ZHelper.resort_dict(result)  # for easier checking
            zlog(wrap_color(f"=> {result}", bcolor='blue'))
            return
        # --
        while self.step_curr <= self.step_stop:
            ii, ss = self.step_curr
            ss_name = self.step2name(ss)
            self.task.setup_iter(ii)  # setup AL iter!
            zlog("# -----")
            with Timer(f"RUN {ii}.{ss_name} ({ss})"):  # note: will throw err if problems
                result = getattr(self, f"run_{ss_name}")()
                if isinstance(result, dict):
                    result = ZHelper.resort_dict(result)  # for easier checking
            self.records.append([self.step_curr, Timer.ctime(), result])
            zlog(wrap_color(f"=> {self.records[-1]}", bcolor='blue'))
            # next step
            ss += 1
            if ss >= len(self.PROCEDURE):  # next iter!
                ii, ss = ii+1, 0
            self.step_curr = (ii, ss)
            # save record (after each step!!)
            default_json_serializer.save_iter(self.records, conf.file_record)
        # --

    # --
    # specials

    def run_conf(self):
        _file = self.conf.file_conf
        if _file:
            default_json_serializer.to_file(self.conf, _file)  # simply save conf
        return f"Save conf to {_file}"

    def run_stat(self):
        reader = self.conf.R.get_reader(input_allow_std=False)
        cc = Counter()
        for inst in reader:
            cc1 = self.task.setup_inst(inst, mark_unn=False)  # reuse setup!
            cc += cc1
        return cc

    # --
    # AL running

    def run_setup(self):
        conf: ALConf = self.conf
        curr_ii, _ = self.step_curr
        # --
        curr_dir = self.get_dir_name()
        mkdir_p(curr_dir)  # make a new dir
        curr_output = self.get_output_name()
        if curr_ii == 0:  # read from external
            cc_all = {}
            doc_ids = set()
            with conf.W.get_writer(output_path=curr_output) as writer, \
                    conf.W.get_writer(output_path=conf.file_ref) as writerR:
                for data, data_type in zip([conf.setup_dataL, conf.setup_dataU], ['L', 'U']):
                    reader = conf.R.get_reader(input_path=data, input_allow_std=False)
                    data_isU = (data_type == 'U')
                    cc = Counter()
                    for inst in reader:
                        # --
                        if inst.id is None:
                            inst.set_id(f'd{len(doc_ids)}')
                        assert inst.id not in doc_ids
                        doc_ids.add(inst.id)
                        # --
                        if conf.simul and data_isU:  # in simul mode, U actually has labels!
                            writerR.write_inst(inst)
                        # --
                        cc1 = self.task.setup_inst(inst, mark_unn=data_isU)
                        writer.write_inst(inst)
                        cc += cc1
                    cc_all[data_type] = cc
            # --
            # prepare dev
            if conf.setup_dataD:
                reader = conf.R.get_reader(input_path=conf.setup_dataD, input_allow_std=False)
                sampled_insts = self.task.sample_data(reader)
                cc = Counter()
                dev_ids = set()
                with conf.W.get_writer(output_path=conf.file_dev) as writerD:
                    for inst in sampled_insts:
                        cc2 = self.task.setup_inst(inst, mark_unn=False)
                        # --
                        if inst.id is None:
                            inst.set_id(f'dev{len(dev_ids)}')
                        assert inst.id not in dev_ids
                        dev_ids.add(inst.id)
                        # --
                        writerD.write_inst(inst)
                        cc += cc2
                cc_all['D'] = cc
            else:
                zwarn("No DEV set provided!")
            # --
            # also write conf in this round
            if conf.file_conf:
                default_json_serializer.to_file(conf, conf.file_conf)
            # --
            return cc_all
        else:  # symlink from last iteration
            last_data = self.get_output_name(curr_ii-1, 'comb')  # combined data
            self.checking(os.path.isfile(last_data), f"LastData not exist: {last_data}")
            resymlink(os.path.join("..", last_data), curr_output)
            return f"Symlink data {last_data} -> {curr_output}"
        # --

    # todo(+N): change name to dispatch, not elegant!! ...
    def run_query(self, qann_ii=''):
        conf: ALConf = self.conf
        curr_ii, _ = self.step_curr
        # --
        _qdata = self.get_output_name(offset=-1)  # from last step
        if conf.debug_query_use_refii >= 0 and curr_ii >= conf.debug_query_use_refii:
            _qdata0 = _qdata
            _qdata = ZHelper.insert_path(_qdata0, 'debug_with_ref', position=-2)
            self.task.prep_query_with_ref(_qdata0, _qdata, conf.file_ref)
        last_model = self.get_last_model()
        # note: delegate to task-specific ones, may need inference, assign-embeddings, ...
        # data prepared for query!
        qdata, qdataD = getattr(self.task, 'prep_query'+qann_ii)(_qdata, last_model, conf.file_dev)
        # actually do the query!
        _stream = conf.R.get_reader(input_path=qdata)
        _streamD = conf.R.get_reader(input_path=qdataD) if qdataD else None  # dev
        _streamDR = conf.R.get_reader(input_path=conf.file_dev) if qdataD else None  # dev ref
        _streamR = conf.R.get_reader(input_path=conf.file_ref) if conf.simul else None  # unlabel-pool ref
        query_insts, query_cc = getattr(self.task, 'do_query'+qann_ii)(
            _stream, _streamD, ref_stream=_streamR, refD_stream=_streamDR, no_strg=(last_model is None))
        assert len(query_insts) > 0, "No remaining un-annotated instances!"
        # write
        curr_output = self.get_output_name()
        with conf.W.get_writer(output_path=curr_output) as writer:
            writer.write_insts(query_insts)
        return query_cc

    def run_ann(self, qann_ii=''):
        conf: ALConf = self.conf
        # --
        curr_query = self.get_output_name(offset=-1)  # from last step
        curr_output = self.get_output_name()
        if conf.simul:  # "ann" data from ref!
            ref_insts = list(conf.R.get_reader(input_path=conf.file_ref))  # reference gold
            ref_map = {z.id: z for z in ref_insts}
            assert len(ref_map) == len(ref_insts)
            query_insts = list(conf.R.get_reader(input_path=curr_query))  # query ones
            ret = getattr(self.task, 'do_simul_ann'+qann_ii)(query_insts, ref_map, last_model=self.get_last_model())
            if isinstance(ret, tuple):
                query_insts, cc = ret  # note: replacing!
            else:
                cc = ret
            # write
            with conf.W.get_writer(output_path=curr_output) as writer:
                writer.write_insts(query_insts)
            # --
            return cc
        else:
            raise NotImplementedError(f"Please annotate externally: {curr_query} => {curr_output}")
        # --

    # --
    # simply reuse!
    def run_query2(self): return self.run_query('2')
    def run_ann2(self): return self.run_ann('2')
    # --

    def run_comb(self):
        conf: ALConf = self.conf
        # --
        ann_insts = list(conf.R.get_reader(input_path=self.get_output_name(offset=-1)))
        trg_insts = list(conf.R.get_reader(input_path=self.get_output_name(ss='setup')))
        trg_map = {z.id: z for z in trg_insts}
        assert len(trg_map) == len(trg_insts)
        cc = self.task.do_comb(ann_insts, trg_map)
        # write
        curr_output = self.get_output_name()
        with conf.W.get_writer(output_path=curr_output) as writer:
            writer.write_insts(trg_insts)
        # --
        # combine full
        zlog(f"Print out cc before final combine full with annpred: {ZHelper.resort_dict(cc)}")
        full_output = ZHelper.insert_path(curr_output, 'full', position=-2)
        self.task.do_comb_annpred(curr_output, full_output, self.get_last_model())
        # --
        return cc

    def run_train(self):
        conf: ALConf = self.conf
        # --
        train_data = self.get_output_name(ss='comb')
        output_model = self.get_output_name()
        last_model = self.get_last_model()
        cc = self.task.do_train(train_data, output_model, last_model, conf.file_dev)
        return cc

# --
# b mspx/tools/al/core:222
