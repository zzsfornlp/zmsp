#

import math
from typing import Dict

from msp import utils
from msp.utils import Timer, Random, Constants, Conf, JsonRW, StatRecorder, Helper
from msp.utils import GLOBAL_RECORDER

# record the results for one point
class RecordResult:
    def __init__(self, results: Dict, score: float=None):
        self.results = results
        if score is None:
            self.score = Constants.REAL_PRAC_MIN      # should be enough
        else:
            self.score = score

    #
    def to_builtin(self):
        return self.results

    def __getitem__(self, item):
        return self.results[item]

    # return the one float score, better eval get larger scores
    def __float__(self):
        return self.score

    def __repr__(self):
        return "Result(%s): %s" % (str(self.score), str(self.results))

# records stats of training-fb and dev, coupled with TrainingRunner
class TrainingProgressRecorder(object):
    def __init__(self, patience, anneal_restarts_cap):
        # hyper-parameters
        self.patience = patience
        self.anneal_restarts_cap = anneal_restarts_cap
        # idxes
        self.eidx = 0       # epochs
        self.iidx = 0       # insts
        self.uidx = 0       # updates: can be diff with uidx
        self.estop = False
        # checkpoints
        self.chp_names = []
        self.train_records = []
        self.dev_records = []
        # about dev checks
        self.best_dev_record = RecordResult({})
        self.best_point = -1
        self.bad_counter = 0
        self.bad_points = []
        self.anneal_restarts_done = 0
        self.anneal_restarts_points = []

    #
    def current_suffix(self):
        # 4 digits should be enough
        sname = ".c%0.3d-e%d-i%dK-u%d" % (len(self.chp_names), self.eidx, int(self.iidx/1000.), self.uidx)
        return sname

    def current_idxes(self):
        return {"aidx": self.anneal_restarts_done, "eidx": self.eidx, "iidx": self.iidx, "uidx": self.uidx, "cidx": len(self.chp_names)}

    def best_info(self):
        return [self.best_point, self.chp_names[self.best_point], str(self.best_dev_record)]

    # called after one dev, log dev result and restart train-record
    def checkpoint(self, train_result: RecordResult, dev_result: RecordResult):
        # log down
        sname = self.current_suffix()
        self.chp_names.append(sname)
        self.train_records.append(train_result)
        self.dev_records.append(dev_result)
        #
        if_best = if_anneal = False
        if float(dev_result) > float(self.best_dev_record):
            self.bad_counter = 0
            self.best_dev_record = dev_result
            self.best_point = len(self.chp_names)-1
            if_best = True
        else:
            cur_info = [sname, self.best_point, self.chp_names[self.best_point], str(self.best_dev_record)]
            self.bad_counter += 1
            # [cur-name, best-score-idx, best-score-name, best-score]
            self.bad_points.append(cur_info)
            utils.zlog("Bad++, now bad/anneal is %s/%s." % (self.bad_counter, self.anneal_restarts_done), func="warn")
            if self.bad_counter >= self.patience:
                self.bad_counter = 0
                self.anneal_restarts_points.append(cur_info)
                if self.anneal_restarts_done < self.anneal_restarts_cap:
                    self.anneal_restarts_done += 1
                    utils.zlog("Anneal++, now %s." % (self.anneal_restarts_done,), func="warn")
                    if_anneal = True
                else:
                    utils.zlog("Sorry, Early Stop !!", func="warn")
                    self.estop = True
        return if_best, if_anneal
#

class RConf(Conf):
    def __init__(self):
        #
        self.skip_batch = 0.            # rate of randomly skipping each batch
        self.split_batch = 1            # num of batch splits, default 1 means no splitting
        self.flag_debug = False
        self.flag_verbose = True
        # mostly based on uidx: number of updates
        self.report_freq = 1000
        self.valid_freq = Constants.INT_MAX
        self.validate_epoch = True
        self.validate_first = False
        self.max_epochs = 100
        self.max_updates = 1000000
        # annealing
        self.anneal_times = 100
        self.patience = 3
        # write model
        self.model_overwrite = True
        self.model_name = "zmodel"
        self.suffix_curr = ".curr"
        self.suffix_best = ".best"
        #
        # lrate schedule
        self.lrate = SVConf().init_from_kwargs(init_val=0.001, which_idx="aidx", mode="exp", k=0.75)
        self.lrate_warmup = 0       # linear increasing lrate as warmup for how many steps (minus values means epoch)

# common practice for training
class TrainingRunner(object):
    def __init__(self, rconf, model, batch_size_f=None):
        self.model = model
        #
        self._tp = TrainingProgressRecorder(rconf.patience, rconf.anneal_times)
        self.train_recorder = StatRecorder(True)
        self.rconf = rconf
        #
        self.batch_size_f = batch_size_f
        if self.batch_size_f is None:       # default is count number of instances
            self.batch_size_f = lambda x: len(x)
        #
        self.lrate = ScheduledValue("lrate", rconf.lrate)
        self.lrate_warmup_steps = 0         # set at the start of run()
        self.scheduled_values = [self.lrate]
        self.add_scheduled_values(model.get_scheduled_values())

    def add_scheduled_values(self, sv_list):
        for one in sv_list:
            utils.zlog("Adding scheduled value %s in training." % (one,))
            self.scheduled_values.append(one)

    def _finished(self):
        return self._tp.estop or self._tp.eidx >= self.rconf.max_epochs \
                or self._tp.uidx >= self.rconf.max_updates

    def current_name(self):
        return self._tp.current_suffix()

    # training for one batch
    def _fb_batch(self, insts):
        num_splits = self.rconf.split_batch
        splitted_insts = Helper.split_list(insts, num_splits)
        with self.train_recorder.go():
            for one_insts in splitted_insts:
                res = self._run_fb(one_insts)
                self.train_recorder.record(res)
        self._tp.iidx += self.batch_size_f(insts)

    # dev all of them
    def _validate(self, dev_streams):
        rconf = self.rconf
        # report & reset training stat
        if self._tp.uidx > 0:
            train_result = self._run_train_report()              # first report training stat
            # todo(warn): reset when validation
            GLOBAL_RECORDER.reset()
            self.train_recorder.reset()     # reset training stat
        else:   # for validate_first
            train_result = RecordResult({})
        # dev
        ss = self.current_name()
        with Timer(tag="valid", info="Valid %s" % ss, print_date=True):
            # validate
            dev_result = self._run_validate(dev_streams)
            # record
            if_best, if_anneal = self._tp.checkpoint(train_result, dev_result)
            # checkpoint - save curr & best
            self.save(rconf.model_name+rconf.suffix_curr)
            if not rconf.model_overwrite:
                self.save(rconf.model_name+ss)
            if if_best:
                self.save(rconf.model_name+rconf.suffix_best)
                utils.zlog("Curr is best: " + str(self._tp.best_info()), func="result")
            else:
                utils.zlog("Curr not best, the best is " + str(self._tp.best_info()), func="result")
        # schedule values
        cur_idxes = self._tp.current_idxes()
        for one_sv in self.scheduled_values:
            one_sv.adjust_at_ckp(ss, cur_idxes)
        utils.zlog("")

    def run(self, train_stream, dev_streams):
        rconf = self.rconf
        last_report_uidx, last_dev_uidx = 0, 0
        if rconf.validate_first:
            self._validate(dev_streams)
        # for lrate warm
        if rconf.lrate_warmup < 0:
            # calculate epochs
            steps_per_epoch = 0
            for _ in train_stream:
                steps_per_epoch += 1
            n_epoch = -rconf.lrate_warmup
            n_steps = n_epoch * steps_per_epoch
            utils.zlog(f"Calculating warmup steps for {n_epoch} epochs: {steps_per_epoch} steps per epoch.")
        elif rconf.lrate_warmup > 0:
            n_steps = rconf.lrate_warmup
        else:
            n_steps = 0
        utils.zlog(f"For lrate-warmup, will go with the first {n_steps} steps.")
        self.lrate_warmup_steps = n_steps
        #
        num_splits = rconf.split_batch
        while not self._finished():
            with Timer(tag="Train-Iter", info="Iter %s" % self._tp.eidx, print_date=True) as et:
                # for batches
                for insts in train_stream:
                    # skip this batch
                    if Random.random_bool(rconf.skip_batch, task="train"):
                        continue
                    # train on batch, return a dictionary
                    # possibly split batch to save memory
                    self._fb_batch(insts)
                    self._tp.uidx += 1
                    # get the effective lrate
                    act_lrate = self.lrate.value / num_splits       # compensate for batch-split
                    if self._tp.uidx < self.lrate_warmup_steps:
                        act_lrate *= (self._tp.uidx / self.lrate_warmup_steps)
                    #
                    self._run_update(act_lrate)
                    # report on training process
                    if rconf.flag_verbose and (self._tp.uidx-last_report_uidx)>=rconf.report_freq:
                        self._run_train_report()
                        last_report_uidx = self._tp.uidx
                    # time for validating
                    if (self._tp.uidx-last_dev_uidx)>=rconf.valid_freq:
                        self._validate(dev_streams)
                        last_dev_uidx = self._tp.uidx
                        last_report_uidx = self._tp.uidx
                        if self._finished():
                            break
                if self._finished():
                    break
                # validate at the end of epoch?
                if rconf.validate_epoch:
                    self._validate(dev_streams)
                    last_dev_uidx = self._tp.uidx
                    last_report_uidx = self._tp.uidx
                self._tp.eidx += 1
        utils.zlog("zzzzzfinal: After training, the best point is: %s." % (str(self._tp.best_info())))

    # save & load
    def save(self, base_name):
        JsonRW.save_to_file(self._tp, base_name+".pr.json")
        self.model.save(base_name)
        utils.zlog("Save TrainRunner to <%s*>." % (base_name,), func="io")

    def load(self, base_name, load_process):
        if load_process:
            JsonRW.load_from_file(self._tp, base_name+".pr.json")
        self.model.load(base_name)
        utils.zlog("Load TrainRunner from <%s*> (load-pr=%s)." % (base_name, load_process), func="io")

    # to be implemented
    # return one train result
    def _run_fb(self, insts):
        raise NotImplementedError()

    # return None
    def _run_update(self, lrate: float):
        raise NotImplementedError()

    # print and return train summary
    def _run_train_report(self) -> RecordResult:
        raise NotImplementedError()

    # run and return dev results
    def _run_validate(self, dev_streams) -> RecordResult:
        raise NotImplementedError()

#
# scheduled values
#

class SVConf(Conf):
    def __init__(self):
        self.init_val = 0.          # should be set properly
        # how to schedule the value
        self.which_idx = "eidx"     # count steps on which: aidx, eidx, iidx, uidx
        self.mode = "none"                  # none, linear, exp, isigm
        #
        self.start_bias = 0
        self.scale = 1.0
        self.min_val = Constants.REAL_PRAC_MIN
        self.k = 1.0

class ScheduledValue(object):
    def __init__(self, name, sv_conf):
        self.name = name
        self.sv_conf = sv_conf
        #
        self.init_val = sv_conf.init_val
        self.cur_val = self.init_val
        #
        mode = sv_conf.mode
        k = sv_conf.k
        if mode == "linear":
            self._ff = lambda idx: 1.0-k*idx
        elif mode == "exp":
            self._ff = lambda idx: math.pow(k, idx)
        elif mode == "isigm":
            self._ff = lambda idx: k/(k+math.exp(idx/k))
        elif mode == "div":
            self._ff = lambda idx: 1/(1.+k*idx)
        elif mode == "none":
            self._ff = lambda idx: 1.0
        else:
            raise NotImplementedError(mode)

    @property
    def value(self):
        return self.cur_val

    def transform_idx(self, idx):
        x = max(0, idx-self.sv_conf.start_bias)
        return x/self.sv_conf.scale

    def __repr__(self):
        return "SV-%s=%s" % (self.name, self.cur_val)

    # adjust at checkpoint
    def adjust_at_ckp(self, sname, cur_idxes):
        the_idx = cur_idxes[self.sv_conf.which_idx]
        #
        new_idx = self.transform_idx(the_idx)
        old_val = self.cur_val
        self.cur_val = max(self.sv_conf.min_val, self.init_val * self._ff(new_idx))
        if self.cur_val != old_val:
            utils.zlog("Change scheduled value %s at %s: %s => %s." % (self.name, sname, old_val, self.cur_val))
        else:
            utils.zlog("Keep scheduled value %s at %s as %s." % (self.name, sname, self.cur_val))
