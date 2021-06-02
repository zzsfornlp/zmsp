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
        return float(self.score)

    def __repr__(self):
        return "Result(%s): %s" % (str(self.score), str(self.results))

# records stats of training-fb and dev, coupled with TrainingRunner
class TrainingProgressRecorder(object):
    def __init__(self, patience, anneal_restarts_cap, bad_start_eidx, bad_start_uidx):
        # hyper-parameters
        self.patience = patience
        self.anneal_restarts_cap = anneal_restarts_cap
        self.bad_start_eidx = bad_start_eidx
        self.bad_start_uidx = bad_start_uidx
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
        # all best
        self.best_dev_record = RecordResult({})
        self.best_point = -1
        # deal with anneal with all bests
        self.bad_counter = 0
        self.bad_points = []
        self.anneal_restarts_done = 0
        self.anneal_restarts_points = []
        # only save best
        self.save_best_dev_record = RecordResult({})
        self.save_best_point = -1

    #
    def current_suffix(self):
        # 4 digits should be enough
        sname = ".c%0.3d-e%d-i%dK-u%d" % (len(self.chp_names), self.eidx, int(self.iidx/1000.), self.uidx)
        return sname

    def current_idxes(self):
        return {"aidx": self.anneal_restarts_done, "eidx": self.eidx, "iidx": self.iidx, "uidx": self.uidx, "cidx": len(self.chp_names)}

    def info_best(self):
        if self.best_point < 0:
            return [-1, "None", "-inf"]
        else:
            return [self.best_point, self.chp_names[self.best_point], str(self.best_dev_record)]

    def info_save_best(self):
        if self.save_best_point < 0:
            return [-1, "None", "-inf"]
        else:
            return [self.save_best_point, self.chp_names[self.save_best_point], str(self.save_best_dev_record)]

    # called after one dev, log dev result and restart train-record
    def checkpoint(self, train_result: RecordResult, dev_result: RecordResult, use_save_best: bool):
        # log down
        sname = self.current_suffix()
        self.chp_names.append(sname)
        self.train_records.append(train_result)
        self.dev_records.append(dev_result)
        # anneal is for all the points
        if_best = if_save_best = if_anneal = False
        if float(dev_result) > float(self.best_dev_record):
            self.bad_counter = 0
            self.best_dev_record = dev_result
            self.best_point = len(self.chp_names)-1
            if_best = True
        elif self.eidx < self.bad_start_eidx or self.uidx < self.bad_start_uidx:
            # not starting bad counter
            pass
        else:
            cur_info = [sname, self.best_point, self.chp_names[self.best_point], str(self.best_dev_record)]
            self.bad_counter += 1
            # [cur-name, best-score-idx, best-score-name, best-score]
            self.bad_points.append(cur_info)
            utils.zlog("Bad++, now bad/anneal is %s/%s." % (self.bad_counter, self.anneal_restarts_done), func="warn")
            if self.bad_counter >= self.patience:
                self.bad_counter = 0
                self.anneal_restarts_points.append(cur_info)
                # there can be chances of continuing when restarts enough (min training settings)
                self.anneal_restarts_done += 1
                utils.zlog("Anneal++, now %s." % (self.anneal_restarts_done,), func="warn")
                if_anneal = True
                if self.anneal_restarts_done >= self.anneal_restarts_cap:
                    utils.zlog("Sorry, Early Stop !!", func="warn")
                    self.estop = True
        # save is only for certain points
        if use_save_best:
            if float(dev_result) > float(self.save_best_dev_record):
                self.save_best_dev_record = dev_result
                self.save_best_point = len(self.chp_names) - 1
                if_save_best = True
        return if_best, if_save_best, if_anneal
#

class RConf(Conf):
    def __init__(self):
        #
        self.skip_batch = 0.            # rate of randomly skipping each batch
        # todo(+N): the effect of split_batch can influence lr in more advanced optimizer? (since we are div lr)
        self.split_batch = 1            # num of batch splits, default 1 means no splitting
        # (now div loss) self.grad_div_for_sb = 0        # divide grad for split-batch mode rather than div lrate
        self.flag_debug = False
        self.flag_verbose = True
        # mostly based on uidx: number of updates
        self.report_freq = 1000
        self.valid_freq = Constants.INT_PRAC_MAX * 100  # this may be enough...
        self.validate_epoch = True
        self.validate_first = False
        self.min_epochs = 0
        self.min_save_epochs = 0
        self.max_epochs = 100
        self.min_updates = 0
        self.min_save_updates = 0
        self.max_updates = 10000000
        self.save_freq = 10000000  # save every ?? check point
        # bad counter start (start bad counter after these)
        self.bad_start_eidx = 0
        self.bad_start_uidx = 0
        # annealing
        self.anneal_times = 100
        self.patience = 3
        self.anneal_restore = False  # restore previous best model when annealing
        # write model
        self.model_overwrite = True
        self.model_name = "zmodel"
        self.suffix_curr = ".curr"
        self.suffix_best = ".best"
        #
        # lrate schedule
        self.lrate = SVConf().init_from_kwargs(val=0.001, which_idx="aidx", mode="exp", m=0.75, min_val=0.00001)
        self.lrate_warmup = 0       # linear increasing lrate as warmup for how many steps (minus values means epoch)
        self.lrate_anneal_alpha = 0.  # similar to ATT-is-ALL-you-Need, sth like -0.5 (after warmup, step^anneal)

    def do_validate(self):
        # min_* must be >= min_save_*
        self.min_epochs = max(self.min_epochs, self.min_save_epochs)
        self.min_updates = max(self.min_updates, self.min_save_updates)

# optimizer conf
class OptimConf(Conf):
    def __init__(self):
        self.optim = "adam"
        self.sgd_momentum = 0.85  # for "sgd"
        self.adam_betas = [0.9, 0.9]  # for "adam"
        self.adam_eps = 1e-4  # for "adam"
        self.adadelta_rho = 0.9  # for "adadelta"
        self.grad_clip = 5.0
        self.no_step_lrate0 = True  # no step when lrate<=0., even no momentum accumulating
        self.weight_decay = 0.

# common practice for training
class TrainingRunner(object):
    def __init__(self, rconf, model, batch_size_f=None):
        self.model = model
        #
        self._tp = TrainingProgressRecorder(rconf.patience, rconf.anneal_times, rconf.bad_start_eidx, rconf.bad_start_uidx)
        self.train_recorder = StatRecorder(True)
        self.rconf = rconf
        #
        self.batch_size_f = batch_size_f
        if self.batch_size_f is None:       # default is count number of instances
            self.batch_size_f = lambda x: len(x)
        #
        self.lrate = ScheduledValue("lrate", rconf.lrate)
        self.lrate_warmup_steps = 0         # set at the start of run()
        self._scheduled_values = [self.lrate]

    @property
    def scheduled_values(self):
        return self._scheduled_values + self.model.get_scheduled_values()

    def add_scheduled_values(self, sv_list):
        for one in sv_list:
            utils.zlog("Adding scheduled value %s in training." % (one,))
            self._scheduled_values.append(one)

    def _adjust_scheduled_values(self):
        # schedule values
        ss = self.current_name()
        cur_idxes = self._tp.current_idxes()
        for one_sv in self.scheduled_values:
            if one_sv.changeable:
                one_sv.adjust_at_ckp(ss, cur_idxes)

    # =====
    # ending criteria

    # already reach any of the right point of training (max-epoch/update or max-restart)
    def _reach_right_end(self):
        return self._tp.estop or self._tp.eidx >= self.rconf.max_epochs \
               or self._tp.uidx >= self.rconf.max_updates

    # already reach all of the left point of training (min-epoch/update)
    def _reach_left_end(self):
        return self._tp.eidx >= self.rconf.min_epochs and self._tp.uidx >= self.rconf.min_updates

    # reach all of the min save conditions
    def _reach_save_end(self):
        return self._tp.eidx >= self.rconf.min_save_epochs and self._tp.uidx >= self.rconf.min_save_updates

    # only finish when reaching both ends
    def _finished(self):
        return self._reach_left_end() and self._reach_right_end()

    # =====

    def current_name(self):
        return self._tp.current_suffix()

    # training for one batch
    def _fb_batch(self, insts):
        num_splits = self.rconf.split_batch
        loss_factor = 1. / num_splits
        splitted_insts = Helper.split_list(insts, num_splits)
        with self.train_recorder.go():
            for one_insts in splitted_insts:
                res = self._run_fb(one_insts, loss_factor)
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
        cur_c_idx = self._tp.current_idxes().get('cidx', 0)
        with Timer(tag="valid", info="Valid %s" % ss, print_date=True):
            # validate
            if len(dev_streams) == 0:  # simply use train if there are no dev
                utils.zlog("Use training results for dev since there are no dev set provided!", func="warn")
                dev_result = train_result
            else:
                dev_result = self._run_validate(dev_streams)
            # record
            cur_use_save_best = self._reach_save_end()
            if_best, if_save_best, if_anneal = self._tp.checkpoint(train_result, dev_result, cur_use_save_best)
            # checkpoint - save curr & best
            self.save(rconf.model_name+rconf.suffix_curr)
            if not rconf.model_overwrite:
                self.save(rconf.model_name+ss)
            if if_best:
                self.save(rconf.model_name+rconf.suffix_best)
                utils.zlog("Curr is best: " + str(self._tp.info_best()), func="result")
            else:
                utils.zlog("Curr not best, the best is " + str(self._tp.info_best()), func="result")
                if if_save_best:
                    # todo(+2): here overwrite the previous best point, will this small mismatch damage reloading?
                    utils.zlog("But Curr is save_best, overwrite the best point!")
                    self.save(rconf.model_name + rconf.suffix_best)
            if cur_c_idx > 0 and cur_c_idx % rconf.save_freq == 0:
                utils.zlog("Save at whole check point: " + ss)
                self.save(rconf.model_name + ss)
            if if_anneal and self.rconf.anneal_restore:
                utils.zlog("Restore from previous best model!!")
                self.load(rconf.model_name+rconf.suffix_best, False)
        utils.zlog("")

    def run(self, train_stream, dev_streams):
        rconf = self.rconf
        last_report_uidx, last_dev_uidx = 0, 0
        if rconf.validate_first:
            self._validate(dev_streams)
        # =====
        # for lrate warmup and annealing
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
        max_lrate = self.lrate.value
        # final_lrate = lrate * anneal_factor * (step)^lrate_anneal_alpha
        # final_lrate(n_steps) = max_lrate
        lrate_anneal_alpha = rconf.lrate_anneal_alpha
        if n_steps > 0:
            anneal_factor = 1. / (n_steps**lrate_anneal_alpha)
        else:
            anneal_factor = 1.
        self.lrate_warmup_steps = n_steps
        utils.zlog(f"For lrate-warmup, will go with the first {n_steps} steps up to {max_lrate}, "
                   f"then anneal with lrate*{anneal_factor}*step^{lrate_anneal_alpha}")
        # =====
        while not self._finished():
            # todo(note): epoch start from 1!!
            self._tp.eidx += 1
            with Timer(tag="Train-Iter", info="Iter %s" % self._tp.eidx, print_date=True) as et:
                self._adjust_scheduled_values()  # adjust at the start of each epoch
                act_lrate = 0.
                # for batches
                for insts in train_stream:
                    # skip this batch
                    if Random.random_bool(rconf.skip_batch):
                        continue
                    # train on batch, return a dictionary
                    # possibly split batch to save memory
                    self._fb_batch(insts)
                    self._tp.uidx += 1
                    # get the effective lrate
                    act_lrate = self.lrate.value
                    if self._tp.uidx < self.lrate_warmup_steps:
                        act_lrate *= (self._tp.uidx / self.lrate_warmup_steps)
                    else:
                        act_lrate *= anneal_factor * (self._tp.uidx**lrate_anneal_alpha)
                    #
                    self._run_update(act_lrate, 1.)
                    # report on training process
                    if rconf.flag_verbose and (self._tp.uidx-last_report_uidx)>=rconf.report_freq:
                        utils.zlog(f"Current act_lrate is {act_lrate}.")
                        self._run_train_report()
                        last_report_uidx = self._tp.uidx
                    # time for validating
                    if (self._tp.uidx-last_dev_uidx)>=rconf.valid_freq:
                        self._validate(dev_streams)
                        last_dev_uidx = self._tp.uidx
                        last_report_uidx = self._tp.uidx
                        # todo(+N): do we need to adjust sv at a finer grained?
                        self._adjust_scheduled_values()  # adjust after uidx validation
                        if self._finished():
                            break
                # validate at the end of epoch?
                utils.zlog(f"End of epoch: Current act_lrate is {act_lrate}.")
                if rconf.validate_epoch:
                    self._validate(dev_streams)
                    last_dev_uidx = self._tp.uidx
                    last_report_uidx = self._tp.uidx
            utils.zlog("")
        utils.zlog("zzzzzfinal: After training, the best point is: %s." % (str(self._tp.info_save_best())))

    # save & load
    def save(self, base_name):
        JsonRW.to_file(self._tp, base_name+".pr.json")
        self.model.save(base_name)
        utils.zlog("Save TrainRunner to <%s*>." % (base_name,), func="io")

    def load(self, base_name, load_process):
        if load_process:
            JsonRW.from_file(self._tp, base_name+".pr.json")
        self.model.load(base_name)
        utils.zlog("Load TrainRunner from <%s*> (load-pr=%s)." % (base_name, load_process), func="io")

    # to be implemented
    # return one train result
    def _run_fb(self, insts, loss_factor: float):
        raise NotImplementedError()

    # return None
    def _run_update(self, lrate: float, grad_factor: float):
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
        # self.init_val = 0.          # should be set properly (deprecated because of confusing name)
        self.val = 0.  # basic value
        # how to schedule the value
        self.which_idx = "eidx"     # count steps on which: aidx, eidx, iidx, uidx
        self.mode = "none"                  # none, linear, exp, isigm
        #
        self.start_bias = 0
        self.scale = 1.0
        self.min_val = Constants.REAL_PRAC_MIN
        self.max_val = Constants.REAL_PRAC_MAX
        self.b = 0.
        self.k = 1.0
        self.m = 1.0  # specific one

class ScheduledValue(object):
    def __init__(self, name, sv_conf, special_ff=None):
        self.name = name
        self.sv_conf = sv_conf
        self.val = sv_conf.val
        self.cur_val = None
        #
        mode = sv_conf.mode
        k = sv_conf.k
        b = sv_conf.b
        m = sv_conf.m
        #
        if special_ff is not None:
            assert mode == "none", "Confusing setting for schedule function!"
        #
        self.changeable = True
        if mode == "linear":
            self._ff = lambda idx: b+k*m*idx
        elif mode == "poly":
            self._ff = lambda idx: b+k*(idx**m)
        elif mode == "exp":
            self._ff = lambda idx: b+k*(math.pow(m, idx))
        elif mode == "isigm":
            self._ff = lambda idx: b+k*(m/(m+math.exp(idx/m)))
        elif mode == "div":
            self._ff = lambda idx: b+k*(1/(1.+m*idx))
        elif mode == "none":
            if special_ff is not None:
                self._ff = lambda idx: b+k*special_ff(idx)  # self-defined schedule: lambda idx: return ...
            else:
                self._ff = lambda idx: 1.0
                self.changeable = False  # no need to add as scheduled value
        else:
            raise NotImplementedError(mode)
        # init setting
        self._set(0)
        utils.zlog(f"Init scheduled value {self.name} as {self.cur_val} (changeable={self.changeable}).")

    @property
    def value(self):
        return self.cur_val

    def __float__(self):
        return float(self.cur_val)

    def __int__(self):
        return int(self.cur_val)

    def transform_idx(self, idx):
        x = max(0, idx-self.sv_conf.start_bias)
        return x/self.sv_conf.scale

    def __repr__(self):
        return "SV-%s=%s" % (self.name, self.cur_val)

    def _set(self, the_idx):
        new_idx = self.transform_idx(the_idx)
        old_val = self.cur_val
        # todo(note): self.val as the basis, multiplied by the factor
        self.cur_val = max(self.sv_conf.min_val, self.val * self._ff(new_idx))
        self.cur_val = min(self.sv_conf.max_val, self.cur_val)
        return old_val, self.cur_val

    # adjust at checkpoint
    def adjust_at_ckp(self, sname, cur_idxes):
        the_idx = cur_idxes[self.sv_conf.which_idx]
        old_val, new_val = self._set(the_idx)
        if self.cur_val != old_val:
            utils.zlog("Change scheduled value %s at %s: %s => %s." % (self.name, sname, old_val, self.cur_val))
        else:
            utils.zlog("Keep scheduled value %s at %s as %s." % (self.name, sname, self.cur_val))
