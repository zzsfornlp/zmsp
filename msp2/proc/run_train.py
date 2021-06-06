#

# for the training process

from typing import List, Iterable, Callable
from collections import OrderedDict
from msp2.utils import Conf, Constants, StatRecorder, zlog, default_json_serializer, zwarn, Timer, Random, OtherHelper, DictHelper
from msp2.data.stream import Streamer
from .help import SVConf, ScheduledValue, TrainingProgressRecord, ResultRecord
from .model import ZModel
from .run_test import TestingRunner

# basic confs for RunTraining
class TRConf(Conf):
    def __init__(self):
        # --
        # about training batches
        self.skip_batch = 0.  # rate of randomly skipping each batch
        # todo(+N): the effect of split_batch can influence lr in more advanced optimizer? (since we are div lr)
        # self.split_batch = 1  # num of batch splits, default 1 means no splitting
        self.accu_batch = 1  # how many fb count as one update (same effect as split_batch!!)
        # --
        # some flags
        self.flag_debug = False
        self.flag_verbose = True
        # --
        # max/min counts and freq values
        # report
        self.report_ufreq = Constants.INT_PRAC_MAX ** 2  # report every this uidx
        # valid
        self.valid_ufreq = Constants.INT_PRAC_MAX ** 2  # do valid every this udix
        self.valid_epoch = True  # do valid at the end of each epoch
        self.valid_first = False  # valid once at the very start
        self.valid_start_eidx = 0  # start valid only >= these
        self.valid_start_uidx = 0
        # store all insts at start?
        self.store_all_insts = False  # whether store all insts at start?
        # run
        self.min_eidx = 0  # min epochs (before early stop)
        self.max_eidx = Constants.INT_PRAC_MAX ** 2  # max epochs
        self.min_uidx = 0  # min updated (before early stop)
        self.max_uidx = Constants.INT_PRAC_MAX ** 2  # max update
        # record best point
        self.record_best_cidx = 0  # starting from where to record best
        # extra save
        self.save_start_cidx = Constants.INT_PRAC_MAX  # save special ones starting from what cidx
        self.save_cfreq = Constants.INT_PRAC_MAX  # save (specific name model) every cidx
        # bad counter start (start bad counter after these)
        self.bad_start_eidx = 0
        self.bad_start_uidx = 0
        # annealing
        self.anneal_times = 100
        self.anneal_patience = 5
        self.anneal_restore = False  # restore previous best model when annealing
        # write model
        self.model_prefix = "zmodel"
        self.model_suffix_curr = ".curr"
        self.model_suffix_best = ".best"
        # lrate schedule
        self.lrate = SVConf().direct_update(val=0.001, which_idx="aidx", mode="exp", m=0.75, min_val=0.000001)
        self.lrate_warmup_eidx = 0  # linear increasing lrate as warmup for how many eidx
        self.lrate_warmup_uidx = 0  # linear increasing lrate as warmup for how many uidx
        self.lrate_decrease_alpha = 0.  # as the one in Transformer, sth like -0.5 (after warmup, uidx**alpha)

class TrainingRunner:
    def __init__(self, conf: TRConf, model: ZModel, train_stream: Streamer,
                 train_batch_f: Callable, train_discard_batch_f: Callable, dev_runners: List[TestingRunner], **kwargs):
        self.conf = conf
        self.kwargs = kwargs
        # --
        self.model = model
        self.train_stream = train_stream
        self.train_batch_f = train_batch_f if train_batch_f is not None else lambda x: 1  # inst -> int
        self.train_discard_batch_f = train_discard_batch_f if train_discard_batch_f is not None else lambda x: False  # List[inst] -> bool
        self.dev_runners = dev_runners
        # some records
        self.tp = TrainingProgressRecord()
        self.train_recorder = StatRecorder(timing=True)
        # -- special!!
        # store all insts for future usage (do not use if input is large)
        self.stored_all_insts = None
        if conf.store_all_insts:
            _all_insts = []
            for _one_insts in self.train_stream:
                _all_insts.extend(_one_insts)
            self.stored_all_insts = _all_insts
        # --
        # scheduled values
        self.lrate = ScheduledValue("lrate", conf.lrate)
        self.lrate_warmup_steps = self._determine_warmup_steps(conf.lrate_warmup_eidx, conf.lrate_warmup_uidx, train_stream)
        if self.lrate_warmup_steps > 0:
            self.lrate_warmup_factor = 1. / (self.lrate_warmup_steps**conf.lrate_decrease_alpha)
            zlog(f"For lrate-warmup, first {self.lrate_warmup_steps} steps up to {self.lrate.value}, "
                 f"then decrease with lrate*{self.lrate_warmup_factor}*step^{conf.lrate_decrease_alpha}", func="plain")
        else:
            self.lrate_warmup_factor = 1.
        self.scheduled_values = OrderedDict([("_lrate", self.lrate)])
        DictHelper.update_dict(self.scheduled_values, model.get_scheduled_values())  # values to be scheduled as training goes

    def _determine_warmup_steps(self, eidx: int, uidx: int, stream: Streamer):
        # set warmup steps if using eidx
        if eidx > 0:
            _epoch_step_size = len(list(stream))
            _warmup_steps = _epoch_step_size * eidx
            zlog(f"Determine warmup steps: {eidx} x {_epoch_step_size} = {_warmup_steps}", func="plain")
        else:
            _warmup_steps = 0
        _warmup_steps = max(_warmup_steps, uidx)
        return _warmup_steps

    def current_name(self):
        return self.tp.current_suffix()

    def add_scheduled_value(self, v: ScheduledValue, key_prefix=''):
        key = key_prefix + v.name
        assert key not in self.scheduled_values
        self.scheduled_values[key] = v

    def adjust_scheduled_values(self):
        # adjust schedule values
        ss = self.current_name()
        for one_name, one_sv in self.scheduled_values.items():
            if one_sv.changeable:
                one_sv.adjust_at_ckp(ss, self.tp, extra_info=one_name)

    # -----
    # saving and loading related
    def save_progress(self, file: str):
        default_json_serializer.to_file(self.tp, file)
        zlog(f"Save training progress to {file}", func="io")

    def load_progress(self, file: str, forward_stream=False):
        old_uidx = self.tp.uidx
        d = default_json_serializer.from_file(file)
        self.tp.from_json(d)
        if forward_stream:
            if old_uidx > self.tp.uidx:
                zwarn(f"Cannot go to the past: {old_uidx} -> {self.tp.uidx}, skip this!")
            else:
                _s = self.train_stream
                for _ in range(self.tp.uidx - old_uidx):
                    _, _eos = _s.next_and_check()
                    if _eos:  # restart and get one
                        _s.restart()
                        _s.next()
                zlog(f"Forward to the future: {old_uidx} -> {self.tp.uidx}!", func="io")
        zlog(f"Load training progress from {file}", func="io")
        self.adjust_scheduled_values()  # also adjust values!

    def save(self, prefix: str):
        self.save_progress(prefix+".tp.json")
        self.model.save(prefix+".m")

    def load(self, prefix: str, load_progress=False, forward_stream=False, load_strict=None):
        if prefix.endswith(".m"):
            prefix = prefix[:-2]
        if load_progress:
            self.load_progress(prefix+".tp.json", forward_stream)
        self.model.load(prefix+".m", strict=load_strict)

    def get_train_stream(self):
        return self.train_stream

    # =====
    # run until the end of training
    def run(self):
        conf = self.conf
        last_report_uidx, last_dev_uidx = 0, 0
        # --
        if conf.valid_first:  # valid before training
            self.validate()
        # --
        _lrate_warmup_factor, _lrate_warmup_steps = self.lrate_warmup_factor, self.lrate_warmup_steps
        _skip_batch = conf.skip_batch
        _gen0 = Random.get_generator("train")
        _gen = Random.stream(_gen0.random_sample)
        # --
        _accu_checker = 0
        _accu_batch = conf.accu_batch
        # --
        # start before loop
        self.adjust_scheduled_values()
        # loop
        act_lrate = None
        while True:  # loop over and over
            _train_stream = self.get_train_stream()  # we may change train_stream!!
            # --
            if _train_stream.is_inactive():  # check to avoid restart after load_progress
                _train_stream.restart()
            insts, _eos = _train_stream.next_and_check()
            if _eos:  # end of epoch
                zlog(f"End of epoch at {self.tp.current_suffix(False)}: Current act_lrate is {act_lrate}.",
                     func="plain", timed=True)
                if conf.valid_epoch:
                    last_dev_uidx = self.tp.uidx
                    self.validate()
                    # todo(+N): do we need to adjust sv at a finer grained?
                    self.adjust_scheduled_values()  # adjust after validation
                if self._finished():
                    break
                self.tp.update_eidx(1)
                continue
            # skip batch?
            if _skip_batch>0 and next(_gen)<_skip_batch:
                continue
            if self.train_discard_batch_f(insts):
                continue  # discard this batch due to some specific reasons (like noevt)
            # run fb (possibly split batch)
            self.fb_batch(insts, 1./_accu_batch)
            self.tp.update_iidx(len(insts))
            # ==
            # only update for certain accu fb runs
            _accu_checker += 1
            if _accu_checker % _accu_batch == 0:
                self.tp.update_uidx(1)
                cur_uidx = self.tp.uidx
                # get the effective lrate and update
                act_lrate = float(self.lrate.value)  # start with the lrate.value
                if cur_uidx < _lrate_warmup_steps:  # linear increase
                    act_lrate *= (cur_uidx / _lrate_warmup_steps)
                else:  # decrease
                    act_lrate *= _lrate_warmup_factor * (cur_uidx**conf.lrate_decrease_alpha)
                self._run_update(act_lrate, 1.)
                # --
                # report on training process
                if conf.flag_verbose and (cur_uidx-last_report_uidx)>=conf.report_ufreq:
                    zlog(f"Report at {self.tp.current_suffix(False)}: Current act_lrate is {act_lrate}.",
                         func="plain", timed=True)
                    self._run_train_report()
                    last_report_uidx = cur_uidx
                # valid?
                if (cur_uidx-last_dev_uidx)>=conf.valid_ufreq:
                    last_dev_uidx = self.tp.uidx
                    self.validate()
                    # todo(+N): do we need to adjust sv at a finer grained?
                    self.adjust_scheduled_values()  # adjust after validation
                    if self._finished():
                        break
            # =====
        zlog(f"Finish training because of: {self._reach_ends()}", func="plain")
        zlog(f"zzzzzfinal: After training, the best point is: {self.tp.info_best()}.", func="report")

    # only finish when reaching any of the endings
    def _reach_ends(self):
        conf = self.conf
        cur_eidx, cur_uidx, cur_aidx = self.tp.eidx, self.tp.uidx, self.tp.aidx
        return cur_eidx>=conf.max_eidx, cur_uidx>=conf.max_uidx, cur_aidx>=conf.anneal_times

    def _finished(self):
        conf = self.conf
        cur_eidx, cur_uidx, cur_aidx = self.tp.eidx, self.tp.uidx, self.tp.aidx
        return (cur_eidx>=conf.min_eidx) and (cur_uidx>=conf.min_uidx) and any(self._reach_ends())

    # training for one batch
    def fb_batch(self, insts: List, loss_factor: float):
        with self.train_recorder.go():
            res = self._run_fb(insts, loss_factor)
            self.train_recorder.record(res)
        # --

    # do validation and record checkpoints
    def validate(self):
        conf = self.conf
        # report & reset training stat
        if self.tp.uidx > 0:
            train_result = self._run_train_report()  # first report training stat
            self.train_recorder.reset()     # reset training stat
        else:   # for validate_first
            train_result = None
        # dev
        ss, cur_cidx = self.current_name(), self.tp.cidx
        zlog("", func="plain")  # empty line
        with Timer(info=f"Valid {ss}", print_date=True), self.model.ema_wrap_dev():
            # no validation if specified
            if (self.tp.eidx < conf.valid_start_eidx) or (self.tp.uidx < conf.valid_start_uidx):
                zlog("No validation since not the time yet!\n", func="plain")
                return
            # validate
            if len(self.dev_runners) == 0:  # simply use train if there are no dev
                zlog("Use training results for dev since there are no dev set provided!", func="warn")
                dev_result = train_result
            else:
                dev_result = self._run_validate(self.dev_runners)
            # record
            cur_no_bad = (self.tp.eidx<conf.bad_start_eidx) or (self.tp.uidx<conf.bad_start_uidx)
            cur_record_best = (self.tp.cidx >= conf.record_best_cidx)
            if_overall_best, if_best, if_anneal = self.tp.update_checkpoint(train_result, dev_result, cur_no_bad, cur_record_best, conf.anneal_patience)
            # save curr & best
            self.save(conf.model_prefix + conf.model_suffix_curr)
            if if_overall_best:
                zlog("Curr is overall best " + str(self.tp.info_overall_best()), func="result")
            else:
                zlog("Curr not overall best, the overall best is " + str(self.tp.info_overall_best()), func="result")
            if if_best:
                self.save(conf.model_prefix + conf.model_suffix_best)
                zlog("Curr is best: " + str(self.tp.info_best()), func="result")
            else:
                zlog("Curr not best, the best is " + str(self.tp.info_best()), func="result")
            if cur_cidx >= conf.save_start_cidx and cur_cidx % conf.save_cfreq == 0:
                self.save(conf.model_prefix + ss)  # speical save
            if if_anneal and conf.anneal_restore:
                zlog("Restore from previous best model!!", func="plain")
                self.load(conf.model_prefix + conf.model_suffix_best, False)
        zlog("", func="plain")  # empty line

    # =====
    # template methods which can be overridden

    # return one train result
    def _run_fb(self, insts: List, loss_factor: float):
        res = self.model.loss_on_batch(insts, loss_factor=loss_factor)
        return res

    # return None
    def _run_update(self, lrate: float, grad_factor: float):
        self.model.update(lrate, grad_factor)

    # print and return train summary
    def _run_train_report(self) -> ResultRecord:
        x = self.train_recorder.summary()
        zlog(f"Train-Info: {OtherHelper.printd_str(x, sep=' ')}")
        return ResultRecord(results=x, description=None)

    # run and return dev results
    def _run_validate(self, dev_runners: List[TestingRunner]) -> ResultRecord:
        if len(dev_runners) == 0: return ResultRecord.get_nil()
        all_records: List[ResultRecord] = [r.run() for r in dev_runners]
        # note: use devs[0] as the criterion, assuming that is the dev itself!!
        r = ResultRecord(results={f"res{ii}": v.results for ii,v in enumerate(all_records)},
                         description={f"res{ii}": str(v) for ii,v in enumerate(all_records)},
                         score=all_records[0].score)
        return r
