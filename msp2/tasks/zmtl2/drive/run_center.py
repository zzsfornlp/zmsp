#

__all__ = [
    "RunCenterConf", "RunCenter", "ResultAggregator",
]

from collections import OrderedDict
import numpy as np
from msp2.nn import BK
from msp2.utils import zlog, Conf, Constants, DictHelper, StatRecorder, zwarn, default_json_serializer, OtherHelper, Timer
from msp2.proc import ResultRecord, TrainingProgressRecord, SVConf, ScheduledValue
from .task_center import TaskCenter
from .data_center import DataCenter, ZDataset

# --
class RunCenterConf(Conf):
    def __init__(self):
        # save and load model
        self.model_save_prefix = "zmodel"
        self.model_save_suffix_curr = ".curr"
        self.model_save_suffix_best = ".best"
        self.train_preload_model = ""  # preload model name before training?
        self.model_load_name = "zmodel.best.m"  # (test time) load name
        # train
        self.accu_batch = 1  # how many fb count as one update
        self.min_uidx = 0  # min updated (before early stop)
        self.max_uidx = 1000 * 100  # max update
        # valid
        self.valid_ufreq = 1000  # do valid every this udix
        self.valid_first = False  # valid once at the very start
        self.valid_start_uidx = 0  # do valid >=this
        # record best & save cidx
        self.record_best_start_cidx = 0  # starting from where to record best
        self.save_special_start_cidx = Constants.INT_PRAC_MAX  # save special ones starting from what cidx
        self.save_special_cfreq = Constants.INT_PRAC_MAX  # save (specific name model) every cidx
        # lrate schedule
        self.lrate = SVConf().direct_update(val=0.001, which_idx="uidx", mode="none", m=0.75, min_val=0.000001)
        self.lrate_warmup_uidx = 0  # linear increasing lrate as warmup for how many uidx
        self.lrate_decrease_alpha = 0.  # as the one in Transformer, sth like -0.5 (after warmup, uidx**alpha)

class RunCenter:
    def __init__(self, conf: RunCenterConf, model, t_center: TaskCenter, d_center: DataCenter):
        self.conf = conf
        self.model = model
        self.t_center = t_center
        self.d_center = d_center
        # ==
        # for train
        self.tp = TrainingProgressRecord()
        self.train_recorder = StatRecorder(timing=True)
        # --
        self.lrate = ScheduledValue("lrate", conf.lrate)
        self.scheduled_values = OrderedDict([("_lrate", self.lrate)])  # add all scheduled values
        DictHelper.update_dict(self.scheduled_values, model.get_scheduled_values())
        DictHelper.update_dict(self.scheduled_values, d_center.get_scheduled_values())
        # --
        self.lrate_warmup_steps = conf.lrate_warmup_uidx
        if self.lrate_warmup_steps > 0:
            self.lrate_warmup_factor = 1. / (self.lrate_warmup_steps**conf.lrate_decrease_alpha)
            zlog(f"For lrate-warmup, first {self.lrate_warmup_steps} steps up to {self.lrate.value}, "
                 f"then decrease with lrate*{self.lrate_warmup_factor}*step^{conf.lrate_decrease_alpha}", func="plain")
        else:
            self.lrate_warmup_factor = 1.
        # ==
        # for ddp
        self.ddp_world_size = BK.ddp_world_size()
        self.ddp_rank = BK.ddp_rank()
        self.is_main_process = BK.is_main_process()  # handle the savings!
        if self.ddp_world_size > 1:
            assert conf.accu_batch == 1, "accu_batch and ddp may conflict!!"
        # --

    # --
    # helpers

    def current_name(self):
        return self.tp.current_suffix()

    def adjust_scheduled_values(self):
        # adjust schedule values
        ss = self.current_name()
        for one_name, one_sv in self.scheduled_values.items():
            if one_sv.changeable:
                one_sv.adjust_at_ckp(ss, self.tp, extra_info=one_name)

    # saving and loading related
    def save_progress(self, file: str):
        default_json_serializer.to_file(self.tp, file)
        zlog(f"Save training progress to {file}", func="io")

    def load_progress(self, file: str, forward_stream=False):
        d = default_json_serializer.from_file(file)
        self.tp.from_json(d)
        assert not forward_stream, "Error: 'forward_stream' not supported in this mode!!"
        zlog(f"Load training progress from {file}", func="io")
        self.adjust_scheduled_values()  # also adjust values!

    def save(self, prefix: str):
        if self.is_main_process:  # note: do saving only with main process
            self.save_progress(prefix+".tp.json")
            self.model.save(prefix+".m")

    def load(self, prefix: str, load_progress=False, forward_stream=False):
        if prefix.endswith(".m"):
            prefix = prefix[:-2]
        if load_progress:
            self.load_progress(prefix+".tp.json", forward_stream)
        self.model.load(prefix+".m")

    # go
    # training for one batch
    def fb_batch(self, ibatch, loss_factor: float):
        with self.train_recorder.go():
            res = self.model.loss_on_batch(ibatch, loss_factor)
            self.train_recorder.record(res)
        # --

    def train_finished(self):
        conf = self.conf
        return self.tp.uidx >= conf.max_uidx

    # print and return train summary
    def run_train_report(self) -> ResultRecord:
        x = self.train_recorder.summary()
        zlog(f"Train-Info: {OtherHelper.printd_str(x, sep=' ')}")
        # also report uidx_counter/iidx_counter
        zlog(f"UidxCounter: {self.tp.uidx_counter}")
        zlog(f"IidxCounter: {self.tp.iidx_counter}")
        return ResultRecord(results=x, description=None)

    # run test on one dataset
    def run_test_dataset(self, dataset: ZDataset):
        test_recorder = StatRecorder(timing=True)
        for ibatch in dataset.yield_batches(loop=False):
            with test_recorder.go():
                one_res = self.model.predict_on_batch(ibatch)
                test_recorder.record(one_res)
        # --
        # write output
        if self.is_main_process:  # note: do saving only with main process
            dataset.write_insts(None)  # use dataset's conf
        # --
        # eval
        x = test_recorder.summary()
        zlog(f"Test-Info: {OtherHelper.printd_str(x, sep=' ')}")
        aggr = ResultAggregator()
        for task in self.t_center.tasks.values():
            if task.name not in dataset.tasks:
                continue
            tn_res: ResultRecord = task.eval_insts(dataset.gold_insts, dataset.insts, quite=False)
            if tn_res is None:
                continue
            aggr.add(task.name, tn_res, task.conf.eval_weight)
        ret = aggr.get_res()
        return ret

    # --
    # main runs

    def do_train(self):
        model, t_center, d_center = self.model, self.t_center, self.d_center
        conf = self.conf
        last_dev_uidx = 0
        # --
        if conf.valid_first:  # valid before training
            self.do_dev()
        # --
        _lrate_warmup_factor, _lrate_warmup_steps = self.lrate_warmup_factor, self.lrate_warmup_steps
        _accu_batch = conf.accu_batch
        self.adjust_scheduled_values()  # once before train
        train_yielder = d_center.yield_train_yielder()
        while not self.train_finished():  # loop over and over
            # sample batch
            cur_yielder = next(train_yielder)
            # fb for accu_batch/ddp steps (use the same dataset/yielder)
            cur_dname = None
            for _i0 in range(self.ddp_world_size):
            # if 1:
                for _i1 in range(_accu_batch):
                    cur_ibatch = next(cur_yielder)
                    cur_dname = cur_ibatch.dataset.name
                    if _i0 == self.ddp_rank:  # only for current rank!!
                    # if 1:
                        self.fb_batch(cur_ibatch, 1./_accu_batch)
                        self.tp.update_iidx(len(cur_ibatch), cur_dname)
            # update
            self.tp.update_uidx(1, cur_dname)
            cur_uidx = self.tp.uidx
            # get the effective lrate and update
            act_lrate = float(self.lrate.value)  # start with the lrate.value
            if cur_uidx < _lrate_warmup_steps:  # linear increase
                act_lrate *= (cur_uidx / _lrate_warmup_steps)
            else:  # decrease
                act_lrate *= _lrate_warmup_factor * (cur_uidx**conf.lrate_decrease_alpha)
            with self.train_recorder.go('update'):  # also record this!
                self.model.update(act_lrate, 1.)
            # valid?
            if (cur_uidx - last_dev_uidx) >= conf.valid_ufreq:
                last_dev_uidx = cur_uidx
                self.do_dev()
                # todo(+N): do we need to adjust sv at a finer grained?
                self.adjust_scheduled_values()  # adjust after validation
        # --
        zlog(f"zzzzzfinal: After training, the best point is: {self.tp.info_best()}.", func="report")

    def do_dev(self):
        conf = self.conf
        # report & reset training stat
        if self.tp.uidx > 0:
            train_result = self.run_train_report()  # first report training stat
            self.train_recorder.reset()  # reset training stat
        else:  # for validate_first
            train_result = ResultRecord.get_nil()
        # dev
        ss, cur_cidx = self.current_name(), self.tp.cidx
        zlog("", func="plain")  # empty line
        with Timer(info=f"Valid {ss}", print_date=True):
            # no validation if specified
            if self.tp.uidx < conf.valid_start_uidx:
                zlog("No validation since not the time yet!\n", func="plain")
                return
            # validate
            if len(self.d_center.get_datasets(wset="dev"))==0:  # simply use train if there are no dev
                zlog("Use training results for dev since there are no dev set provided!", func="warn")
                dev_result = train_result
            else:
                dev_result = self.do_test("dev")
            # record
            cur_record_best = (self.tp.cidx >= conf.record_best_start_cidx)
            if_overall_best, if_best, if_anneal = self.tp.update_checkpoint(train_result, dev_result, record_best=cur_record_best)
            # save curr & best
            self.save(conf.model_save_prefix + conf.model_save_suffix_curr)
            if if_overall_best:
                zlog("Curr is overall best " + str(self.tp.info_overall_best()), func="result")
            else:
                zlog("Curr not overall best, the overall best is " + str(self.tp.info_overall_best()), func="result")
            if if_best:
                self.save(conf.model_save_prefix + conf.model_save_suffix_best)
                zlog("Curr is best: " + str(self.tp.info_best()), func="result")
            else:
                zlog("Curr not best, the best is " + str(self.tp.info_best()), func="result")
            if cur_cidx >= conf.save_special_start_cidx and cur_cidx % conf.save_special_cfreq == 0:
                self.save(conf.model_save_prefix + ss)  # speical save
        # --
        zlog("", func="plain")  # empty line

    def do_test(self, wset="test"):
        model, t_center, d_center = self.model, self.t_center, self.d_center
        conf = self.conf
        # --
        to_test_datasets = d_center.get_datasets(wset=wset)
        t_center.prepare_datasets(to_test_datasets)  # re-prepare!!
        aggr = ResultAggregator()
        for one_ii, one_dataset in enumerate(to_test_datasets):
            with Timer(info=f"Test({one_ii+1}/{len(to_test_datasets)}): {one_dataset}", print_date=True):
                one_res = self.run_test_dataset(one_dataset)
                aggr.add(one_dataset.name, one_res, one_dataset.conf.group_eval_weight)
        ret = aggr.get_res()
        return ret

# helper
class ResultAggregator:
    def __init__(self):
        self.all_numbers = []
        self.all_weights = []
        self.all_results = OrderedDict()

    def add(self, key: str, res: ResultRecord, weight: float):
        self.all_numbers.append(res.score)
        self.all_weights.append(weight)
        self.all_results[key] = res.results  # assign the dict!

    def get_res(self):
        if len(self.all_numbers) == 0:
            final_score = 0.
        else:  # weighted sum!!
            final_score = (np.asarray(self.all_numbers) * np.asarray(self.all_weights)).sum() / sum(self.all_weights)
            final_score = final_score.item()
        return ResultRecord(self.all_results, score=final_score)

# --
# b msp2/tasks/zmtl2/drive/run_center:129
