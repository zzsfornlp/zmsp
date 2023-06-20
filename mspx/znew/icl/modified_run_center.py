#

# a quick modified run_center!

#

__all__ = [
    "CheckpointManager", "ResultAggregator",
    "ZRunCenterConf", "ZRunCenter",
]

from collections import OrderedDict
import os
import time
import numpy as np
from mspx.utils import zlog, Conf, Constants, StatRecorder, zwarn, default_json_serializer, ZHelper, \
    Timer, Logger, ZObject, ZResult, WithWrapper
from mspx.nn import BK, OptimConf
from mspx.proc.run.helper import *

# --
# Helpers

class CheckpointManager:
    def __init__(self, save_bestn: int):
        self.n = save_bestn
        self.bestn_states = []  # List[(name, score, state_dict)]
        # --

    def __repr__(self):
        return f"{[z[:2] for z in self.bestn_states]}"

    def add(self, name: str, score: float, model):
        if self.n <= 0:
            return
        if len(self.bestn_states) < self.n or score > self.bestn_states[-1][1]:
            zlog(f"Add checkpoint {name}[{score}].")
            _s = OrderedDict([(k, v.to(BK.CPU_DEVICE).clone()) for k, v in model.state_dict().items()])
            self.bestn_states.append((name, score, _s))
            self.bestn_states.sort(key=(lambda x: x[1]), reverse=True)  # highest first
            self.bestn_states = self.bestn_states[:self.n]  # store only bestn!
        else:
            zlog(f"No adding worse checkpoint: {name}[{score}]")
        zlog(f"Current nbest checkpoints: {self}")

    def average_model(self):
        if len(self.bestn_states) <= 0:
            zlog(f"No states available, skip: {self}!")
            return
        ds = [z[2] for z in self.bestn_states]
        avg_d = OrderedDict()
        for d in ds:
            for k, v in d.items():
                if k not in avg_d:
                    avg_d[k] = [v]
                else:
                    avg_d[k].append(v)
        for k in list(avg_d.keys()):
            vs = avg_d[k]
            if len(vs) < len(ds):
                zwarn(f"When average_model, key is not full: {len(vs)}")
            avg_d[k] = BK.stack(vs, 0).mean(0)  # simply average!
        return avg_d

class ResultAggregator:
    def __init__(self):
        self.all_numbers = []
        self.all_weights = []
        self.all_results = OrderedDict()

    def add(self, key: str, res: ZResult, weight: float):
        self.all_numbers.append(float(res))
        self.all_weights.append(weight)
        self.all_results[key] = res

    def aggr(self):
        if len(self.all_numbers) == 0:
            final_score = 0.
        else:  # weighted sum!!
            final_score = (np.asarray(self.all_numbers) * np.asarray(self.all_weights)).sum() / sum(self.all_weights)
            final_score = final_score.item()
        return ZResult(self.all_results, res=final_score)

# --
# Runner

class ZRunCenterConf(Conf):
    def __init__(self):
        # save and load model
        self.model_save_prefix = "zmodel"  # as those allowed by "parse_save_load_name"
        self.model_save_suffix_curr = ".curr"
        self.model_save_suffix_best = ".best"
        self.train_preload_model = ""  # preload model name before training?
        self.model_load_name = "zmodel.best.m"  # (test time) load name
        self.debug_ddp_save_all = False  # debug for ddp to save all models
        self.debug_print_step = False
        # train
        self.accu_batch = 1  # how many fb count as one update
        self.accu_ddp = True  # unless ddp is handled in data-loader
        self.min_uidx = 0  # min updated (before early stop)
        self.max_uidx = 1000  # max update
        # valid
        self.valid_ufreq = 100  # do valid every this udix
        self.valid_first = False  # valid once at the very start
        self.valid_start_uidx = 0  # do valid >=this
        self.valid_with_prev = 0  # do eval for valid with K previous results (check by pred-stable)
        # st: special testing (like inf for self-training)
        self.st_trgs = []  # target datasets for st
        self.st_specs = [0, 0]  # starting-cidx/cfreq
        self.st_kwargs = {}  # kwargs for st
        # record best & save cidx
        self.record_best_start_cidx = 0  # starting from where to record best
        self.save_special_start_cidx = Constants.INT_PRAC_MAX  # save special ones starting from what cidx
        self.save_special_cfreq = Constants.INT_PRAC_MAX  # save (specific name model) every cidx
        # -- best N checkpoints
        self.save_bestn = 0  # save bestn checkpoints (if >0)
        self.model_save_suffix_bestn = ".bestn"  # finally average bestn checkpoints
        self.lastn_as_bestn = False  # make lastn as bestn (typically when auto-eval is not good)
        # --
        # lrate schedule
        self.optim = OptimConf()
        self.lrate = SVConf.direct_conf(val=0.00002, val_range=[0.1, 1.])
        self.lrate_warmup_uidx = 0  # linear increasing lrate as warmup for how many uidx
        self.lrate_decrease_alpha = 0.  # as the one in Transformer, sth like -0.5 (after warmup, uidx**alpha)
        # --
        self.train_do_iter = False
        self.test_do_iter = False  # for testing mode
        self.iter_test_models = []  # what models to load?
        self.iter_test_conditions = []  # eval these at the training-end testing!
        # --

class ZRunCenter:
    def __init__(self, conf: ZRunCenterConf, model):
        self.conf = conf
        # self.t_center = t_center
        # self.d_center = d_center
        # ==
        # for train
        self.tp = TrainingProgressRecord()
        self.train_recorder = StatRecorder()
        # --
        self.model = model  # first set to this!
        self.optimizer = None  # setup when starting training!
        self.lrate = ScheduledValue(conf.lrate, name='lrate')
        self.scheduled_values = OrderedDict([("_lrate", self.lrate)])  # add all scheduled values
        ZHelper.update_dict(self.scheduled_values, self.model.get_scheduled_values())
        # if d_center is not None:
        #     ZHelper.update_dict(self.scheduled_values, d_center.get_scheduled_values())
        self.ckp_manager = CheckpointManager(conf.save_bestn)
        self.last_dev_time = time.time()  # start the game
        # --
        # warm up
        self.lrate_warmup_steps = conf.lrate_warmup_uidx
        if self.lrate_warmup_steps > 0:
            self.lrate_warmup_factor = 1. / (self.lrate_warmup_steps ** conf.lrate_decrease_alpha)
            zlog(f"For lrate-warmup, first {self.lrate_warmup_steps} steps up to {self.lrate.value}, "
                 f"then decrease with lrate*{self.lrate_warmup_factor}*step^{conf.lrate_decrease_alpha}", func="plain")
        else:
            self.lrate_warmup_factor = 1.
        # --
        # ddp
        self.ddp_world_size = BK.ddp_world_size()
        self.ddp_rank = BK.ddp_rank()
        self.is_main_process = BK.is_main_process()  # handle the savings!
        # if self.ddp_world_size > 1:  # todo(+W): why?
        #     assert conf.accu_batch == 1, "accu_batch and ddp may conflict!!"
        # --
        # amp
        self.amp: BK.AmpManager = None
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
        # also check current lrate
        if self.optimizer is None:
            zwarn("Cannot check lrate: optimizer is None")
        else:
            lrates = [pg['lr'] for pg in self.optimizer.param_groups]
            zlog(f"Current lrates: {lrates}")
        # --

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

    def save(self, save_name: str, suffix='', mod=None):
        for ss, flag in zip([f".ddp{self.ddp_rank}", ""], [self.conf.debug_ddp_save_all, self.is_main_process]):
            if flag:
                path_json = BK.change_slname(save_name, add_suffix=f'{suffix}{ss}.tp.json', rm_suffixes=['.m'], rm_specs=True)
                self.save_progress(path_json)
                path_model = BK.change_slname(save_name, add_suffix=f'{suffix}{ss}.m', rm_suffixes=['.m'], rm_specs=False)
                if mod is not None:
                    BK.save_mod(mod, path_model, quite=False)
                else:
                    BK.save_mod(self.model, path_model, quite=False)
        # --

    def load(self, load_name: str, suffix='', mod=None, load_progress=False):
        if load_progress:
            path_json = BK.change_slname(load_name, add_suffix=f'{suffix}.tp.json', rm_suffixes=['.m'], rm_specs=True)
            self.load_progress(path_json)
        path_model = BK.change_slname(load_name, add_suffix=f'{suffix}.m', rm_suffixes=['.m'], rm_specs=False)
        if mod is None:
            mod = self.model
        BK.load_mod(mod, path_model, quite=False)
        # --

    # go
    # training for one batch
    def fb_batch(self, ibatch, loss_factor: float):
        _dps = self.conf.debug_print_step
        amp, model = self.amp, self.model
        with self.train_recorder.go():
            model.train()  # note: remember to make it train!
            if _dps: zlog("Before forward")
            # --
            if amp is not None:
                with amp.with_autocast():
                    loss, res = self.forward_model(model, ibatch)
            else:
                loss, res = self.forward_model(model, ibatch)
            if loss_factor != 1.:
                loss = loss * loss_factor
            # --
            if _dps: zlog("After forward")
            if BK.is_expr(loss) and loss.requires_grad:  # note: if requires_grad
                # --
                if _dps: zlog(f"Before loss_backward: {loss}")
                if amp is not None:
                    with amp.with_scale(loss, self.optimizer) as scaled_loss:
                        if _dps: zlog(f"After scale_loss: {scaled_loss}")
                        scaled_loss.backward()
                else:
                    loss.backward()
                if _dps: zlog("After loss_backward")
                # --
            else:
                assert not BK.use_ddp(), "Should not bypass backward in DDP mode!"
                res["fb0"] = 1
            # --
            if _dps: zlog("After backward")
            self.train_recorder.record(res)
            return res
        # --

    def train_finished(self):
        conf = self.conf
        return self.tp.uidx >= conf.max_uidx

    # print and return train summary
    def run_train_report(self):
        x = self.train_recorder.summary()
        zlog(f"Train-Info: {ZHelper.printd_str(x, sep=' ')}")
        # also report uidx_counter/iidx_counter
        zlog(f"UidxCounter: {self.tp.uidx_counter}")
        zlog(f"IidxCounter: {self.tp.iidx_counter}")
        return ZResult(x)

    # --
    # main runs

    def do_train(self, train_data, dev_data):
        # t_center, d_center = self.t_center, self.d_center
        conf = self.conf
        last_dev_uidx = 0
        # =====
        # prepare training (DDP & optim)
        model, optimizer = self.model, None
        # optim
        _last_lrate = self.lrate.value
        _params = [p for n,p in self.model.named_parameters()]
        optimizer = BK.get_optimizer(conf.optim, _params, _last_lrate)
        # amp
        if BK.use_amp():
            self.amp = BK.AmpManager()
            model, optimizer = self.amp.initialize(model, optimizer)
        # ddp
        if BK.use_ddp():
            model = BK.wrap_ddp_model(model)
        # reset things!
        self.model, self.optimizer = model, optimizer
        # =====
        # start training
        self.last_dev_time = time.time()  # start the game
        if conf.valid_first:  # valid before training
            self.do_dev(dev_data)
        # --
        _accu_batch = conf.accu_batch
        _accu_ddp = self.ddp_world_size if conf.accu_ddp else 1
        _lrate_warmup_factor, _lrate_warmup_steps = self.lrate_warmup_factor, self.lrate_warmup_steps
        self.adjust_scheduled_values()  # once before train
        # batch_yielder = d_center.yield_batches(each_time=_accu_ddp * _accu_batch)
        while not self.train_finished():  # loop over and over
            optimizer.zero_grad()
            # fb for accu_batch/ddp steps (use the same dataset/yielder)
            fb_res, cur_dname = None, None
            for _i0 in range(_accu_ddp):
                for _ in range(_accu_batch):
                    with self.train_recorder.go('fetch'):  # also record this!
                        # cur_ibatch = next(batch_yielder)
                        cur_ibatch = next(train_data)
                    cur_run_flag = (not conf.accu_ddp) or (_i0==self.ddp_rank)
                    if conf.debug_print_step:
                        _info = [f"{z.inst}[{z.inst.id}]" for z in cur_ibatch.items[:10]]
                        zlog(f"Current ibatch for {self.ddp_rank}[run={cur_run_flag}] = {_info}")
                    if cur_run_flag:  # only for current rank!!
                        if hasattr(cur_ibatch, 'dataset'):
                            cur_dname = cur_ibatch.dataset.name
                        else:
                            cur_dname = 'unk_data'
                        fb_res = self.fb_batch(cur_ibatch, 1./_accu_batch)
                        self.tp.update_iidx(len(cur_ibatch), cur_dname)
            # step/update
            self.tp.update_uidx(1, cur_dname)
            cur_uidx = self.tp.uidx
            # --
            if conf.debug_print_step:
                zlog(f"Step {cur_uidx}[-1/{_accu_batch}]: {fb_res}", timed=True)
            with self.train_recorder.go('update'):  # also record this!
                act_lrate = float(self.lrate.value)  # current lrate
                if cur_uidx < _lrate_warmup_steps:  # linear increase
                    act_lrate *= (cur_uidx / _lrate_warmup_steps)
                else:  # decrease
                    act_lrate *= _lrate_warmup_factor * (cur_uidx ** conf.lrate_decrease_alpha)
                if _last_lrate != act_lrate:  # if we change larte
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = act_lrate
                    _last_lrate = act_lrate
                if self.amp is not None:
                    self.amp.step(optimizer)
                    self.amp.update()
                else:
                    optimizer.step()
                # self.model.update_regs(_last_lrate)  # after update
            # valid?
            if (cur_uidx - last_dev_uidx) >= conf.valid_ufreq:
                last_dev_uidx = cur_uidx
                self.do_dev(dev_data)
                # todo(+N): do we need to adjust sv at a finer grained?
                self.adjust_scheduled_values()  # adjust after validation
        # --
        if self.is_main_process:
            # average bestn points
            dd = self.ckp_manager.average_model()
            if dd is not None:
                self.save(conf.model_save_prefix, suffix=conf.model_save_suffix_bestn, mod=dd)
        # --
        info_best = self.tp.info_best()
        zlog(f"zzzzzfinal: After training, the best point is: {info_best[-1].to_dict(store_all_fields=True)}.", func="report")
        zlog(f"zzzzz-----: After training, the best point is: {info_best}.", func="report")

    def do_dev(self, dev_data):
        conf = self.conf
        # report & reset training stat
        if self.tp.uidx > 0:
            train_result = self.run_train_report()  # first report training stat
            self.train_recorder.reset()  # reset training stat
        else:  # for validate_first
            train_result = ZResult()
        # dev
        ss, cur_cidx = self.current_name(), self.tp.cidx
        # --
        zlog("", func="plain")  # empty line
        with Timer(info=f"Valid {ss}", print_date=True):
            # no validation if specified
            if self.tp.uidx < conf.valid_start_uidx:
                zlog("No validation since not the time yet!\n", func="plain")
                return
            # save curr before validate to see what might go wrong?
            if conf.model_save_suffix_curr:
                self.save(conf.model_save_prefix, suffix=conf.model_save_suffix_curr)
            # validate
            if dev_data is None:  # simply use train if there are no dev
                zlog("Use training results for dev since there are no dev set provided!", func="warn")
                dev_result = train_result
            else:
                dev_result = self.do_test(dev_data)
            # record
            cur_record_best = (self.tp.cidx >= conf.record_best_start_cidx)
            if_overall_best, if_best, if_anneal = self.tp.update_checkpoint(train_result, dev_result, record_best=cur_record_best)
            # save curr & best
            if if_overall_best:
                zlog("Curr is overall best " + str(self.tp.info_overall_best()), func="result")
            else:
                zlog("Curr not overall best, the overall best is " + str(self.tp.info_overall_best()), func="result")
            if if_best:
                if conf.model_save_suffix_best:
                    self.save(conf.model_save_prefix, suffix=conf.model_save_suffix_best)
                zlog("Curr is best: " + str(self.tp.info_best()), func="result")
            else:
                zlog("Curr not best, the best is " + str(self.tp.info_best()), func="result")
            if cur_cidx >= conf.save_special_start_cidx and cur_cidx % conf.save_special_cfreq == 0:
                self.save(conf.model_save_prefix, suffix=ss)  # special save
            # --
            # bestn?
            if cur_record_best or conf.lastn_as_bestn:
                # if lastn_as_bestn, simply use idx as it gets larger!
                _cur_result = float(self.tp.cidx) if conf.lastn_as_bestn else float(dev_result)
                self.ckp_manager.add(ss, _cur_result, self.model)
            # --
        # --
        # check last-dev-time
        _stamp = time.time()
        zlog(f"END dev at {time.ctime()} ||| {_stamp-self.last_dev_time:.2f} secs from last_dev.")
        self.last_dev_time = _stamp
        zlog("", func="plain")  # empty line
        Logger.get_singleton_logger().flush_cached_logs()

    def do_test(self, test_data):
        conf = self.conf
        self.model.eval()  # note: remember to make it eval!
        # --
        raise NotImplementedError()

    # note: this can be changed in sub-class
    def forward_model(self, model, ibatch):
        # return model(ibatch, do_loss=True)
        raise NotImplementedError()

# --
# b mspx/znew/icl/modified_run_center
