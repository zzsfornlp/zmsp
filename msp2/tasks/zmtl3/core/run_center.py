#

__all__ = [
    "RunCenterConf", "RunCenter", "ResultAggregator",
]

from collections import OrderedDict
import os
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from msp2.utils import zlog, Conf, Constants, DictHelper, StatRecorder, zwarn, default_json_serializer, OtherHelper, \
    Timer, Logger, ZObject
from msp2.proc import ResultRecord, TrainingProgressRecord, SVConf, ScheduledValue
from msp2.nn import BK
from .task_center import TaskCenter
from .data_center import DataCenter

# --
class CheckpointManager:
    def __init__(self, save_bestn: int):
        self.n = save_bestn
        self.bestn_states = []  # List[(name, score, state_dict)]
        # --

    def __repr__(self):
        return f"{[z[:2] for z in self.bestn_states]}"

    def add(self, name, score, model):
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

    def average_model(self, prefix: str, suffix: str):
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
            avg_d[k] = torch.stack(vs, 0).mean(0)  # simply average!
        torch.save(avg_d, prefix + suffix)
        zlog(f"Try to average models: {self} -> {prefix}{suffix}")

# --
class RunCenterConf(Conf):
    def __init__(self):
        # save and load model
        self.model_save_prefix = "zmodel"
        self.model_save_suffix_curr = ".curr"
        self.model_save_suffix_best = ".best"
        self.model_save_dels = []  # deleted mods when saving
        self.train_preload_model = ""  # preload model name before training?
        self.model_load_name = "zmodel.best.m"  # (test time) load name
        self.debug_ddp_save_all = False  # debug for ddp to save all models
        self.debug_print_step = False
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
        # -- best N checkpoints
        self.save_bestn = 0  # save bestn checkpoints (if >0)
        self.model_save_suffix_bestn = ".bestn"  # finally average bestn checkpoints
        self.lastn_as_bestn = False  # make lastn as bestn (typically when auto-eval is not good)
        # --
        # lrate schedule
        self.lrate = SVConf().direct_update(val=0.00002, which_idx="uidx", mode="none", m=0.75, min_val=0.000001)
        self.loss_scale = 0.
        self.optim = 'adam'
        # --
        self.train_do_iter = False
        self.test_do_iter = False  # for testing mode
        self.iter_test_models = []  # what models to load?
        self.iter_test_conditions = []  # eval these at the training-end testing!
        # --

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
        self.optimizer = None
        self.lrate = ScheduledValue("lrate", conf.lrate)
        self.scheduled_values = OrderedDict([("_lrate", self.lrate)])  # add all scheduled values
        DictHelper.update_dict(self.scheduled_values, model.get_scheduled_values())
        if d_center is not None:
            DictHelper.update_dict(self.scheduled_values, d_center.get_scheduled_values())
        # --
        self.chp_manager = CheckpointManager(conf.save_bestn)
        # ==
        # for ddp
        self.ddp_world_size = BK.ddp_world_size()
        self.ddp_rank = BK.ddp_rank()
        self.is_main_process = BK.is_main_process()  # handle the savings!
        if self.ddp_world_size > 1:
            assert conf.accu_batch == 1, "accu_batch and ddp may conflict!!"
        # --
        self.amp = None
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

    def save(self, prefix: str):
        del_mods = self.conf.model_save_dels
        if self.conf.debug_ddp_save_all:
            self.model.save(prefix+f".m{self.ddp_rank}", del_mods=del_mods)
        elif self.is_main_process:  # note: do saving only with main process
            self.save_progress(prefix+".tp.json")
            self.model.save(prefix+".m", del_mods=del_mods)
        # --

    def load(self, prefix: str, load_progress=False, forward_stream=False):
        # --
        # parse prefix name
        cut_mods, del_mods = [], []
        if '###' in prefix:
            prefix, specs = prefix.rsplit("###", 1)
            for spec in specs.split(","):
                if spec[0] == 'C':
                    cut_mods.append(spec[1:])
                elif spec[0] == 'D':
                    del_mods.append(spec[1:])
                else:
                    raise NotImplementedError()
            zlog(f"Parse prefix: {prefix}###{specs} -> {cut_mods} {del_mods}")
        # --
        if prefix.endswith(".m"):
            prefix = prefix[:-2]
        if load_progress:
            self.load_progress(prefix+".tp.json", forward_stream)
        self.model.load(prefix+".m", cut_mods=cut_mods, del_mods=del_mods)
        # --

    # go
    # training for one batch
    def fb_batch(self, ibatch, loss_factor: float):
        _dps = self.conf.debug_print_step
        with self.train_recorder.go():
            self.model.train()  # note: remember to make it train!
            if _dps: zlog("Before forward")
            # --
            loss, res = self.model(ibatch, do_loss=True)
            if loss_factor != 1.:
                loss = loss * loss_factor
            # --
            if _dps: zlog("After forward")
            if loss.requires_grad:  # note: if requires_grad
                # --
                if self.amp is not None:
                    if _dps: zlog(f"Before scale_loss: {loss}")
                    with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        if _dps: zlog(f"Before loss_backward: {scaled_loss}")
                        scaled_loss.backward()
                        if _dps: zlog("After loss_backward")
                else:
                    loss.backward()
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
    def run_train_report(self) -> ResultRecord:
        x = self.train_recorder.summary()
        zlog(f"Train-Info: {OtherHelper.printd_str(x, sep=' ')}")
        # also report uidx_counter/iidx_counter
        zlog(f"UidxCounter: {self.tp.uidx_counter}")
        zlog(f"IidxCounter: {self.tp.iidx_counter}")
        return ResultRecord(results=x, description=None)

    # --
    # main runs

    def get_lr_scheduler(self):
        def lr_lambda(current_step: int):
            return self.lrate.value
        return LambdaLR(self.optimizer, lr_lambda)

    def do_train(self):
        t_center, d_center = self.t_center, self.d_center
        conf = self.conf
        last_dev_uidx = 0
        # =====
        # prepare training (DDP & optim)
        nn_conf = BK.get_global_conf()
        # optim
        _last_lrate = self.lrate.value
        _params = [p for n,p in self.model.named_parameters()]
        if nn_conf.amp_opt_level:
            try:
                from apex import amp
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
            assert conf.optim == 'adam'
            optimizer = FusedAdam(_params, betas=(0.9, 0.98), lr=_last_lrate, eps=1e-08)
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=nn_conf.amp_opt_level)
            self.amp = amp
        else:
            optim_type = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}[conf.optim]
            optimizer = optim_type(_params, betas=(0.9, 0.98), lr=_last_lrate, eps=1e-08)
        # ddp
        if BK.use_ddp():
            self.model = BK.wrap_ddp_model(self.model)
        # --
        self.optimizer = optimizer
        # scheduler = self.get_lr_scheduler()
        # =====
        # --
        if conf.valid_first:  # valid before training
            self.do_dev()
        # --
        _accu_batch = conf.accu_batch
        # --
        # to setup, note: the model should be eval mode but this is mainly for training purpose
        self.model.eval()
        gr = ZObject(d_center=self.d_center, training=True)
        self.model.apply(lambda x: (x.setup(gr=gr) if hasattr(x, 'setup') else None))
        self.adjust_scheduled_values()  # once before train
        # --
        train_dataset_yielder = d_center.yield_dataset()
        dataset_yielders = {}  # id(dataset) -> yielder
        while not self.train_finished():  # loop over and over
            # sample batch
            cur_train_dataset = next(train_dataset_yielder)  # select dataset
            cur_dname = cur_train_dataset.name
            if id(cur_train_dataset) not in dataset_yielders:
                dataset_yielders[id(cur_train_dataset)] = cur_train_dataset.yield_batches()
            cur_yielder = dataset_yielders[id(cur_train_dataset)]
            # fb for accu_batch/ddp steps (use the same dataset/yielder)
            fb_res = None
            for _i0 in range(self.ddp_world_size):
                for _ in range(_accu_batch):
                    cur_ibatch = next(cur_yielder)
                    if _i0 == self.ddp_rank:  # only for current rank!!
                        cur_dname = cur_ibatch.dataset.name
                        fb_res = self.fb_batch(cur_ibatch, 1./_accu_batch)
                        self.tp.update_iidx(len(cur_ibatch), cur_dname)
            # step/update
            self.tp.update_uidx(1, cur_dname)
            cur_uidx = self.tp.uidx
            # --
            # if cur_uidx == 1 and torch.cuda.is_available():
            #     # emptying the CUDA cache after the first step can reduce the chance of OOM (from fairseq)
            #     torch.cuda.empty_cache()
            # --
            if conf.debug_print_step:
                zlog(f"Step {cur_uidx}[-1/{_accu_batch}]: {fb_res}", timed=True)
            with self.train_recorder.go('update'):  # also record this!
                if _last_lrate != self.lrate.value:  # if we change larte
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = _last_lrate
                    _last_lrate = self.lrate.value
                self.model.update_regs(True, _last_lrate)  # before update
                optimizer.step()
                optimizer.zero_grad()
                self.model.update_regs(False, _last_lrate)  # after update
                # scheduler.step()  # Update learning rate schedule
            # valid?
            if (cur_uidx - last_dev_uidx) >= conf.valid_ufreq:
                last_dev_uidx = cur_uidx
                # --
                # to setup, note: the model should be eval mode but this is mainly for training purpose
                self.model.eval()
                gr = ZObject(d_center=self.d_center, training=True)
                self.model.apply(lambda x: (x.setup(gr=gr) if hasattr(x, 'setup') else None))
                # --
                self.do_dev()
                # todo(+N): do we need to adjust sv at a finer grained?
                self.adjust_scheduled_values()  # adjust after validation
        # --
        if self.is_main_process:
            # average bestn points
            self.chp_manager.average_model(conf.model_save_prefix + conf.model_save_suffix_bestn, ".m")
            if conf.train_do_iter:
                self.do_iter_test(include_dev=True)
        # --
        zlog(f"zzzzzfinal: After training, the best point is: {self.tp.info_best()}.", func="report")

    def do_iter_test(self, include_dev=False):
        conf = self.conf
        # --
        ii = 0
        for mname in conf.iter_test_models:
            zlog("# -- -- --")
            # try load model
            try:
                self.load(mname)
                zlog(f"Try load model of {mname} and use it to eval!")
            except:
                zwarn(f"Try load model file of {mname}, but failed, skip this!")
                continue
            # try run conditions
            _conditions = ["None"] if len(conf.iter_test_conditions) == 0 else conf.iter_test_conditions
            for _ee in _conditions:
                try:
                # if 1:
                #     _rr = eval(_ee)
                    _rr = exec(_ee)
                    zlog(f"# --\nRunning with condition {_ee}: {_rr}")
                    _res_dt = {}
                    if include_dev:
                        _res_dt.update(self.do_test(wset="dev", run_ii=str(ii)).results)
                    _res_dt.update(self.do_test(wset="test", run_ii=str(ii)).results)
                    _this_res = {"m": mname, "condition": _ee, "res": _res_dt}
                    zlog(f"RUN_ONE_RES = {_this_res}")
                except:
                    zwarn(f"Running with {_ee}, but failed, skip this!")
                    continue
                ii += 1
        # --

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
        # --
        zlog("", func="plain")  # empty line
        with Timer(info=f"Valid {ss}", print_date=True):
            # no validation if specified
            if self.tp.uidx < conf.valid_start_uidx:
                zlog("No validation since not the time yet!\n", func="plain")
                return
            # save curr before validate to see what might go wrong?
            if conf.model_save_suffix_curr:
                self.save(conf.model_save_prefix + conf.model_save_suffix_curr)
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
            if if_overall_best:
                zlog("Curr is overall best " + str(self.tp.info_overall_best()), func="result")
            else:
                zlog("Curr not overall best, the overall best is " + str(self.tp.info_overall_best()), func="result")
            if if_best:
                if conf.model_save_suffix_best:
                    self.save(conf.model_save_prefix + conf.model_save_suffix_best)
                zlog("Curr is best: " + str(self.tp.info_best()), func="result")
            else:
                zlog("Curr not best, the best is " + str(self.tp.info_best()), func="result")
            if cur_cidx >= conf.save_special_start_cidx and cur_cidx % conf.save_special_cfreq == 0:
                self.save(conf.model_save_prefix + ss)  # special save
            # --
            # bestn?
            if cur_record_best or conf.lastn_as_bestn:
                # if lastn_as_bestn, simply use idx as it gets larger!
                _cur_result = float(self.tp.cidx) if conf.lastn_as_bestn else float(dev_result)
                self.chp_manager.add(ss, _cur_result, self.model)
            # --
        zlog("", func="plain")  # empty line
        Logger.get_singleton_logger().flush_cached_logs()

    def do_test(self, wset="test", run_ii=""):
        t_center, d_center = self.t_center, self.d_center
        conf = self.conf
        # --
        self.model.eval()  # note: remember to make it eval!
        to_test_datasets = d_center.get_datasets(wset=wset)
        # --
        # prepare_test (simply setup)
        gr = ZObject(d_center=d_center, training=False)
        self.model.apply(lambda x: (x.setup(gr=gr) if hasattr(x, 'setup') else None))
        # --
        aggr = ResultAggregator()
        for one_ii, one_dataset in enumerate(to_test_datasets):
            with Timer(info=f"Test({one_ii+1}/{len(to_test_datasets)}): {one_dataset}", print_date=True):
                one_res = self.run_test_dataset(one_dataset, run_ii=run_ii)
                aggr.add(one_dataset.name, one_res, one_dataset.conf.group_eval_weight)
        ret = aggr.get_res()
        Logger.get_singleton_logger().flush_cached_logs()
        return ret

    # run test for decode
    def run_test_dataset(self, dataset, run_ii=""):
        conf = self.conf
        # --
        _test_with_loss = dataset.conf.test_with_loss
        test_recorder = StatRecorder(timing=True)
        if _test_with_loss > 0:
            _count = 0
            all_losses = []
            with BK.no_grad_env():
                for ibatch in dataset.yield_batches(loop=True):
                    with test_recorder.go():
                        one_loss, one_res = self.model(ibatch, do_loss=True)
                        all_losses.append(one_loss)
                        test_recorder.record(one_res)
                        _count += 1
                    if _count >= _test_with_loss:
                        break
            mean_negloss = - BK.stack(all_losses, 0).mean().item()
            x = test_recorder.summary()
            x['mean_negloss'] = mean_negloss
            zlog(f"Test-Info: {OtherHelper.printd_str(x, sep=' ')}")
            ret = ResultRecord(x, score=mean_negloss)
            return ret
        else:
            if dataset.conf.special_test:
                # special test, simply pass full dataset in
                with test_recorder.go():
                    self.model.apply(lambda x: (x.do_special_test(dataset) if hasattr(x, 'do_special_test') else None))
            else:
                _count = 0
                for ibatch in dataset.yield_batches(loop=False):
                    with test_recorder.go():
                        one_res = self.model(ibatch, do_pred=True)
                        test_recorder.record(one_res)
                        _count += len(ibatch)
                        if conf.debug_print_step:
                            zlog(f"Test Inst={_count}: {one_res}", timed=True)
        # --
        # write output
        if self.is_main_process:  # note: do saving only with main process
            dataset.write_insts(None)
        # --
        # eval
        x = test_recorder.summary()
        try:  # get inst/sec
            x['inst_sec'] = round(x['inst'] / x['_time'], 3)
        except:
            pass
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
        ret.results['info'] = x  # further add this in!
        return ret

# =====
# helpers

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
# b msp2/tasks/zmtl3/core/run_center:
