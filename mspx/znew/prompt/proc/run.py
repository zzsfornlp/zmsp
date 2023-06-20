#

# main running processes, such as training and testing

import time
import torch
from collections import OrderedDict
from tqdm.auto import tqdm
from typing import Dict, Union
from mspx.utils import zlog, zwarn, Conf, Configurable, StatRecorder, ZResult, ZHelper, Timer, Constants, Logger
from .run_helper import *

# --
class RunnerConf(Conf):
    def __init__(self):
        # optim
        self.optim = OptimizerConf()
        # train
        self.accu_batch = 1  # how many fb count as one update
        self.accu_ddp = False  # for parallel mode
        self.min_uidx = 0  # min updated (before early stop)
        self.max_uidx = 1000 * 10  # max update
        self.max_eidx = Constants.INT_PRAC_MAX  # num of epoch
        # valid
        self.valid_ufreq = 1000  # do valid every this udix
        self.valid_first = False  # valid once at the very start
        self.valid_start_uidx = 0  # do valid >=this
        self.pass_cidx_f = "False"  # whether pass train?
        # model save
        self.model_save_prefix = "zmodel"  # as those allowed by "parse_save_load_name"
        self.model_save_suffix_curr = ""
        self.model_save_suffix_best = ""
        # record best & save cidx
        self.record_best_start_cidx = 0  # starting from where to record best
        self.save_special_start_cidx = Constants.INT_PRAC_MAX  # save special ones starting from what cidx
        self.save_special_cfreq = Constants.INT_PRAC_MAX  # save (specific name model) every cidx
        # -- best N checkpoints
        self.model_bestn = 1  # save bestn checkpoints (if >0)
        # self.model_save_suffix_bestn = ".bestn"  # finally average bestn checkpoints
        self.model_save_suffix_bestn = ""  # finally average bestn checkpoints
        self.lastn_as_bestn = False  # make lastn as bestn (typically when auto-eval is not good)
        # model loading
        self.model_load_name = ''  # loading nema (for both trainning and testing)
        # lrate-scheduler
        self.warmup_uidx = 0  # warming up
        self.lr_scheduler_type = 'constant_with_warmup'
        self.lr_scheduler_final = 0.1  # final target lrate if linear decreasing
        # misc
        self.print_progress = True  # progress bar
        self.debug_print_step = False  # printing for debugging
        # testing
        self.test_with_dropout = False
        self.test_no_grad = False

class MyRunner(Configurable):
    def __init__(self, conf: RunnerConf, model, task, accelerator, extra_scheduled_values=None):
        super().__init__(conf)
        conf: RunnerConf = self.conf
        # --
        self.raw_model = model
        self.task = task
        self.accelerator = accelerator
        # parallel specifications
        self.par_world_size, self.par_rank, self.par_is_main = \
            accelerator.num_processes, accelerator.process_index, accelerator.is_main_process
        self.scheduled_values = OrderedDict()  # add all scheduled values
        ZHelper.update_dict(self.scheduled_values, model.get_scheduled_values())
        ZHelper.update_dict(self.scheduled_values, task.get_scheduled_values())
        if extra_scheduled_values:
            ZHelper.update_dict(self.scheduled_values, extra_scheduled_values)
        # --

    # --
    # the main training loop

    def do_train(self, train_dataloader, dev_dataloader):
        conf: RunnerConf = self.conf
        model, accelerator, task, scheduled_values = self.raw_model, self.accelerator, self.task, self.scheduled_values
        _par_rank, _par_world_size, _par_is_main = self.par_rank, self.par_world_size, self.par_is_main
        # --
        # optimizer
        optimizer = get_optim(conf.optim, model)
        # --
        # lrate scheduler
        from transformers import get_scheduler
        # note: extend max steps to 1/(1-final)
        _all_steps = int(conf.warmup_uidx + (conf.max_uidx - conf.warmup_uidx) / (1-conf.lr_scheduler_final))
        lr_scheduler = get_scheduler(name=conf.lr_scheduler_type, optimizer=optimizer,
                                     num_warmup_steps=conf.warmup_uidx, num_training_steps=_all_steps)
        # --
        # prepare things with accelerator
        # model, optimizer, train_dataloader, dev_dataloader, lr_scheduler = \
        #     accelerator.prepare(model, optimizer, train_dataloader, dev_dataloader, lr_scheduler)
        # note: no wrapping dataloader with accelarator!!
        model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
        # --
        zlog(f"=====> Start to run training par=[{_par_rank}/{_par_world_size}, is_main={_par_is_main}]")
        tp = TrainingProgressRecord()
        train_recorder = StatRecorder()
        data_iter = iter(train_dataloader)
        ckp_manager = CheckpointManager(conf.model_bestn)
        progress_bar = tqdm(range(conf.max_uidx), disable=not (conf.print_progress and _par_is_main))
        # valid first?
        dev_info = {'last_uidx': 0, 'last_time': time.time()}
        if conf.valid_first:  # valid before training
            self.do_dev(model, dev_dataloader, tp, train_recorder, ckp_manager, dev_info)
        self.adjust_scheduled_values(tp, scheduled_values, optimizer)  # once before train
        _pass_cidx_f = ZHelper.eval_ff(conf.pass_cidx_f, 'cidx')
        if _pass_cidx_f(tp.cidx-1):
            with Timer(info=f"PassTrain {tp.current_suffix()}", print_date=True):
                task.pass_train(train_dataloader)  # task-specific
        # --
        _accu_ddp = _par_world_size if conf.accu_ddp else 1
        while tp.uidx < conf.max_uidx:  # until the maximum steps
            fb_res, cur_dname = None, None
            for _i0 in range(_accu_ddp):
                for _ in range(conf.accu_batch):
                    with train_recorder.go('fetch'):  # also record this!
                        try:
                            cur_data = next(data_iter)
                        except StopIteration:
                            data_iter = iter(train_dataloader)  # restart!
                            tp.update_eidx(1)
                    cur_ibatch = cur_data['ibatch']  # note: special name!
                    cur_run_flag = (not conf.accu_ddp) or (_i0 == _par_rank)
                    if conf.debug_print_step:
                        _info = [f"{z.inst}[{z.inst.id}]" for z in cur_ibatch.items[:10]]
                        zlog(f"Current ibatch for {_par_rank}[run={cur_run_flag}] = {_info}")
                    if cur_run_flag:  # only for current rank!!
                        cur_dname = cur_ibatch.dataset.wset
                        fb_res = self.fb_batch(model, task, cur_data, train_recorder, accelerator, 1./conf.accu_batch)
                        tp.update_iidx(len(cur_ibatch), cur_dname)
            # step/update
            tp.update_uidx(1, cur_dname)
            cur_uidx = tp.uidx
            if conf.debug_print_step:
                zlog(f"Step {cur_uidx}[-1/{conf.accu_batch}]: {fb_res}", timed=True)
            with train_recorder.go('update'):  # also record this!
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            # valid
            if (cur_uidx - dev_info['last_uidx']) >= conf.valid_ufreq:
                self.do_dev(model, dev_dataloader, tp, train_recorder, ckp_manager, dev_info)
                self.adjust_scheduled_values(tp, scheduled_values, optimizer)  # adjust after validation
                if _pass_cidx_f(tp.cidx-1):
                    with Timer(info=f"PassTrain {tp.current_suffix()}", print_date=True):
                        task.pass_train(train_dataloader)  # task-specific
        # --
        # average bestn points
        dd = ckp_manager.average_model()
        if _par_is_main and dd is not None and conf.model_save_suffix_bestn:
            self.save_model(dd, conf.model_save_prefix, suffix=conf.model_save_suffix_bestn)
        # --
        info_best = tp.info_best()
        # zlog(f"zzzzzfinal: After training, the best point is: {info_best[-1].to_dict()}.", func="report")
        zlog(f"zzzzzdevfinal: {info_best[-1].to_dict(store_all_fields=True)}.")
        zlog(f"zzzzz-----: After training, the best point is: {info_best}.", func="report")
        # --
        # return best state_dict
        return dd

    def do_dev(self, model, dev_dataloader, tp, train_recorder, ckp_manager, dev_info):
        conf: RunnerConf = self.conf
        _par_is_main = self.par_is_main
        # --
        # report & reset training stat
        if tp.uidx > 0:
            train_result = self.run_train_report(train_recorder, tp)  # first report training stat
            train_recorder.reset()  # reset training stat
        else:  # for validate_first
            train_result = ZResult()
        # dev
        ss, cur_cidx = tp.current_suffix(), tp.cidx
        # --
        zlog("")  # empty line
        with Timer(info=f"Valid {ss}", print_date=True):
            # no validation if specified
            if tp.uidx < conf.valid_start_uidx:
                zlog("No validation since not the time yet!")
                return
            # save curr before validate to see what might go wrong?
            if conf.model_save_suffix_curr and _par_is_main:
                self.save_model(model, conf.model_save_prefix, suffix=conf.model_save_suffix_curr)
            # validate
            if dev_dataloader is None:  # simply use train if there are no dev
                zlog("Use training results for dev since there are no dev set provided!", func="warn")
                dev_result = train_result
            else:
                dev_result = self.do_test(model, dev_dataloader)
            # record
            cur_record_best = (tp.cidx >= conf.record_best_start_cidx)
            if_overall_best, if_best, if_anneal = tp.update_checkpoint(train_result, dev_result, record_best=cur_record_best)
            # save curr & best
            if if_overall_best:
                zlog("Curr is overall best " + str(tp.info_overall_best()), func="result")
            else:
                zlog("Curr not overall best, the overall best is " + str(tp.info_overall_best()), func="result")
            if if_best:
                if conf.model_save_suffix_best and _par_is_main:
                    self.save_model(model, conf.model_save_prefix, suffix=conf.model_save_suffix_best)
                zlog("Curr is best: " + str(tp.info_best()), func="result")
            else:
                zlog("Curr not best, the best is " + str(tp.info_best()), func="result")
            if cur_cidx >= conf.save_special_start_cidx and cur_cidx % conf.save_special_cfreq == 0 and _par_is_main:
                self.save_model(model, conf.model_save_prefix, suffix=ss)  # special save
            # --
            # bestn?
            _lastn_as_bestn = (conf.lastn_as_bestn or dev_dataloader is None)  # lastN if no dev!
            # if cur_record_best or _lastn_as_bestn:
            if 1:
                # if lastn_as_bestn, simply use idx as it gets larger!
                _cur_result = float(tp.cidx) if _lastn_as_bestn else float(dev_result)
                ckp_manager.add(ss, _cur_result, model)
            # --
        # --
        _stamp = time.time()
        zlog(f"END dev at {time.ctime()} ||| {_stamp-dev_info['last_time']:.2f} secs from last_dev.")
        dev_info.update({'last_uidx': tp.uidx, 'last_time': _stamp})
        zlog("")  # empty line
        Logger.get_singleton_logger().flush_cached_logs()

    def do_test(self, model, dataloader):
        conf: RunnerConf = self.conf
        task = self.task
        _par_is_main = self.par_is_main
        # --
        if conf.test_with_dropout:
            model.train()  # special mode!
        else:
            model.eval()  # note: remember to make it eval!
        # --
        evaler = task.new_evaler()
        test_recorder = StatRecorder()
        hit_datasets = {}  # id->Dataset
        if conf.print_progress:
            progress_bar = tqdm(range(len(list(dataloader))), disable=not (conf.print_progress and _par_is_main))
        else:
            progress_bar = None
        with Timer(info=f"Test", print_date=True):
            _count = 0
            for step, cur_data in enumerate(dataloader):
                cur_ibatch = cur_data['ibatch']
                with test_recorder.go('forward'):
                    with torch.no_grad():
                        outputs = model(task=task, do_test=True, **cur_data)
                    if progress_bar:
                        progress_bar.update(1)
                    test_recorder.record(outputs.info)
                    _count += len(cur_ibatch)
                    if conf.debug_print_step:
                        zlog(f"Test Inst={_count}: {outputs.info}", timed=True)
                    # record dataset
                    _dataset = cur_ibatch.dataset
                    if _dataset is not None and id(_dataset) not in hit_datasets:
                        hit_datasets[id(_dataset)] = _dataset
                # eval
                with test_recorder.go('pred'):
                    task.pred(cur_ibatch, outputs, evaler)
        # --
        if _par_is_main:  # try to write outputs
            for _dataset in hit_datasets.values():
                _dataset.write_insts()
        # --
        x = test_recorder.summary()
        try:  # get inst/sec
            x['inst_sec'] = round(x['inst'] / x['_time'], 3)
        except:
            pass
        zlog(f"Test-Info: {ZHelper.printd_str(x, sep=' ')}")
        ret = evaler.get_res()
        ret.info = x
        return ret

    # --
    # other helpers

    # forward/backward for one batch
    def fb_batch(self, model, task, data, train_recorder, accelerator, loss_factor: float):
        _dps = self.conf.debug_print_step
        with train_recorder.go('fb'):
            model.train()  # note: remember to make it train!
            if _dps: zlog("Before forward")
            outputs = model(task=task, do_loss=True, **data)
            loss = outputs.loss
            if loss_factor != 1.:
                loss = loss * loss_factor
            info = outputs.info
            if _dps: zlog(f"Before backward: {loss}")
            accelerator.backward(loss)
            if _dps: zlog("After backward")
            info['fb'] = 1
            train_recorder.record(info)
            return info
        # --

    def adjust_scheduled_values(self, tp, scheduled_values, optimizer=None):
        # adjust schedule values
        ss = tp.current_suffix()
        for one_name, one_sv in scheduled_values.items():
            if one_sv.changeable:
                one_sv.adjust_at_ckp(ss, tp, extra_info=one_name)
        # also check current lrate
        if optimizer is None:
            zwarn("Cannot check lrate: optimizer is None")
        else:
            lrates = [pg['lr'] for pg in optimizer.param_groups]
            zlog(f"Current lrates: {lrates}")
        # --

    # print and return train summary
    def run_train_report(self, train_recorder, tp):
        x = train_recorder.summary()
        zlog(f"Train-Info: {ZHelper.printd_str(x, sep=' ')}")
        # also report uidx_counter/iidx_counter
        zlog(f"UidxCounter: {tp.uidx_counter}")
        zlog(f"IidxCounter: {tp.iidx_counter}")
        return ZResult(x)

    # --
    # save & load

    @staticmethod
    def get_model_state_and_description(model):
        if isinstance(model, dict):
            state_dict = model  # directly passing a state_dict
            description = f'D[{len(state_dict)}]'
        else:
            state_dict = model.tosave_state_dict()
            description = str(model)
        description = f"{description}[{len(state_dict)}|{sum([z.numel() for z in state_dict.values()])}]"
        return state_dict, description

    @staticmethod
    def save_model(model, save_name: str, suffix='', quiet=False):
        path = save_name + suffix
        state_dict, description = MyRunner.get_model_state_and_description(model)
        torch.save(state_dict, path)
        if not quiet:
            zlog(f"Save {description} to {path}", func="io")
        # --

    @staticmethod
    def load_model(model, path: Union[Dict, str], quiet=False, strict=None, map_location=None):
        if isinstance(path, dict):
            state_dict = path
            path_desc = f"[{len(state_dict)}|{sum([z.numel() for z in state_dict.values()])}]"
        else:
            state_dict = torch.load(path, map_location=map_location)
            path_desc = path
        _, description = MyRunner.get_model_state_and_description(state_dict)
        if isinstance(model, dict):
            model.update(state_dict)  # simply update!
        else:
            model.from_state_dict(state_dict, strict=strict)
        if not quiet:
            zlog(f"Load {description} from {path_desc}", func="io")
        # --

# --
# b mspx/znew/prompt/proc/run:
