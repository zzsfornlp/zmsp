#

# task-specific helpers
# (make them fine-grained enough!)
# note: mostly use CMDs to make things separable (at the cost of more R&W)

__all__ = [
    "ALTaskConf", "ALTaskHelper"
]

import os
import numpy as np
from collections import Counter
from itertools import chain
from mspx.utils import Conf, Configurable, ConfEntryChoices, zwarn, zlog, ZHelper, Random, mkdir_p, zglob1, ZObject
from mspx.proc.run import DataSamplerConf, DataSampler
from mspx.data.inst import yield_sents
from .selector import *

class ALTaskConf(Conf):
    def __init__(self):
        super().__init__()
        from .helper_repr import QueryReprHelperConf
        # --
        # sample dev
        self.sampler = DataSamplerConf().direct_update(sample_s=4000, size_f='sum(len(z) for z in x.sents)')
        # external data for ST?
        self.external_st_data = ""
        # query
        self.query_partial = True  # partial or not: tok/sent
        self.query_i0_partial = True  # partial for iter0 (if in overall partial mode!)
        self.query_use_unc = True  # query by uncertainty (marginal probs)
        self.query_emb_bname = ""  # calculate embs
        self.query_emb_use_model = False  # use last-model for emb?
        self.query_use_t0 = False  # use last t0 rather than tz for querying!
        self.selector = SelectorConf()
        self.qann_with_past = True  # also include past-ann in querying files
        self.qann_with_hit = False  # only include hit(query) docs in querying files
        self.qann_store_topk = 5  # store topK idx and score (for easier analysis)
        # query: special mode of selv2
        self.query_selv2 = False  # use v2: selecting
        self.query_repr = QueryReprHelperConf()
        self.selv2_only_empty = True  # only selecting empty sents?
        self.selv2_ratio_sentwise = None  # deprecated!
        self.selv2_sc0 = 1  # at least how many queries per sent?
        self.selv2_sc1 = 1
        self.selv2_ratios0 = ['1.']  # selecting rates: low:high or high
        self.selv2_ratios1 = ['1.']
        self.selv2_threshs0 = [-1.]  # selecting threshs (>this)
        self.selv2_threshs1 = [-1.]
        # budget
        self.budget = [4000.]  # tok(or percentage) budget per iter (by default, continuing the last one ...)
        self.budget_group = 0  # sent budget constraint
        # training
        self.train_st = 0  # number of self-train iters, 0 means no self-train
        self.train_preload_last = False  # preload last model?
        self.train_no_ann_i0 = False  # indicator for no ann data (useful for transfer setting!)
        # specifications; eg: conf_sbase:bert_name:xlm-roberta-base ...
        self.specs_vocdir = ""  # base specs of common dir!
        self.specs_base = ""  # base specs for all
        self.specs_train = ""  # model training confs (for the final one)
        self.specs_trainE = ""  # extra model training confs (for the final one)
        self.specs_train00 = ""  # further extra ones for iter0.t0 (useful for transfer setting!)
        self.specs_train0p = ""  # further extra ones for iter0.t+ (useful for transfer setting!)
        self.specs_trainST = ""  # extra basic model training confs (for ST)
        self.specs_trainST1 = ""  # extra basic model training confs (for ST-t1)
        self.specs_trainST2 = ""  # extra basic model training confs (for ST-t2)
        self.specs_infQ = ""  # model inf confs for Query
        self.specs_extra_infQD = ""  # extra for query dev!
        self.specs_infT = ""  # model inf confs for Training(self-train)
        self.specs_qemb = ""  # emb for query
        self.inf_with_caliQ = ''  # do calibration for inf-Q!
        self.inf_with_caliT = ''  # do calibration for inf-T!
        # --

    def curr_is_partial(self, curr_iter: int):
        assert isinstance(curr_iter, int) and curr_iter >= 0
        if curr_iter == 0:
            return self.query_partial and self.query_i0_partial
        else:
            return self.query_partial

    @staticmethod
    def get_entries():  # note: for different tasks
        from .zext import ALZextConf
        from .zdpar import ALZdparConf
        from .zrel import ALZrelConf
        return ConfEntryChoices({
            'zext': ALZextConf(), 'zdpar': ALZdparConf(), 'zrel': ALZrelConf(),
        })

@ALTaskConf.conf_rd()
class ALTaskHelper(Configurable):
    def __init__(self, conf: ALTaskConf, **kwargs):
        super().__init__(conf, **kwargs)
        self.sampler: DataSampler = conf.sampler.make_node()
        self.curr_iter = -1
        self.name: str = None
        self._auto_ths = [None, None]  # currently at most two!
        # --

    @property
    def tasks_out(self): return [self.name]  # all output tasks, by default only one
    @property
    def tasks_all(self): return [self.name]  # all tasks, possibly including enc

    @property
    def curr_selv2_ths(self):
        conf: ALTaskConf = self.conf
        _ret = []
        for ii, vv0 in enumerate([self._get_curr_val(conf.selv2_threshs0), self._get_curr_val(conf.selv2_threshs1)]):
            vv1 = self._auto_ths[ii]
            vv = vv0 if vv1 is None else vv1
            _ret.append(vv)
        return _ret

    # setup common running specs
    def _init_specs(self):
        conf: ALTaskConf = self.conf
        # --
        # running specs (with some default ones)
        if 'vocab_load_dir:' in conf.specs_base:  # note: make it compatible with previous ones!
            _idx = conf.specs_base.index('vocab_load_dir:')
            _dir = conf.specs_base[_idx:].split()[0].split(":", 1)[1]
            conf.specs_vocdir = _dir
        _specs_base = f"vocab_load_dir:{conf.specs_vocdir} {conf.specs_base}"  # specific for voc!
        _n = self.name
        _specs_train = f"{_specs_base} {conf.specs_train} {conf.specs_trainE}"
        _specs_trainST = f"{_specs_train} {conf.specs_trainST}"  # base ST ones
        self.specs_trains = [_specs_train, f"{_specs_trainST} {conf.specs_trainST1}", f"{_specs_trainST} {conf.specs_trainST2}"]  # usually we only need to specify this much ...
        _specs_infQ, _specs_infT, _specs_infT0 = f"{_specs_base}", f"{_specs_base}", f"{_specs_base}"
        for _n in self.tasks_out:
            _specs_infQ += f" {_n}.pred_use_partial:1 {_n}.pred_do_strg:1 {_n}.pred_do_dec:0"
            _specs_infT += f" {_n}.pred_use_partial:1 {_n}.pred_do_strg:1 {_n}.pred_do_dec:0"
        self.specs_infQ = f"{_specs_infQ} {conf.specs_infQ}"
        self.specs_infT = f"{_specs_infT} {conf.specs_infT}"
        self.specs_qemb = f"{conf.specs_qemb}"  # note: no base for qemb!
        # --

    def __repr__(self):
        return self.__class__.__name__

    # helpers
    def run_cmd(self, cmd: str, ret_output=False):
        from mspx.utils import system
        if ret_output:
            zlog(f"RUN (with ret_output): {cmd}")
            return system(cmd, pp=False, ass=True, popen=True)
        else:
            return system(cmd, pp=True, ass=True, popen=False)

    # base args for test
    def base_args_test(self, model_file: str, input_data: str, output_data: str, load_conf=True):
        if model_file is None:
            model_file = ""
        if model_file and load_conf:  # load confs and vocab
            model_dir = os.path.dirname(model_file) + "/"
            ret = f"{model_dir}/_conf vocab_load_dir:{model_dir}"
        else:
            ret = ""
        ret += f" model_load_name:{model_file} fs:test log_stderr:0 device:0 test0.group_files:{input_data} test0.output_file:{output_data} test0.input_dir: test0.output_dir: d_input_dir:"
        return ret

    def _do_inf(self, model_file: str, input_data: str, output_data: str, extra_args: str,
                inf_with_cali: str, dev_in: str, dev_out='', dev_extra_args=''):
        base_args = self.base_args_test(model_file, input_data, output_data)
        CMD = f"python3 -m {self.main_entry} {base_args} d_tasks:{','.join(self.tasks_all)} {extra_args}"
        # --
        # inf with the unlabeled pool
        if inf_with_cali:
            for one_out_name in self.tasks_out:
                CMD2 = CMD + f" {one_out_name}.pred_use_partial:0 {one_out_name}.pred_for_cali:{inf_with_cali} test0.group_files:{dev_in} test0.output_file:{output_data}.devcali"
                self.run_cmd(CMD2)
                CMD3 = f"python3 -m mspx.scripts.tools.calibrate input_path:{output_data}.devcali log_stderr:1 key_cali:{one_out_name}_cali"
                output = self.run_cmd(CMD3, ret_output=True)
                best_temp = [float(z.split()[-1]) for z in output.split("\n") if z.strip().startswith("BEST_TEMP =")][-1]
                zlog(f"Get best temp for cali of: {best_temp}")
                CMD = CMD + f" {one_out_name}.{'pred_m_tau' if inf_with_cali=='logm' else 'inf_tau'}:{best_temp}"
        self.run_cmd(CMD)
        # --
        # inf with the dev for later usage
        if dev_in and dev_out:
            CMD_DEV = CMD
            for one_out_name in self.tasks_out:
                CMD_DEV += f" {one_out_name}.pred_use_partial:0 {one_out_name}.pred_for_cali:m"
            CMD_DEV += f" test0.group_files:{dev_in} test0.output_file:{dev_out} {dev_extra_args}"
            self.run_cmd(CMD_DEV)
            # auto-ths
            # if autoths is not None:
            #     for ii, one_out_name in enumerate(self.tasks_out):  # note: must has the same order!
            #         _autoth = autoths[ii]
            #         if not _autoth:  # not applied!
            #             continue
            #         CMD3 = f"python3 -m mspx.scripts.tools.calibrate input_path:{output_data}.devautoth log_stderr:1 key_cali:{one_out_name}_cali mode:th {_autoth}"  # extra args here!
            #         output = self.run_cmd(CMD3, ret_output=True)
            #         best_th = [float(z.split()[-1]) for z in output.split("\n")
            #                    if z.strip().startswith("BEST_TH =")][-1]
            #         zlog(f">>>\n{output}\n>>>")
            #         zlog(f"Get best auto-th for {one_out_name}: {best_th}")
            #         self._auto_ths[ii] = best_th
        # --

    def _do_emb(self, bname: str, model_file: str, input_data: str, output_data: str, extra_args: str):
        # assert len(self.tasks_out) == 1, "Currently only support one out task for this mode!"
        # _n = self.tasks_out[0]
        # note: use "specs_qemb:conf_sbase2:task_name:enc0 bert_lidx:6"
        base_args = self.base_args_test(model_file, input_data, output_data, load_conf=False)  # reuse it!
        CMD = f"python3 -m mspx.tools.misc.assign_emb {base_args} conf_sbase:bert_name:{bname} {extra_args}"
        self.run_cmd(CMD)

    def _do_train(self, train0: str, train1: str, preload_model: str, extra_args: str):
        conf: ALTaskConf = self.conf
        if preload_model is None:
            preload_model = ""
        _strg_args = " ".join([f"{z}.strg_ratio.val:1" for z in self.tasks_out])
        base_args = f"fs:build,train,test log_stderr:0 device:0 conf_output:_conf log_file:_log train0.group_files:{train0} train1.group_files:{train1} train_preload_model:{preload_model} {_strg_args}"
        CMD = f"python3 -m {self.main_entry} {base_args} d_tasks:{','.join(self.tasks_all)} {extra_args}"
        self.run_cmd(CMD)

    def _yield_items(self, inst, cc, **kwargs):
        raise NotImplementedError()

    @property
    def main_entry(self):
        raise NotImplementedError()

    @property
    def always_do_pred_prep_query(self):
        return False  # some tasks might need this!

    # store topk score for easier analysis
    def _qann_score_topk(self, item, src_name: str, trg_name: str):
        import torch  # note: easier topk!
        _k = self.conf.qann_store_topk
        if _k > 0:
            arr0 = item.arrs.get(src_name)
            if arr0 is not None:
                t0 = torch.as_tensor(arr0.copy()).view([arr0.shape[0], -1])  # flatten, [L, ...]
                _vv, _ii = t0.topk(_k, dim=-1)  # [L, K]
                # item.arrs[trg_name+"I"] = _ii.numpy().astype(np.int16)
                # item.arrs[trg_name+"V"] = _vv.numpy().astype(np.float16)
                item.arrs[trg_name+"I"] = _ii.numpy()
                item.arrs[trg_name+"V"] = _vv.numpy()

    # --
    # main ones

    def setup_iter(self, curr_iter: int):
        zlog(f"Setting from {self.curr_iter} to {curr_iter}")
        self.curr_iter = curr_iter

    def _get_curr_val(self, specs, curr_iter=None):
        if curr_iter is None:
            curr_iter = self.curr_iter
        # if isinstance(specs, (int, float)):
        #     return specs  # iter-agnostic
        if isinstance(specs, (list, tuple)):
            ii = min(curr_iter, len(specs)-1)  # last one if more
            return specs[ii]
        else:
            return specs  # iter-agnostic

    @property
    def curr_budget(self):
        return self._get_curr_val(self.conf.budget)

    # setup insts
    def setup_inst(self, inst, mark_unn: bool):
        raise NotImplementedError()

    # downsample for dev set
    def sample_data(self, stream):
        ret = list(self.sampler.prep_insts(stream))
        return ret

    # prepare query model
    def prep_query_model(self, last_model: str):
        conf: ALTaskConf = self.conf
        if last_model is None:
            zwarn("No model provided, this is probably the first round!")
        else:
            if conf.query_use_t0:  # use t0 instead of tz!
                last_model_orig = last_model
                last_model = last_model_orig.replace('tz/zmodel', 't0/zmodel')
                # assert last_model != last_model_orig
                if last_model == last_model_orig:
                    zwarn(f"Strange last model provided: {last_model}")
                zlog(f"Use t0 ({last_model}) instead of tz ({last_model_orig})")
        return last_model

    # prepare data for query
    def prep_query(self, orig_data: str, last_model: str, dev_data: str):
        conf: ALTaskConf = self.conf
        last_model = self.prep_query_model(last_model)
        curr_data, curr_dataD = orig_data, dev_data
        if last_model is not None:
            if conf.query_use_unc or self.always_do_pred_prep_query:  # do inference!
                idata = ZHelper.insert_path(orig_data, 'unc', position=-2)
                idataD = idata + ".dev"
                if os.path.isfile(idata) and (dev_data and os.path.isfile(idataD)):
                    zlog(f"Use the cached file: {idata} // {idataD}")
                elif conf.selector.strg_f.startswith('bald'):
                    bald_times = int(conf.selector.strg_f[len('bald'):])
                    for ii in range(bald_times):
                        if os.path.isfile(f"{idata}_{ii}") and (dev_data and os.path.isfile(f"{idataD}_{ii}")):
                            continue  # use existing ones
                        _specs = self.specs_infQ + f" test_with_dropout:1 seed0:{ii} msp_seed:{ii}"
                        self._do_inf(last_model, curr_data, f"{idata}_{ii}", _specs, conf.inf_with_caliQ,
                                     dev_data, f"{idataD}_{ii}", conf.specs_extra_infQD)
                    # merge strg_arr results
                    self.merge_arr_strg([f"{idata}_{ii}" for ii in range(bald_times)], idata)
                    self.merge_arr_strg([f"{idataD}_{ii}" for ii in range(bald_times)], idataD)
                else:
                    self._do_inf(last_model, curr_data, idata, self.specs_infQ, conf.inf_with_caliQ,
                                 dev_data, idataD, conf.specs_extra_infQD)
                curr_data, curr_dataD = idata, (idataD if dev_data else '')
        if conf.query_emb_bname:  # calculate embs for diversity-query
            edata = ZHelper.insert_path(orig_data, 'emb', position=-2)
            qmodel = last_model if conf.query_emb_use_model else None  # whether try using last-model for emb
            if os.path.isfile(edata):
                zlog(f"Use the cached file: {edata}")
            else:
                self._do_emb(conf.query_emb_bname, qmodel, curr_data, edata, self.specs_qemb)
            curr_data = edata
        return curr_data, curr_dataD

    # special preparing!
    def prep_query_with_ref(self, input_data: str, output_data: str, ref_data: str):
        raise NotImplementedError()

    # obtain query-data
    def do_query(self, data_stream, dev_stream, ref_stream=None, refD_stream=None, no_strg=False):
        conf: ALTaskConf = self.conf
        # --
        # prepare sent repr
        repr_helper = None
        curr_weight_repr = self._get_curr_val(conf.query_repr.repr_weights)
        if curr_weight_repr > 0.:
            from .helper_repr import QueryReprHelper
            data_stream = list(data_stream)
            sents_qA, sents_qU = [], []  # not-queried before, queried before
            for _sent in yield_sents(data_stream):
                if _sent.info.get('query_iters'):
                    sents_qA.append(_sent)
                else:
                    sents_qU.append(_sent)
            repr_helper = QueryReprHelper(sents_qA, sents_qU, curr_weight_repr, conf.query_repr)
        # --
        # querying
        ret_docs, q_hit_sents, cc = self._do_query(data_stream, dev_stream, ref_stream, refD_stream, no_strg, repr_helper)
        # --
        # add query info
        for _sent in q_hit_sents.values():
            if 'query_iters' not in _sent.info:
                _sent.info['query_iters'] = []
            _sent.info['query_iters'].append(self.curr_iter)
            _sent.info[f'query_i{self.curr_iter:02d}'] = 1
        # delete extra things for compactness
        for sent in yield_sents(ret_docs):
            trgs = [sent] + sent.get_frames()
            for item in trgs:
                for _key in list(item.arrs.keys()):
                    if any(_key.endswith(z) for z in ["_strg", "_strg0"]):
                        self._qann_score_topk(item, _key, "qannK")
                    if any(_key.endswith(z) for z in ["_strg", "_strg0", "_strg1", "_hid"]):
                        del item.arrs[_key]
        # --
        return ret_docs, cc

    # obtain query-data
    def _do_query(self, data_stream, dev_stream, ref_stream=None, refD_stream=None, no_strg=False, repr_helper=None):
        raise NotImplementedError()

    # simulated annotation
    def do_simul_ann(self, query_insts, ref_map, last_model=None):
        raise NotImplementedError()

    # combine curr annotations with previous
    def do_comb(self, ann_insts, trg_map):
        conf: ALTaskConf = self.conf
        # --
        cc_ann = Counter()
        cc_before, cc_after = Counter(), Counter()
        # stat before comb
        for inst in trg_map.values():
            for one in self._yield_items(inst, cc_before):
                pass
        # comb
        has_ann_smap = {}  # dsids -> sent
        for inst in ann_insts:
            self._do_comb(inst, trg_map, cc_ann)
        for a_sent in yield_sents(ann_insts):  # update query info
            t_sent = trg_map[a_sent.doc.id].sents[a_sent.sid]
            q_info = {k: v for k, v in a_sent.info.items() if k.startswith("query_")}
            if q_info:
                t_sent.info.update(q_info)
        # stat after comb
        for inst in trg_map.values():
            for one in self._yield_items(inst, cc_after):
                pass
            # todo(+1): re-mark finished instances?
        # --
        return {'cc0_ann': cc_ann, 'cc1_before': cc_before, 'cc2_after': cc_after,
                'cc_hasann_sent': len(has_ann_smap)}

    # combine one
    def _do_comb(self, ann_inst, trg_map, cc):
        raise NotImplementedError()

    # ann + pred
    def do_comb_annpred(self, input_data: str, output_data: str, last_model: str):
        if last_model:
            last_model = self.prep_query_model(last_model)  # follow the query one!
            zwarn(f"Inf ann-pred with {last_model} for {input_data} -> {output_data}")
            _specs = self.specs_infT
            _specs = _specs.replace("pred_do_dec:0", "pred_do_dec:1")
            _specs = _specs.replace("pred_do_strg:1", "pred_do_strg:0")
            self._do_inf(last_model, input_data, output_data, _specs, '', '')
        else:
            zwarn(f"No model to ann-pred for {input_data} -> {output_data}")

    # do training
    def do_train(self, input_data: str, output_model: str, last_model: str, cali_dev: str):
        conf: ALTaskConf = self.conf
        # --
        output_dir = os.path.dirname(output_model)
        cc = {'models': []}
        # --
        curr_last_model = last_model
        trg_ii = max(1, conf.train_st)  # at least one!
        for ii in range(trg_ii):
            if ii == trg_ii - 1:  # last one
                rdir = output_dir
            else:  # make another one!
                rdir = os.path.join(os.path.dirname(output_dir), f't{ii}')
            mkdir_p(rdir)
            # skip if already finished training!
            next_last_model = os.path.join(rdir, 'zmodel.best.m')
            if not os.path.isfile(next_last_model):  # skip if existing
                # prepare training data
                train0 = os.path.relpath(input_data, rdir)  # orig train
                train1 = ""  # ST train
                if (not curr_last_model) or (ii==0):  # no ST!
                    pass
                else:  # use ST in some way
                    # orig train data
                    rel_tdata = '_train.json'
                    tdata = os.path.join(rdir, rel_tdata)
                    self._do_inf(curr_last_model, input_data, tdata, self.specs_infT, conf.inf_with_caliT, cali_dev)
                    self._process_st_train(tdata)
                    train1 = rel_tdata
                    # using external unlab data?
                    if conf.external_st_data:
                        ext_data = zglob1(conf.external_st_data)
                        rel_tdataE = '_trainE.json'
                        tdataE = os.path.join(rdir, rel_tdataE)
                        # note: treating extra data as all unlabeled!!
                        _specsE = str(self.specs_infT).replace("pred_use_partial:1", "pred_use_partial:0")
                        self._do_inf(curr_last_model, ext_data, tdataE, _specsE, conf.inf_with_caliT, cali_dev)
                        train1 = rel_tdataE  # use this instead!
                # change dir & train it!
                _specs_train = self.specs_trains[ii] if ii<len(self.specs_trains) else self.specs_trains[-1]
                _extra_args = f"zdir:{rdir} train0.info:strgR:0 train1.info:strgR:1 {_specs_train}"
                if self.curr_iter == 0:
                    if ii == 0:
                        _extra_args += f" {conf.specs_train00}"  # extra ones for 00!
                    else:
                        _extra_args += f" {conf.specs_train0p}"  # extra ones for 0p!
                    if conf.train_no_ann_i0:  # no train0 data!
                        # assert curr_last_model
                        if train1 and not (('train1.group_files: ' in _extra_args) or _extra_args.endswith('train1.group_files:')):  # no train0 but has train1
                            _extra_args += " train0.group_files:"
                        else:  # no training at all!
                            _extra_args += " record_best_start_cidx:0 valid_first:1 max_uidx:0"
                self._do_train(train0, train1, (curr_last_model if conf.train_preload_last else None), _extra_args)
                cc['models'].append(curr_last_model)
            else:
                zlog(f"Skip training for {next_last_model}")
            # --
            curr_last_model = next_last_model
        return cc

    def _process_st_train(self, file: str):
        # return self._process_st_strg(file)
        pass

    @staticmethod
    def _process_st_strg(file: str):
        import traceback
        from mspx.data.inst import yield_sents
        from mspx.data.rw import ReaderGetterConf, WriterGetterConf
        try:  # simply delete all arrs!
            _keys = ["_strg", "_strg0", "_strg1", "_cali"]
            insts = list(ReaderGetterConf().get_reader(input_path=file))
            for sent in yield_sents(insts):
                for key in list(sent.arrs.keys()):
                    if any(key.endswith(z) for z in _keys):
                        del sent.arrs[key]
                for frame in sent.get_frames():
                    for key in list(frame.arrs.keys()):
                        if any(key.endswith(z) for z in _keys):
                            del frame.arrs[key]
            with WriterGetterConf().get_writer(output_path=file+".sjson") as writer:
                writer.write_insts(insts)
        except:
            zwarn(f"#== Error in _process_st_train:\n{traceback.format_exc()}\n#==")

    @staticmethod
    def merge_arr_strg(input_files, output_file):
        from mspx.data.inst import yield_sents
        from mspx.data.rw import ReaderGetterConf, WriterGetterConf
        # note: add one more index at idx=1
        insts = list(ReaderGetterConf().get_reader(input_path=input_files[0]))
        all_sents = list(yield_sents(insts))
        _keys = [k for k in all_sents[0].arrs.keys() if any(k.endswith(suffix) for suffix in ['_strg', '_strg0', '_cali'])]
        all_strgs = {k: [[s.arrs[k]] for s in all_sents] for k in _keys}
        for ff in input_files[1:]:
            insts2 = list(ReaderGetterConf().get_reader(input_path=ff))
            for ii, sent in enumerate(yield_sents(insts2)):
                assert sent.seq_word.vals == all_sents[ii].seq_word.vals
                for k in _keys:
                    all_strgs[k][ii].append(sent.arrs[k])
        # combine them
        for k in _keys:
            for ii, sent in enumerate(all_sents):
                sent.arrs[k] = np.stack(all_strgs[k][ii], 1)  # [L, ...] -> [L, C, ...]
        with WriterGetterConf().get_writer(output_path=output_file) as writer:
            writer.write_insts(insts)

    def _yield_dev_cands(self, dev_stream, key_cali: str):
        from mspx.data.inst import yield_sents
        for sent in yield_sents(dev_stream):
            _arr = sent.arrs[key_cali]
            for _slice in _arr:
                one_cand = ZObject(type='UNK', gid=id(sent), sent=sent, budget=1)
                if len(_slice.shape) == 1:  # plain 1d
                    one_cand.arr_strg = _slice[1:]
                    one_cand.gold_idx = int(_slice[0].item())  # note: special format from cali!
                else:  # possibly for bald
                    one_cand.arr_strg = _slice[..., 1:]
                    _gi = _slice[0, 0]
                    assert all(_gi == z for z in _slice[..., 0])
                    one_cand.gold_idx = int(_gi)
                    # breakpoint()
                yield one_cand
        # --

    def _get_ref_helper(self, ref_stream):
        from .helper_ref import QueryRefHelper
        return QueryRefHelper(ref_stream, self, self.conf.specs_vocdir)

# --
# b mspx/tools/al/tasks/base:130
