#

# single sequence classification (with CLM)

__all__ = [
    "TaskClfSingleConf", "TaskClfSingle",
]

import torch
from transformers.utils import ModelOutput
from mspx.utils import ZObject, zlog
from .base import TaskConf, MyTask, OutConf, OutSpec
from ..model.storage import StorageConf, MyStorage, SelectorConf, MySelector
from .info_clf0 import _INFOS, _TEMPLATES

@TaskConf.rd('clf0')
class TaskClfSingleConf(TaskConf):
    def __init__(self):
        super().__init__()
        # --
        # various different modes:
        self.clf_modes = ["logp", "logp"]  # modes for train/pred; logp: whole-sequence log-prob; head: head-repr
        # --
        self.logp_loss = [1., 0., 0.]  # loss weights: LM,UL,LN
        self.logp_aggr_methods = ['avg', 'avg']  # aggr for train/pred
        self.head_use0 = False   # use idx=0 instead of "_head"
        self.head_out_conf = OutConf()
        self.head_init_stra = ''  # avg or first-token
        # others
        self.debug_print_first = 5
        # storage & knn
        self.storage = StorageConf()
        self.pred_knn_k = 3  # knn
        self.pred_knn_lambda = 0.  # mixing?
        # icl selector
        self.demo_select = SelectorConf()

@TaskClfSingleConf.conf_rd()
class TaskClfSingle(MyTask):
    def __init__(self, conf: TaskClfSingleConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: TaskClfSingleConf = self.conf
        # --
        self.template = self.obtain_template(_TEMPLATES, conf.template_base, conf.template_choice, conf.template_kwargs)
        self.info = self.obtain_info(_INFOS, conf.info_choice, conf.info_kwargs, conf.mapper_kwargs)
        if self.template.conf.extra_info:
            self.info.update(self.template.conf.extra_info)
        self.mapper = self.info.get('mapper', {})
        self.head_out = OutSpec(conf.head_out_conf, self.model, self.info.get('choices'))
        # --
        self.CKEY = "_C"
        self.debug_print_first = conf.debug_print_first
        self.add_modules_to_model({'head_out': self.head_out})
        # --
        self.storage = None
        self.demo_selector = None

    def remove_cache(self, insts):
        for inst in insts:
            inst.del_cache(self.CKEY)

    def preprocess_inst(self, inst, dataset):
        conf: TaskClfSingleConf = self.conf
        # --
        if not inst.has_cache(self.CKEY):
            _is_train = dataset.is_train
            _model = self.model
            _mode = conf.clf_modes[int(not _is_train)]
            # --
            self._get_mapped_inst(inst)  # process this!
            res = None
            demos = ()
            if self.demo_selector is not None:
                demos = self.demo_selector.select([inst])[0]
            if _mode == 'logp':  # prepare full sequences?
                _need_ext = (not _is_train) or (_is_train and sum(conf.logp_loss[1:]) > 0)  # need choice extension?
                if _model.is_clm:  # record the breaking idx
                    res = self._process_inst(inst, demos, expand_choices=_need_ext,
                                             idx_keys={'idx_trg0': '_trg0', 'idx_trg1': '_trg1'})
                elif _model.is_s2s:  # actually do breaking for s2s
                    res = self._process_inst(inst, demos, expand_choices=_need_ext, break_key='_break',
                                             idx_keys={'idx_out_trg0': '_trg0', 'idx_out_trg1': '_trg1'})
            elif _mode == 'head':
                break_key = None if _model.is_mlm else '_break'  # no need to break for MLM!
                res = self._process_inst(inst, demos, break_key=break_key, idx_keys={'idx_head': '_head'})
            elif _mode == 'gen':
                raise RuntimeError("No need of this mode for CLF!")
            # --
            assert res is not None
            inst.set_cache(self.CKEY, res)
        # --
        return inst

    def collate_fn(self, ibatch):
        ret = {'ibatch': ibatch}
        _caches = [z.get_cache(self.CKEY) for z in ibatch.items]
        spec_left_truncate = {}
        if self.model.is_clm:
            spec_left_truncate['input'] = True  # note: left for input of CLM!
        ret2 = self._collate_caches(_caches, spec_left_truncate=spec_left_truncate)
        ret.update(ret2)
        return ret

    def model_forward(self, ibatch, do_loss=False, do_test=False, **kwargs):
        conf: TaskClfSingleConf = self.conf
        _mode = conf.clf_modes[int(not do_loss)]
        # --
        ret = ModelOutput()
        ret['info'] = {'inst': len(ibatch.items)}
        if _mode == 'logp':  # prepare full sequences?
            self.forward_logp(ibatch, ret, do_loss=do_loss, do_test=do_test, **kwargs)
        elif _mode == 'head':
            self.forward_head(ibatch, ret, do_loss=do_loss, do_test=do_test, **kwargs)
        else:
            raise NotImplementedError()
        # --
        # print for debugging
        if self.debug_print_first > 0:
            _toker = self.model.toker
            t_input, t_output = kwargs['t_input'], kwargs.get('t_output', None)
            for ii in range(len(t_input)):
                zlog(f"#--\nDebug instance #{self.debug_print_first}:\n(Input)\n{_toker.decode(t_input[ii])}\n(Output)\n{_toker.decode(t_output[ii]) if t_output is not None else None}")
                self.debug_print_first -= 1
                if self.debug_print_first <= 0:
                    break
        # --
        # breakpoint()
        return ret

    def forward_logp(self, ibatch, ret, do_loss=False, do_test=False, **kwargs):
        conf: TaskClfSingleConf = self.conf
        _model = self.model
        _base_model = _model.base_model
        # --
        if _model.is_clm:
            t_input, t_input_mask, t_idx_trg0, t_idx_trg1 = [kwargs[z] for z in ['t_input', 't_input_mask', 't_idx_trg0', 't_idx_trg1']]
            outputs = _base_model(input_ids=t_input, attention_mask=t_input_mask, return_dict=True)
            t_content, t_content_mask = t_input, t_input_mask
        elif _model.is_s2s:
            t_input, t_input_mask, t_output, t_output_mask, t_idx_trg0, t_idx_trg1 = [kwargs[z] for z in ['t_input', 't_input_mask', 't_output', 't_output_mask', 't_idx_out_trg0', 't_idx_out_trg1']]
            outputs = _base_model(input_ids=t_input, attention_mask=t_input_mask, decoder_input_ids=t_output, decoder_attention_mask=t_output_mask, return_dict=True)
            t_content, t_content_mask = t_output, t_output_mask
        else:
            raise NotImplementedError()
        # --
        _logits, _labels = outputs.logits[..., :-1, :], t_content[..., 1:]  # [bs, L-1, *]
        _t_arange = torch.arange(t_content.shape[-1]).to(_labels)  # [L]
        _label_masks0 = ((t_idx_trg0 <= _t_arange) & (_t_arange < t_idx_trg1)).to(t_content_mask) * t_content_mask  # [L]
        _label_masks = _label_masks0[..., 1:]  # note: shift one!
        # --
        bs, n_items = t_input.shape[0], len(ibatch.items)
        assert bs % n_items == 0
        num_c = bs // n_items
        # --
        _aggr0, _aggr1 = conf.logp_aggr_methods
        if do_loss:
            _w_ml, _w_ul, _w_ln = conf.logp_loss
            if num_c == 1:  # only LM loss
                assert _w_ul==0 and _w_ln==0
                loss = - self._get_logprobs(_logits, _labels, _label_masks, aggr_method=_aggr0).mean() * _w_ml
            else:  # multiple loss
                t_gold = kwargs['t_gold']  # [bs, 1]
                _logprob0 = self._get_logprobs(_logits, _labels, _label_masks, aggr_method=_aggr0)  # [bs?]
                _logprob1 = _logprob0.view([-1, num_c])  # [bs, C]
                loss_ml = - _logprob1.gather(-1, t_gold).squeeze(-1).mean()
                ret['info']['loss_ml'] = loss_ml.item()
                loss = loss_ml * _w_ml
                if _w_ul > 0:
                    from ..model.misc import extend_idxes
                    t_gold_mask = extend_idxes(t_gold.squeeze(-1), num_c)  # [bs, C]
                    _ul0 = self._get_logprobs(_logits, _labels, _label_masks, aggr_method=_aggr0, ul=True)  # [bs?]
                    _ul1 = _ul0.view([-1, num_c])  # [bs, C]
                    loss_ul = - _ul1[t_gold_mask == 0.].mean()
                    loss = loss + loss_ul * _w_ul
                    ret['info']['loss_ul'] = loss_ul.item()
                if _w_ln > 0:
                    loss_ln = - _logprob1.log_softmax(-1).gather(-1, t_gold).squeeze(-1).mean()
                    loss = loss + loss_ln * _w_ln
                    ret['info']['loss_ln'] = loss_ln.item()
            ret['loss'] = loss
        # --
        if do_test:
            _aggr_res = self._get_logprobs(_logits, _labels, _label_masks, aggr_method=_aggr1)  # [bs?]
            _res = _aggr_res.view([-1, num_c])  # [bs, C]
            ret['res'] = _res
        # --

    # up until get reprs
    def forward_head0(self, **kwargs):
        conf: TaskClfSingleConf = self.conf
        _model = self.model
        _base_model = _model.base_model
        # --
        t_input, t_input_mask, t_idx_head = [kwargs[z] for z in ['t_input', 't_input_mask', 't_idx_head']]
        if _model.is_s2s:
            t_output, t_output_mask = [kwargs[z] for z in ['t_output', 't_output_mask']]
            more_kwargs = {'decoder_input_ids': t_output, 'decoder_attention_mask': t_output_mask}
        else:
            more_kwargs = {}
        outputs = _base_model(input_ids=t_input, attention_mask=t_input_mask, output_hidden_states=True, return_dict=True, **more_kwargs)
        if conf.head_use0:
            t_idx_head = t_idx_head * 0  # simply use idx=0
            assert not _model.is_clm, "No meaning of use0 for clm"
        else:
            assert not _model.is_s2s, "Please use0 for head in s2s"
        # --
        model_outs = ZObject(logits=outputs.logits, hidden_states=(outputs.decoder_hidden_states if _model.is_s2s else outputs.hidden_states))
        t_repr, t_score = self.head_out(model_outs, t_idx_head)  # [bs, R]
        return t_repr, t_score

    def forward_head(self, ibatch, ret, do_loss=False, do_test=False, **kwargs):
        conf: TaskClfSingleConf = self.conf
        # --
        t_repr, t_score = self.forward_head0(**kwargs)
        if do_loss:
            from ..model.misc import loss_nll
            t_gold = kwargs['t_gold']  # [bs, 1]
            loss = loss_nll(t_score, t_gold.squeeze(-1)).mean()  # [bs]
            ret['loss'] = loss
            ret['info']['loss'] = loss.item()
        if do_test:
            _pred_knn_lambda = conf.pred_knn_lambda
            if _pred_knn_lambda > 0:
                t_knn_distr = self.storage.search_and_distr(t_repr, conf.pred_knn_k, 't_gold')
                t_orig_distr = t_score.softmax(-1)
                t_new_distr = _pred_knn_lambda * t_knn_distr + (1.-_pred_knn_lambda) * t_orig_distr
                t_score = t_new_distr  # note: simply use distr!
            ret['res'] = t_score
        # breakpoint()

    def pred(self, ibatch, outputs, evaler):
        conf: TaskClfSingleConf = self.conf
        _mode = conf.clf_modes[1]
        if _mode == 'logp' or _mode == 'head':
            arr_choices = outputs.res.argmax(-1).cpu().numpy()  # [bs]
            for ii, inst in enumerate(ibatch.items):
                _cache = inst.get_cache(self.CKEY)
                _pred = _cache['_choices'][arr_choices[ii]]
                _gold = _cache['_gold']
                inst.update({'label_pred': _pred, 'label_gold': _gold})  # write it down!
                if evaler is not None:
                    evaler.add(pred=_pred, gold=_gold)
                # breakpoint()
        else:
            raise NotImplementedError()
        return

    def pass_train(self, train_loader):
        conf: TaskClfSingleConf = self.conf
        _mode = conf.clf_modes[0]
        # --
        if conf.demo_select.sel_k > 0:
            all_insts = [self.preprocess_inst(z, train_loader.dataset.dataset) for z in train_loader.dataset.dataset.yield_insts()]  # note: yield raw ones!
            self.demo_selector = MySelector(conf.demo_select, all_insts)
            self.remove_cache(all_insts)  # remove caches!
        # --
        if _mode == 'head':
            # --
            # prepare head initial
            _hinit = conf.head_init_stra
            if _hinit == 'tok':  # simply take first sub-token!
                _toker = self.model.toker
                # todo(+N): always assume a leading space!!
                t_idx = torch.as_tensor([_toker.convert_tokens_to_ids(_toker.tokenize(" "+z))[0] for z in self.info['choices']])
                self.head_out.reset_param(t_idx=t_idx)
            # --
            # prepare storage for KNN
            if conf.pred_knn_lambda > 0 or _hinit == 'avg':
                all_reprs, all_golds, all_insts = [], [], []
                for cur_data in train_loader:
                    with torch.no_grad():
                        _repr, _ = self.forward_head0(**cur_data)
                    ibatch = cur_data['ibatch']
                    all_reprs.append(_repr)
                    all_golds.append(cur_data['t_gold'])
                    all_insts.extend(ibatch.items)
                t_repr = torch.cat(all_reprs, 0)
                t_gold = torch.cat(all_golds, 0)
                if conf.pred_knn_lambda > 0:
                    self.storage = MyStorage(conf.storage)
                    self.storage.add(t_repr, t_gold=t_gold, insts=all_insts)
                if _hinit == 'avg':
                    self.head_out.reset_param(t_repr=t_repr, t_gold=t_gold)
            # --
        # --
        pass

# --
# b mspx/znew/prompt/task/task_clf0:176
