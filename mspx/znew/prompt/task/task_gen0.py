#

# for generation

__all__ = [
    "TaskGenConf", "TaskGen",
]

import torch
from transformers.utils import ModelOutput
from mspx.utils import ZObject, zlog, zwarn
from .base import TaskConf, MyTask, OutConf, OutSpec
from .info_gen0 import _TEMPLATES, _INFOS

@TaskConf.rd('gen0')
class TaskGenConf(TaskConf):
    def __init__(self):
        super().__init__()
        # --
        # for decoding
        self.max_new_tokens = 128
        self.temperature = 1.
        self.top_k = 50
        self.top_p = 1.
        self.num_beams = 1
        self.do_sample = False
        # --
        # others
        self.debug_print_first = 5

@TaskGenConf.conf_rd()
class TaskGen(MyTask):
    def __init__(self, conf: TaskGenConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: TaskGenConf = self.conf
        # --
        self.template = self.obtain_template(_TEMPLATES, conf.template_base, conf.template_choice, conf.template_kwargs)
        self.info = self.obtain_info(_INFOS, conf.info_choice, conf.info_kwargs, conf.mapper_kwargs)
        if self.template.conf.extra_info:
            self.info.update(self.template.conf.extra_info)
        self.mapper = self.info.get('mapper', {})
        # --
        self.CKEY = "_C"
        self.debug_print_first = conf.debug_print_first
        # --
        self.storage = None
        self.demo_selector = None

    def remove_cache(self, insts):
        for inst in insts:
            inst.del_cache(self.CKEY)

    def preprocess_inst(self, inst, dataset):
        conf: TaskGenConf = self.conf
        # --
        if not inst.has_cache(self.CKEY):
            _is_train = dataset.is_train
            _model = self.model
            # --
            self._get_mapped_inst(inst)  # process this!
            demos = ()
            if self.demo_selector is not None:
                demos = self.demo_selector.select([inst])[0]
            # --
            assert _model.is_clm or _model.is_s2s
            if _model.is_clm and _is_train:
                res = self._process_inst(inst, demos, idx_keys={'idx_trg': '_trg0'})
            else:
                res = self._process_inst(inst, demos, break_key='_break')
            inst.set_cache(self.CKEY, res)
        # --
        return inst

    def collate_fn(self, ibatch):
        ret = {'ibatch': ibatch}
        _caches = [z.get_cache(self.CKEY) for z in ibatch.items]
        _is_train = ibatch.dataset.is_train
        _spec_left = {}
        if self.model.is_clm:
            _spec_left['input'] = True  # note: left for input of CLM!
        _surroundings = self.model.model_helper.surroundings
        _spec_tok = {'input': _surroundings, 'output': _surroundings}
        _spec_tok['output'][1] = 1  # add EOS for output
        if self.model.is_clm:
            if _is_train:  # complete it for training
                _spec_tok['input'][1] = 1
            else:  # open-ended for decoding
                _spec_tok['input'][1] = 0
        ret2 = self._collate_caches(_caches, spec_tok_extras=_spec_tok, spec_left_pad=_spec_left, spec_left_truncate=_spec_left)
        ret.update(ret2)
        return ret

    def model_forward(self, ibatch, do_loss=False, do_test=False, do_enc=False, debug_print=False, **kwargs):
        conf: TaskGenConf = self.conf
        _model = self.model
        _base_model = _model.base_model
        # --
        ret = ModelOutput()
        ret['info'] = {'inst': len(ibatch.items)}
        t_input, t_input_mask, t_output, t_output_mask, t_idx_trg = [kwargs.get(z) for z in ['t_input', 't_input_mask', 't_output', 't_output_mask', 't_idx_trg']]
        # --
        # print for debugging
        if self.debug_print_first > 0 or debug_print:
            _toker = self.model.toker
            t_input, t_output = kwargs['t_input'], kwargs.get('t_output', None)
            for ii in range(len(t_input)):
                zlog(f"#--\nDebug instance #{self.debug_print_first}:\n(Input)\n{_toker.decode(t_input[ii])}\n(Output)\n{_toker.decode(t_output[ii]) if t_output is not None else None}")
                self.debug_print_first -= 1
                if self.debug_print_first <= 0:
                    break
        # --
        if do_enc:
            if _model.is_clm:
                t_posi = (t_input_mask.cumsum(-1) - 1).clamp(min=0).to(t_input)
                outputs = _base_model(input_ids=t_input, attention_mask=t_input_mask, position_ids=t_posi, output_hidden_states=True, return_dict=True)
                t_enc = outputs['hidden_states'][-1][:,-1]
            elif _model.is_s2s:
                outputs = _base_model(input_ids=t_input, attention_mask=t_input_mask, decoder_input_ids=t_output, decoder_attention_mask=t_output_mask, output_hidden_states=True, return_dict=True)
                t_enc = outputs['decoder_hidden_states'][-1][:,0]
            else:
                raise NotImplementedError()
            ret['t_enc'] = t_enc
        if do_loss:
            if _model.is_clm:
                t_posi = (t_input_mask.cumsum(-1) - 1).clamp(min=0).to(t_input)
                outputs = _base_model(input_ids=t_input, attention_mask=t_input_mask, position_ids=t_posi, return_dict=True)
                t_content, t_content_mask = t_input, t_input_mask.clone()
                t_arange = torch.arange(t_content_mask.shape[-1]).to(t_content_mask)  # [L]
                t_content_mask = t_content_mask * (t_arange >= t_idx_trg).to(t_content_mask)  # [bs, L]
            elif _model.is_s2s:
                outputs = _base_model(input_ids=t_input, attention_mask=t_input_mask, decoder_input_ids=t_output, decoder_attention_mask=t_output_mask, return_dict=True)
                t_content, t_content_mask = t_output, t_output_mask
            else:
                raise NotImplementedError()
            _logits, _labels = outputs.logits[..., :-1, :], t_content[..., 1:]  # [bs, L-1, *]
            _label_masks = t_content_mask[..., 1:]  # note: shift one!
            loss_lm = - self._get_logprobs(_logits, _labels, _label_masks, aggr_method='avg').mean()
            ret['info']['loss_lm'] = loss_lm.item()
            ret['loss'] = loss_lm
            # breakpoint()
        if do_test:
            from transformers import GenerationConfig
            toker = self.model.toker
            generation_config = GenerationConfig(temperature=conf.temperature, top_p=conf.top_p, top_k=conf.top_k, num_beams=conf.num_beams, max_new_tokens=conf.max_new_tokens, do_sample=conf.do_sample, eos_token_id=toker.eos_token_id, pad_token_id=toker.pad_token_id)
            # breakpoint()
            generation_output = _base_model.generate(inputs=t_input, attention_mask=t_input_mask, generation_config=generation_config, return_dict_in_generate=True, output_scores=True)
            ret['gen_out'] = generation_output
            ret['info']['l_input'] = t_input.shape[-1]  # [length-input]
            # breakpoint()
        # --
        return ret

    def pred(self, ibatch, outputs, evaler):
        from string import punctuation
        conf: TaskGenConf = self.conf
        # --
        predictions = outputs.gen_out.sequences.cpu().numpy()  # [bs, L]
        if self.model.is_clm:  # no inputs for CLM
            predictions = predictions[:, outputs.info['l_input']:]
        for ii, inst in enumerate(ibatch.items):
            _cache = inst.get_cache(self.CKEY)
            _pred_seq = self.model.toker.decode(predictions[ii], skip_special_tokens=True)
            inst.update({'output_pred': _pred_seq})
            # --
            # match choices
            _gold = inst.get('output', '')
            _choices = inst.get('choices', self.info.get('choices'))
            if _choices is not None:
                # find first tok match
                _pred_choice = 'UNK'
                # breakpoint()
                for tt in _pred_seq.strip().split(".")[0].lower().split():
                    tt = tt.rstrip(punctuation)
                    for tt2 in _choices:
                        if tt2.lower() == tt:
                            _pred_choice = tt2
                            break
                # breakpoint()
                inst.update({'output_pred': _pred_choice})
                inst.update({'output_orig': _pred_seq})
                _gold = inst.get_cache(self.CKEY)['_gold']
            # --
            if evaler is not None:
                evaler.add(pred=inst['output_pred'], gold=_gold)
            # breakpoint()
        return

# --
# b mspx/znew/prompt/task/task_gen0:134
