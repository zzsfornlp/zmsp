#
import math

import torch
from copy import deepcopy
from torch.utils.data import IterableDataset, DataLoader
from mspx.utils import Conf, Configurable, Registrable, zlog, zwarn
from ..model.template import TemplateConf, Template
from ..eval import EvalConf, Evaler

@Registrable.rd('Task')
class TaskConf(Conf):
    def __init__(self):
        # template
        self.template_base = TemplateConf()
        self.template_choice = ""  # template choice
        self.template_kwargs = {}
        # info
        self.info_choice = []  # allow multiple ones, updating previous ones!
        self.info_kwargs = {}
        self.mapper_kwargs = {}
        # model
        self.max_seq_len = 2048
        # eval
        self.eval = EvalConf()

    @staticmethod
    def find_conf(s: str):
        import importlib
        importlib.import_module(f'.task_{s}', package=__package__)
        return TaskConf.key2cls(s)()

    @classmethod
    def get_base_conf_type(cls): return TaskConf
    @classmethod
    def get_base_node_type(cls): return MyTask

@Registrable.rd('_Task')
class MyTask(Configurable):
    def __init__(self, conf: TaskConf, model, **kwargs):
        super().__init__(conf, **kwargs)
        conf: TaskConf = self.conf
        # --
        from ..model import MyModel
        self.model: MyModel = model
        self.template: Template = None  # template
        self.info = None  # other information
        self.mapper = None  # mapper: inst -> template's input

    def get_scheduled_values(self):
        return {}

    def new_evaler(self):
        return Evaler(self.conf.eval)

    def add_modules_to_model(self, modules):
        for k, m in modules.items():
            zlog(f"Add module {k} to the model!")
            assert not hasattr(self.model, f"mod_{k}")
            setattr(self.model, f"mod_{k}", m)

    # --
    # data processing

    # preprocess instance
    def preprocess_inst(self, inst, dataset):
        raise NotImplementedError()

    # batch data
    def collate_fn(self, ibatch):
        raise NotImplementedError()

    # forward model
    def model_forward(self, *kwargs):
        raise NotImplementedError()

    # prediction
    def pred(self, ibatch, outputs, evaler):
        raise NotImplementedError()

    # pass through full training data (for special purposes)
    def pass_train(self, train_loader):
        pass

    # --
    # other helpers

    def get_dataloader(self, dataset, loop: bool):
        iter_dataset = MyIterableDataset(dataset, loop, self.preprocess_inst)
        loader = DataLoader(iter_dataset, batch_size=None, collate_fn=self.collate_fn)
        return loader

    def obtain_template(self, options, template_base, template_choice, template_kwargs):
        template_d = deepcopy(options[template_choice])
        template_d.update(template_kwargs)
        ret = Template(conf=template_base, toker=self.model.toker, **template_d)
        return ret

    def obtain_info(self, options, info_choice, info_kwargs, mapper_kwargs):
        ret = {}
        if not isinstance(info_choice, (list, tuple)):
            info_choice = [info_choice]
        if '_base' not in info_choice and '_base' in options:
            info_choice = ['_base'] + list(info_choice)  # note: add _base by default!
        for one in info_choice:
            ret.update(options[one])
        if info_kwargs:
            ret.update(info_kwargs)
        if mapper_kwargs:
            ret['mapper'].update(mapper_kwargs)
        return ret

    def _get_mapped_inst(self, inst):
        if '_map' not in inst:
            inst['_map'] = {k: f(inst, self.info) for k, f in self.mapper.items()}
        return inst['_map']

    def _process_inst(self, inst, demos, expand_choices=False, break_key=None, idx_keys=None):
        template: Template = self.template
        # --
        mapped_inst = self._get_mapped_inst(inst)
        mapped_demos = [self._get_mapped_inst(z) for z in demos]
        final_ret = {}
        # add choices
        _choice_key = inst.get('choice_key', self.info.get('choice_key'))
        _choices = inst.get('choices', self.info.get('choices'))
        if _choices is not None:
            _gold_idx = _choices.index(mapped_inst[_choice_key])
            final_ret.update({'_choices': _choices, '_gold': mapped_inst[_choice_key], 'gold': [_gold_idx]})
        if expand_choices:
            prep_res = [template.prepare(mapped_inst, demos=mapped_demos, inst_extras={_choice_key: z}) for z in _choices]
        else:
            prep_res = [template.prepare(mapped_inst, demos=mapped_demos)]
        # --
        # collect more info
        _input_toks = [z[0] for z in prep_res]  # first one is input-ids
        _output_toks = None
        if break_key is not None:
            _idx_breaks = [z[1]['instance'][break_key] for z in prep_res]
            assert all(zi[1]==0 for zi in _idx_breaks), "_break itself should not be filled!"
            _output_toks = [zz[ii[0]:] for zz,ii in zip(_input_toks, _idx_breaks)]
            _input_toks = [zz[:ii[0]] for zz,ii in zip(_input_toks, _idx_breaks)]
        if idx_keys is not None:
            _offsets = None
            if _output_toks is not None:
                _offsets = [len(z) for z in _input_toks]
            for _trg_name, _inst_key in idx_keys.items():
                assert _trg_name.startswith("idx_")
                _idxes = [z[1]['instance'][_inst_key][0] for z in prep_res]
                if _trg_name.startswith("idx_out_"):  # note: target idx!
                    _idxes = [zz-ii for zz,ii in zip(_idxes, _offsets)]
                final_ret.update({_trg_name: _idxes})
        final_ret.update({'input': _input_toks})
        if _output_toks is not None:
            final_ret.update({'output': _output_toks})
        # --
        return final_ret

    def _collate_caches(self, caches, spec_tok_extras=None, spec_left_pad=None, spec_left_truncate=None):
        _toker = self.model.toker
        _max_len = min(self.conf.max_seq_len, _toker.model_max_length)
        _surroundings = self.model.model_helper.surroundings
        if spec_tok_extras is None:
            spec_tok_extras = {}  # c_key -> surroundings
        if spec_left_pad is None:
            spec_left_pad = {}  # c_key -> whether pad/truncate left?
        if spec_left_truncate is None:
            spec_left_truncate = {}
        # --
        ret = {}
        _io_keys = {'input', 'output'}
        _keys = sorted(caches[0].keys(), key=(lambda k: 0 if k in _io_keys else 1))
        _io_idx_offsets = {}
        for c_key in _keys:
            if c_key.startswith("_"): continue  # ignore these fields
            c_vals = sum([z[c_key] for z in caches], [])
            _is_io = c_key in _io_keys
            if _is_io:  # pad with PAD_ID!!
                _add_cls = spec_tok_extras.get(c_key, [1,1])[0] * _surroundings[0]
                _add_sep = spec_tok_extras.get(c_key, [1,1])[1] * _surroundings[1]
                _extras = ([_toker.cls_token_id] if _add_cls else [], [_toker.sep_token_id] if _add_sep else [])
                _leftP = spec_left_pad.get(c_key, False)
                _leftT = spec_left_truncate.get(c_key, False)
                if c_key == 'output':
                    _m_conf = self.model.base_model.config
                    if getattr(_m_conf, 'is_encoder_decoder', False) and getattr(_m_conf, 'decoder_start_token_id', None) is not None:
                        _extras = ([_m_conf.decoder_start_token_id], _extras[1])  # for s2s, we should use a decoder-start!
                t_vals, t_mask, t_offset = self.model.batch_2d(c_vals, pad_val=_toker.pad_token_id, max_len=_max_len, max_len_mul=8, extras=_extras, left_pad=_leftP, left_truncate=_leftT, warn_truncate=True)
                _io_idx_offsets[c_key] = t_offset.unsqueeze(-1)
            else:
                if not isinstance(c_vals[0], (list, tuple)):
                    c_vals = [[z] for z in c_vals]  # make it 2D!
                t_vals = self.model.batch_2d(c_vals)[0]
                t_mask = None
            # adjust offset for idxes
            if c_key.startswith("idx_out_"):
                _offset = _io_idx_offsets.get('output')
                if _offset is not None:
                    t_vals = t_vals + _offset
            elif c_key.startswith("idx_"):
                _offset = _io_idx_offsets.get('input')
                if _offset is not None:
                    t_vals = t_vals + _offset
            # --
            ret[f't_{c_key}'] = t_vals
            if t_mask is not None:
                ret[f't_{c_key}_mask'] = t_mask
        # --
        # put on device!
        _device = self.model.my_first_device()
        for kk in list(ret.keys()):
            ret[kk] = ret[kk].to(_device)
        return ret

    def _get_logprobs(self, logits, labels, label_mask, ul=False, aggr_method=None):
        _shape = list(logits.shape)
        _logprobs_full = logits.log_softmax(-1)  # [bs, ?, V]
        if ul:  # unlikelihood
            _logprobs_full = torch.clamp((1.0 - _logprobs_full.exp()), min=1e-8).log()
        _logprobs0 = _logprobs_full.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [bs, ?]
        _ret = _logprobs0 * label_mask  # [bs, ?]
        if aggr_method:
            _ret = self._aggr_logprobs(_ret, label_mask, aggr_method)
        return _ret

    def _aggr_logprobs(self, logprobs, masks, method: str):
        if method == 'sum':
            ret = logprobs.sum(-1)
        elif method == 'avg':
            ret = logprobs.sum(-1) / masks.sum(-1)
        elif method == 'first':
            from ..model.misc import mask2idx
            t_idxes, t_valid = mask2idx(masks)
            ret = logprobs.gather(-1, t_idxes[..., 0:1]).squeeze(-1)
        else:
            raise NotImplementedError(f"UNK method {method}")
        return ret

# iterable dataset
class MyIterableDataset(IterableDataset):
    def __init__(self, dataset, loop, preprocess):
        self.dataset = dataset
        self.loop = loop
        self.preprocess = preprocess

    def __iter__(self):
        yield from self.dataset.yield_batches(loop=self.loop, processors=[self.preprocess])

# output specifier
class OutConf(Conf):
    def __init__(self):
        self.use_what = 'logits'
        self.layers_hid = [-1]  # which layers for layer-mode?
        self.logit_logprob = True  # do logprob over logits

class OutSpec(torch.nn.Module):
    def __init__(self, conf: OutConf, model, choices):
        super().__init__()
        self.conf = conf
        # --
        base_model = model.base_model
        if conf.use_what == 'logits':
            _dim = base_model.config.vocab_size
        elif conf.use_what == 'hiddens':
            _dim = base_model.config.hidden_size
        else:
            raise NotImplementedError()
        self.output_dim = _dim
        self.vocab_dim = len(choices)
        # --
        if choices is not None:
            self.linear = torch.nn.Linear(_dim, len(choices))
            with torch.no_grad():
                self.linear.bias.zero_()  # zero it!
        else:
            self.linear = None
        self.to(model.my_first_param())
        # --

    def forward(self, outputs, t_idx):
        conf: OutConf = self.conf
        # --
        # get t_full: [bs, L, ??]
        if conf.use_what == 'logits':
            t_full = outputs.logits
            if conf.logit_logprob:
                t_full = t_full.log_softmax(-1)
        elif conf.use_what == 'hiddens':
            t_full = torch.stack([outputs.hidden_states[z] for z in conf.layers_hid], dim=0).mean(0)  # note: currently simply average!
        else:
            raise NotImplementedError()
        t_arange = torch.arange(t_idx.shape[0]).unsqueeze(-1).to(t_idx)  # [bs, 1]
        t_repr = t_full[t_arange, t_idx].squeeze(1)   # [bs, R]
        t_score = None
        if self.linear is not None:
            t_score = self.linear(t_repr)  # [bs, C]
        return t_repr, t_score

    def reset_param(self, t_repr=None, t_gold=None, t_idx=None):
        if self.linear is not None:
            with torch.no_grad():
                self.linear.bias.zero_()  # zero it!
                _C = self.vocab_dim
                if t_idx is not None:
                    _weight = self.linear.weight
                    _weight.zero_()
                    t_arange = torch.arange(_C).to(_weight.device)
                    _weight[t_arange, t_idx.to(_weight.device)] = 1.
                else:
                    t_gold = t_gold.squeeze(-1)
                    t_hit = (torch.arange(_C).unsqueeze(-1).to(t_gold) == t_gold).to(t_repr)  # [C, N]
                    t_count = t_hit.sum(-1, keepdims=True)  # [C, 1]
                    if torch.any(t_count == 0):
                        zwarn(f"Get no repr for some entries: {t_count}")
                    t_weight = t_hit / t_count.clamp(min=1)  # [C, N]
                    t_avg = torch.matmul(t_weight, t_repr)  # [C, D]
                    t_avg = t_avg / math.sqrt(t_avg.shape[-1])  # 1/sqrt(D)
                    self.linear.weight.set_(t_avg)
                # breakpoint()
        else:
            zwarn("No output layer to set!!")

# --
# b mspx/znew/prompt/task/base:??
