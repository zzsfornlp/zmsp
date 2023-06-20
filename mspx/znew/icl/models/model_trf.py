#

# models from transformers

__all__ = [
    'NmTrfConf', 'NmTrf'
]

import torch
from mspx.data.inst import DataPadder
from mspx.utils import zlog, zwarn
from mspx.nn import NnConf, NnLayer, BK
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import *

@NnConf.rd('nm_trf')
class NmTrfConf(NewBaseModelConf):
    def __init__(self):
        super().__init__()
        # --
        self.model_name = 'gpt2'
        self.model_load_path = ''  # specific loading
        self.toker_name = None  # by default the same as model_name
        self.cache_dir = None
        self.gen_kwargs = {}  # kwargs for generation
        self.max_length = 1024  # max seq size
        # --
        # special ones
        self.auto_device_map = False
        self.load_in_8bit = False
        self.load_half = True
        # --

# --
# fixing problems
def fix_gpt2_attn_bias(model):
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    for mod in model.modules():
        if isinstance(mod, GPT2Attention):
            mod.register_buffer('bias', mod.bias.to(torch.bool))
# --

@NmTrfConf.conf_rd()
class NmTrf(NewBaseModel):
    def __init__(self, conf):
        super().__init__(conf)
        conf: NmTrfConf = self.conf
        # --
        self.model, self.toker = self.load_model_toker(conf)
        # --

    def load_model_toker(self, conf):
        other_kwargs = {}
        if conf.load_in_8bit:
            other_kwargs['load_in_8bit'] = True
        if conf.auto_device_map:
            other_kwargs['device_map'] = 'auto'
        if conf.load_half:
            other_kwargs['torch_dtype'] = torch.float16
        if 'llama' in conf.model_name:
            from transformers import LlamaForCausalLM, LlamaTokenizer
            AutoModelCls, AutoTokerCls = LlamaForCausalLM, LlamaTokenizer
        else:
            AutoModelCls, AutoTokerCls = AutoModelForCausalLM, AutoTokenizer
        # model
        if conf.model_load_path:
            zlog(f"Loading model {conf.model_name} from {conf.model_load_path}")
            state_dict = torch.load(conf.model_load_path)
            model = AutoModelCls.from_pretrained(conf.model_name, state_dict=state_dict, **other_kwargs)
        else:
            zlog(f"Creating model {conf.model_name} from {conf.cache_dir}")
            model = AutoModelCls.from_pretrained(conf.model_name, cache_dir=conf.cache_dir, **other_kwargs)
        # --
        # note: special fixing for gpt2 with 8bit!
        if conf.load_in_8bit and 'gpt2' in conf.model_name:
            fix_gpt2_attn_bias(model)
        # --
        if not conf.auto_device_map:
            model = model.to(BK.DEFAULT_DEVICE)
        # --
        # toker
        toker = AutoTokerCls.from_pretrained(conf.toker_name if conf.toker_name else conf.model_name, cache_dir=conf.cache_dir)
        if toker.pad_token is None:  # add padding!
            toker.pad_token = toker.eos_token
        # --
        return model, toker

    def make_logprob_inputs(self, inputs0, inputs1):
        toker = self.toker
        assert len(inputs0) == len(inputs1)
        # --
        all_toks, all_tids = [], []
        for t0, t1 in zip(inputs0, inputs1):
            l0, l1 = toker.encode(t0, add_special_tokens=False), toker.encode(t1, add_special_tokens=False)
            all_toks.append(l0 + l1)
            all_tids.append([0]*len(l0) + [1]*len(l1))
            # breakpoint()
        t_tok, t_mask = DataPadder.batch_2d(all_toks, self.toker.pad_token_id, ret_mask=True, ret_tensor=True)
        t_tid, _ = DataPadder.batch_2d(all_tids, 0, ret_tensor=True)
        return t_tok, t_mask, t_tid

    def run_logprob(self, inputs0, inputs1):
        t_tok, t_mask, t_tid = self.make_logprob_inputs(inputs0, inputs1)
        _logprobs = self.forward_logprob(t_tok, t_mask, t_tid)
        ret = []
        for a, b in zip(_logprobs.tolist(), t_tid[..., -_logprobs.size(-1):].tolist()):
            ret.append([v for v,flag in zip(a,b) if flag])
        return ret, None

    def forward_logprob(self, input_ids, attention_mask, token_type_ids):
        # --
        conf: NmTrfConf = self.conf
        if input_ids.size(-1) > conf.max_length:
            _size0 = input_ids.size(-1)
            input_ids, attention_mask, token_type_ids = input_ids[...,-conf.max_length:], attention_mask[...,-conf.max_length:], token_type_ids[...,-conf.max_length:]
            zwarn(f"Truncate since original length is larger than max_length: {_size0} > {conf.max_length}")
        # --
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[..., :-1, :].contiguous()  # [bs, L-1, V]
        _shape = list(logits.shape)
        _shape01 = _shape[0] * _shape[1]
        # --
        labels = input_ids[..., 1:].contiguous()  # [bs, L-1]
        label_mask = token_type_ids[..., 1:].contiguous().to(logits)  # [bs, L-1]
        _logprobs = logits.view([-1, _shape[-1]]).log_softmax(-1)[torch.arange(_shape01), labels.view(-1)]  # [bs * L-1]
        _logprobs = _logprobs.view(_shape[:2]) * label_mask  # [bs, L-1]
        return _logprobs

    def forward_gen(self, input_ids, attention_mask, **gen_kwargs):
        _final_gen_kwargs = self.conf.gen_kwargs.copy()
        _final_gen_kwargs.update(gen_kwargs)
        # --
        outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **_final_gen_kwargs)
        raise NotImplementedError("TODO(!)")

# --
# b mspx/znew/icl/models/model_trf:52
