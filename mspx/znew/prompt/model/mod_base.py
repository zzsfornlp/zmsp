#
import os

# base models from transformers

import torch
from collections import defaultdict
from mspx.nn import BK
from mspx.utils import Conf, zlog, get_global_conf, zglob1

class BaseModConf(Conf):
    def __init__(self):
        # basic
        self.model_name = 'gpt2'  # model's name
        self.model_type = 'AUTO'  # clm/s2s/mlm
        self.model_load_path = ''  # specific loading
        self.toker_name = None  # by default the same as model_name
        # loading specific
        self.auto_device_map = True
        self.load_in_8bit = False
        self.load_half = False
        # peft
        self.peft = PeftConf()

def load_model_toker(conf: BaseModConf, **kwargs):
    if kwargs:
        conf = BaseModConf.direct_conf(conf, copy=True, **kwargs)
    cache_dir = zglob1(get_global_conf(['utils', 'global_cache_dir']))
    # --
    # auto judge "model_type"
    model_helper = ModelHelper(conf.model_name)
    if conf.model_type == "AUTO":
        model_type = model_helper.model_type
    else:
        model_type = conf.model_type
    # --
    other_kwargs = {}
    if conf.load_in_8bit:
        other_kwargs['load_in_8bit'] = True
    if conf.auto_device_map:
        if torch.cuda.is_available():
            other_kwargs['device_map'] = 'auto'
        else:  # all on cpu!
            other_kwargs['device_map'] = {'':'cpu'}
    if conf.load_half:
        other_kwargs['torch_dtype'] = torch.float16
    if 'llama' in conf.model_name:
        from transformers import LlamaForCausalLM, LlamaTokenizer
        assert model_type == 'clm'
        AutoModelC, AutoTokerC = LlamaForCausalLM, LlamaTokenizer
    else:
        from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, AutoTokenizer
        AutoModelC, AutoTokerC = {'clm': AutoModelForCausalLM, 'mlm': AutoModelForMaskedLM, 's2s': AutoModelForSeq2SeqLM}[model_type], AutoTokenizer
    # model
    zlog(f"Creating model {conf.model_name}[type={model_type}] from {cache_dir}")
    model = AutoModelC.from_pretrained(conf.model_name, cache_dir=cache_dir, **other_kwargs)
    # --
    # peft
    if conf.peft.peft_type:
        model = wrap_peft(model, model_type, conf.peft)
    # --
    # note: special fixing for gpt2 with 8bit!
    if conf.load_in_8bit and 'gpt2' in conf.model_name:
        fix_gpt2_attn_bias(model)
    if not conf.auto_device_map:
        model = model.to(BK.DEFAULT_DEVICE)
    zlog(f"Obtain model of {type(model)}: \n{info_trainable_parameters(model).to_string()}")
    if conf.model_load_path:  # further loading
        _weights = torch.load(conf.model_load_path)
        model.load_state_dict(_weights, strict=False)
        zlog(f"Loaded weights from {conf.model_load_path} ...")
    # --
    # toker
    toker_name = conf.toker_name if conf.toker_name else conf.model_name
    if 't5-v1_1' in toker_name:  # todo(!): strange error with regard to protobuf
        os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
    toker = AutoTokerC.from_pretrained(toker_name, cache_dir=cache_dir)
    if toker.pad_token is None:  # add padding!
        toker.pad_token = toker.eos_token
    if toker.cls_token is None:  # add [bos]
        toker.cls_token = toker.bos_token
    if toker.sep_token is None:  # add [sep]
        toker.sep_token = toker.eos_token
    # --
    return model_helper, model, toker

# --
# fixing problems
def fix_gpt2_attn_bias(model):
    from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
    for mod in model.modules():
        if isinstance(mod, GPT2Attention):
            mod.register_buffer('bias', mod.bias.to(torch.bool))
# --

class PeftConf(Conf):
    def __init__(self):
        self.peft_type = ''
        self.peft_kwargs = {}
        # --
        self.lora_r = 8
        self.lora_alpha = 16
        self.lora_dropout = 0.1

def wrap_peft(model, model_type, conf: PeftConf):
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
    # todo(+N): simply reuse CAUSAL_LM for mlm for now
    task_type = {'clm': TaskType.CAUSAL_LM, 'mlm': TaskType.CAUSAL_LM, 's2s': TaskType.SEQ_2_SEQ_LM}[model_type]
    peft_kwargs = conf.peft_kwargs.copy()
    if conf.peft_type == 'lora':
        peft_kwargs.update({'r': conf.lora_r, 'lora_alpha': conf.lora_alpha, 'lora_dropout': conf.lora_dropout})
        peft_config = LoraConfig(task_type=task_type, **peft_kwargs)
    else:
        raise NotImplementedError()
    ret = get_peft_model(model, peft_config)
    return ret

def info_trainable_parameters(model):
    import pandas as pd
    # --
    def _get_key(_name):
        _fields = _name.split(".")
        while len(_fields) > 0 and _fields[-1] in ['weight', 'bias']:
            _fields.pop()
        return _fields[-1] if len(_fields)>0 else 'UNK'
    # --
    info_all, info_trainable = defaultdict(int), defaultdict(int)
    for param_name, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        # --
        keys = ['ALL', _get_key(param_name)]
        for kk in keys:
            info_all[kk] += num_params
            if param.requires_grad:
                info_trainable[kk] += num_params
    # --
    all_keys = sorted(info_all.keys(), key=(lambda x: -info_all[x]))
    data = pd.DataFrame({'all': [info_all[z] for z in all_keys], 'trainable': [info_trainable[z] for z in all_keys]}, index=all_keys)
    data['perc'] = data['trainable'] / data['all']
    return data

# --
class ModelHelper:
    INFO = {  # information table
        'gpt': {
            'model_type': 'clm', 'surroundings': [0, 0],
        },
        'pythia': {
            'model_type': 'clm', 'surroundings': [0, 0],
        },
        'bert': {
            'model_type': 'mlm', 'surroundings': [1, 1],
        },
        't5': {
            'model_type': 's2s', 'surroundings': [0, 1],
        },
        'llama': {
            'model_type': 'clm', 'surroundings': [1, 0],
        },
    }

    def __init__(self, name):
        self.name = name
        _name = name.split("/")[-1]
        self.normed_name = None
        self.info = None
        for k, v in self.INFO.items():
            if k in name:
                self.normed_name = k
                self.info = v
                break
        assert self.normed_name is not None
        # --

    @property
    def model_type(self):
        return self.info['model_type']

    @property
    def surroundings(self):
        return self.info['surroundings']

# --
# b mspx/znew/prompt/model/base:
