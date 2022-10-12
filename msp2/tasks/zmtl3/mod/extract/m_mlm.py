#

# module of MLM

__all__ = [
    "ZTaskMlmConf", "ZTaskMlm", "ZModMlmConf", "ZModMlm",
]

import math
from typing import List
from collections import Counter
import numpy as np
from msp2.proc import FrameEvalConf, FrameEvaler, ResultRecord
from msp2.data.inst import yield_sents, yield_frames, set_ee_heads, DataPadder, SimpleSpanExtender, Sent
from msp2.utils import zlog
from msp2.nn import BK
from msp2.nn.l3 import *
from .base import *

class ZTaskMlmConf(ZTaskBaseEConf):
    def __init__(self, name: str = None):
        super().__init__()
        self.name = name if name is not None else 'mlm'
        self.mlm_conf = ZModMlmConf()
        # --

    def build_task(self):
        return ZTaskMlm(self)

class ZTaskMlm(ZTaskBaseE):
    def __init__(self, conf: ZTaskMlmConf):
        super().__init__(conf)
        conf: ZTaskMlmConf = self.conf
        # --

    def build_vocab(self, datasets: List):
        return None  # no need to build since directly from pre-trained vocabs

    def eval_insts(self, gold_insts: List, pred_insts: List, quite=False):
        _losses0 = [z.info.get('loss_mlm', 0) for z in yield_sents(pred_insts)]
        _losses1 = [z for z in _losses0 if z>0.]
        avg_nloss = (- np.mean(_losses1).item()) if len(_losses1)>0 else 0.
        zlog(f"{self.name} detailed results:\n\tloss={avg_nloss}", func="result")
        res = ResultRecord(results={'nloss': avg_nloss, 'size0': len(_losses0), 'size1': len(_losses1)},
                           description='', score=avg_nloss)
        return res

    # build mod
    def build_mod(self, model):
        return self.conf.mlm_conf.make_node(self, model)

# --
class ZModMlmConf(ZModBaseEConf):
    def __init__(self):
        super().__init__()
        # --
        self.mlm_mrate = 0.15  # how much to mask?
        self.mlm_repl_ranges = [0.8, 0.9]  # cumsum: [MASK], random, remaining unchanged!
        # --

@node_reg(ZModMlmConf)
class ZModMlm(ZModBaseE):
    def __init__(self, conf: ZModMlmConf, ztask, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZModMlmConf = self.conf
        # --

    def do_loss(self, rc: ZRunCache, *args, **kwargs): return self.do_forward(rc.ibatch, is_testing=False)
    def do_predict(self, rc: ZRunCache, *args, **kwargs): return self.do_forward(rc.ibatch, is_testing=True)

    def do_forward(self, ibatch, is_testing: bool):
        conf: ZModMlmConf = self.conf
        all_sents = [z.sent for z in ibatch.items]
        # --
        _tokenizer = self.tokenizer
        _pad_id, _cls_id, _sep_id, _mask_id = \
            _tokenizer.pad_token_id, _tokenizer.cls_token_id, _tokenizer.sep_token_id, _tokenizer.mask_token_id
        _list_ids = [[_cls_id] + sum(self.prep_sent(s), []) + [_sep_id] for s in all_sents]
        _arr_ids = DataPadder.go_batch_2d(_list_ids, _pad_id)
        _ids_t = BK.input_idx(_arr_ids)  # [?bs, Lf0]
        _ids_mask_t = (_ids_t != _pad_id).float()
        # -- prepare mlm
        _shape = _ids_t.shape
        # sample mask
        mlm_mask = ((BK.rand(_shape) < conf.mlm_mrate) & (_ids_t != _cls_id) & (_ids_t != _sep_id)).float() \
                   * _ids_mask_t  # [*, elen]
        # sample repl
        _repl_sample = BK.rand(_shape)  # [*, elen], between [0, 1)
        mlm_repl_ids = BK.constants_idx(_shape, _mask_id)  # [*, elen] [MASK]
        _repl_rand, _repl_origin = conf.mlm_repl_ranges
        mlm_repl_ids = BK.where(_repl_sample > _repl_rand, (BK.rand(_shape) * _tokenizer.vocab_size).long(),
                                mlm_repl_ids)
        mlm_repl_ids = BK.where(_repl_sample > _repl_origin, _ids_t, mlm_repl_ids)
        mlm_input_ids = BK.where(mlm_mask > 0., mlm_repl_ids, _ids_t)  # [*, elen]
        # forward & loss
        bert_out = self.bmod.forward_enc(mlm_input_ids, _ids_mask_t)
        t_out = self.bmod.forward_lmhead(bert_out.last_hidden_state)  # [*, elen, V]
        t_loss = BK.loss_nll(t_out, _ids_t)  # [bs, elen]
        loss_item = LossHelper.compile_leaf_loss("mlm", (t_loss * mlm_mask).sum(), mlm_mask.sum(), loss_lambda=1.)
        if is_testing:
            t_sloss = (t_loss * mlm_mask).sum(-1) / mlm_mask.sum(-1).clamp(min=1.)  # [bs]
            l_sloss = BK.get_value(t_sloss).tolist()
            for bidx, evt in enumerate(all_sents):
                evt.info['loss_mlm'] = l_sloss[bidx]
            return {}
        else:
            ret = LossHelper.compile_component_loss(self.name, [loss_item])
            return ret, {}
        # --
