#

# MLM

__all__ = [
    "ZTaskMlmConf", "ZTaskMlm", "ZDecoderMlmConf", "ZDecoderMlm",
]

from typing import List
import numpy as np
from collections import Counter
from msp2.data.inst import yield_sents, yield_sent_pairs
from msp2.data.vocab import SimpleVocab
from msp2.utils import AccEvalEntry, zlog, Constants
from msp2.proc import ResultRecord, DparEvalConf, DparEvaler
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.nn.modules import LossHelper
from .base import *
from .base_idec import *
from ..common import ZMediator, ZLabelConf, ZlabelNode
from ..enc import ZEncoder

# --

class ZTaskMlmConf(ZTaskDecConf):
    def __init__(self):
        super().__init__()
        self.name = "mlm"
        # --
        self.mlm_conf = ZDecoderMlmConf()

    def build_task(self):
        return ZTaskMlm(self)

class ZTaskMlm(ZTaskDec):
    def __init__(self, conf: ZTaskMlmConf):
        super().__init__(conf)
        conf: ZTaskMlmConf = self.conf
        # --

    # no need for any preparations!
    def build_vocab(self, datasets: List): return None
    def prep_inst(self, inst, dataset): pass
    def prep_item(self, item, dataset): pass

    def build_mod(self, model):
        return ZDecoderMlm(self.conf.mlm_conf, self, model.encoder)

# --

class ZDecoderMlmConf(ZDecoderConf):
    def __init__(self):
        super().__init__()
        # --
        self.idec_mlm = IdecConf.make_conf('score')
        self.idec_mlm.node.conn.do_dsel = False  # on the enc-seq!
        # note: do the softmax here! no fix-nil!
        self.lab_mlm = ZLabelConf().direct_update(emb_size=768, input_act='elu', loss_do_sel=True)
        self.loss_mlm = 1.
        # mlm specific
        self.mlm_use_input_embed = False  # share input embed as output
        self.mlm_mrate = 0.15  # how much to mask?
        self.mlm_repl_rates = [0.8, 0.1, 0.1]  # rates of: [MASK], random, unchange

    def get_repl_ranges(self):
        _arr = np.asarray(self.mlm_repl_rates)
        _a, _b, _c = (_arr/_arr.sum()).cumsum().tolist()
        assert _c == 1.
        return _a, _b

@node_reg(ZDecoderMlmConf)
class ZDecoderMlm(ZDecoder):
    def __init__(self, conf: ZDecoderMlmConf, ztask, main_enc: ZEncoder, **kwargs):
        super().__init__(conf, ztask, main_enc, **kwargs)
        conf: ZDecoderMlmConf = self.conf
        # --
        # mlm
        _enc_dim, _head_dim = main_enc.get_enc_dim(), main_enc.get_head_dim()
        # --
        _W = main_enc.get_embed_w()  # get input embeddings: [nword, D]
        self.target_size = BK.get_shape(_W, 0)
        self.mask_token_id = main_enc.tokenizer.mask_token_id  # note: specific one!!
        self.repl_ranges = conf.get_repl_ranges()
        # --
        self.lab_mlm = ZlabelNode(conf.lab_mlm, _csize=self.target_size)
        self.idec_mlm = conf.idec_mlm.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=self.lab_mlm.get_core_csize())
        self.reg_idec('mlm', self.idec_mlm)
        if conf.mlm_use_input_embed:
            zlog(f"Use input embed of {_W.T.shape} for output!")
            self.lab_mlm.aff_final.put_external_ws([(lambda: _W.T)])
        # --

    def prep_enc(self, med: ZMediator, *args, **kwargs):
        conf: ZDecoderMlmConf = self.conf
        # note: we have to use enc-mask
        # todo(+W): currently do mlm for full seq regardless of center or not!
        sinfo = med.ibatch.seq_info
        enc_ids, enc_mask = sinfo.enc_input_ids, sinfo.enc_input_masks  # [*, elen]
        _shape = enc_ids.shape
        # sample mask
        mlm_mask = (BK.rand(_shape) < conf.mlm_mrate).float() * enc_mask  # [*, elen]
        # sample repl
        _repl_sample = BK.rand(_shape)  # [*, elen], between [0, 1)
        mlm_repl_ids = BK.constants_idx(_shape, self.mask_token_id)  # [*, elen] [MASK]
        _repl_rand, _repl_origin = self.repl_ranges
        mlm_repl_ids = BK.where(_repl_sample>_repl_rand, (BK.rand(_shape)*self.target_size).long(), mlm_repl_ids)
        mlm_repl_ids = BK.where(_repl_sample>_repl_origin, enc_ids, mlm_repl_ids)
        # final prepare
        mlm_input_ids = BK.where(mlm_mask>0., mlm_repl_ids, enc_ids)  # [*, elen]
        med.set_cache('eff_input_ids', mlm_input_ids)
        med.set_cache('mlm_mask', mlm_mask)
        # --

    def loss(self, med: ZMediator, *args, **kwargs):
        conf: ZDecoderMlmConf = self.conf
        # --
        loss_items = []
        if conf.loss_mlm > 0.:
            origin_ids = med.ibatch.seq_info.enc_input_ids
            mlm_mask = med.get_cache('mlm_mask')
            loss_items.extend(self.loss_from_lab(self.lab_mlm, 'mlm', med, origin_ids, mlm_mask, conf.loss_mlm))
        # --
        ret_loss = LossHelper.combine_multiple_losses(loss_items)
        return ret_loss, {}

    def predict(self, med: ZMediator, *args, **kwargs):
        raise NotImplementedError("Currently no need for predict!")

# --
# b msp2/tasks/zmtl2/zmod/dec/dec_mlm:??
