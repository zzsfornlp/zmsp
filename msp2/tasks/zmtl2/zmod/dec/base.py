#

# the base of dec

__all__ = [
    "ZTaskDecConf", "ZTaskDec", "ZDecoderConf", "ZDecoder",
]

from typing import List
from collections import OrderedDict
from msp2.nn.layers import *
from msp2.nn.modules import LossHelper
from msp2.nn import BK
from msp2.utils import ConfEntryChoices, zlog, StrHelper
from msp2.proc import ResultRecord
from ...core import ZMod, ZModConf, ZTask, ZTaskConf
from ..common import ZMediator
from ..enc import ZEncoder
from .base_idec import *

# =====
# the overall decoder

class ZTaskDecConf(ZTaskConf):
    def __init__(self):
        super().__init__()
        # --

class ZTaskDec(ZTask):
    def __init__(self, conf: ZTaskConf):
        super().__init__(conf)
        # --

    @staticmethod
    def do_eval(name: str, evaler, gold_insts: List, pred_insts: List, quite: bool):
        evaler.reset()
        res0 = evaler.eval(gold_insts, pred_insts)
        res = ResultRecord(results=res0.get_summary(), description=res0.get_brief_str(), score=float(res0.get_result()))
        if not quite:
            res_detailed_str0 = res0.get_detailed_str()
            res_detailed_str = StrHelper.split_prefix_join(res_detailed_str0, '\t', sep='\n')
            zlog(f"{name} detailed results:\n{res_detailed_str}", func="result")
        return res

class ZDecoderConf(ZModConf):
    def __init__(self):
        super().__init__()
        # --
        # the adapter idecs
        self.idec_ff = ConfEntryChoices({"yes": IdecConf.make_conf('ff'), "no": None}, "no")
        self.idec_satt = ConfEntryChoices({"yes": IdecConf.make_conf('satt'), "no": None}, "no")
        # --
        # how to deal with msent in training and testing (for some decoders)
        self.msent_loss_center = False
        self.msent_pred_center = False
        # --

@node_reg(ZDecoderConf)
class ZDecoder(ZMod):
    def __init__(self, conf: ZDecoderConf, ztask, main_enc: ZEncoder, **kwargs):
        super().__init__(conf, ztask, **kwargs)
        conf: ZDecoderConf = self.conf
        # --
        self._active_output = False  # for the upcoming runs, whether this task is activated
        self.idecs: OrderedDict[str, IdecNode] = OrderedDict()# --
        self.max_app_lidx = -1  # by default -1
        # --
        _isize, _nhead = main_enc.get_enc_dim(), main_enc.get_head_dim()
        self.idec_ff = None if conf.idec_ff is None else conf.idec_ff.make_node(_isize=_isize, _nhead=_nhead)
        self.idec_satt = None if conf.idec_satt is None else conf.idec_satt.make_node(_isize=_isize, _nhead=_nhead)
        self.reg_idec('ff', self.idec_ff)
        self.reg_idec('satt', self.idec_satt)
        # --

    def set_activate_output(self, flag: bool):
        self._active_output = flag

    @property
    def activate_output(self):  # for loss/pred/score/...
        return self._active_output

    # internal usage
    def reg_idec(self, name: str, idec: IdecNode):
        if idec is None:
            return
        assert name not in self.idecs
        self.idecs[name] = idec
        self.max_app_lidx = max(self.max_app_lidx, idec.max_app_lidx)  # note: update here!
        # --

    # forwards after each layer
    def layer_end(self, med: ZMediator):
        lidx = med.lidx
        activate_output = self.activate_output
        rets = []
        for name, idec in self.idecs.items():
            if idec.has_layer(lidx):
                # note: if forced_feed, then this idec would already be part of the whole model, thus must feed!
                if activate_output or idec.has_feed(lidx):
                    scores_t, feeds_t = idec.forward(med)
                    if scores_t is not None:  # store the score if we have one!
                        med.set_cache((self.name, name), scores_t, app=True, app_info=lidx)
                    rets.append(feeds_t)  # add feeds for ret
        # --
        return rets, (lidx >= self.max_app_lidx)

    # ==
    # to be implemented

    def prep_enc(self, med: ZMediator, *args, **kwargs):  # sth to prepare before enc?
        pass  # by default nothing!

    def loss(self, med: ZMediator, *args, **kwargs):
        raise NotImplementedError()

    def predict(self, med: ZMediator, *args, **kwargs):
        raise NotImplementedError()

    def score(self, med: ZMediator, *args, **kwargs):
        raise NotImplementedError()

    # --
    # common routines

    def get_dec_mask(self, ibatch, center_only: bool):
        if center_only:
            center_idxes = BK.input_idx([z.center_sidx for z in ibatch.items]).unsqueeze(-1)  # [bs, 1]
            ret_mask = (ibatch.seq_info.dec_sent_idxes == center_idxes).float()  # [bs, dlen]
        else:  # otherwise, simply further exclude CLS/PAD
            ret_mask = (ibatch.seq_info.dec_sent_idxes >= 0).float()  # [*, dlen]
        # ret_mask *= ibatch.seq_info.dec_sel_masks  # [*, dlen], note: no need for this
        return ret_mask

    def loss_from_lab(self, lab_node, score_name: str, med: ZMediator, label_t, mask_t, loss_lambda: float,
                      loss_neg_sample: float = None):
        score_cache = med.get_cache((self.name, score_name))
        loss_items = []
        # note: simply collect them all
        all_losses = lab_node.gather_losses(score_cache.vals, label_t, mask_t, loss_neg_sample=loss_neg_sample)
        for ii, vv in enumerate(all_losses):
            nn = score_cache.infos[ii]
            _loss_t, _mask_t = vv
            _loss_item = LossHelper.compile_leaf_loss(
                f'{score_name}_{nn}', _loss_t.sum(), _mask_t.sum(), loss_lambda=loss_lambda)
            loss_items.append(_loss_item)
        return loss_items
