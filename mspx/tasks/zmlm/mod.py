#

# masked lm

__all__ = [
    "ZTaskMlmConf", "ZTaskMlm", "ZModMlmConf", "ZModMlm",
]

import numpy as np
from typing import List
from mspx.utils import MathHelper, zlog, ZResult
from mspx.nn import BK, ZTaskConf, ZModConf, ZTaskSbConf, ZTaskSb, ZModSbConf, ZModSb, ZRunCache

# --

@ZTaskConf.rd('mlm')
class ZTaskMlmConf(ZTaskSbConf):
    def __init__(self):
        super().__init__()
        # --
        self.mod = ZModMlmConf()
        self.eval = None
        # --

@ZTaskMlmConf.conf_rd()
class ZTaskMlm(ZTaskSb):
    def __init__(self, conf: ZTaskMlmConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZTaskMlmConf = self.conf
        # --
        self._eval_res = {}  # note: for simplicity, we accumulate while testing!
        self.eval_reset()
        # --

    def eval_reset(self):
        self._eval_res = {'loss': 0., 'inst': 0, 'toks': 0, 'count': 0, 'corr1': 0, 'corrA': 0}

    def eval_record(self, pred_insts: List):
        res = self._eval_res
        for inst in pred_insts:
            res['inst'] += 1
            res['toks'] += len(inst)
            for item in inst.info['mlm_info'].values():
                if item is not None:
                    res['loss'] += item['loss']
                    res['count'] += 1
                    res['corr1'] += int(item['hit1'])
                    res['corrA'] += int(item['hitA'])
        # --

    def eval_insts(self, pred_insts: List, gold_insts: List, quite=False):
        res = self._eval_res
        res['avg_loss'] = MathHelper.safe_div(res['loss'], res['count'])
        res['acc1'] = MathHelper.safe_div(res['corr1'], res['count'])
        res['accA'] = MathHelper.safe_div(res['corrA'], res['count'])
        if not quite:
            zlog(f"=>Result of mlm_eval: {res}")
        ret = ZResult(res, res=-res['avg_loss'])
        self.eval_reset()  # note: reset here!
        return ret

@ZModConf.rd('mlm')
class ZModMlmConf(ZModSbConf):
    def __init__(self):
        super().__init__('bmod2')  # note: use our own by default
        # --
        self.label_smooth = 0.  # label smoothing
        # mlm specific
        self.mlm_mrate = [0.15, 0.15]  # how much to mask?
        self.mlm_repl_rates = [0.8, 0.1, 0.1]  # rates of: [MASK], random, unchanged
        # --

    def get_repl_ranges(self):
        _arr = np.asarray(self.mlm_repl_rates)
        _a, _b, _c = (_arr/_arr.sum()).cumsum().tolist()
        assert _c == 1.
        return _a, _b

@ZModMlmConf.conf_rd()
class ZModMlm(ZModSb):
    def __init__(self, conf: ZModMlmConf, ztask: ZTaskMlm, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZModMlmConf = self.conf
        # --
        self.repl_ranges = conf.get_repl_ranges()
        self.target_size = len(self.bmod.toker)
        # --

    def do_prep(self, rc: ZRunCache, *args, **kwargs):
        conf: ZModMlmConf = self.conf
        # --
        toker = self.bmod.toker
        ID_MASK, ID_CLS, ID_SEP = toker.mask_token_id, toker.cls_token_id, toker.sep_token_id
        # --
        t_ids, t_mask, _, _ = self.prepare_ibatch(rc.ibatch, False)
        _shape = t_ids.shape  # [*, L]
        # sample mask
        _m1, _m2 = conf.mlm_mrate  # allow a range
        mlm_mask = ((BK.rand(_shape) < (_m1+BK.rand(_shape)*(_m2-_m1))) & (t_ids != ID_CLS) &
                    (t_ids != ID_SEP)).to(BK.DEFAULT_FLOAT) * t_mask  # [*, elen]
        # sample repl
        _repl_sample = BK.rand(_shape)  # [*, L], between [0, 1)
        mlm_repl_ids = BK.constants(_shape, ID_MASK, dtype=BK.DEFAULT_INT)  # [*, elen] [MASK]
        _repl_rand, _repl_origin = self.repl_ranges
        mlm_repl_ids = BK.where(_repl_sample>_repl_rand, (BK.rand(_shape, dtype=BK.float32)*self.target_size).long(), mlm_repl_ids)
        mlm_repl_ids = BK.where(_repl_sample>_repl_origin, t_ids, mlm_repl_ids)
        # final prepare
        mlm_input_ids = BK.where(mlm_mask>0., mlm_repl_ids, t_ids)  # [*, elen]
        # --
        rc.set_cache((self.name, 'input'), (t_ids, t_mask, mlm_input_ids, mlm_mask))
        # --

    def calc_output(self, rc: ZRunCache):
        t_ids, t_mask, mlm_input_ids, mlm_mask = rc.get_cache((self.name, 'input'))
        bout = self.bmod.forward_enc(mlm_input_ids, t_mask=t_mask)  # note: input masked ones!
        t_hid = self.bout.comb_hid(bout)  # [bs, L, D]
        # --
        valid_t = (mlm_mask > 0.)
        flatten_hid = t_hid[valid_t]  # [??, D]
        flatten_trgs = t_ids[valid_t]  # [??]
        flatten_output = self.bmod.forward_lmhead(flatten_hid)
        return mlm_mask, flatten_trgs, flatten_output

    def do_loss(self, rc: ZRunCache, *args, **kwargs):
        conf: ZModMlmConf = self.conf
        # --
        mlm_mask, flatten_trgs, flatten_output = self.calc_output(rc)
        loss_t = BK.loss_nll(flatten_output, flatten_trgs, label_smoothing=conf.label_smooth)  # [??]
        one_loss = self.compile_leaf_loss('out', loss_t.sum(), BK.input_real(len(loss_t)))
        ret = self.compile_losses([one_loss])
        return ret, {}

    def do_predict(self, rc: ZRunCache, *args, **kwargs):
        toker = self.bmod.toker
        # --
        mlm_mask, flatten_trgs, flatten_output = self.calc_output(rc)
        loss_t = BK.loss_nll(flatten_output, flatten_trgs)  # [??]
        prob_t = flatten_output.softmax(-1)  # [??, V]
        topk_probs_t, topk_idxes_t = prob_t.topk(5, dim=-1)  # [??, K]
        hit1_t, hitA_t = (topk_idxes_t[...,0]==flatten_trgs), ((topk_idxes_t==flatten_trgs.unsqueeze(-1)).sum(-1)>0)
        # assign output
        arr_mlm_mask, arr_loss, arr_topk_probs, arr_topk_idxes, arr_hit1, arr_hitA = \
            [BK.get_value(z) for z in (mlm_mask, loss_t, topk_probs_t, topk_idxes_t, hit1_t, hitA_t)]
        ii = 0
        for bidx, inst in enumerate(rc.ibatch.items):
            sent = inst.sent
            mlm_info = {}
            for tidx, mm in enumerate(arr_mlm_mask[bidx]):
                if mm > 0.:
                    mlm_info[tidx] = {
                        'loss': arr_loss[ii].item(), 'topk_toks': toker.convert_ids_to_tokens(arr_topk_idxes[ii]),
                        'topk_probs': arr_topk_probs[ii].tolist(),
                        'hit1': arr_hit1[ii].item(), 'hitA': arr_hitA[ii].item(),
                    }
                    ii += 1
            sent.info['mlm_info'] = mlm_info
        assert ii == len(arr_loss)
        self.ztask.eval_record([z.sent for z in rc.ibatch.items])
        # --
        return {}

# --
# b mspx/tasks/zmlm/mod:109
