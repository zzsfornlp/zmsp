#

# classification

__all__ = [
    "ZTaskClfConf", "ZTaskClf", "ZModClfConf", "ZModClf",
]

from mspx.data.inst import yield_sents, get_label_gs
from mspx.data.vocab import Vocab
from mspx.proc.eval import ClfEvalConf
from mspx.utils import zlog
from mspx.nn import BK, ZTaskConf, ZModConf, ZTaskSbConf, ZTaskSb, ZModSbConf, ZModSb, MlpConf, ZRunCache

# --

@ZTaskConf.rd('clf')
class ZTaskClfConf(ZTaskSbConf):
    def __init__(self):
        super().__init__()
        self.mod = ZModClfConf()
        self.eval = ClfEvalConf()
        self.lab_f = "_info:label"
        self.do_regression = False  # target is regression

    def _validate(self):  # overwrite eval's
        self.eval.trg_f = self.lab_f
        self.eval.do_regr = self.do_regression

@ZTaskClfConf.conf_rd()
class ZTaskClf(ZTaskSb):
    def __init__(self, conf: ZTaskClfConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZTaskClfConf = self.conf
        # --
        self.lab_getter, self.lab_setter = get_label_gs(conf.lab_f)

    def build_vocab(self, datasets):
        conf: ZTaskClfConf = self.conf
        # --
        if conf.do_regression:
            return None  # no need to build!
        else:
            voc = Vocab.build_empty(f"voc_{self.name}")
            for dataset in datasets:
                if dataset.name.startswith('train'):
                    for sent in yield_sents(dataset.yield_insts()):
                        lab = self.lab_getter(sent)
                        voc.feed_one(lab)
            voc.build_sort()
            zlog(f"Finish building for: {voc}")
            return (voc, )
        # --

@ZModConf.rd('clf')
class ZModClfConf(ZModSbConf):
    def __init__(self):
        super().__init__('bmod2')
        self.scorer = MlpConf()
        self.label_smooth = 0.  # label smoothing

@ZModClfConf.conf_rd()
class ZModClf(ZModSb):
    def __init__(self, conf: ZModClfConf, ztask: ZTaskClf, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZModClfConf = self.conf
        # --
        _ztconf = ztask.conf
        self.do_regression = _ztconf.do_regression
        if _ztconf.do_regression:
            self.voc = None
            _osize = 1
        else:
            self.voc = ztask.vpack[0]
            _osize = len(self.voc)
        _ssize = self.bout.dim_out_hid()
        self.scorer = conf.scorer.make_node(isize=_ssize, osize=_osize)

    def _forw_logits(self, rc: ZRunCache):
        t_hid = self._do_forward(rc)
        t_hid_aggr = t_hid[:, 0]  # simply use CLS
        t_logits = self.scorer(t_hid_aggr)
        return t_logits

    def do_loss(self, rc: ZRunCache, *args, **kwargs):
        t_logits = self._forw_logits(rc)  # [bs, V]
        label_getter = self.ztask.lab_getter
        trg_labs = [label_getter(z.inst) for z in rc.ibatch.items]
        if self.do_regression:
            t_gold = BK.input_real(trg_labs)  # [bs]
            loss_t = (t_gold - t_logits.squeeze(-1)) ** 2
        else:
            t_gold = BK.input_idx(self.voc.seq_word2idx(trg_labs))  # [bs]
            loss_t = BK.loss_nll(t_logits, t_gold, label_smoothing=self.conf.label_smooth)  # [bs]
        one_loss = self.compile_leaf_loss('lab', loss_t.sum(), BK.input_real(len(loss_t)))
        ret = self.compile_losses([one_loss])
        return (ret, {})

    def do_predict(self, rc: ZRunCache, *args, **kwargs):
        label_setter = self.ztask.lab_setter
        t_logits = self._forw_logits(rc)  # [bs, V]
        if not BK.is_zero_shape(t_logits):
            if self.do_regression:
                list_preds = BK.get_value(t_logits).tolist()
                for ii, item in enumerate(rc.ibatch.items):
                    label_setter(item.inst, list_preds[ii])
            else:
                _, t_preds = t_logits.max(-1)
                list_preds = BK.get_value(t_preds).tolist()
                list_labs = self.voc.seq_idx2word(list_preds)
                for ii, item in enumerate(rc.ibatch.items):
                    label_setter(item.inst, list_labs[ii])
        # --
        return {}

# --
# b mspx/tasks/zclf/mod:??
