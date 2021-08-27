#

# UDEP

__all__ = [
    "ZTaskUdep", "ZTaskUdepConf", "ZDecoderUdepConf", "ZDecoderUdep",
]

from typing import List
import numpy as np
from collections import Counter
from msp2.data.inst import yield_sents, yield_sent_pairs
from msp2.data.vocab import SimpleVocab
from msp2.utils import AccEvalEntry, zlog, Constants, zwarn, ConfEntryChoices
from msp2.proc import ResultRecord, DparEvalConf, DparEvaler
from msp2.nn import BK
from msp2.nn.layers import *
from msp2.nn.modules import LossHelper
from .base import *
from .base_idec import *
from .base_idec2 import *
from ..common import ZMediator, ZLabelConf, ZlabelNode
from ..enc import ZEncoder

# --

class ZTaskUdepConf(ZTaskDecConf):
    def __init__(self):
        super().__init__()
        self.name = "udep"  # a default name
        # --
        # eval
        self.udep_eval: DparEvalConf = DparEvalConf().direct_update(deplab_l1=True)
        self.use_l1 = True  # process to only lookat L1
        self.udep_pred_clear = True  # clear exiting ones for all
        self.udep_prep_all = False
        # model
        self.udep_conf = ZDecoderUdepConf()
        # --

    def build_task(self):
        return ZTaskUdep(self)

class ZTaskUdep(ZTaskDec):
    def __init__(self, conf: ZTaskUdepConf):
        super().__init__(conf)
        conf: ZTaskUdepConf = self.conf
        # --
        self.evaler = DparEvaler(conf.udep_eval)
        # --

    # build vocab (simple gather all)
    def build_vocab(self, datasets: List):
        conf: ZTaskUdepConf = self.conf
        # --
        voc_udep = SimpleVocab.build_empty(self.name)
        for dataset in datasets:
            for sent in yield_sents(dataset.insts):
                _vals = sent.tree_dep.seq_label.vals
                if conf.use_l1:
                    _vals = [z.split(":")[0] for z in _vals]
                voc_udep.feed_iter(_vals)
        voc_udep.build_sort()
        _, udep_direct_range = voc_udep.non_special_range()  # range of direct labels
        zlog(f"Finish building voc_udep: {voc_udep}")
        return (voc_udep, udep_direct_range)

    # prepare one instance
    def prep_inst(self, inst, dataset):
        conf: ZTaskUdepConf = self.conf
        wset = dataset.wset
        # --
        if wset == "train" or conf.udep_prep_all:
            voc_udep, udep_direct_range = self.vpack
            for sent in yield_sents(inst):
                _tree = sent.tree_dep
                if _tree is None:
                    continue
                _vals = _tree.seq_label.vals
                if conf.use_l1:
                    _vals = [z.split(":")[0] for z in _vals]
                idxes_labs = [voc_udep.get_else_unk(z) for z in _vals]
                _tree.seq_label.set_idxes(idxes_labs)
                # note: refresh the cache
                _mat = _tree.label_matrix  # [m,h]
        elif conf.udep_pred_clear:  # clear if there are
            for sent in yield_sents(inst):
                sent.build_dep_tree([0]*len(sent), ["UNK"]*len(sent))
        # --

    # prepare one input_item
    def prep_item(self, item, dataset):
        pass  # leave to the mod to handle!!

    # eval
    def eval_insts(self, gold_insts: List, pred_insts: List, quite=False):
        return ZTaskDec.do_eval(self.name, self.evaler, gold_insts, pred_insts, quite)

    # build mod
    def build_mod(self, model):
        return ZDecoderUdep(self.conf.udep_conf, self, model.encoder)

# --

# --
# for special label filters (include both v1 and v2)
UD_CATEGORIES={
    "CoreN": ["nsubj", "obj", "iobj"] + ["nsubjpass", "dobj"],  # Nominals
    "CoreC": ["csubj", "ccomp", "xcomp"] + ["csubjpass"],  # Clauses
    "NCoreN": ["obl", "vocative", "expl", "dislocated"],
    "NCoreC": ["advcl"],
    "NCoreM": ["advmod", "discourse"] + ["neg"],  # Modifier words
    "NCoreF": ["aux", "cop", "mark"] + ["auxpass"],  # Function Words
    "NomN": ["nmod", "appos", "numod"],
    "NomC": ["acl"],
    "NomM": ["amod"],
    "NomF": ["det", "clf", "case"] + ["neg"],
    "Coord": ["conj", "cc"],
    "MWE": ["compound", "fixed", "flat"] + ["name", "mwe", "foreign"],
    "others": ["list", "parataxis", "orphan", "goeswith", "reparandum", "punct", "root", "dep"] + ["remnant"],
}
# shortcuts
UD_CATEGORIES["exclude1"] = sum([UD_CATEGORIES[z] for z in ["Coord", "MWE", "others", "NomF", "NCoreF"]], [])
# --

class ZDecoderUdepConf(ZDecoderConf):
    def __init__(self):
        super().__init__()
        # --
        # udep pairwise label
        self.idec_udep_lab = ConfEntryChoices({'idec1': IdecConf.make_conf('pairwise'), 'idec2': Idec2Conf()}, 'idec1')
        self.loss_udep_lab = 1.
        self.lab_udep = ZLabelConf().direct_update(fixed_nil_val=0., loss_neg_sample=-2.)  # 2 neg samples
        # special for labels: note: only utilized in training!!
        self.udep_ignore_labels = []  # ignore these (by masking out)
        self.udep_nilout_labels = []  # make them nil
        # root score (simply binary)
        self.idec_udep_root = IdecConf.make_conf('score')
        self.loss_udep_root = 0.  # make it smaller weight
        self.lab_root = ZLabelConf().direct_update(fixed_nil_val=0.)
        # --
        # decoding
        self.dec_no_root_score = True  # no root score when decoding
        self.udep_no_decode = False  # simply skip decoding (used in special cases!)
        # --

@node_reg(ZDecoderUdepConf)
class ZDecoderUdep(ZDecoder):
    def __init__(self, conf: ZDecoderUdepConf, ztask, main_enc: ZEncoder, **kwargs):
        super().__init__(conf, ztask, main_enc, **kwargs)
        conf: 'ZDecoderUdepConf' = self.conf
        self.voc, _ = self.ztask.vpack
        # --
        # dep-lab
        _enc_dim, _head_dim = main_enc.get_enc_dim(), main_enc.get_head_dim()
        self.lab_udep = ZlabelNode(conf.lab_udep, _csize=len(self.voc))
        self.idec_udep = conf.idec_udep_lab.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=self.lab_udep.get_core_csize())
        self.reg_idec('udep', self.idec_udep)
        # special masks
        self.udep_ignore_masks = self.get_label_mask(conf.udep_ignore_labels)
        self.udep_nilout_masks = self.get_label_mask(conf.udep_nilout_labels)
        # --
        # binary root
        self._label_idx_root = self.voc.get("root")  # get root's index for decoding
        self.lab_root = ZlabelNode(conf.lab_root, _csize=2)
        self.idec_root = conf.idec_udep_root.make_node(_isize=_enc_dim, _nhead=_head_dim, _csize=self.lab_root.get_core_csize())
        self.reg_idec('root', self.idec_root)
        # --

    def get_label_mask(self, sels: List[str]):
        expand_sels = []
        for s in sels:
            if s in UD_CATEGORIES:
                expand_sels.extend(UD_CATEGORIES[s])
            else:
                expand_sels.append(s)
        expand_sels = sorted(set(expand_sels))
        voc = self.voc
        # --
        ret = np.zeros(len(voc))
        _cc = 0
        for s in expand_sels:
            if s in voc:
                ret[voc[s]] = 1.
                _cc += voc.word2count(s)
            else:
                zwarn(f"UNK dep label: {s}")
        _all_cc = voc.get_all_counts()
        zlog(f"Get label mask with {expand_sels}: {len(expand_sels)}=={ret.sum().item()} -> {_cc}/{_all_cc}={_cc/(_all_cc+1e-5)}")
        return BK.input_real(ret)

    def loss(self, med: ZMediator, *args, **kwargs):
        conf: ZDecoderUdepConf = self.conf
        # --
        # prepare info
        ibatch = med.ibatch
        expr_udep_labels, expr_isroot = self.prepare(ibatch)  # [bs, dlen, dlen], [bs, dlen]
        base_mask_t = self.get_dec_mask(ibatch, conf.msent_loss_center)  # [bs, dlen]
        # get losses
        loss_items = []
        _loss_udep_lab = conf.loss_udep_lab
        if _loss_udep_lab > 0.:
            # extra masks: force same sent!
            _dec_sent_idxes = ibatch.seq_info.dec_sent_idxes  # [bs, dlen]
            _mask_t = (_dec_sent_idxes.unsqueeze(-1) == _dec_sent_idxes.unsqueeze(-2)).float()  # [bs, dlen, dlen]
            _mask_t *= base_mask_t.unsqueeze(-1)
            _mask_t *= base_mask_t.unsqueeze(-2)  # [bs, dlen, dlen]
            # special handlings
            _mask_t *= (1.-self.udep_ignore_masks[expr_udep_labels])  # [bs, dlen, dlen]
            expr_udep_labels2 = (expr_udep_labels * (1.-self.udep_nilout_masks[expr_udep_labels])).long()  # [bs, dlen, dlen]
            # --
            loss_items.extend(self.loss_from_lab(self.lab_udep, 'udep', med, expr_udep_labels2, _mask_t, _loss_udep_lab))
        _loss_udep_root = conf.loss_udep_root
        if _loss_udep_root > 0.:
            loss_items.extend(self.loss_from_lab(self.lab_root, 'root', med, expr_isroot, base_mask_t, _loss_udep_root))
        # --
        ret_loss = LossHelper.combine_multiple_losses(loss_items)
        return ret_loss, {}

    def predict(self, med: ZMediator, *args, **kwargs):
        if not self.conf.udep_no_decode:
            self._pred_udep(med)
        return {}

    # decoding!
    def _pred_udep(self, med: ZMediator):
        conf: ZDecoderUdepConf = self.conf
        # --
        # pairwise
        udep_score_cache = med.get_cache((self.name, 'udep'))
        udep_scores_t = self.lab_udep.score_labels(udep_score_cache.vals, None)  # [*, h, m, L]
        udep_logprobs_t = udep_scores_t.log_softmax(-1)  # [*, h, m, L]
        # root
        if conf.dec_no_root_score:  # not using model
            root_logprobs_t = None
        else:
            root_score_cache = med.get_cache((self.name, 'root'))
            root_scores_t = self.lab_root.score_labels(root_score_cache.vals, None)  # [*, dlen, 2]
            root_logprobs_t = root_scores_t.log_softmax(-1).narrow(-1, 0, 1).squeeze(-1)  # [*, dlen]
        # --
        self.decode_udep(med.ibatch, udep_logprobs_t, root_logprobs_t)
        # --

    # --
    # helpers

    # prepare gold labels
    def prepare(self, ibatch):
        b_seq_info = ibatch.seq_info
        bsize, dlen = BK.get_shape(b_seq_info.dec_sel_masks)
        arr_udep_labels = np.full([bsize, dlen, dlen], 0, dtype=np.int)  # by default 0
        arr_head = np.full([bsize, dlen], -1, dtype=np.int)  # the 0 ones are root
        for bidx, item in enumerate(ibatch.items):  # for each item in the batch
            _dec_offsets = item.seq_info.dec_offsets
            for sidx, sent in enumerate(item.sents):  # for each sent in the msent item
                tree = sent.tree_dep
                _start = _dec_offsets[sidx]
                _slen = len(sent)
                # note: here transpose it: (h,m), arti-root not included!
                arr_udep_labels[bidx, _start:_start+_slen, _start:_start+_slen] = tree.label_matrix[:, 1:].T
                arr_head[bidx, _start:_start+_slen] = tree.seq_head.vals
        # --
        expr_udep_labels = BK.input_idx(arr_udep_labels)  # [bs, dlen, dlen]
        expr_isroot = (BK.input_idx(arr_head) == 0).long()  # [bs, dlen]
        return expr_udep_labels, expr_isroot

    # decode with scores
    # [*, h, m, L], [*, dlen]
    def decode_udep(self, ibatch, udep_logprobs_t: BK.Expr, root_logprobs_t: BK.Expr):
        conf: ZDecoderUdepConf = self.conf
        # --
        arr_udep = BK.get_value(udep_logprobs_t.transpose(-2,-3))  # [*, m, h, L]
        arr_root = None if root_logprobs_t is None else BK.get_value(root_logprobs_t)  # [*, dlen]
        _dim_label = arr_udep.shape[-1]
        _neg = -10000.  # should be enough!!
        _voc, _lab_range = self.ztask.vpack
        _idx_root = self._label_idx_root
        # --
        for bidx, item in enumerate(ibatch.items):  # for each item in the batch
            _dec_offsets = item.seq_info.dec_offsets
            for sidx, sent in enumerate(item.sents):
                if conf.msent_pred_center and (sidx != item.center_sidx):
                    continue  # skip non-center sent in this mode!
                _start = _dec_offsets[sidx]
                _len = len(sent)
                _len_p1 = _len + 1
                # --
                _arr = np.full([_len_p1, _len_p1, _dim_label], _neg, dtype=np.float32)  # [1+m, 1+h, L]
                # assign label scores
                _arr[1:_len_p1, 1:_len_p1, 1:_lab_range] = arr_udep[bidx, _start:_start+_len, _start:_start+_len, 1:_lab_range]
                # assign root scores
                if arr_root is not None:
                    _arr[1:_len_p1, 0, _idx_root] = arr_root[bidx, _start:_start+_len]
                else:  # todo(+N): currently simply assign a smaller "neg-inf"
                    _arr[1:_len_p1, 0, _idx_root] = -99.
                # --
                from msp2.tools.algo.nmst import mst_unproj  # decoding algorithm
                arr_ret_heads, arr_ret_labels, arr_ret_scores = \
                    mst_unproj(_arr[None], np.asarray([_len_p1]), labeled=True)  # [*, 1+slen]
                # assign
                list_dep_heads = arr_ret_heads[0, 1:_len_p1].tolist()
                list_dep_lidxes = arr_ret_labels[0, 1:_len_p1].tolist()
                list_dep_labels = _voc.seq_idx2word(list_dep_lidxes)
                sent.build_dep_tree(list_dep_heads, list_dep_labels)
                # sent.tree_dep.seq_label.set_idxes(list_dep_lidxes)
                # --
        # --

# --
# b tasks/zmtl2/zmod/dec/dec_udep:?
