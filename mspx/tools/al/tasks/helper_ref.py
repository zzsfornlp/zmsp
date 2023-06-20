#

__all__ = [
    "QueryRefHelper"
]

import os
from mspx.data.inst import yield_sents
from mspx.data.vocab import SeqVocab
from mspx.utils import zlog, zwarn, zglob1, default_pickle_serializer
from .base import ALTaskHelper

# todo(+N): somehow bad codings ...
class QueryRefHelper:
    def __init__(self, ref_stream, task_helper: ALTaskHelper, voc_dir: str):
        self.ref_stream = ref_stream
        self.task_helper = task_helper
        self.voc_dir = voc_dir
        # --
        self._ref_maps = None  # lazy loading!
        self.NIL_IDX = 0  # note: always assume 0 as NIL!
        self.BAD_IDX = -100
        # --
        _task_mod = task_helper.__module__.split('.')[-1]  # simply judge by module name
        self.do_load_f = getattr(self, f"_load_{_task_mod}")
        self.do_eval_f = getattr(self, f"_eval_{_task_mod}")
        # --

    @property
    def ref_maps(self):  # lazy init!
        if self._ref_maps is None:
            if self.ref_stream is None:
                zwarn("No ref!!")
            else:  # load data!
                _ref_insts = list(self.ref_stream)
                self._ref_maps = self.do_load_f(_ref_insts)
        return self._ref_maps

    def sent2id(self, sent):
        return (sent.doc.id, sent.sid)

    # loading
    def _load_zext(self, ref_insts, task_helper=None):
        if task_helper is None:
            task_helper = self.task_helper
        # --
        voc = default_pickle_serializer.from_file(zglob1(os.path.join(self.voc_dir, f'v_{task_helper.name}.pkl')))[0]
        use_cate_label = any(("___" in z) for z in voc.full_i2w)
        svoc = SeqVocab(voc)
        frame_cate = task_helper.frame_cate
        _ret = {}
        for sent in yield_sents(ref_insts):
            sent_id = self.sent2id(sent)
            gold_frames = sent.get_frames(cates=frame_cate)
            gold_tags = svoc.spans2tags_str([z.mention.get_span() + (z.cate_label if use_cate_label else z.label,) for z in gold_frames], len(sent))[0][0]
            gold_tids = svoc.seq_word2idx(gold_tags)
            _ret[sent_id] = gold_tids
        zlog(f"RefHelper: Load {len(_ret)} sents for zext")
        # --
        return [_ret]

    def _load_zdpar(self, ref_insts):
        voc = default_pickle_serializer.from_file(zglob1(os.path.join(self.voc_dir, f'v_{self.task_helper.name}.pkl')))[0]
        _ret = {}
        _vlen = len(voc)
        for sent in yield_sents(ref_insts):
            sent_id = self.sent2id(sent)
            gold_lids = voc.seq_word2idx(sent.tree_dep.seq_label.vals)
            _ret[sent_id] = [(a*_vlen+b) for a,b in zip(sent.tree_dep.seq_head.vals, gold_lids)]
        zlog(f"RefHelper: Load {len(_ret)} sents for zdpar")
        return [_ret]

    def _load_zrel(self, ref_insts):
        _ret_extH = self._load_zext(ref_insts, self.task_helper.ext_helper)[0]  # for ext sub-task!
        # --
        vocR = default_pickle_serializer.from_file(zglob1(os.path.join(self.voc_dir, f'v_{self.task_helper.name}.pkl')))[0]
        cateHs, cateTs = self.task_helper.conf.rconf.cateHs, self.task_helper.conf.rconf.cateTs
        _ret = {}
        for sent in yield_sents(ref_insts):
            sent_id = self.sent2id(sent)
            _gold_ones = {}
            for frameH in sent.get_frames(cates=cateHs):
                _keyH = (sent_id, frameH.mention.get_span())
                _gold_ones[_keyH] = frameH.label
                for alink in frameH.args:
                    if alink.arg.cate in cateTs:  # get a valid link!
                        _asent = alink.mention.sent
                        _key = (sent_id, frameH.mention.get_span(),
                                self.sent2id(_asent), alink.mention.get_span())
                        if _key not in _gold_ones:
                            _gold_ones[_key] = set()
                        _gold_ones[_key].add(vocR.word2idx(alink.label))
            for frameT in sent.get_frames(cates=cateTs):
                _keyT = (sent_id, frameT.mention.get_span())
                _gold_ones[_keyT] = frameT.label
            _ret[sent_id] = _gold_ones
        zlog(f"RefHelper: Load {len(_ret)} sents for zrel")
        # --
        return [_ret, _ret_extH]

    # eval
    def _eval_zext(self, cands, ref_maps):
        _map = ref_maps[0]
        for one_cand in cands:
            _skey = self.sent2id(one_cand.sent)
            if _skey not in _map:
                zwarn(f"Cannot find {_skey}")
                _gold_idx = self.BAD_IDX
            else:
                _gold_idx = _map[_skey][one_cand.widx]
            one_cand.gold_idx = _gold_idx

    def _eval_zdpar(self, cands, ref_maps):
        return self._eval_zext(cands, ref_maps)  # note: actually the same procedure

    def _eval_zrel(self, cands, ref_maps):
        _tok_cands = [c for c in cands if c.type=='tok']
        _alink_cands = [c for c in cands if c.type=='alink']
        assert len(_tok_cands) + len(_alink_cands) == len(cands)
        # --
        self._eval_zext(_tok_cands, [ref_maps[1]])
        _map = ref_maps[0]
        for one_cand in _alink_cands:
            _skey = self.sent2id(one_cand.sent)
            if _skey not in _map:
                zwarn(f"Cannot find {_skey}")
                _gold_idx = self.BAD_IDX
            else:
                _map2 = _map[_skey]
                q_alink = one_cand.alink
                q_key = (self.sent2id(q_alink.main.sent), q_alink.main.mention.get_span(),
                         self.sent2id(q_alink.arg.sent), q_alink.arg.mention.get_span())
                _keyH, _keyT = q_key[:2], q_key[2:]
                if q_key in _map2:
                    _lids = list(_map2[q_key])
                    if len(_lids) > 1:
                        _lids.sort(key=lambda z: -one_cand.arr_strg[z])  # prefer high-score one!
                    _gold_idx = _lids[0]
                else:  # No alinks!
                    _gold_idx = 0
                    # if _keyH not in _map2 or _keyT not in _map2:  # note: nope, too much!
                    #     _gold_idx = self.BAD_IDX  # frame-error!
                    #     # breakpoint()
                # --
                # note: extra info for calculating recall later!
                one_cand.gold_ninfo = (_skey, len([z for z in _map2.values() if any(z2!=self.NIL_IDX for z2 in z)]))
                # --
            one_cand.gold_idx = _gold_idx

    def eval_cands(self, cands):
        ref_maps = self.ref_maps
        if ref_maps is None:  # no assign!
            for one_cand in cands:
                one_cand.gold_idx = self.BAD_IDX
        else:
            self.do_eval_f(cands, ref_maps)

# --
# b mspx/tools/al/tasks/helper_ref:
