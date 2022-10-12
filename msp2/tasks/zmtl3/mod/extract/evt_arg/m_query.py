#

# query module

__all__ = [
    "QmodConf", "QmodLayer", "build_onto_reprs",
]

from typing import List, Dict
from collections import Counter
import re
import numpy as np
from msp2.utils import default_pickle_serializer, zwarn, ZRuleFilter, zlog, Random, zglob1z
from msp2.data.inst import DataPadder
from msp2.nn import BK
from msp2.nn.l3 import *
from . import onto as zonto

# --
# build reprs (build static ones on the fly)
def build_onto_reprs(onto, spec_str: str, np_getter=None):
    from transformers import AutoTokenizer, AutoModel
    # --
    if np_getter is None:
        np_getter = lambda x: x.np
    # parse spec
    _specs = spec_str.split("###", 1)
    bname, blayer = _specs[0], (int(_specs[1]) if len(_specs)>1 else -1)
    bsize = 10
    zlog(f"Ready to build_onto_reprs for {onto} with {bname}(#L{blayer})")
    # --
    tokenizer = AutoTokenizer.from_pretrained(bname)
    device = BK.DEFAULT_DEVICE
    model = AutoModel.from_pretrained(bname).to(device)
    _cls_id, _pad_id, _sep_id = tokenizer.cls_token_id, tokenizer.pad_token_id, tokenizer.sep_token_id
    # --
    rets = []
    for descriptions in [[" " + z.vp for z in onto.frames], [" " + np_getter(z) for z in onto.roles]]:
        arrs = []
        for ii in range(0, len(descriptions), bsize):
            ids = [[_cls_id] + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s)) + [_sep_id]
                   for s in descriptions[ii:ii + bsize]]
            arr_ids = DataPadder.go_batch_2d(ids, _pad_id)  # [bs, L]
            t_ids = BK.input_idx(arr_ids)  # [bs, L]
            t_masks = (t_ids != _pad_id).float()  # [bs, L]
            res = model(input_ids=t_ids, attention_mask=t_masks,
                        output_attentions=True, output_hidden_states=True, return_dict=True)
            t_hid = res.hidden_states[blayer]  # [bs, L, D]
            t_valid = ((t_ids != _cls_id) & (t_ids != _pad_id) & (t_ids != _sep_id)).float()  # [bs, L]
            assert (t_valid.sum(-1) > 0).all()
            t_w = t_valid / t_valid.sum(-1, keepdims=True)  # [bs, L]
            t_repr = (t_hid * t_w.unsqueeze(-1)).sum(-2)  # [bs, D]
            one_arr = BK.get_value(t_repr)
            arrs.append(one_arr)
        one_ret = np.concatenate(arrs, 0)  # [*, D]
        rets.append(one_ret)
    # --
    zlog(f"Build ok with: {[z.shape for z in rets]}")
    return tuple(rets)
# --

class QmodConf(ZlayerConf):
    def __init__(self):
        super().__init__()
        # --
        self.mdim = -1  # model dim
        # reprs
        self.load_reprs = ""  # load element reprs rather than random init
        self.build_reprs = ""  # build element reprs rather than random init, eg: roberta-base###-1
        self.set_sn_reprs = False  # set sn with the build/load reprs!
        self.freeze_reprs = False  # whether freeze reprs
        self.aff_frame = AffineConf.direct_conf(osize=0, use_bias=False)  # affine for frame repr
        self.aff_role = AffineConf.direct_conf(osize=0, use_bias=False)  # affine for role repr
        self.emb_initscale = 1.  # init scale for embs
        self.sn_frame = SimpleNormConf()  # simple-norm
        self.sn_role = SimpleNormConf()  # simple-norm
        # special mode!!
        self.default_frame_name = ""  # if frame not found, then find a default one by this name!
        # --
        # query
        # common ones
        self.filter_noncore = []  # rules for filtering noncore roles
        self.train_sample_ncr = [1., 1.1]  # rate range for sampling noncore roles?
        self.train_mask_vv = 0.  # mask VV with <mask> for the template?
        self.np_getter = 'np'  # np, vn, fn
        self.query_add_dot = False  # add dot to the template?
        self.shuf_template_rate = 0.  # shuf of template?
        self.reorder_template = False  # reorder by name?
        self.vp_template_real = False  # use real trig ids instead of vp
        self.np_template_plh = False  # use placeholder instead of np (only for tpl flexible template mode!)
        # -> for *_trig_pat: <T> = real trigger
        # -> for *_role_pat: <RE> = role embedding, <RN> = role np, <R> = query/ar_placeholder, <P> = preposition
        # -> for level0 ones: <T> = trigger, <R> = role for the question, <Role-name> = specific role
        # clf: [cls] trig_pat [sep] ... [sep]
        self.clf_trig_pat = " <T>"  # '' means no seq; other examples: ' " <T> "'
        # mrc: [cls] question [sep] ... [sep]
        self.mrc_trig_pat = " <T>"  # for example: what is the <role> of the <trig>?
        self.mrc_use_other_qwords = True  # use other qwords rather than only "What"
        self.mrc_query_qword = True  # otherwise use "<R>"
        self.mrc_use_rques = False  # use role_questions provided in the onto
        self.mrc_use_tques = False  # construct role_questions with template
        self.mrc_use_eques = False  # use questions provided in the evt (for qa datasets!)
        self.mrc_add_t = False  # add "in the <T> event" at the end of the question
        self.mrc_query_version = 0  # query version
        self.mrc_role_pat = " <RN>"  # role pattern in question; other examples: ' <RE> ( <RN> )'
        # tpl: [cls] template [sep] ... [sep]
        self.tpl_trig_pat = " <T>:"  # at the start of the template
        self.tpl_role_pat = " <P> <R> <RN>"  # role pattern for template-query; other examples: ' <P> <R> <RE> ( <RN> )'
        # s2s: [cls] ... [sep] -> [cls] template_for_autoR [sep]
        self.s2s_trig_pat = " <T>:"  # at the start of the template
        self.s2s_role_pat = " <P> <R> <RN>"  # or " <P> <RN> <R2>" for ATR, <R2> is replaced with [MASK] and used for ATR
        # gen: same as s2s
        self.gen_trig_pat = " <T>:"
        self.gen_role_pat = " <P> <RN> <R2>"
        self.gen_fill_head = True  # fill only head word or full span?
        # --

@node_reg(QmodConf)
class QmodLayer(Zlayer):
    def __init__(self, conf: QmodConf, onto: zonto.Onto, sub_toker, **kwargs):
        super().__init__(conf, **kwargs)
        conf: QmodConf = self.conf
        self.sub_toker = sub_toker
        # --
        self.idx_offset = 1
        self._gen = Random.get_generator(f'sample_Qmod')
        # --
        # patterns
        _toker = sub_toker.tokenizer
        for k, v in conf.__dict__.items():
            if k.endswith("_pat"):
                setattr(self, k, PartialSeq.parse(v, _toker))
        _queries = [
            [  # v0
                ('what', f"What is the <R> of the <T> event?"),
                ('who', f"Who is the <R> in the <T> event?"),
                ('where', f"Where does the <T> event take place?"),
                ('where2', f"Where is the <R> of the <T> event?"),  # maybe origin/destination/...
            ],
            [  # v1 (... in the <T> event?)
                ('what', f"What is the <R> in the <T> event?"),
                ('who', f"Who is the <R> in the <T> event?"),
                ('where', f"Where is the <R> in the <T> event?"),
                ('where2', f"Where is the <R> in the <T> event?"),  # same as where!
            ],
            [  # v2 (with all <R>s)
                ('what', f"What is the <R> of the <T> event?"),
                ('who', f"Who is the <R> in the <T> event?"),
                ('where', f"Where is the <R> of the <T> event?"),
                ('where2', f"Where is the <R> of the <T> event?"),
            ],
        ][conf.mrc_query_version]  # level0 pattern for the questions
        self.question_patterns = {k: PartialSeq.parse(v, _toker) for k, v in _queries}
        self.np_getter = {
            "np": lambda x: x.np,
            "vn": lambda x: x.info.get("np_vn", x.np),  # backoff to np
            "fn": lambda x: x.info.get("np_fn", x.np),  # backoff to np
            "name": lambda x: str.lower(x.name),  # simply use name
        }[conf.np_getter]
        # --
        self._setup(onto)

    # further setup things: note: here we assume that onto is fixed and should never change!!
    def _setup(self, onto):
        conf: QmodConf = self.conf
        # prepare onto
        self.onto = onto
        self.onto.build_idxes(self.idx_offset)  # start with >1
        _nc_filter = ZRuleFilter(
            conf.filter_noncore, {"spec": ['Place', 'Instrument', 'Weapon', 'Vehicle', 'ARGM-LOC']}, True)
        for ff in self.onto.frames:
            ff.build_role_map(nc_filter=(lambda _name: _nc_filter.filter_by_name(_name)), force_rebuild=True)
        # --
        # build reprs
        self.emb_frame = BK.get_emb_with_initscale(len(onto.frames), conf.mdim, conf.emb_initscale)
        self.emb_role = BK.get_emb_with_initscale(len(onto.roles), conf.mdim, conf.emb_initscale)
        self.sn_frame = conf.sn_frame.make_node(dim=conf.mdim)
        self.sn_role = conf.sn_role.make_node(dim=conf.mdim)
        # --
        # build or load
        emb_init_arrs = None
        if conf.build_reprs:
            emb_init_arrs = build_onto_reprs(onto, conf.build_reprs, np_getter=self.np_getter)
        elif conf.load_reprs:
            _path = zglob1z(conf.load_reprs)
            emb_init_arrs = default_pickle_serializer.from_file(_path)
            zlog(f"Load reprs from {_path}.")
        # assign
        if emb_init_arrs is not None:
            arr_frame, arr_role = emb_init_arrs
            # note: also apply initscale here!!
            BK.set_value(self.emb_frame.weight, arr_frame*conf.emb_initscale)
            BK.set_value(self.emb_role.weight, arr_role*conf.emb_initscale)
            zlog(f"Finished assigning reprs to embeddings with init={conf.emb_initscale}")
            if conf.set_sn_reprs:
                for _arr, _sn in zip([arr_frame, arr_role], [self.sn_frame, self.sn_role]):
                    _t = BK.input_real(_arr)
                    _b, _w = _t.mean(0), 1./(_t.std(0) + 1e-6)
                    _sn.set(_b, _w)
                zlog("Finished assigning sn layers.")
        if conf.freeze_reprs:
            assert emb_init_arrs is not None
            for _e in [self.emb_frame, self.emb_role]:
                for p in _e.parameters():
                    p.requires_grad = False
        # --
        self.aff_frame = AffineLayer(conf.aff_frame, isize=conf.mdim)
        self.aff_role = AffineLayer(conf.aff_role, isize=conf.mdim)
        # --

    # get reprs
    def get_repr_frame(self, idx_t):
        ret0 = self.emb_frame((idx_t-self.idx_offset).clamp(min=0))  # [*, D]
        ret1 = self.sn_frame(ret0)
        ret = self.aff_frame(ret1)
        return ret

    def get_repr_role(self, idx_t):
        ret0 = self.emb_role((idx_t-self.idx_offset).clamp(min=0))  # [*, D]
        ret1 = self.sn_role(ret0)
        ret = self.aff_role(ret1)
        # breakpoint()
        return ret

    # valid for core or active noncore
    def arg2role(self, arg):
        frame = self.evt2frame(arg.main)
        ret = frame.find_role(arg.label, None)[0]
        return ret

    def evt2frame(self, evt):
        ret = self.onto.find_frame(evt.label)
        if ret is None:
            _df = self.conf.default_frame_name
            if _df:
                ret = self.onto.find_frame(_df)
        return ret

    def ridx2role(self, ridx: int):
        return self.onto.roles[ridx-self.idx_offset]

    # --
    # helpers for queries

    def _get_rs(self, evts, is_testing: bool):
        conf: QmodConf = self.conf
        # --
        r1, r2 = conf.train_sample_ncr[:2]
        _gen = self._gen
        # --
        rets = []
        ffs = []
        for evt in evts:
            ff = self.evt2frame(evt)
            ffs.append(ff)
            _ffmap = ff.role_map
            # --
            crs = ff.core_roles  # always put core roles!
            ncrs = ff.active_noncore_roles
            _len_ncrs = len(ncrs)
            if is_testing or r1>=1. or _len_ncrs==0:  # query all or nothing to add
                query_rs = crs + ncrs
            else:  # need to sample
                hit_ncr = set()  # hit set of ncrs
                # first put already existing ones
                for arg in evt.args:
                    if arg.label in _ffmap:
                        hit_ncr.add(_ffmap[arg.label].idx)
                # then do sample
                _nn = min(1., _gen.random() * (r2-r1) + r1)
                _nn = int(_len_ncrs * _nn)
                for z in _gen.choice(_len_ncrs, _nn, replace=False):
                    hit_ncr.add(ncrs[z].idx)
                # finally concat
                query_rs = crs + [z for z in ncrs if z.idx in hit_ncr]
            # --
            rets.append(query_rs)
        # --
        arr_rs = DataPadder.go_batch_2d(rets, None, dtype=object)
        # note: pad 0!
        arr_rids = np.asarray([z.idx if z is not None else 0 for z in arr_rs.flatten()]).reshape(arr_rs.shape)
        return arr_rs, arr_rids, ffs

    def _get_trig_subids(self, evt, trig_pat: 'PartialSeq'):
        _trig_key = f"_trig_{self.sub_toker.key}"
        # --
        _trig_ids = evt._cache.get(_trig_key)
        if _trig_ids is None:
            _trig0 = evt.mention.get_words(concat=True)
            _trig_ids, _ = trig_pat.fill(T=_trig0)  # note: currently does not care about trig position!
            evt._cache[_trig_key] = _trig_ids
        # --
        return _trig_ids

    def _get_qwords(self, rr):
        conf: QmodConf = self.conf
        # qwords = ['what']  # note: nope, a small bug here!
        qwords = []
        if conf.mrc_use_other_qwords:
            if rr.qwords:
                qwords.extend(rr.qwords)
            elif rr.name.lower() in ['place', 'argm-loc']:  # note: special for Place!
                qwords.extend(['where', 'where2'])
        qwords.append("what")
        return qwords

    def _norm_ques(self, q: str):
        q = q.strip()
        if self.conf.mrc_add_t and '<T>' not in q:
            if q[-1] == '?':
                q = q[:-1]
            q = f"{q} in the <T> event?"
        return q

    def _construct_tques(self, ff):
        # construct role-questions according to the template and qwords
        assert isinstance(ff.template, list), "Bad template to use!"
        _sub_toker = self.sub_toker
        # first divide left and right
        left_rrs, right_rrs = [], []
        curr_rrs = left_rrs
        role_vals = {}
        for r_name, r_preps in ff.template:
            if r_name is None:
                curr_rrs = right_rrs  # switch
            if r_name not in ff.role_map:
                continue  # non-active one!
            role, is_core = ff.role_map[r_name]
            if len(r_preps) == 0 and is_core:  # ignore the ones with preps
                curr_rrs.append((r_name, "the " + self.np_getter(role)))
        # create the templates
        is_left = True
        left_ss = f"someone" if len(left_rrs)==0 else f"{left_rrs[-1][-1]}"  # if no left, put a dummy
        right_ss = "" if len(right_rrs)==0 else f"{right_rrs[0][-1]}"  # if no right, simply empty
        full_ss = f"{left_ss} {ff.vp} {right_ss}"
        left_ss = f"{left_ss} {ff.vp}"
        right_ss = f"{ff.vp} {right_ss}"
        for r_name, r_preps in ff.template:
            if r_name is None:
                is_left = False
            if r_name not in ff.role_map:
                continue  # non-active one!
            role = ff.role_map[r_name][0]
            role_qprefixes = []
            for qw in self._get_qwords(role):
                if qw == 'what':
                    one = f"What {self.np_getter(role)}"
                elif qw == 'who':
                    one = "Who"
                elif qw == 'where':
                    one = "Where"
                elif qw == 'where2':
                    continue
                else:
                    raise NotImplementedError(f"Unknown qword {qw}")
                role_qprefixes.append((qw, one))
            if is_left:  # simply put as the subj and ignore preps
                questions = [f"{one} {right_ss}" for qw, one in role_qprefixes]
            elif len(r_preps) == 0:  # view it as obj
                questions = [f"{one} did {left_ss}" for qw, one in role_qprefixes]
            else:  # view it as obl (simply use the first prep!)
                questions = [f"{one} did {full_ss} {'' if qw=='where' else r_preps[0]}"
                             for qw, one in role_qprefixes]
            role_vals[r_name] = {qw: PartialSeq.parse(self._norm_ques(q), _sub_toker.tokenizer)
                                 for (qw,_), q in zip(role_qprefixes,questions)}
        # --
        return role_vals

    def _get_ques(self, evt, ff, rr, role_pat: 'PartialSeq', _trig_ids, is_testing: bool):
        conf: QmodConf = self.conf
        _sub_toker = self.sub_toker
        _rr_key = f"_rrQ_{ff.name}_{_sub_toker.key}"  # note: need this!!
        _ms = [_sub_toker.tokenizer.mask_token_id]
        # --
        # get ids for the role
        _cache = rr.info.get(_rr_key)
        if _cache is None:
            _rques = ff.info.get("role_questions", {}).get(rr.name)
            _rques_qt = None if _rques is None else PartialSeq.parse(self._norm_ques(_rques), _sub_toker.tokenizer)
            _cache = (role_pat.fill(RE=_ms, RN=self.np_getter(rr), R=[]), _rques_qt)
            rr.info[_rr_key] = _cache
        (_role_ids, _role_k2p), _rques_qt = _cache
        # --
        # put things together
        if conf.mrc_use_eques:
            _cache = evt._cache.get(_rr_key)
            if _cache is None:
                _cache = PartialSeq.parse(self._norm_ques(evt.info['question']), _sub_toker.tokenizer)
                evt._cache[_rr_key] = _cache
            qt = _cache  # use the one in the data!
        elif conf.mrc_use_rques:
            qt = _rques_qt  # use pre-defined ones!
        else:  # use default ones
            # select question template with qword!
            qwords = self._get_qwords(rr)
            if is_testing or len(qwords) == 1:
                qw = qwords[0]
            else:
                qw = qwords[self._gen.choice(len(qwords))]
            # --
            if conf.mrc_use_tques:  # construct from template!
                # --
                _ff_key = f"_ffQ_{_sub_toker.key}"
                _qcache = ff.info.get(_ff_key)
                if _qcache is None:  # get the template
                    _qcache = self._construct_tques(ff)
                    ff.info[_ff_key] = _qcache
                # --
                if qw == 'where2':
                    qw = 'where'  # no specific where2 in this mode!
                qt = _qcache[rr.name][qw]
            else:  # simply use the default ones
                qt = self.question_patterns[qw]  # get the template
            # --
        # --
        ques_ids, ques_k2p = qt.fill(_add_cls=True, _add_sep=True, R=_role_ids, T=_trig_ids)  # fill it
        e_ridxes = [0] * len(ques_ids)  # for embedding inputs
        if 'R' in ques_k2p and 'RE' in _role_k2p:
            e_ridxes[ques_k2p['R'] + _role_k2p['RE']] = rr.idx
        rq_idx = 1 if conf.mrc_query_qword else (ques_k2p['R'] + _role_k2p['R'])
        # note: simply use idx0 as the query!
        return ques_ids, rq_idx, e_ridxes

    def _get_template(self, ff, roles, role_pat: 'PartialSeq', _trig_ids, is_testing: bool, is_s2s: bool):
        conf: QmodConf = self.conf
        _reorder_template, _shuf_template_rate, _vp_template_real = \
            conf.reorder_template, conf.shuf_template_rate, conf.vp_template_real
        _sub_toker = self.sub_toker
        _ff_key = f"_ffA_{_sub_toker.key}" if is_s2s else f"_ffT_{_sub_toker.key}"
        _ms = [_sub_toker.tokenizer.mask_token_id]
        # --
        # template
        _cache = ff.info.get(_ff_key)
        if _cache is None:  # get the template
            assert ff.template is not None, "No template to use?"
            vv_val = PartialSeq.parse(" <VV>", _sub_toker.tokenizer).fill(VV=ff.vp)[0]
            if isinstance(ff.template, str):
                frame_pat0 = "<T> " + ff.template  # already written!
                # only one case (preps are already in template)
                role_vals = {k: [role_pat.fill(RE=_ms, RN=self.np_getter(v[0]), P=[], R=[], R2=_ms)] for k,v in ff.role_map.items()}
                assert _shuf_template_rate <= 0., "Cannot shuffle things in str mode!"
            else:
                # construct one on the fly!
                frame_pat0_s = []
                role_vals = {}
                r_ii = 0
                _tmp_name_order = {}
                for r_name, r_preps in ff.template:
                    if len(r_preps) == 0:
                        r_preps = [[]]
                    if r_name is None:  # VP
                        frame_pat0_s.append("<VV>")  # allow later dynamic mask
                        _tmp_name_order["<VV>"] = ""  # first
                    else:
                        if r_name not in ff.role_map:
                            continue  # non-active one!
                        frame_pat0_s.append(f"<{r_name}>")
                        role = ff.role_map[r_name][0]
                        _rn = self.np_getter(role)
                        _tmp_name_order[f"<{r_name}>"] = _rn  # use NL name
                        if conf.np_template_plh:
                            _rn = _rn.lower() if isinstance(_rn, str) else _rn[0].lower()
                            _rn = _rn if _rn.startswith('arg') else f"arg{r_ii}"
                        _rns = [_rn] if isinstance(_rn, str) else _rn
                        assert isinstance(_rns, list)
                        role_vals[r_name] = [role_pat.fill(RE=_ms, RN=_rn0, P=_p, R=[], R2=_ms)
                                             for _p in r_preps for _rn0 in _rns]
                        r_ii += 1
                if _reorder_template:  # simply: V names ...
                    frame_pat0_s.sort(key=lambda x: _tmp_name_order[x])
                if (not is_testing) and (_shuf_template_rate > 0.) and (self._gen.random() < _shuf_template_rate):  # shuf this!
                    self._gen.shuffle(frame_pat0_s)
                frame_pat0 = " ".join(["<T>"] + frame_pat0_s)
            frame_pat = PartialSeq.parse(frame_pat0, _sub_toker.tokenizer)
            _cache = (frame_pat, vv_val, role_vals)
            if _shuf_template_rate <= 0.:  # todo(+2): currently simply no store if shuf!
                ff.info[_ff_key] = (frame_pat, vv_val, role_vals)
        # --
        frame_pat, vv_val, role_vals = _cache
        # --
        # fill in
        _m = {ncr.name: [] for ncr in ff.active_noncore_roles}  # for ncr, by default let it be EMPTY
        # todo(+N): here if use trig_ids, simply remove the last ":"!!
        _m["VV"] = _ms if (not is_testing) and (self._gen.random() < conf.train_mask_vv) \
            else (_trig_ids[:-1] if _vp_template_real else vv_val)
        _m["T"] = _trig_ids
        _r_k2p = []
        for rii, rr in enumerate(roles):
            if rr is not None:
                _vals = role_vals[rr.name]
                if is_testing or len(_vals) == 1:
                    _role_ids, _role_k2p = _vals[0]
                else:
                    _role_ids, _role_k2p = _vals[self._gen.choice(len(_vals))]
                _m[rr.name] = _role_ids  # store this!
            else:
                _role_k2p = None
            _r_k2p.append(_role_k2p)
        query_ids, query_k2p = frame_pat.fill(_m, _add_cls=True, _add_sep=True, _add_dot=conf.query_add_dot)  # fill it
        # --
        # get extra info
        rq_idxes = [0] * len(roles)  # role-idx to query-idx
        qr_idxes = [0] * len(query_ids)  # query-idx to role-idx
        e_ridxes = [0] * len(query_ids)  # extra embeddings
        for rii, (rr, k2p) in enumerate(zip(roles, _r_k2p)):
            if rr is not None:
                _offset0 = query_k2p[rr.name]  # offset in the overall seq
                _query_ii = _offset0 + (k2p["R2"] if 'R2' in k2p else k2p["R"])  # prefer R2
                qr_idxes[_query_ii] = rr.idx  # role idx
                rq_idxes[rii] = _query_ii  # query position
                if 'RE' in k2p:
                    e_ridxes[_offset0 + k2p["RE"]] = rr.idx  # emb position
        # --
        return query_ids, rq_idxes, qr_idxes, e_ridxes

    # --
    # queries

    def query_clf(self, evts, is_testing: bool):
        conf: QmodConf = self.conf
        _toker = self.sub_toker.tokenizer
        _cls, _sep = [_toker.cls_token_id], [_toker.sep_token_id]
        # --
        arr_rs, arr_rids, _ = self._get_rs(evts, is_testing)
        if conf.clf_trig_pat:
            ids_query = [_cls + self._get_trig_subids(evt, self.clf_trig_pat) + _sep for evt in evts]
        else:  # otherwise simply no seq1
            ids_query = [_cls for _ in evts]
        # -> [bs, R], [bs, R], List[bs, lq]
        return arr_rs, arr_rids, ids_query

    def query_mrc(self, evts, is_testing: bool):
        conf: QmodConf = self.conf
        # --
        arr_rs, arr_rids, ffs = self._get_rs(evts, is_testing)
        # prepare all the questions
        arr_bidx, arr_rbidx = [], arr_rids*0  # [qbs], [bs, R]
        arr_rq_idxes = []  # [qbs]
        ids_query, arr_e_ridx = [], []  # List[qbs, lq], [qbs, lq]
        qbs = 0
        for bidx, (evt, ff) in enumerate(zip(evts, ffs)):
            _trig_ids = self._get_trig_subids(evt, self.mrc_trig_pat)
            for rii, rr in enumerate(arr_rs[bidx]):
                if rr is not None:  # we have a question
                    # put batch idxes
                    arr_bidx.append(bidx)
                    arr_rbidx[bidx, rii] = qbs
                    qbs += 1
                    # get a question
                    _ques_ids, _rq_idx, _e_ridxes = \
                        self._get_ques(evt, ff, rr, self.mrc_role_pat, _trig_ids, is_testing)
                    ids_query.append(_ques_ids)
                    arr_rq_idxes.append(_rq_idx)
                    arr_e_ridx.append(_e_ridxes)
        # --
        arr_rq_idxes = np.asarray(arr_rq_idxes)
        arr_e_ridxes = DataPadder.go_batch_2d(arr_e_ridx, 0)
        if np.any(arr_e_ridxes>0):
            t_e = self.get_repr_role(BK.input_idx(arr_e_ridxes))  # [qbs, lq, D]
        else:  # no valid ones!
            arr_e_ridxes, t_e = None, None
        # -> [bs, R], [bs, R], List[qbs, lq], [qbs], [qbs], [bs, R], ([qbs, lq], T[qbs, lq, D])
        return arr_rs, arr_rids, ids_query, arr_rq_idxes, arr_bidx, arr_rbidx, (arr_e_ridxes, t_e)

    def _query_with_template(self, evts, is_testing: bool, is_s2s: bool):
        conf: QmodConf = self.conf
        _trig_pat = self.s2s_trig_pat if is_s2s else self.tpl_trig_pat
        _role_pat = self.s2s_role_pat if is_s2s else self.tpl_role_pat
        # --
        arr_rs, arr_rids, ffs = self._get_rs(evts, is_testing)  # [bs, R]
        arr_rq_idxes = arr_rids*0  # [bs, R], role-query-idx
        arr_qr_idxes = []  # [bs, lq]
        ids_query, arr_e_ridx = [], []  # List[bs, lq], [bs, lq]
        for bidx, (evt, ff) in enumerate(zip(evts, ffs)):
            _trig_ids = self._get_trig_subids(evt, _trig_pat)
            _query_ids, _rq_idxes, _qr_idxes, _e_ridxes = \
                self._get_template(ff, arr_rs[bidx], _role_pat, _trig_ids, is_testing, is_s2s)
            ids_query.append(_query_ids)
            arr_rq_idxes[bidx] = _rq_idxes
            arr_qr_idxes.append(_qr_idxes)
            arr_e_ridx.append(_e_ridxes)
        # --
        arr_qr_idxes = DataPadder.go_batch_2d(arr_qr_idxes, 0)
        arr_e_ridxes = DataPadder.go_batch_2d(arr_e_ridx, 0)
        if np.any(arr_e_ridxes>0):
            t_e = self.get_repr_role(BK.input_idx(arr_e_ridxes))  # [qbs, lq, D]
        else:  # no valid ones!
            arr_e_ridxes, t_e = None, None
        # -> [bs, R], [bs, R], List[bs, lq], [bs, R], [bs, lq], ([bs, lq], T[bs, lq, D])
        return arr_rs, arr_rids, ids_query, arr_rq_idxes, arr_qr_idxes, (arr_e_ridxes, t_e)
        # --

    def query_tpl(self, evts, is_testing: bool): return self._query_with_template(evts, is_testing, False)
    def query_s2s(self, evts, is_testing: bool): return self._query_with_template(evts, is_testing, True)

    # note: gen returns totally different things!
    def query_gen(self, evts, is_testing: bool):
        conf: QmodConf = self.conf
        _trig_pat, _role_pat = self.gen_trig_pat, self.gen_role_pat
        _tokenizer = self.sub_toker.tokenizer
        # --
        arr_rs, arr_rids, ffs = self._get_rs(evts, is_testing)  # [bs, R]
        arr_rq_idxes = arr_rids*0  # [bs, R], role-query-idx
        ids_query, ids_trg = [], []
        for bidx, (evt, ff) in enumerate(zip(evts, ffs)):
            _trig_ids = self._get_trig_subids(evt, _trig_pat)
            _query_ids, _rq_idxes, _, _ = \
                self._get_template(ff, arr_rs[bidx], _role_pat, _trig_ids, is_testing, True)  # same as s2s!
            _query_ids2 = _query_ids.copy()  # useful as reference for decoding!
            # fill in roles
            for rii, rr in enumerate(arr_rs[bidx]):
                if rr is not None:
                    _rr_args = sorted([a for a in evt.args if a.label==rr.name],
                                      key=(lambda x: (x.mention.sent.sid, x.mention.get_span())))
                    if len(_rr_args) > 0:
                        _spans = [a.mention.get_span(shead=conf.gen_fill_head) for a in _rr_args]
                        _s = " " + " and ".join([" ".join(a.mention.sent.seq_word.vals[x:x+y])
                                                 for (a, (x,y)) in zip(_rr_args, _spans)])
                        _query_ids2[_rq_idxes[rii]] = _tokenizer.convert_tokens_to_ids(_tokenizer.tokenize(_s))
            # flatten
            _trg_ids = []
            for z in _query_ids2:
                if isinstance(z, list):
                    _trg_ids.extend(z)
                else:
                    _trg_ids.append(z)
            # --
            ids_query.append(_query_ids)
            ids_trg.append(_trg_ids)
            arr_rq_idxes[bidx] = _rq_idxes
            # --
        # -> [bs, R], [bs, R], List[bs, lq], [bs, R], [bs, lt]
        return arr_rs, arr_rids, ids_query, arr_rq_idxes, ids_trg
        # --

# --
# partially filled seq
class PartialSeq:
    _PATTERN = re.compile(r"<[-_a-zA-Z0-9]+>")

    def __init__(self, sub_ids: List[int], var_i2k: List[str], var_i2bb: List[bool], var_k2i: Dict[str, int], tokenizer=None):
        self.sub_ids = sub_ids
        self.var_i2k = var_i2k
        self.var_i2bb = var_i2bb
        self.var_k2i = var_k2i
        self.tokenizer = tokenizer

    def has_var(self, s):
        return s in self.var_k2i

    def __repr__(self):
        if self.tokenizer is None:
            return str(self.sub_ids)
        else:
            ii = 0
            rets = []
            for _id in self.sub_ids:
                if _id is None:
                    rets.append(f"<{self.var_i2k[ii]}>")
                    ii += 1
                else:
                    rets.append(self.tokenizer.convert_ids_to_tokens([_id])[0])
            return str(rets)
        # --

    # key -> str or List[int]
    def fill(self, _m=None, _add_cls=False, _add_sep=False, _add_dot=False, **kwargs):
        m = {} if _m is None else _m
        m.update(kwargs)
        # --
        # fill it!
        _dot_id = self.tokenizer.convert_tokens_to_ids(["."])[0]
        ret_ids = [self.tokenizer.cls_token_id] if _add_cls else []
        ret_k2posi = {}  # key -> position
        cur_ii = len(ret_ids)  # cur full subtoken idx
        cur_vi = 0  # cur var idx
        for one_id in self.sub_ids:
            if one_id is not None:
                ret_ids.append(one_id)
                cur_ii += 1
            else:
                _k = self.var_i2k[cur_vi]
                ret_k2posi[_k] = cur_ii
                fill_ids = m.get(_k)
                if isinstance(fill_ids, str):
                    _ss = (" " if self.var_i2bb[cur_vi] else "") + fill_ids
                    fill_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(_ss))
                if fill_ids is None:
                    zwarn(f"Unfilled item: {_k}")
                    fill_ids = []
                # update
                cur_vi += 1
                cur_ii += len(fill_ids)
                ret_ids.extend(fill_ids)
        if _add_dot:
            ret_ids.append(_dot_id)
        if _add_sep:
            ret_ids.append(self.tokenizer.sep_token_id)
        # --
        # note: one can use "self.sub_toker.tokenizer.decode" to see the full seq
        return ret_ids, ret_k2posi

    # accept special patterns: ... <key1> ...
    @staticmethod
    def parse(pat: str, tokenizer):
        _pp = PartialSeq._PATTERN
        mask_token = tokenizer.mask_token
        mask_token_id = tokenizer.mask_token_id
        # --
        # parse pat & tok
        var_i2k, var_i2bb = [], []  # idx to key, idx to blank_before??
        for m in _pp.finditer(pat):
            a, b = m.start(), m.end()
            var_i2k.append(pat[a:b][1:-1])
            var_i2bb.append(a>0 and str.isspace(pat[a-1]))
        var_k2i = {k:i for i,k in enumerate(var_i2k)}  # key to idx
        assert len(var_k2i) == len(var_i2k), f"Currently no support for repeated keys: {var_i2k}"
        pat2 = _pp.sub(mask_token, pat)  # note: simply replace with mask
        sub_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(pat2))  # subtoken ids
        # special for slots!
        n_slots = 0
        for ii, tt in enumerate(sub_ids):
            if tt == mask_token_id:
                sub_ids[ii] = None  # special marking!
                n_slots += 1
        assert n_slots == len(var_i2k), "Unmatched slot number"
        # --
        ret = PartialSeq(sub_ids, var_i2k, var_i2bb, var_k2i, tokenizer=tokenizer)
        return ret

# --
# b msp2/tasks/zmtl3/mod/extract/evt_arg/m_query:
