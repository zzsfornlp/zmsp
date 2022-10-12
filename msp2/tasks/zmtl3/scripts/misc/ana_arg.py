#

# specific arg analyzer

import sys
from collections import Counter

import numpy as np

from msp2.utils import init_everything, zglob1z, zlog, OtherHelper, MathHelper
from msp2.tools.analyze import FrameAnalyzer, FrameAnalyzerConf, FrameAnnotationTask
from msp2.data.inst import set_ee_heads

# --
class MyAnalyzerConf(FrameAnalyzerConf):
    def __init__(self):
        super().__init__()
        # --
        self.align_gold_sent = True
        self.gold_set_ee_heads = True
        self.pred_set_ee_heads = True
        # --
        self.pfilter = 'lambda x: True'  # for example: 'lambda x: x.label.startswith("syn:")'
        # --
        self.filter_noncore = ['Place', 'Instrument', 'Weapon', 'Vehicle', 'ARGM-LOC']
        self.onto = ""
        self.ana_fall = False
        self.do_loop = True
        # --

class MyAnalyzer(FrameAnalyzer):
    def __init__(self, conf: MyAnalyzerConf):
        super().__init__(conf)
        conf: MyAnalyzerConf = self.conf
        self.pfilter = eval(conf.pfilter)
        # --
        if conf.onto:
            from msp2.tasks.zmtl3.mod.extract.evt_arg import onto as zonto
            _path = zglob1z(conf.onto)
            self.onto = zonto.Onto.load_onto(_path)
        else:
            self.onto = None
        self.filter_noncore = set(conf.filter_noncore)
        # --

    def ana_path(self, arg, only_syn=False):
        # first locate the last pred for graph!
        asent = arg.arg.sent
        if hasattr(asent, 'gsent'):
            asent = asent.gsent
        asent = asent.psents[-1]
        # --
        if only_syn:
            _pfilter = (lambda x: self.pfilter(x) and x.label.startswith("syn:"))
        else:
            _pfilter = self.pfilter
        # --
        evt_hwidx, arg_hwidx = arg.main.mention.shead_widx, arg.arg.mention.shead_widx
        path_edges = asent.get_tp_graph(use_cache=False).shortest_path(evt_hwidx, arg_hwidx, _pfilter)
        path_labs = [z.label for z in path_edges]
        cur_labs = path_labs.copy()
        # --
        # category it
        # type1: direct edge
        DIRECT_LS = {f"syn:{z}" for z in ['nmod', 'nsubj', 'obj', 'iobj', 'obl']}
        is_dlink = (lambda xs: len(xs)==0 or (len(xs)==1 and (xs[0] in DIRECT_LS or 'ARG' in xs[0]) and ("^ARG" not in xs[0])))
        if is_dlink(cur_labs):
            t = "t1"
        else:
            FINAL_LS = {f'syn:{z}' for z in ['nmod', 'appos', '^appos' 'nummod', 'flat', '^flat', 'compound', 'conj', '^conj']}
            # type2: slight mismatch ending
            if len(cur_labs)>0 and cur_labs[-1] in FINAL_LS:
                cur_labs = cur_labs[:-1]
            if is_dlink(cur_labs) and len(cur_labs)>=1:
                t = "t2"
            else:
                # type3: special predicate
                SPEC_LS = {f'syn:{z}' for z in ['^nmod', '^amod', '^compound']}
                while len(cur_labs)>0 and cur_labs[0] in SPEC_LS:
                    cur_labs = cur_labs[1:]
                if is_dlink(cur_labs) and len(cur_labs) >= 1:
                    t = "t3"
                else:
                    # type4: need to go to another frame
                    GO_LS = {f'syn:{z}' for z in ['^nmod', 'conj', '^conj', 'acl', '^acl', 'advcl', '^advcl', 'csubj', '^csubj', 'ccomp', '^ccomp', 'xcomp', '^xcomp']}
                    if all('ARG' in z or z in GO_LS for z in cur_labs[:-1]) and is_dlink(cur_labs[-1:]) and len(cur_labs) >= 1:
                        t = f't4-{len(cur_labs)-1}'
                    else:  # UNK?
                        t = 't5'
        # --
        return t, path_labs

    def get_frame(self, evt, df=None):
        return self.onto.find_frame(evt.label, df)

    def print_frame(self, evt):
        from msp2.tasks.zmtl3.main.stream import show_outputs
        show_outputs([evt])

    # analyze one frame
    def ana_frame(self, evt, score_thr=0., topk=1, margin=2., tau=1., eoff_alpha=0., topk_k=2, topk_thr=0.):
        frame = self.get_frame(evt)
        frame.build_role_map(nc_filter=(lambda _name: _name in self.filter_noncore), force_rebuild=True)
        # --
        arg_names = list(frame.role_map.keys())
        gold_args = {k: [] for k in arg_names}
        for arg in evt.args:
            set_ee_heads(arg.mention.sent)
            if arg.label in gold_args:
                gold_args[arg.label].append(arg.mention.shead_token.get_indoc_id(True))
        pred_args = {k: [] for k in arg_names}
        if 'arg_scores' in evt.info:
            for k, vs in evt.info['arg_scores'].items():
                if k in pred_args:
                    _eoff = 0.
                    # --
                    if eoff_alpha > 0:  # find max-explicit
                        for k2 in arg_names:
                            if k2 != k:
                                for nn in gold_args[k2]:
                                    _eoff = max(vs.get(nn,0.), _eoff)
                    _eoff *= eoff_alpha
                    # --
                    # breakpoint()
                    ranked_vs = [((s-_eoff)/tau, k2) for k2,s in vs.items()]
                    ranked_vs.sort(reverse=True)
                    # --
                    if topk_thr > 0:
                        _tmp_vs = MathHelper.softmax(np.asarray([z[0] for z in ranked_vs] + [score_thr]))
                        if sum(_tmp_vs[:-1][:topk_k]) < topk_thr:
                            continue  # no adding for this one!
                    # --
                    ranked_vs = [(s,k2) for s,k2 in ranked_vs[:topk] if s>score_thr]
                    if len(ranked_vs) > 0:
                        _thr = ranked_vs[0][0] - margin
                        pred_args[k].extend([k2 for s,k2 in ranked_vs if s>_thr])
        hit_args = {k: list(set(pred_args[k]) & set(gold_args[k])) for k in arg_names}
        # --
        # get counts
        cc = Counter({
            'frame': 1,
            'gold': sum(len(z) for z in gold_args.values()),
            'gold_hit': sum(len(z) for z in hit_args.values()),
            'pred': sum(len(z) for z in pred_args.values()),
            'pred_hit': sum(len(z) for z in hit_args.values()),
            'a': len(arg_names),
            'a_gold': sum(len(gold_args[a])>0 for a in arg_names),
            'a_pred': sum(len(pred_args[a])>0 for a in arg_names),
            'a_hit': sum(len(hit_args[a])>0 for a in arg_names),
            'a_predE': sum(len(pred_args[a])>0 and len(gold_args[a])==0 for a in arg_names),
        })
        return cc

    def ana_frames(self, frames, **kwargs):
        cc = Counter()
        if isinstance(frames, str):
            frames = self.get_var(frames)
        for f in frames:
            if self.get_frame(f) is not None:
                cc += self.ana_frame(f, **kwargs)
        # --
        OtherHelper.printd(cc, try_div=True)
        return cc

    @classmethod
    def get_ann_type(cls):
        return MyAnnotationTask

class MyAnnotationTask(FrameAnnotationTask):
    def obj_info(self, obj, **kwargs) -> str:
        # --
        from msp2.tasks.zmtl3.main.stream import show_outputs
        for z in obj:
            show_outputs([z])
        # --
        # return None  # if None, then no info!
        return super().obj_info(obj, **kwargs)

    def do_ap(self, sent_evt=1, **kwargs):
        super().do_ap(sent_evt=sent_evt, **kwargs)
        zlog(f"Cur_obj's ea: {getattr(self.cur_obj, 'ea_info', None)}")

    def do_aa(self, *p_ea: str):
        _obj = self.cur_obj
        _gold, _preds = self.cur_obj.gold, self.cur_obj.preds
        _eas = [[[z3 for z3 in z2.split(",") if z3] for z2 in z.split("/")] for z in list(p_ea)]
        if len(_preds) != len(_eas) or any(len(zz)!=2 or len(zz[0])!=len(_gold.args)
                                           or len(zz[1])!=len(_preds[ii].args) for ii,zz in enumerate(_eas)):
            zlog(f"No change since len(arg) mismatch!")
        else:
            _obj.ea_info = _eas
        zlog(f"EA_INFO after change: {getattr(_obj, 'ea_info', None)}")
        # --

# --
def main(*args):
    conf = init_everything(MyAnalyzerConf(), args)
    ana = MyAnalyzer(conf)
    # --
    if conf.ana_fall:
        fl = ana.get_var('fl')
        for alpha in [0, 1]:
            zlog(f"Run with alpha={alpha}")
            for i in range(len(fl[0].preds)):
                ana.ana_frames([z[i+1] for z in fl], eoff_alpha=alpha)
    # --
    if conf.do_loop:
        ana.main()
    # --

# --
if __name__ == '__main__':
    main(*sys.argv[1:])

"""
# --
python3 -m pdb -m msp2.tasks.zmtl3.scripts.misc.ana_arg span_mode_frame:shead span_mode_arg:shead gold:? preds:?
# analyze pb/nb/fn & syn
python3 -m pdb -m msp2.tasks.zmtl3.scripts.misc.ana_arg span_mode_frame:shead span_mode_arg:shead 'pfilter:lambda x: x.label.startswith("syn:")' gold:../../events/data/data21f/en.ewt.dev.ud.json preds:../../events/data/data21f/en.ewt.dev.ud.json
-> (with dev)
fg al 'd.gold is not None and d.gold.label not in ["V", "C-V"]' 'len(self.ana_path(d.gold)[1])'
# fn: 1)0.6145, 0)0.8259, 2)0.9518, 3)0.9876, 4)0.9949
# pb: 1)0.8203, 2)0.9859, 3)0.9981
# nb: 1)0.6727, 2)0.8273, 0)0.9626, 3)0.9936
# ace: 1)0.4295, 2)0.7220, 3)0.8590, 4)0.9289, 5)0.9578, 6)0.9829
# ere: 1)0.5123, 2)0.7836, 3)0.9041, 4)0.9616, 5)0.9795, 6)0.9877
# --
python3 -m pdb -m msp2.tasks.zmtl3.scripts.misc.ana_arg onto:pbfn span_mode_frame:shead span_mode_arg:shead gold:../../events/data/data21f/en.ewt.dev.ud.json preds:../qadistill/en.ewt.dev.ud.q1.json
=> self.ana_frames([z[1] for z in self.get_var('fl')])
"""
