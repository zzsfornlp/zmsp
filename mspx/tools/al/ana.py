#

# some analyzing

import pandas as pd
import numpy as np
from mspx.utils import default_pickle_serializer, init_everything, zlog, zglob1
from mspx.data.vocab import SeqVocab
from mspx.data.rw import WriterGetterConf
from mspx.proc.eval import FrameEvalConf
from mspx.tools.analyze import AnalyzerConf, Analyzer, MatchedList

class AlAnalyzerConf(AnalyzerConf):
    def __init__(self):
        super().__init__()
        self.vocab_path = ""
        self.task = ""  # task_type:*args
        self.output_file = ""
        self.frame_eval = FrameEvalConf()

@AlAnalyzerConf.conf_rd()
class AlAnalyzer(Analyzer):
    def __init__(self, conf: AlAnalyzerConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: AlAnalyzerConf = self.conf
        # --
        # load vocab
        self.vpack = default_pickle_serializer.from_file(zglob1(conf.vocab_path))
        # task
        self.task_spec = conf.task.split(":")
        self.frame_eval = conf.frame_eval.make_node()
        # load data
        self.all_insts, lists0, lists1 = self.read_data()
        self.set_var("sl", lists0, explanation="init")  # sent pair
        self.set_var("tl", lists1, explanation="init")  # token pair
        # --

    def get_topk(self, arrs, k=5, trg_name='qannK'):
        if (trg_name + "I") in arrs:
            return
        # --
        # simply check keys
        for strg_name in ["ext0_strg0", "extH_strg0", "dpar0_strg"]:
            if strg_name not in arrs:
                continue
            arr_strg = arrs[strg_name]
            # --
            import torch  # note: easier topk!
            _k = k
            t0 = torch.as_tensor(arr_strg.copy()).view([arr_strg.shape[0], -1])  # flatten, [L, ...]
            _vv, _ii = t0.topk(_k, dim=-1)  # [L, K]
            # arrs[trg_name + "I"] = _ii.numpy().astype(np.int16)
            # arrs[trg_name + "V"] = _vv.numpy().astype(np.float16)
            arrs[trg_name + "I"] = _ii.numpy()
            arrs[trg_name + "V"] = _vv.numpy()
            # delete origin ones
            del arrs[strg_name]
            if strg_name[:-1] + '1' in arrs:
                del arrs[strg_name[:-1] + '1']
            break
        # --

    def read_data(self):
        all_files, all_insts, all_sents, all_tokens = self.read_basic_data()
        s_lists = [MatchedList(z) for z in zip(*all_sents)]
        t_lists = [MatchedList(z) for z in zip(*all_tokens)]
        # --
        _task_spec = self.task_spec
        # doing AL specific processing!
        if _task_spec[0] == 'ner':
            _voc = self.vpack[0]
            use_cate_label = any(("___" in z) for z in _voc.full_i2w)
            _svoc = SeqVocab(_voc)
            _frame_cate = _task_spec[1:]
            for ms in s_lists:
                gold_s = ms[0]
                gold_frames = gold_s.get_frames(cates=_frame_cate)
                gold_tags = _svoc.spans2tags_str(
                    [z.mention.get_span() + (z.cate_label if use_cate_label else z.label,) for z in gold_frames], len(gold_s))[0][0]
                for gold_t in gold_s.get_tokens():
                    gold_t.gold_tag = gold_tags[gold_t.widx]
                for pred_s in ms[1:]:
                    _has_gold = len([z2 for z2 in pred_s.get_frames(cates=_frame_cate) if not z2.info.get('is_pred', False)])>0  # sentence has gold?
                    _has_query = any(z2.label=="_Q_" for z2 in pred_s.get_frames(cates=_frame_cate))  # sentence has query?
                    self.get_topk(pred_s.arrs)  # note: topk if needed
                    query_frames = pred_s.get_frames(cates=_frame_cate)
                    query_tags = _svoc.spans2tags_str(
                        [z.mention.get_span() + (z.label,) for z in query_frames], len(gold_s))[0][0]
                    query_tags = [z.split("-", 1)[-1] for z in query_tags]  # no need BIO here!
                    for pred_t in pred_s.get_tokens():  # cached!
                        pred_t.topk_vals = pred_s.arrs['qannKV'][pred_t.widx]  # [K]
                        pred_t.margin = pred_t.topk_vals[0] - pred_t.topk_vals[1]  # margin criterion
                        pred_t.topk_tags = _svoc.seq_idx2word(pred_s.arrs['qannKI'][pred_t.widx])  # [K]
                        pred_t.gold_tag = gold_tags[pred_t.widx]
                        pred_t.query_tag = query_tags[pred_t.widx]
                        pred_t.has_gold = _has_gold  # sent has_gold
                        pred_t.has_query = _has_query  # sent has_query
        elif _task_spec[0] == 'dpar':
            _voc = self.vpack[0]
            _lv = len(_voc)
            for ms in s_lists:
                gold_tree = ms[0].tree_dep
                gold_ones = list(zip(gold_tree.seq_head.vals, gold_tree.seq_label.vals))
                for pred_s in ms[1:]:
                    _has_gold = (pred_s.tree_dep is not None and any(z2>=0 for z2 in pred_s.tree_dep.seq_head.vals))
                    _has_query = (pred_s.tree_dep is not None and any(z2=='_Q_' for z2 in pred_s.tree_dep.seq_label.vals))
                    self.get_topk(pred_s.arrs)  # note: topk if needed
                    query_tree = pred_s.tree_dep
                    query_tags = list(query_tree.seq_label.vals)
                    for pred_t in pred_s.get_tokens():  # cached!
                        pred_t.topk_vals = pred_s.arrs['qannKV'][1+pred_t.widx]  # [K]
                        pred_t.margin = pred_t.topk_vals[0] - pred_t.topk_vals[1]  # margin criterion
                        pred_t.topk_hls = [(int(z)//_lv, _voc.idx2word(int(z)%_lv))
                                           for z in pred_s.arrs['qannKI'][1+pred_t.widx]]  # [K]
                        pred_t.gold_hl = gold_ones[pred_t.widx]
                        pred_t.gold_tag, pred_t.topk_tags = pred_t.gold_hl, pred_t.topk_hls  # for convenience!
                        pred_t.query_tag = query_tags[pred_t.widx]
                        pred_t.has_gold = _has_gold  # sent has_gold
                        pred_t.has_query = _has_query  # sent has_query
        elif _task_spec[0] == 'evt':  # evt argument
            _voc = self.vpack[0]
            from mspx.tools.analyze import AnalyzerFrame
            _evaler = self.frame_eval
            _, a_lists = AnalyzerFrame.read_frame_data(all_files, all_insts, _evaler)
            t_lists = a_lists  # simply replace that!
            # todo(+N): nope, the non-Q ones have been deleted, need the unc one!
        # --
        return all_insts, s_lists, t_lists

# --
def ana_main(*args):
    conf = AlAnalyzerConf()
    conf: AlAnalyzerConf = init_everything(conf, args)
    ana: AlAnalyzer = conf.make_node()
    ana.main()
    if conf.output_file:
        with WriterGetterConf().get_writer(output_path=conf.output_file) as writer:
            writer.write_insts(ana.all_insts[1])  # note: simply save pred[1]
    # --

# PYTHONPATH=../src/ python3 -mpdb -m mspx.tools.al.ana ...
if __name__ == '__main__':
    import sys
    ana_main(*sys.argv[1:])

# --
"""
python3 -mpdb -m mspx.tools.al.ana vocab_path: task: gold: preds:
python3 -mpdb -m mspx.tools.al.ana vocab_path:__vocabs/dpar_en/v_dpar0.pkl task:dpar gold:./*/run_try0908dpar_0/al.ref.json preds:./*/run_try0908dpar_0/iter01/data.query.json
python3 -mpdb -m mspx.tools.al.ana vocab_path:__vocabs/ner_en/v_ext0.pkl task:ner:ef gold:./*/run_try0908ner_0/al.ref.json preds:./*/run_try0908ner_0/iter01/data.query.json
# --
python3 -mpdb -m mspx.tools.al.ana vocab_path:__vocabs/ner_en/v_ext0.pkl task:ner:ef gold:__data/ner/data/en.train.json preds:??
python3 -mpdb -m mspx.tools.al.ana vocab_path:__vocabs/dpar_en/v_dpar0.pkl task:dpar gold:__data/ud/data/en_ewt.train.json preds:??
# do_loop:0 output_file:_tmp.json
# --
>> margin distr.
fg tl 'd[1].query_tag.startswith("_Q")' 'int(d[1].margin*10)' -- sum_key:id
>> query distr.
fg tl 'd[1].query_tag.startswith("_Q")' 'd[1].gold_tag' -- sum_key:ncount
fg tl 'd[1].query_tag.startswith("_Q")' 'min(10,abs(d[1].gold_hl[0]-d[1].widx))' -- sum_key:ncount
>> -- 'print_head:RES '
>> avg dist
eval 'np.mean([abs(t[1].gold_hl[0]-t[1].widx) for t in self.vars["tl"] if t[1].query_tag.startswith("_Q")])'
>> avg margin
eval 'np.mean([t[1].margin for t in self.vars["tl"] if t[1].query_tag.startswith("_Q")])'
>> accuracy
eval 'np.mean([t[1].gold_hl==t[1].topk_hls[0] for t in self.vars["tl"] if t[1].query_tag.startswith("_Q")])'
eval 'np.mean([t[1].gold_tag==t[1].topk_tags[0] for t in self.vars["tl"] if t[1].query_tag.startswith("_Q")])'
>> bins
fg tl 'not d[1].query_tag.startswith("**")' 'int(d[1].margin*10), d[1].gold_tag==d[1].topk_tags[0]' -- sum_key:id
fg tl 'not d[1].query_tag.startswith("**") and not(d[1].gold_tag=="O" and d[1].topk_tags[0]=="O")' 'int(d[1].margin*1), d[1].gold_tag==d[1].topk_tags[0]' -- sum_key:id
fg tl 'not d[1].query_tag.startswith("**") and not(d[1].gold_tag=="O" and d[1].topk_tags[0]=="O")' 'int(d[1].topk_vals[0]>0.9), d[1].gold_tag==d[1].topk_tags[0]' -- sum_key:id
# --
>> compare two for ST
zz=filter tl 'not d[1].query_tag.startswith("**") and not d[2].query_tag.startswith("**") and d[1].margin<1 and d[2].margin<1'
corr tl 'd[1].margin' 'd[2].margin'
corr zz 'd[1].margin' 'd[2].margin'
fg tl 'd[1].query_tag.startswith("_Q")' 'd[1].query_tag == d[2].query_tag' -- sum_key:id
fg tl 'd[1].query_tag.startswith("_Q")' 'int(d[1].margin < 0.99), d[1].query_tag == d[2].query_tag' -- sum_key:id
fg tl 'd[1].query_tag.startswith("_Q")' 'int(min(0.999,d[1].margin)*3), d[1].query_tag == d[2].query_tag' -- sum_key:id
fg tl '1' 'd[1].topk_tags[0] == d[2].topk_tags[0]' -- sum_key:id
fg tl '1' 'int(d[1].margin*5), d[1].topk_tags[0] == d[2].topk_tags[0]' -- sum_key:id
fg tl 'd[1].gold_tag!="O"' 'd[1].topk_tags[0] == d[2].topk_tags[0]' -- sum_key:id
fg tl 'd[1].gold_tag!="O"' 'int(d[1].margin*5), d[1].topk_tags[0] == d[2].topk_tags[0]' -- sum_key:id
eval 'np.mean([t[1].margin for t in self.vars["tl"] if t[1].query_tag.startswith("_Q")])'
eval 'np.mean([t[2].margin for t in self.vars["tl"] if t[2].query_tag.startswith("_Q")])'
# --
# more ana
fg tl 'd[1].has_gold and not d[1].query_tag.startswith("**")' 'd[1].query_tag.startswith("_Q"), d[1].gold_tag==d[1].topk_tags[0]' -- sum_key:id
fg tl 'not d[1].query_tag.startswith("**")' 'int(d[1].topk_vals[0]>0.99), d[1].gold_tag==d[1].topk_tags[0]' -- sum_key:id
fg tl 'd[1].has_gold and not d[1].query_tag.startswith("**")' 'd[1].query_tag.startswith("_Q"), d[1].gold_hl==d[1].topk_hls[0]' -- sum_key:id
fg tl 'not d[1].query_tag.startswith("**")' 'int(d[1].topk_vals[0]>0.99), d[1].gold_hl==d[1].topk_hls[0]' -- sum_key:id
# --
# check upos
fg tl 'd[0].gold_tag!="O"' 'd[0].upos'
group tl 'd[0].gold_tag!="O", d[0].upos'
group tl 'd[0].upos, d[0].gold_tag!="O"'
"""
