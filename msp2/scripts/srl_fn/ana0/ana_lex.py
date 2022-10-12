#

# especially analyze lexicons

from collections import Counter, defaultdict
from msp2.data.inst import yield_sents
from msp2.data.rw import ReaderGetterConf
from msp2.tasks.zsfp.extract.constrainer import *
from msp2.utils import zlog, wrap_color, DictHelper

# --
def _predict_frame(res):
    if res is None: return None
    # simply rank by 1) count, 2) string
    return min(res.keys(), key=lambda x: (-res[x], x))

def _stat_cons(cons_lex):
    nitem_counts = {}
    for k, v in cons_lex.cmap.items():
        c = len(v)
        nitem_counts[c] = nitem_counts.get(c,0) + 1
    return DictHelper.get_counts_info_table(nitem_counts).to_string()
# --

def main(file: str, lex0: str, lex1: str):
    # stat = Counter()
    stat = defaultdict(int)
    # --
    reader = ReaderGetterConf().get_reader(input_path=file)
    cons_lex0 = LexConstrainer.load_from_file(lex0)
    cons_lex1 = LexConstrainer.load_from_file(lex1)
    # show the lex
    zlog(f"StatLex0:\n{_stat_cons(cons_lex0)}")
    zlog(f"StatLex1:\n{_stat_cons(cons_lex1)}")
    # --
    for sent in yield_sents(reader):
        for frame in sent.events:
            stat["all"] += 1
            gold_fname = frame.type
            gold_lu = frame.info.get("luName")
            res_lu0 = cons_lex0.get(cons_lex0.lu2feat(gold_lu))
            pred0 = _predict_frame(res_lu0)
            cur_widx, cur_wlen = frame.mention.get_span()
            feat_lu1 = cons_lex1.span2feat(sent, cur_widx, cur_wlen)
            res_lu1 = cons_lex1.get(feat_lu1)
            pred1 = _predict_frame(res_lu1)
            # --
            stat["corr0"] += int(gold_fname == pred0)
            stat["corr1"] += int(gold_fname == pred1)
            stat["none0"] += int(pred0 is None)
            stat["none1"] += int(pred1 is None)
            stat["wc0"] += int(res_lu0 is not None and gold_fname not in res_lu0)  # wrong constraint
            stat["wc1"] += int(res_lu1 is not None and gold_fname not in res_lu1)  # wrong constraint
            stat["mapone0"] += int(res_lu0 is not None and len(res_lu0)==1)
            stat["mapone1"] += int(res_lu1 is not None and len(res_lu1)==1)
            stat["maponecorr0"] += int(gold_fname == pred0 and len(res_lu0)==1)
            stat["maponecorr1"] += int(gold_fname == pred1 and len(res_lu1)==1)
            if pred0 != pred1:
                stat["diff"] += 1
                if pred0 == gold_fname:
                    stat["diff_win0"] += 1
                elif pred1 == gold_fname:
                    stat["diff_win1"] += 1
                else:
                    stat["diff_both_lose"] += 1
                zlog(f"{gold_fname} {wrap_color(gold_lu, bcolor='red')}={res_lu0} {wrap_color(feat_lu1, bcolor='red')}={res_lu1}")
    # --
    zlog(f"#--\n{stat}")

if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# --
# PYTHONPATH=../src/ python3 ana_lex.py ./fn15_fulltext.dev.json ../run_voc/cons_lex{0,2}.json
# simulate with ana_frame
"""
# go!!
PYTHONPATH=../src/ python3 -m pdb -m msp2.cli.analyze frame main.input_path:../fn_parsed/fn15_fulltext.dev.json extra.input_path:_tmp do_eval:1
# load frame
frames = eval "m2.default_json_serializer.from_file('../fn_parsed/fn15_frames.json')" "msp2.utils"
frame_keys = eval "list(vs.frames.keys())"
group frame_keys "(min(20,len(vs.frames[d]['lexUnit'])), )" -- "sum_key:id"
# load lex
lex = eval "m2.LexConstrainer.load_from_file('../run_voc/cons_lex_lp0.json')" "msp2.tasks.zsfp.extract.constrainer"
lex_items = eval "list(vs.lex.cmap.items())"
group lex_items "(len(d[1]), )" -- "sum_key:id"
# on actual data
group ep "((lambda x: 0 if x is None else len(x))(vs.lex.get(vs.lex.lu2feat(d.gold.info['luName']))), )" -- "sum_key:id"
# --
group ep "(vs.lex.lu2feat(d.gold.info['luName'])==vs.lex.span2feat(d.pred.sent, d.pred.mention.widx, d.pred.mention.wlen), d.gold.type==d.pred.type, )"
nm = filter ep "(vs.lex.lu2feat(d.gold.info['luName'])!=vs.lex.span2feat(d.pred.sent, d.pred.mention.widx, d.pred.mention.wlen) and d.gold.type!=d.pred.type)"
eval "print(self.cur_ann_task.cur_obj.gold.info['luName'], self.cur_ann_task.cur_obj.gold.mention.get_tokens()[0].upos)"
# =====
# =====
# load and compare three of them
frames1 = eval "[f for doc in list(m2.ReaderGetterConf().get_reader(input_path='../fn_parsed/fn15_fulltext.dev.json')) for sent in doc.sents for f in sent.events]" "msp2.data.rw"
frames2 = eval "[f for doc in list(m2.ReaderGetterConf().get_reader(input_path='../run_frames/go0921_lab/_out.dev.LU0')) for sent in doc.sents for f in sent.events]" "msp2.data.rw"
frames3 = eval "[f for doc in list(m2.ReaderGetterConf().get_reader(input_path='../run_frames/go0921_lab/_out.dev.LU1')) for sent in doc.sents for f in sent.events]" "msp2.data.rw"
all3 = eval "[list(p3) for p3 in zip(vs.frames1, vs.frames2, vs.frames3)]"
group all3 "(vs.lex.lu2feat(d[0].info['luName'])==vs.lex.span2feat(d[0].sent, d[0].mention.widx, d[0].mention.wlen), d[0].type==d[1].type, d[0].type==d[2].type, d[1].type==d[2].type)"
# --
f3p = filter all3 "vs.lex.lu2feat(d[0].info['luName'])!=vs.lex.span2feat(d[0].sent, d[0].mention.widx, d[0].mention.wlen) and d[0].type==d[2].type and d[1].type!=d[2].type"
"""
