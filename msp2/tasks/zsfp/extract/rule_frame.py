#

# rule-based frame extractor
# -> lexical target + most_freq labeling

__all__ = [
    "RuleFrameExtractorConf", "RuleFrameExtractor",
]

from typing import List, Dict, Union
from msp2.tools.annotate import Annotator, AnnotatorConf
from msp2.data.inst import yield_sents, HeadFinder, Doc, Sent, Token
from .constrainer import LexConstrainer, LexConstrainerConf
from .rule_target import BlacklistRule, BlacklistRule_semafor

# =====

class RuleFrameExtractorConf(AnnotatorConf):
    def __init__(self):
        super().__init__()
        # --
        self.ftag = "evt"
        # =====
        # lex
        self.use_trg = False  # pick up the targets and only predict frames
        self.use_trg_lu = True  # if use_trg, further use its LU for looking up?
        self.lex_conf = LexConstrainerConf()
        self.lex_load_file = ""
        # --
        self.trg_max_wlen = 2  # check how many at most?
        # blacklist
        self.brule_semafor = 1  # use 'BlacklistRule_semafor'?

@Annotator.reg_decorator("rule_frame", conf=RuleFrameExtractorConf)
class RuleFrameExtractor(Annotator):
    def __init__(self, conf: RuleFrameExtractorConf):
        super().__init__(conf)
        conf: RuleFrameExtractorConf = self.conf
        # --
        # lex entries
        self.cons_lex = LexConstrainer(conf.lex_conf)
        self.cons_lex.load(conf.lex_load_file)
        # blacklist: (check each rule for exclusion)
        self.blacklist: List[BlacklistRule] = []
        if conf.brule_semafor:
            self.blacklist.append(BlacklistRule_semafor())

    def annotate(self, insts: List[Union[Doc, Sent]]):
        self.predict(insts)

    def _predict_frame(self, res: Dict):
        # simply rank by 1) count, 2) string
        return min(res.keys(), key=lambda x: (-res[x], x))

    def predict(self, insts: List[Union[Doc, Sent]]):
        conf: RuleFrameExtractorConf = self.conf
        cons_lex = self.cons_lex
        # --
        for sent in yield_sents(insts):
            if not conf.use_trg:
                # do extract
                sent_toks = sent.get_tokens()
                sent.delete_frames(conf.ftag)
                # extract
                cur_widx, cur_slen = 0, len(sent)
                while cur_widx < cur_slen:
                    next_inc = 1
                    for cur_wlen in reversed(range(1,conf.trg_max_wlen+1)):
                        if cur_widx+cur_wlen >= cur_slen: continue
                        res = cons_lex.get(cons_lex.span2feat(sent, cur_widx, cur_wlen))
                        if res is not None:  # hit!!
                            # check blacklist!
                            hwidx = cons_lex.hf.find_shead(sent, cur_widx, cur_wlen)
                            key_tok, mention_toks = sent_toks[hwidx], sent_toks[cur_widx:cur_widx+cur_wlen]
                            if not any(rule.hit(key_tok, mention_toks, sent_toks) for rule in self.blacklist):  # ok!
                                f = sent.make_frame(cur_widx, cur_wlen, conf.ftag)
                                f.set_label(self._predict_frame(res))
                                next_inc = cur_wlen  # make it non-overlapping (greedy)
                    cur_widx += next_inc
            else:
                # only looking up
                for frame in sent.get_frames(conf.ftag):
                    frame.set_label(None)
                    # --
                    if conf.use_trg_lu:
                        res = cons_lex.get(cons_lex.lu2feat(frame.info.get("luName")))
                    else:
                        cur_widx, cur_wlen = frame.mention.get_span()
                        res = cons_lex.get(cons_lex.span2feat(sent, cur_widx, cur_wlen))
                    if res is not None:  # hit!
                        frame.set_label(self._predict_frame(res))
        # --

# --
# test
"""
# decode
# use_trg=0
PYTHONPATH=../src/ python3 -m msp2.cli.annotate msp2.tasks.zsfp.extract.rule_frame/rule_frame R.input_path:fn15_fulltext.dev.json W.output_path:_tmp.json lex_load_file:../run_voc/cons_lex.json  # fn_styled=1
PYTHONPATH=../src/ python3 -m msp2.cli.annotate msp2.tasks.zsfp.extract.rule_frame/rule_frame R.input_path:fn15_fulltext.dev.json W.output_path:_tmp.json lex_load_file:../run_voc/cons_lex2.json  # fn_styled=0
# use_trg=1
PYTHONPATH=../src/ python3 -m msp2.cli.annotate msp2.tasks.zsfp.extract.rule_frame/rule_frame R.input_path:fn15_fulltext.dev.json W.output_path:_tmp.json lex_load_file:../run_voc/cons_lex.json use_trg:1  # fn_styled=1
PYTHONPATH=../src/ python3 -m msp2.cli.annotate msp2.tasks.zsfp.extract.rule_frame/rule_frame R.input_path:fn15_fulltext.dev.json W.output_path:_tmp.json lex_load_file:../run_voc/cons_lex2.json use_trg:1 use_trg_lu:0  # fn_styled=0
# analyze
PYTHONPATH=../src/ python3 -m msp2.cli.analyze frame main.input_path:fn15_fulltext.dev.json extra.input_path:_tmp.json do_eval:1 skip_gold_empty:1
"""
