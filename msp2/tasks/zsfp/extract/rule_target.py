#

# rule-based target extractor
"""
1. features/key: orig_tok, lemma, pos; lowercase?
2. which tok is the start one for matching? (first or last or head?)
3. two phase: whitelist candidates + blacklist pruning
"""

__all__ = [
    "RuleTargetExtractorConf", "RuleTargetExtractor", "BlacklistRule", "BlacklistRule_semafor",
]

import sys
from typing import Iterable, List, Dict, Union
from collections import defaultdict
from msp2.utils import Conf, zlog, zwarn, JsonSerializable, init_everything, default_json_serializer, OtherHelper
from msp2.data.inst import yield_sents, HeadFinder, Doc, Sent, Token
from msp2.data.rw import ReaderGetterConf
from msp2.tools.annotate import Annotator, AnnotatorConf

# =====

class RuleTargetExtractorConf(AnnotatorConf):
    def __init__(self):
        super().__init__()
        # --
        # general
        self.ftag = "evt"  # sent.events
        # used in training
        self.key_tok = "head"  # use which tok to be the key token? 0/-1/head?
        self.feature_tok_f = "lambda t: f'{t.word}|{t.upos}'"  # what feature to extract from Token for matching
        self.set_head = True  # set head by HF
        # the above are from training set
        # =====
        # whitelist
        self.wl_max_wlen = 100  # wlen <= this
        self.wl_min_count = 0  # must >= this
        self.wl_alter_f = "lambda t: False"  # alternative wl filter
        # blacklist rules
        self.brule_semafor = 1  # use 'BlacklistRule_semafor'?
        # =====
        self.preload_file = ''  # load savepoint

    def valid_json_fields(self):  # only save training related ones!!
        return ["ftag", "key_tok", "feature_tok_f", "set_head"]

@Annotator.reg_decorator("rule_target", conf=RuleTargetExtractorConf)
class RuleTargetExtractor(JsonSerializable, Annotator):
    def __init__(self, conf: RuleTargetExtractorConf):
        super().__init__(conf)
        conf: RuleTargetExtractorConf = self.conf
        # whitelist: (items are sorted by matching order)
        self.whitelist = {}  # key-feat => {'items': [{'left': [], 'right': [], 'count': int}], 'count': int}
        # blacklist: (check each rule for exclusion)
        self.blacklist: List[BlacklistRule] = []
        if conf.brule_semafor:
            self.blacklist.append(BlacklistRule_semafor())
        # preload
        if conf.preload_file:
            self.from_json(default_json_serializer.from_file(conf.preload_file))
            zlog(f"Load RuleTargetExtractor from {conf.preload_file}")
        # --
        # compile
        self.feat_tok_f = eval(conf.feature_tok_f)  # lambda Token: features
        self.wl_alter_f = eval(conf.wl_alter_f)

    @classmethod
    def cls_from_json(cls, data: Dict, **kwargs):
        conf = RuleTargetExtractorConf.cls_from_json(data["conf"])
        extractor = cls(conf)
        extractor.from_json(data)
        return extractor

    def to_json(self):
        return {"conf": self.conf.to_json(), "whitelist": self.whitelist}

    def from_json(self, data: Dict):
        conf0, conf1 = data["conf"], self.conf.to_json()
        if conf0 != conf1:
            zwarn(f"Mismtach in load:{conf0} and current:{conf1}")
        self.conf.from_json(conf0)
        self.whitelist = data["whitelist"]  # directly assign

    # =====

    def feat_toks(self, s: Sent):
        sent_toks = s.get_tokens()
        sent_tok_feats = [self.feat_tok_f(t) for t in sent_toks]
        return sent_toks, sent_tok_feats

    # todo(note): some of the prunings (like min_count and blacklist can be applied in collecting for efficiency)
    def predict(self, insts: List[Union[Doc, Sent]]):
        conf: RuleTargetExtractorConf = self.conf
        wl_min_count, wl_max_wlen = conf.wl_min_count, conf.wl_max_wlen
        # --
        for sent in yield_sents(insts):
            sent.delete_frames(conf.ftag)  # first clear original ones!
            # extract
            sent_toks, sent_tok_feats = self.feat_toks(sent)
            for one_widx, one_feat in enumerate(sent_tok_feats):
                if one_feat in self.whitelist:
                    possible_items = self.whitelist[one_feat]['items']
                else:
                    possible_items = []
                # check each one: the order means priority
                hit_span = None
                for one_item in possible_items:
                    if one_item['count'] < wl_min_count:
                        continue  # must >= min_count
                    left_feats, right_feats = one_item['left'], one_item['right']
                    left_len, right_len = len(left_feats), len(right_feats)
                    if left_len + right_len > wl_max_wlen:
                        continue  # must <= max_wlen
                    if sent_tok_feats[max(0,one_widx-left_len):one_widx] == left_feats \
                            and sent_tok_feats[one_widx+1:one_widx+1+right_len] == right_feats:
                        # check blacklist
                        key_tok, mention_toks = sent_toks[one_widx], sent_toks[one_widx-left_len:one_widx+right_len+1]
                        if not any(rule.hit(key_tok, mention_toks, sent_toks) for rule in self.blacklist):
                            hit_span = (one_widx-left_len, left_len+1+right_len)
                            break
                # check alternative wl
                if hit_span is None and self.wl_alter_f(sent_toks[one_widx]):
                    hit_span = (one_widx, 1)  # make this token
                # new one -- directly adding (no frame-type)
                if hit_span is not None:
                    f = sent.make_frame(hit_span[0], hit_span[1], conf.ftag, adding=True)
        # --

    @staticmethod
    def train(stream: Iterable, conf: RuleTargetExtractorConf):
        extractor = RuleTargetExtractor(conf)
        hf = HeadFinder({"ef": "NOUN", "evt": "VERB"}.get(conf.ftag))
        need_set_head = (conf.set_head and conf.key_tok=="head")
        key_tok_offset_f = {"0": lambda xs,shoff: 0, "-1": lambda xs,shoff: len(xs)-1,
                            "head": lambda xs,shoff: shoff}[conf.key_tok]
        # collect whitelist: all positive examples
        whitelist = {}
        for sent in yield_sents(stream):
            _, sent_tok_feats = extractor.feat_toks(sent)
            for frame in sent.get_frames(conf.ftag):
                mention = frame.mention
                if need_set_head:  # set head if needed
                    hf.set_head_for_mention(mention)
                mention_tok_feats = sent_tok_feats[mention.widx:mention.wridx]
                key_offset = key_tok_offset_f(mention_tok_feats, mention.get_shoff())  # which one to regard as key
                # collect features
                left_feats, key_feat, right_feats = \
                    mention_tok_feats[:key_offset], mention_tok_feats[key_offset], mention_tok_feats[key_offset+1:]
                # todo(note): simply enumerate all items for matching!!
                key_entry = whitelist.get(key_feat, None)
                if key_entry is None:
                    whitelist[key_feat] = {'items': [{'left': left_feats, 'right': right_feats, 'count': 1}], 'count': 1}
                else:
                    key_entry['count'] += 1
                    key_items = key_entry['items']
                    hit_item = False
                    for one_item in key_items:
                        if one_item['left']==left_feats and one_item['right']==right_feats:  # hit
                            one_item['count'] += 1
                            hit_item = True
                            break
                    if not hit_item:
                        key_items.append({'left': left_feats, 'right': right_feats, 'count': 1})
        # sort all items in whitelist: by default from larger to smaller span
        for key_entry in whitelist.values():
            key_entry['items'].sort(key=lambda z: len(z['left'])+len(z['right']), reverse=True)
        # setup and return
        extractor.whitelist = whitelist
        # print details
        pp_stat, pp_details = extractor.stat_whitelist()
        zlog(f"Build Extractor with whitelist: {pp_stat}\n {pp_details}")
        return extractor

    def stat_whitelist(self):
        ret = defaultdict(int)
        for entry in self.whitelist.values():
            ret["num_entry"] += 1
            ret["num_item"] += len(entry['items'])
            ret["count_all"] += entry["count"]
            ret["num_item_survive"] += sum((1 if item['count']>=self.conf.wl_min_count else 0) for item in entry['items'])
        # detailed str
        detailed_lines = []
        for ki, key in enumerate(sorted(list(self.whitelist.keys()), key=lambda x: self.whitelist[x]['count'], reverse=True)):
            line = f"#[{ki}] {key} => {self.whitelist[key]}"
            detailed_lines.append(line)
        return ret, "\n".join(detailed_lines)

    def annotate(self, insts: List[Union[Doc, Sent]]):
        self.predict(insts)

# -----
# rules
class BlacklistRule:
    def hit(self, key_tok: Token, mention_toks: List[Token], sent_toks: List[Token]):
        raise NotImplementedError()  # hit blacklist rule

# the one adopted in semafor, modified from J&N'07
# note: this one is only for English!!
class BlacklistRule_semafor(BlacklistRule):
    def __init__(self):
        # prep set
        self.prep_set = set("above,against,at,below,beside,by,in,on,over,under,after,before,into,to,through,"
                            "as,for,so,with,of".split(","))

    def hit(self, key_tok: Token, mention_toks: List[Token], sent_toks: List[Token]):
        key_lemma = key_tok.lemma.lower()
        # have was retained only if had an object
        if key_lemma == "have":
            if any(ch_tok.deplab.startswith("obj") for ch_tok in key_tok.ch_toks):
                pass
            else:
                return True
        # be was retained only if it was preceded bythere
        if key_lemma == "be":
            if key_tok.widx==0 or sent_toks[key_tok.widx-1].lemma.lower() != "there":
                return True
        # will was removed in its modal sense
        if key_lemma == "will" and key_tok.deplab.startswith("aux"):
            return True
        # remove two special ones
        str_all_lemmas = " ".join([z.lemma.lower() for z in mention_toks])
        if str_all_lemmas in ["of course", "in particular"]:
            return True
        # modify 1: remove all prepositions
        if len(mention_toks) == 1:
            key_upos = key_tok.upos
            if key_upos == "ADP" or key_lemma in self.prep_set:
                return True
        # modify 2: no remove supp.
        # --
        return False

# =====
# training (collecting whitelist)

class TrainConf(Conf):
    def __init__(self):
        self.train = ReaderGetterConf()
        self.econf = RuleTargetExtractorConf()
        self.save_name = "rule.model.json"

def main(*args):
    conf: TrainConf = init_everything(TrainConf(), args)
    # --
    reader = conf.train.get_reader()
    inputs = yield_sents(reader)
    extractor = RuleTargetExtractor.train(inputs, conf.econf)
    # save
    zlog(f"Save extractor to {conf.save_name}")
    default_json_serializer.to_file(extractor.to_json(), conf.save_name)

if __name__ == '__main__':
    main(*sys.argv[1:])

# b msp2/tasks/zsfp/extract/rule_target:
# train and test
"""
# to train
PYTHONPATH=../src/ python3 -m pdb -m msp2.tasks.zsfp.extract.rule_target train.input_path:fn15_fulltext.train.json |& tee _log.rt
# to decode
PYTHONPATH=../src/ python3 -m pdb -m msp2.cli.annotate msp2.tasks.zsfp.extract.rule_target/rule_target R.input_path:fn15_fulltext.dev.json W.output_path:_tmp.json preload_file:rule.model.json
# analyze
PYTHONPATH=../src/ python3 -m pdb -m msp2.cli.analyze frame main.input_path:fn15_fulltext.dev.json extra.input_path:_tmp.json do_eval:1 skip_gold_empty:1
"""
