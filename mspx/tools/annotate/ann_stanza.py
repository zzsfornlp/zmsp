#

# annotate with Stanza

__all__ = [
    "AnnotatorStanzaConf", "AnnotatorStanza",
]

from typing import List, Union
from mspx.data.inst import Doc, Sent
from mspx.utils import zwarn
from .annotator import *

"""
pip install stanza
import stanza
stanza.download('en')
nlp = stanza.Pipeline('en')
"""

@AnnotatorConf.rd('stanza')
class AnnotatorStanzaConf(AnnotatorConf):
    def __init__(self):
        super().__init__()
        # specify the stanza pipeline
        self.stanza_lang = "en"
        self.stanza_dir = ""
        self.stanza_processors = 'tokenize,pos,lemma,depparse'.split(',')  # tokenize,mwt,pos,lemma,depparse,ner
        self.stanza_use_gpu = False
        self.stanza_dpar_level = ''
        self.stanza_others = {}  # other options
        # --
        self.ann_input_mode = "tokenized"  # input mode: raw / ssplit / tokenized
        self.ann_rm_text = True  # remove text after tokenization

@AnnotatorStanzaConf.conf_rd()
class AnnotatorStanza(Annotator):
    def __init__(self, conf: AnnotatorStanzaConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: AnnotatorStanzaConf = self.conf
        # init options
        import stanza
        pipeline_options = {"processors": ",".join(conf.stanza_processors), "use_gpu": conf.stanza_use_gpu}
        if conf.stanza_lang:
            pipeline_options["lang"] = conf.stanza_lang
        if conf.stanza_dir:
            pipeline_options["dir"] = conf.stanza_dir
        pipeline_options.update(conf.stanza_others)
        # special options for ssplit and tokenize
        tok_options = {"raw": {},
                       "ssplit": {"tokenize_no_ssplit": True},
                       "tokenized": {"tokenize_pretokenized": True}}[conf.ann_input_mode]
        self.ann_ff = getattr(self, f"ann_{conf.ann_input_mode}")
        pipeline_options.update(tok_options)
        # --
        self.pipeline = stanza.Pipeline(**pipeline_options)
        # has which components?
        self.processors_set = set(conf.stanza_processors)
        self.pred_upos, self.pred_lemma, self.pred_dep = [z in self.processors_set for z in ["pos","lemma","depparse"]]

    def annotate(self, insts: List[Doc]):
        return self.ann_ff(insts)

    def ann_raw(self, insts: List[Doc]):
        for doc in insts:  # note: process one by one!
            text = doc.get_text()
            res = self.pipeline(text)
            self.put_doc(doc, res)

    def ann_ssplit(self, insts: List[Doc]):
        all_sents = [s for d in insts for s in d.sents]
        res = self.pipeline([s.get_text() for s in all_sents])
        assert len(res.sentences) == len(all_sents)
        for orig_sent, nlp_sent in zip(all_sents, res.sentences):
            self.put_sent(orig_sent, nlp_sent, put_positions=False)

    def ann_tokenized(self, insts: List[Doc]):
        all_sents = [s for d in insts for s in d.sents]
        res = self.pipeline([list(s.seq_word.vals) for s in all_sents])
        assert len(res.sentences) == len(all_sents)
        for orig_sent, nlp_sent in zip(all_sents, res.sentences):
            self.put_sent(orig_sent, nlp_sent, put_tokens=False, put_positions=False)

    def put_doc(self, orig_doc: Doc, nlp_doc):
        _ann_rm_text = self.conf.ann_rm_text
        # --
        assert orig_doc.get_text() == nlp_doc.text, "Error: Input & Output text not match!"
        orig_doc.clear_sents()  # clean sents if there are originally
        # process nlp_doc
        sent_positions = []
        new_nlp_sents = nlp_doc.sentences
        for nlp_sent in new_nlp_sents:
            if len(nlp_sent.tokens) == 0:
                continue  # ignore empty ones
            sent_start_char, sent_end_char = nlp_sent.tokens[0].start_char, nlp_sent.tokens[-1].end_char
            new_s = Sent()
            self.put_sent(new_s, nlp_sent)  # annotate sents
            orig_doc.add_sent(new_s)  # add sent
            sent_positions.append((sent_start_char, sent_end_char-sent_start_char))
        if _ann_rm_text:
            orig_doc.build_text(None)
        else:
            orig_doc.build_sent_positions(sent_positions)  # put positions
        # --

    def put_sent(self, orig_sent: Sent, nlp_sent, put_tokens=True, put_positions=True):
        _ann_rm_text = self.conf.ann_rm_text
        _dpar_level = self.conf.stanza_dpar_level
        if _dpar_level is not None and len(_dpar_level) > 0:
            _dpar_level = int(_dpar_level)
        else:
            _dpar_level = None
        # --
        # here we process the words!
        list_words = []
        list_word_positions = []
        list_uposes = []
        list_lemmas = []
        list_dep_heads = []
        list_dep_labels = []
        # find them!!
        for w in nlp_sent.words:
            list_words.append(w.text)
            if getattr(w, 'start_char', None) is not None:
                list_word_positions.append((w.start_char, w.end_char-w.start_char))  # [widx, wlen]
            else:  # note: deal with mwt for certain languages, todo(+N): currently use words
                _wt = w.parent
                list_word_positions.append((_wt.start_char, _wt.end_char-_wt.start_char))  # [widx, wlen]
            list_uposes.append(w.upos)
            list_lemmas.append(w.lemma)
            list_dep_heads.append(w.head)
            list_dep_labels.append(w.deprel)
        # add them
        if put_tokens:
            orig_sent.build_words(list_words)
            if not _ann_rm_text:
                orig_sent.build_text(nlp_sent.text)
                if put_positions:
                    orig_sent.build_word_positions(list_word_positions)
        if self.pred_upos:
            orig_sent.build_uposes(list_uposes)
        if self.pred_lemma:
            orig_sent.build_lemmas(list_lemmas)
        if self.pred_dep:
            orig_sent.build_dep_tree(list_dep_heads, list_dep_labels)
            if _dpar_level is not None:
                orig_sent.tree_dep.build_labels(orig_sent.tree_dep.get_labels(_dpar_level))
        # --

# --
# b mspx/tools/annotate/ann_stanza:124
