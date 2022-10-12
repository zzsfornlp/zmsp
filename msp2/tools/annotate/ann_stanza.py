#

# annotator with Stanza

__all__ = [
    "AnnotatorStanzaConf", "AnnotatorStanza",
]

from typing import List, Union
from msp2.data.inst import DataInstance, Doc, Sent
from .base import AnnotatorConf, Annotator
from .helper import InputHelper

# =====
"""
stanza.download('en')
nlp = stanza.Pipeline('en')
"""

class AnnotatorStanzaConf(AnnotatorConf):
    def __init__(self):
        super().__init__()
        # specify the stanza pipeline
        self.stanza_lang = ""
        self.stanza_dir = ""
        self.stanza_processors = []  # tokenize,mwt,pos,lemma,depparse,ner
        self.stanza_use_gpu = True
        self.stanza_others = {}  # other options
        # speicial input mode
        self.stanza_input_mode = "raw"  # raw / ssplit / tokenized

@Annotator.reg_decorator("stanza", conf=AnnotatorStanzaConf)
class AnnotatorStanza(Annotator):
    def __init__(self, conf: AnnotatorStanzaConf):
        super().__init__(conf)
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
        tok_options = {"raw": {}, "ssplit": {"tokenize_no_ssplit": True},
                       "tokenized": {"tokenize_pretokenized": True}}[conf.stanza_input_mode]
        pipeline_options.update(tok_options)
        # --
        self.pipeline = stanza.Pipeline(**pipeline_options)
        self.input_helper = InputHelper.get_input_helper(conf.stanza_input_mode)
        # has which components?
        self.processors_set = set(conf.stanza_processors)
        self.pred_upos, self.pred_lemma, self.pred_dep = [z in self.processors_set for z in ["pos","lemma","depparse"]]

    def annotate(self, insts: List[Union[Doc, Sent]]):
        conf: AnnotatorStanzaConf = self.conf
        # --
        # prepare inputs
        processing_iters = self.input_helper.prepare(insts)
        for one_pack in processing_iters:
            one_origs, one_inputs = one_pack
            doc = self.pipeline(one_inputs)  # return as one Doc
            # put things back
            if isinstance(one_origs, Doc):
                self.put_doc(one_origs, doc)
            if isinstance(one_origs, Sent):
                self.put_sent(one_origs, doc.sentences[0])
            else:
                assert isinstance(one_origs, list) and len(one_origs)==len(doc.sentences)
                for one_orig_sent, one_nlp_sent in zip(one_origs, doc.sentences):
                    self.put_sent(one_orig_sent, one_nlp_sent)

    # -----
    # put back annotations

    def put_doc(self, orig_doc: Doc, nlp_doc):
        assert orig_doc.get_text() == nlp_doc.text, "Error: Input & Output text not match!"
        orig_doc.clear_sents()  # clean sents if there are originally
        # process nlp_doc
        sent_positions = []
        new_nlp_sents = nlp_doc.sentences
        for nlp_sent in new_nlp_sents:
            if len(nlp_sent.tokens) == 0:
                continue  # ignore empty ones
            sent_start_char, sent_end_char = nlp_sent.tokens[0].start_char, nlp_sent.tokens[-1].end_char
            new_s = Sent.create(text=nlp_doc.text[sent_start_char:sent_end_char])
            self.put_sent(new_s, nlp_sent)  # annotate sents
            orig_doc.add_sent(new_s)  # add sent
            sent_positions.append((sent_start_char, sent_end_char-sent_start_char))
        orig_doc.build_sent_positions(sent_positions)  # put positions
        # --

    def put_sent(self, orig_sent: Sent, nlp_sent):
        text = orig_sent.get_text()
        # here we process the words!
        list_words = []
        list_uposes = []
        list_lemmas = []
        list_dep_heads = []
        list_dep_labels = []
        list_word_positions = []
        cur_word_start = 0
        # find them!!
        for w in nlp_sent.words:
            list_words.append(w.text)
            list_uposes.append(w.upos)
            list_lemmas.append(w.lemma)
            list_dep_heads.append(w.head)
            list_dep_labels.append(w.deprel)
            try:
                # todo(+N): some words can map to the same token if using MWT!
                t = w.parent
                tok_start = text.index(t.text, cur_word_start)  # idx inside the sentence
                list_word_positions.append((tok_start, t.end_char-t.start_char))  # [widx, wlen]
                cur_word_start = sum(list_word_positions[-1])  # start with next one
            except:
                list_word_positions = None
        # add them
        orig_sent.build_words(list_words)
        if self.pred_upos:
            orig_sent.build_uposes(list_uposes)
        if self.pred_lemma:
            orig_sent.build_lemmas(list_lemmas)
        if self.pred_dep:
            orig_sent.build_dep_tree(list_dep_heads, list_dep_labels)
        # note: do not rewrite original ones!
        if list_word_positions is not None and orig_sent.word_positions is None:
            orig_sent.build_word_positions(list_word_positions)
        # --

# example
#PYTHONPATH=../src/ python3 -m pdb -m msp2.cli.annotate 'stanza' input_path:en_pud.txt output_path:_tmp2.json stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:ssplit input_format:raw_sent ann_batch_size:100
#PYTHONPATH=../../zmtl2/src/ python3 -m msp2.cli.annotate 'stanza' input_path:./covid19/covid19.raw.json input_format:zjson_doc stanza_processors:tokenize,pos,lemma,depparse output_path:./covid19/covid19.par.json
