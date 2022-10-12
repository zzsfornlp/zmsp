#

# tokenize

import sys
from msp2.data.inst import Doc, Sent
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import default_json_serializer, Conf

class MainConf(Conf):
    def __init__(self):
        self.input = ""
        self.output = ""
        # --
        self.lang = "en"
        self.toolkit = "corenlp"
        # stanza
        self.stanza_dir = ""
        self.stanza_use_gpu = False
        # corenlp
        self.corenlp_port_plus = 0  # 9000+?
        self.corenlp_threads = 1  # num of threads?
        self.corenlp_max_char = 1000000
        # --

class StanzaNLP:
    def __init__(self, conf: MainConf):
        import stanza
        self.conf = conf
        common_kwargs = {"lang": conf.lang, "use_gpu": conf.stanza_use_gpu}
        if conf.stanza_dir:
            common_kwargs["dir"] = conf.stanza_dir
        self.tokenizer = stanza.Pipeline(processors='tokenize', **common_kwargs)

    def process(self, doc):  # note: modify inplace!
        conf = self.conf
        # --
        res = self.tokenizer(doc.get_text())
        for ss in res.sentences:
            tokens = [tok.text for tok in ss.tokens]
            if len(tokens) > 0:
                sent = Sent.create(tokens)
                doc.add_sent(sent)
        doc.text = None  # clear it!!
        if len(doc.sents) == 0:
            return None
        return doc

class CorenlpNLP:
    def __init__(self, conf: MainConf):
        self.conf = conf
        from stanza.server import CoreNLPClient
        self.client = CoreNLPClient(
            annotators=['tokenize', 'ssplit'], timeout=60000, memory='16G', properties=conf.lang, be_quite=True,
            threads=conf.corenlp_threads, endpoint=f"http://localhost:{9000+conf.corenlp_port_plus}",
            max_char_length=conf.corenlp_max_char,
        )  # be_quite=True

    def __del__(self):
        self.client.stop()

    def process(self, doc):  # note: modify inplace!
        conf = self.conf
        CUT_HINTS = "\n "  # first '\n', then ' '
        # --
        doc_text = doc.get_text()
        max_length = conf.corenlp_max_char
        cur_idx = 0
        while cur_idx < len(doc_text):
            cur_end = min(len(doc_text), cur_idx + max_length)
            if cur_end < len(doc_text):  # try to find a cut point!
                for hint in CUT_HINTS:  # try to find a point!
                    cur_end0 = cur_end
                    while cur_end0 > cur_idx:
                        if doc_text[cur_end0-1] == hint:
                            break
                        cur_end0 -= 1
                    if cur_end0 > cur_idx:  # find it!
                        cur_end = cur_end0
                        break
            # --
            res = self.client.annotate(doc_text[cur_idx:cur_end])
            for ss in res.sentence:
                tokens = [tok.originalText for tok in ss.token]
                if len(tokens) > 0:
                    sent = Sent.create(tokens)
                    doc.add_sent(sent)
            # --
            cur_idx = cur_end
        doc.text = None  # clear it!!
        if len(doc.sents) == 0:
            return None
        return doc

def main(*args):
    conf = MainConf()
    conf.update_from_args(list(args))
    # --
    nlp = {'stanza': StanzaNLP, 'corenlp': CorenlpNLP}[conf.toolkit](conf)
    with WriterGetterConf().get_writer(output_path=conf.output, output_format='zjson_doc') as writer:
        for doc in ReaderGetterConf().get_reader(input_path=conf.input, input_format='zjson_doc'):
            doc2 = nlp.process(doc)
            if doc2 is not None:  # if has content!
                writer.write_inst(doc2)
    # --

# PYTHONPATH=../?? OMP_NUM_THREADS=1 python3 raw2tok.py input:?? output:??
if __name__ == '__main__':
    main(*sys.argv[1:])
