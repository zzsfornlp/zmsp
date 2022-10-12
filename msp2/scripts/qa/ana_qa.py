#

# analyze qa instances
import sys
from typing import List, Tuple
import pandas as pd
from msp2.data.inst import MyPrettyPrinter, QuestionAnalyzer
from msp2.data.rw import ReaderGetterConf
from msp2.tools.analyze import Analyzer, AnalyzerConf, AnnotationTask
from msp2.utils import zlog, Conf, init_everything

# --
class QaAnalyzerConf(AnalyzerConf):
    def __init__(self):
        super().__init__()
        # --

class QaAnalyzer(Analyzer):
    def __init__(self, conf: QaAnalyzerConf):
        super().__init__(conf)
        conf: QaAnalyzerConf = self.conf
        # --

# --
class MainConf(Conf):
    def __init__(self):
        self.R = ReaderGetterConf()
        self.ana = QaAnalyzerConf()
        # --

# --
QANA = QuestionAnalyzer()
class QaInst:
    def __init__(self, doc, qsent):
        self.doc = doc
        self.qsent = qsent
        # --
        self.qinfo = QANA.analyze_question(qsent)
        self.q_temp = QANA.question2template(qsent)
        # --

    @property
    def ctx_sents(self):
        return self.doc.sents[:self.doc.info['context_nsent']]

    @property
    def ans_tid_spans(self):
        # overall tid span
        if len(self.qsent.info['answers']) == 0:
            return [[0,0]]
        else:
            tid_offsets = [0]
            for sent in self.ctx_sents:
                tid_offsets.append(tid_offsets[-1] + len(sent))
            ret = []
            for sid, widx, wlen in self.qsent.info['answers']:
                _offset = tid_offsets[sid]
                ret.append([_offset+widx, _offset+widx+wlen])
        return ret

    def __repr__(self):
        return self.qsent.get_text() + " | " + str(self.qinfo) + "\n" \
               + ' '.join(self.q_temp) + "\n" \
               + MyPrettyPrinter.str_fnode(self.qsent, self.qsent.tree_dep.fnode)

def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    # first load all data
    zlog(f"Start to load data from {conf.R.input_path}")
    reader = conf.R.get_reader()
    all_qas = []
    for doc in reader:
        for qsent in doc.sents[doc.info['context_nsent']:]:
            inst = QaInst(doc, qsent)
            all_qas.append(inst)
    zlog(f"Load {len(all_qas)} qas")
    # --
    analyzer = QaAnalyzer(conf.ana)
    analyzer.set_var('qs', all_qas)
    analyzer.loop()
    # --

# --
# python3 -m msp2.scripts.qa.ana_qa input_path:
if __name__ == '__main__':
    main(*sys.argv[1:])

"""
python3 -m pdb -m msp2.scripts.qa.ana_qa input_path:./squad.dev.ud.json
# ->
x=ann_new qs AnnotationTask
group qs "d.qinfo['q_sig']"
group qs "d.qinfo['q_sig'].split('|',1)[-1]"
group qs "d.qinfo['q_sigD'].split('|',1)[-1]"
zz=filter qs "d.qinfo['q_word']=='UNK'"
zz=filter qs "d.qinfo['q_sig']=='what|det|obl|xcomp|root'"
zz=filter qs "d.qinfo['q_sig'].split('|',1)[-1]=='obl'"
x=ann_new zz AnnotationTask
"""

# --
# ~87%
train_details0 = [
    ('what|det|nsubj', 0.1410, 'What ?? Verb ...', 'X ?? Verb ...'),
    ('what|head', 0.1386, 'What BE ...', '... BE X'),
    ('how|advmod', 0.0977, 'How DO ...', '... by X'),  # how many??
    ('who|nsubj', 0.0753, 'Who ...', 'X ...'),
    ('what|obj', 0.0673, 'What DO ... Verb', '... Verb X'),
    ('what|nsubj', 0.0657, 'What DO ...', 'X DO ...'),
    ('what|det|obj', 0.0579, ),
    ('what|det|obl', 0.0531, ),
    ('when|advmod', 0.0363, ),
    ('which|det|nsubj', 0.0361, ),
    ('where|advmod', 0.0249, ),
    ('what|head|obl', 0.0194, ),
    ('when|head', 0.0186, ),
    ('who|head', 0.0159, ),
    ('what|det|head', 0.0151, ),
    ('UNK', 0.0140, ),
]
train_details1 = [
    ('head', 0.1924, 'What BE ...', '... BE What'),
    ('det|nsubj', 0.1771, 'What ?? ...', 'X ?? ...'),
    ('advmod', 0.1715, 'When/Where ...', '... at X time/place'),
    ('nsubj', 0.1428, 'What ...', 'X ...'),
    ('obj', 0.0752, 'What DO ...', '... Verb X'),
    ('det|obl', 0.0643, 'Prep What ?? ...', '... Prep What ??'),
    ('det|obj', 0.0634, 'What ?? DO ...', '... Verb X ??'),
    ('head|obl', 0.0251, 'Prep What ...', '... Prep What'),
    ('det|head', 0.0167, 'What ?? BE ...', '... BE X ??'),
    ('UNK', 0.0140, '...', '...'),
    ('mark', 0.0071, 'strange question words (when, where) in between', '...'),
    ('head|obj', 0.0068, '??'),
    ('obl', 0.0051, 'What ... Verb ... Prep', '... Verb ... Prep What'),
]
train_details2 = [
    ('what', 0.5929), ('who', 0.1029), ('how', 0.1019), ('which', 0.0632), ('when', 0.0630),
    ('where', 0.0405), ('why', 0.0146), ('UNK', 0.0140), ('whose', 0.0035), ('whom', 0.0034),
]
# --
