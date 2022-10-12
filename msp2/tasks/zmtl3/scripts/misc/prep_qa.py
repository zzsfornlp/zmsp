#

# prepare qa datasets

import os
import re
import math
from collections import Counter, defaultdict, OrderedDict
from msp2.data.inst import yield_sents, yield_frames, Sent, Doc, Frame, set_ee_heads
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import Conf, zlog, zopen, zwarn, init_everything, default_json_serializer, OtherHelper, zglob
from msp2.tools.annotate import AnnotatorStanzaConf, AnnotatorStanza
from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer, PunktSentenceTokenizer
from nltk.corpus import stopwords
import pandas as pd

# --
# todo(note): likely to be nltk's bug
class ModifiedTreebankWordTokenizer(TreebankWordTokenizer):
    def span_tokenize(self, sentence):
        raw_tokens = self.tokenize(sentence)
        # convert
        if ('"' in sentence) or ("''" in sentence):
            # Find double quotes and converted quotes
            matched = [m.group() for m in re.finditer(r"``|'{2}|\"", sentence)]
            # Replace converted quotes back to double quotes
            tokens = [matched.pop(0) if tok in ['"', "``", "''"] else tok for tok in raw_tokens]
        else:
            tokens = raw_tokens
        # align_tokens
        point = 0
        offsets = []
        for token in tokens:
            try:
                start = sentence.index(token, point)
            except ValueError as e:
                # raise ValueError(f'substring "{token}" not found in "{sentence}"') from e
                zwarn(f"Tokenizer skip unfound token: {token} ||| {sentence[point:]}")
                continue  # note: simply skip this one!!
            point = start + len(token)
            offsets.append((start, point))
        return offsets
# --

# --
# word tokenizer
class NTokenizer:
    def __init__(self):
        self.word_toker = ModifiedTreebankWordTokenizer()
        self.sent_toker = PunktSentenceTokenizer()
        # --

    def tokenize(self, text: str):
        # first split sent
        sent_spans = list(self.sent_toker.span_tokenize(text))
        sents = [text[a:b] for a,b in sent_spans]
        # then split tokens
        char2posi = [None] * len(text)  # int -> (sid, tid)
        all_tokens = []
        all_token_spans = []
        for sid, sent in enumerate(sents):
            tok_spans = list(self.word_toker.span_tokenize(sent))
            _toks = [sent[a:b] for a,b in tok_spans]
            _spans = []
            for ii, (a, b) in enumerate(tok_spans):
                _offset = sent_spans[sid][0]
                _s0, _s1 = _offset+a, _offset+b
                char2posi[_s0:_s1] = [(sid, ii)] * (b - a)
                _spans.append((_s0, _s1))
            all_tokens.append(_toks)
            all_token_spans.append(_spans)
        return all_tokens, all_token_spans, char2posi
# --

# --
def read_squad(file: str):
    cc = Counter()
    # --
    toker = NTokenizer()
    data = default_json_serializer.from_file(file)['data']
    docs = []
    for article in data:
        cc['article'] += 1
        title = article['title']
        for para_id, para in enumerate(article['paragraphs']):
            doc_id = f"{title}_{para_id}"
            context = para['context']
            # --
            # tokenize and prepare context
            _tokens, _token_spans, _char2posi = toker.tokenize(context)
            cc['sent'] += len(_tokens)
            sents = [Sent.create(_toks) for _toks in _tokens]
            doc = Doc.create(sents, id=doc_id)
            docs.append(doc)  # actually paragraph
            # --
            # prepare questions
            for qa in para['qas']:
                cc['qas'] += 1
                question = qa['question']
                if qa['is_impossible']:
                    cc['qas_A0'] += 1
                    assert len(qa['answers']) == 0
                    # --
                    frame = sents[0].make_event(0, 1, type='Q')  # note: simply put sent0!
                    frame.info['question'] = question
                else:
                    assert len(qa['answers']) > 0
                    cc[f"qas_A{min(2, len(qa['answers']))}"] += 1
                    _lcc = Counter([z["answer_start"] for z in qa['answers']])
                    # take the (max vote, shortest length) one!
                    _ans = sorted([(-_lcc[z["answer_start"]], len(z['text']), ii, z) for ii,z in enumerate(qa['answers'])])[0][-1]
                    # locate the answer span
                    _start, _end = _ans['answer_start'], _ans['answer_start'] + len(_ans['text'])
                    _tok_idxes = []
                    for _c in range(_start, _end):
                        _posi = _char2posi[_c]
                        if _posi is None: continue
                        if len(_tok_idxes)>0 and _tok_idxes[-1][0] != _posi[0]:
                            zwarn(f"Maybe bad sentence splitting for: {_ans}")
                            cc['qas_missplit'] += 1
                            break  # bad sentence splitting!
                        if len(_tok_idxes)==0 or _posi != _tok_idxes[-1]:
                            _tok_idxes.append(_posi)
                    # get the span
                    _sid, _widx, _wlen = _tok_idxes[0][0], _tok_idxes[0][1], _tok_idxes[-1][1] - _tok_idxes[0][1] + 1
                    _text = " ".join(_tokens[_sid][_widx:_widx+_wlen])
                    if ''.join(_text.split()) != ''.join(_ans['text'].split()):
                        zwarn(f"Maybe bad answer parsing: {_ans} vs {_text}")
                        cc['qas_mismatch'] += 1
                    # --
                    frame = sents[_sid].make_event(0, 1, type='Q')
                    frame.info['question'] = question
                    answer = sents[_sid].make_entity_filler(_widx, _wlen)
                    frame.add_arg(answer, role='A')
    # --
    zlog(f"Read squad from {file}: {cc}")
    return docs

# --
def read_qamr(ann_file: str, data_file: str):
    cc = Counter()
    # first read all sents
    _maps = {'``': "\"", "''": "\"", "-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}"}
    all_sents = OrderedDict()
    with zopen(data_file) as fd:
        for line in fd:
            cc['sent_all'] += 1
            line = line.strip()
            _id, _text = line.split("\t")
            _toks = [_maps.get(t, t) for t in _text.split()]
            all_sents[_id] = Sent.create(_toks, id=_id)
    # then read ann
    hit_ids = set()
    with zopen(ann_file) as fd:
        for line in fd:
            fields = line.strip().split("\t")
            _sid = fields[0]
            _tids = sorted([int(z) for z in fields[4].split()])
            question = fields[5]
            _aids = sorted([int(z) for z in fields[6].split()])
            cc['qas'] += 1
            # --
            hit_ids.add(_sid)
            sent = all_sents[_sid]
            frame = sent.make_event(_tids[0], _tids[-1]-_tids[0]+1, type='Q')
            frame.info['question'] = question
            answer = sent.make_entity_filler(_aids[0], _aids[-1]-_aids[0]+1)
            frame.add_arg(answer, role='A')
    # --
    cc['sent_hit'] = len(hit_ids)
    rets = [z for z in all_sents.values() if z.id in hit_ids]
    # --
    zlog(f"Read qamr from {ann_file}/{data_file}: {cc}")
    return rets

# --
def read_qasrl(file: str):
    cc = Counter()
    # --
    _maps = {'``': "\"", "''": "\"", "-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}"}
    rets = []
    for inst in default_json_serializer.yield_iter(file):
        cc['sent'] += 1
        _toks = [_maps.get(t, t) for t in inst['sentenceTokens']]
        sent = Sent.create(_toks, id=inst['sentenceId'])
        rets.append(sent)
        # read qas
        for vv in inst['verbEntries'].values():
            cc['pred'] += 1
            cc[f"pred_{min(len(vv['questionLabels']), 5)}"] += 1
            frame_widx = vv['verbIndex']
            for qq in vv['questionLabels'].values():
                cc['qas'] += 1
                question = qq['questionString']
                # --
                # merge into one
                _spans = [sorted(z['spans']) for z in qq['answerJudgments'] if z['isValid']]
                cc[f'qas_V{len(_spans)}'] += 1
                if len(_spans) == 0:
                    continue  # simply discard if no valids!
                _lcc = Counter([str(z) for z in _spans])
                _ans = sorted([(-_lcc[str(z)], len(z), ii, z) for ii,z in enumerate(_spans)])[0]  # prefer shorter one!
                cc[f'qas_A{_ans[0]}'] += 1  # agree count
                _ans = _ans[-1]  # simply take the largest span
                cc[f'qas_S{len(_ans)}'] += 1  # segment count
                # --
                frame = sent.make_event(frame_widx, 1, type='Q')  # note: one qas one frame!
                frame.info['question'] = question
                answer = sent.make_entity_filler(_ans[0][0], _ans[-1][-1]-_ans[0][0])
                frame.add_arg(answer, role='A')
    # --
    zlog(f"Read qasrl from {file}: {OtherHelper.printd_str(cc, sep=' | ')}")
    return rets

# --
def read_qanom(file: str):
    cc = Counter()
    # --
    _maps = {'``': "\"", "''": "\"", "-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}"}
    rets = []
    df = pd.read_csv(file)
    for index, row in df.iterrows():
        cc['row'] += 1
        # put the sent
        _orig_toks = row['sentence'].split()
        _id, _toks = row['qasrl_id'], [_maps.get(t, t) for t in _orig_toks]
        if len(rets)>0 and row['qasrl_id'] == rets[-1].id:  # use prev sent!
            sent = rets[-1]
            assert _toks == sent.seq_word.vals
        else:
            cc['sent'] += 1
            sent = Sent.create(_toks, id=_id)
            rets.append(sent)
        # read qas
        if isinstance(row['question'], str):
            cc['row1'] += 1
            cc['qas'] += 1
            frame_widx = row['target_idx']
            question = row['question'].strip()
            assert _orig_toks[frame_widx].lower() == row['noun'].lower()
            # simply put into one span
            _spans = [[int(z) for z in z0.split(":",1)] for z0 in row['answer_range'].split("~!~")]
            try:
                assert "~!~".join([" ".join(_orig_toks[a:b]) for a,b in _spans]).lower() == row['answer'].lower()
            except:
                # breakpoint()
                zwarn(f"Mismatched answer of: {row['answer']}")
            _spans.sort()
            _left, _right = _spans[0][0], _spans[-1][-1]
            # --
            frame = sent.make_event(frame_widx, 1, type='Q')  # note: one qas one frame!
            frame.info['question'] = question
            answer = sent.make_entity_filler(_left, _right-_left)
            frame.add_arg(answer, role='A')
        else:
            cc['row0'] += 1  # no qas!
    # --
    hasframe_rets = [z for z in rets if len(z.events)>0]
    cc['sentV'] = len(hasframe_rets)
    zlog(f"Read qanom from {file}: {OtherHelper.printd_str(cc, sep=' | ')}")
    return rets  # still return all

# --
# modify in place!
def refind_targets(inst_stream, annotator):
    cc = Counter()
    toker = NTokenizer()
    _stopword_set = set(stopwords.words('english'))
    from string import punctuation
    _stopword_set.update(punctuation)
    # --
    for inst in inst_stream:
        cc['all_inst'] += 1
        # --
        # note: need to cache all evts here
        inst_orig_evts = []
        for sent in yield_sents(inst):
            cc['all_sent'] += 1
            for evt in list(sent.events):  # list it since we want to change!
                cc['qas'] += 1
                inst_orig_evts.append(evt)
        # --
        for evt in inst_orig_evts:
            sent = evt.sent
            # --
            _question = evt.info['question']
            _q_toks = sum(toker.tokenize(_question.strip())[0], [])  # question tokens
            _q_sent = Sent.create(_q_toks)
            annotator.annotate([_q_sent])  # parse it!
            # --
            # check answer and context
            answer_lemmas = set(
                sum([[t.lemma.lower() for t in a.mention.get_tokens() if t.lemma is not None] for a in evt.args], []))
            ranked_preds = defaultdict(list)  # lemma -> pred_token
            # we might want to check all sents (for example, the paragraph in squad)
            all_sents = sent.doc.sents if sent.doc is not None else [sent]
            _sid0 = all_sents.index(sent)
            for _sii, _s in enumerate(all_sents):
                _dist = abs(_sii - _sid0)  # still make it closer to the answer
                for _t in _s.tokens:
                    _lemma = _t.lemma.lower() if _t.lemma is not None else None
                    if _lemma is None or _lemma in _stopword_set or _lemma in answer_lemmas:
                        continue
                    _depth = _s.tree_dep.depths[_t.widx]  # we want higher nodes
                    ranked_preds[_lemma].append((_dist, _sii, _depth, _t.widx, _t))  # add more tie-breakers
            for vs in ranked_preds.values():
                vs.sort()
            # --
            # check question
            _cand_qtoks = [
                t for t in _q_sent.tokens if (t.lemma is not None and t.lemma.lower() not in _stopword_set
                                              and t.lemma.lower() not in answer_lemmas)]
            _q_toks = sorted(_cand_qtoks, key=lambda t: (_q_sent.tree_dep.depths[t.widx], t.widx))
            trg_token = None
            for qt in _q_toks:
                cts = ranked_preds[qt.lemma.lower()]
                if len(cts) > 0:
                    trg_token = cts[0][-1]
                    if trg_token is evt.mention.shead_token:  # same token
                        cc['qas_nochange'] += 1  # no need to change!
                    else:
                        cc['qas_changed'] += 1
                        # --
                        # modify inplace!
                        new_evt = trg_token.sent.make_event(trg_token.widx, 1, type='Q')  # new target
                        new_evt.info.update(evt.info)  # note: remember to add info!
                        for arg in evt.args:
                            new_evt.add_arg(arg.arg, role='A')  # add answer
                        sent.delete_frame(evt, 'evt')
                        # --
                    break
            if trg_token is None:  # fail to find one, simply no change
                cc['qas_notfound'] += 1
    # --
    zlog(f"Do refind_targets: {OtherHelper.printd_str(cc, sep=' | ')}")
# --

# --
class MainConf(Conf):
    def __init__(self):
        self.data_type = 'squad'  # squad/qamr/qasrl/qanom//json
        self.input_file = ''
        self.extra_input_file = ''  # for qamr
        self.output_file = ''
        # --
        # special mode
        self.ann = AnnotatorStanzaConf.direct_conf(
            stanza_lang='en', stanza_use_gpu=False, stanza_processors="tokenize,pos,lemma,depparse".split(","),
            stanza_input_mode="tokenized",)
        self.do_refind = False
        # --

# --
def main(*args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    if conf.data_type == 'json':
        reader = ReaderGetterConf().get_reader(input_path=conf.input_file)
        insts = list(reader)
    elif conf.data_type == 'squad':
        insts = read_squad(conf.input_file)
    elif conf.data_type == 'qamr':
        insts = read_qamr(conf.input_file, conf.extra_input_file)
    elif conf.data_type == 'qasrl':
        insts = read_qasrl(conf.input_file)
    elif conf.data_type == 'qanom':
        insts = read_qanom(conf.input_file)
    else:
        raise NotImplementedError(f"UNK data_type: {conf.data_type}")
    # --
    if conf.do_refind:
        annotator = AnnotatorStanza(conf.ann)
        refind_targets(insts, annotator)
    # --
    if conf.output_file:
        with WriterGetterConf().get_writer(output_path=conf.output_file) as writer:
            writer.write_insts(insts)
    # --

# --
def read_probes(file: str):
    from collections import Counter
    cc = Counter()
    ret = {}  # type -> {role -> "question"}
    with open(file) as fd:
        for record in fd.read().split("\n\n"):
            lines = [z.strip() for z in record.split("\n") if z.strip()!='']
            evt_type = lines[0].lower()
            roles = {}
            for one in lines[1:]:
                # print(one)
                name0, question0 = one.split(":", 1)
                name = name0.rsplit("_", 1)[0]
                question = question0.replace("{trigger}", "<T>") + "?"
                assert name not in roles
                roles[name] = question
            assert evt_type not in ret
            ret[evt_type] = roles
            cc['evt'] += 1
            cc['arg'] += len(roles)
    # --
    print(cc)
    print(ret)
    return ret
# --

# python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa data_type:? input_file:?
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# --
"""
# --
# squad v2.0
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa data_type:squad input_file:train-v2.0.json output_file:en.squad.train.json
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa data_type:squad input_file:dev-v2.0.json output_file:en.squad.dev.json
# Read squad from train-v2.0.json: Counter({'qas': 130319, 'sent': 96333, 'qas_A1': 86821, 'qas_A0': 43498, 'qas_mismatch': 1690, 'qas_missplit': 515, 'article': 442})
# Read squad from dev-v2.0.json: Counter({'qas': 11873, 'sent': 6487, 'qas_A0': 5945, 'qas_A2': 5927, 'qas_mismatch': 111, 'article': 35, 'qas_missplit': 16, 'qas_A1': 1})
# --
# qamr
git clone https://github.com/uwnlp/qamr
for wset in train dev test; do
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa data_type:qamr extra_input_file:qamr/data/wiki-sentences.tsv input_file:qamr/data/filtered/${wset}.tsv output_file:en.qamr.${wset}.json
done
# Read qamr from qamr/data/filtered/train.tsv/qamr/data/wiki-sentences.tsv: Counter({'qas': 50615, 'sent_all': 4923, 'sent_hit': 3885})
# Read qamr from qamr/data/filtered/dev.tsv/qamr/data/wiki-sentences.tsv: Counter({'qas': 18908, 'sent_all': 4923, 'sent_hit': 499})
# Read qamr from qamr/data/filtered/test.tsv/qamr/data/wiki-sentences.tsv: Counter({'qas': 18770, 'sent_all': 4923, 'sent_hit': 480})
# --
# qasrl
# git clone https://github.com/julianmichael/qasrl
# cd qasrl
# python scripts/download_data.py
wget http://qasrl.org/data/qasrl-v2_1.tar
tar -x -f ./qasrl-v2_1.tar
for wset in train dev test; do
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa data_type:qasrl input_file:./qasrl-v2_1/orig/${wset}.jsonl.gz output_file:en.qasrl.${wset}.json
done
# Read qasrl from ./qasrl-v2_1/orig/train.jsonl.gz: pred: 95258 | pred_0: 5 | pred_1: 13371 | pred_2: 50605 | pred_3: 25115 | pred_4: 5380 | pred_5: 782 | qas: 215427 | qas_A-1: 35940 | qas_A-2: 82205 | qas_A-3: 97282 | qas_S1: 213845 | qas_S2: 1432 | qas_S3: 96 | qas_S4: 29 | qas_S5: 15 | qas_S6: 5 | qas_S7: 3 | qas_S8: 1 | qas_S9: 1 | qas_V1: 5311 | qas_V2: 20564 | qas_V3: 189533 | qas_V4: 8 | qas_V5: 7 | qas_V6: 4 | sent: 44477
# Read qasrl from ./qasrl-v2_1/orig/dev.jsonl.gz: pred: 17577 | pred_1: 2757 | pred_2: 9757 | pred_3: 4138 | pred_4: 831 | pred_5: 94 | qas: 38487 | qas_A-1: 5844 | qas_A-2: 14644 | qas_A-3: 17999 | qas_S1: 38225 | qas_S2: 242 | qas_S3: 13 | qas_S4: 5 | qas_S6: 1 | qas_S9: 1 | qas_V1: 877 | qas_V2: 3557 | qas_V3: 34052 | qas_V4: 1 | sent: 9078
# Read qasrl from ./qasrl-v2_1/orig/test.jsonl.gz: pred: 20603 | pred_0: 1 | pred_1: 3113 | pred_2: 11377 | pred_3: 5049 | pred_4: 956 | pred_5: 107 | qas: 45387 | qas_A-1: 7054 | qas_A-2: 16904 | qas_A-3: 21429 | qas_S1: 45051 | qas_S2: 307 | qas_S3: 19 | qas_S4: 6 | qas_S5: 1 | qas_S6: 3 | qas_V1: 1133 | qas_V2: 4367 | qas_V3: 39887 | sent: 10453
# --
# qanom
# --
# https://github.com/kleinay/QANom/blob/master/scripts/download_qanom_dataset.sh
function download_gdrive_zip_file {
    ggID=$1
    archive=$2
    ggURL='https://drive.google.com/uc?export=download'
    echo "Downloading ${archive}"
    filename="$(curl -sc /tmp/gcokie "${ggURL}&id=${ggID}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')"
    getcode="$(awk '/_warning_/ {print $NF}' /tmp/gcokie)"
    curl -Lb /tmp/gcokie "${ggURL}&confirm=${getcode}&id=${ggID}" -o "${archive}"
}
download_gdrive_zip_file "1_cTOy9isFo2qglAXETD2rgDTkhxC_OZr" "qanom_dataset.zip"
unzip "qanom_dataset.zip" -d "qanom_dataset"
for wset in train dev test; do
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa data_type:qanom input_file:./qanom_dataset/annot.${wset}.csv output_file:en.qanom.${wset}.json
done
#Read qanom from ./qanom_dataset/annot.train.csv: qas: 15895 | row: 30644 | row0: 14749 | row1: 15895 | sent: 7114 | sentV: 4636
#Read qanom from ./qanom_dataset/annot.dev.csv: qas: 5577 | row: 7660 | row0: 2083 | row1: 5577 | sent: 1557 | sentV: 1252
#Read qanom from ./qanom_dataset/annot.test.csv: qas: 4886 | row: 7023 | row0: 2137 | row1: 4886 | sent: 1517 | sentV: 1163
# --
# parse them all
for ff in en.*.{train,dev,test}.json; do
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:en stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:${ff} "output_path:${ff%.json}.ud2.json"
done |& tee _log_parse
cat en.qamr.{train,dev,test}.ud2.json >en.qamr.all.ud2.json
cat en.qasrl.{train,dev,test}.ud2.json >en.qasrl.all.ud2.json
cat en.qanom.{train,dev,test}.ud2.json >en.qanom.all.ud2.json
# --
# refind targets?
for dset in squad qamr; do
for wset in train dev test; do
python3 -m msp2.tasks.zmtl3.scripts.misc.prep_qa data_type:json do_refind:1 input_file:en.$dset.$wset.ud2.json output_file:en.${dset}R.$wset.ud2.json
done; done |& tee _log_refind
cat en.qamrR.{train,dev,test}.ud2.json >en.qamrR.all.ud2.json
# =>
Do refind_targets: all_inst: 19035 | all_sent: 96333 | qas: 130319 | qas_changed: 124558 | qas_nochange: 3526 | qas_notfound: 2235
Do refind_targets: all_inst: 1204 | all_sent: 6487 | qas: 11873 | qas_changed: 11485 | qas_nochange: 273 | qas_notfound: 115
Do refind_targets: all_inst: 3885 | all_sent: 3885 | qas: 50615 | qas_changed: 35872 | qas_nochange: 11642 | qas_notfound: 3101
Do refind_targets: all_inst: 499 | all_sent: 499 | qas: 18908 | qas_changed: 13306 | qas_nochange: 4453 | qas_notfound: 1149
Do refind_targets: all_inst: 480 | all_sent: 480 | qas: 18770 | qas_changed: 13246 | qas_nochange: 4380 | qas_notfound: 1144
-> Dist with squad.train(arg=0/1: 0.66/0.33): 0(0.83), 1(0.93), 2(0.97)
-> Dist with squad.dev(arg=0/1: 0.5/0.5): 0(0.86), 1(0.95), 2(0.97)
# --
# prepare ACE/ERE questions
# https://github.com/veronica320/Zeroshot-Event-Extraction/tree/4332f6efeffcc1db81c7015a998b3555098b3fd2
wget https://raw.githubusercontent.com/veronica320/Zeroshot-Event-Extraction/master/source/lexicon/probes/ACE/arg_qa_probes_contextualized.txt -O probes.ace.txt
wget https://raw.githubusercontent.com/veronica320/Zeroshot-Event-Extraction/master/source/lexicon/probes/ERE/arg_qa_probes_contextualized.txt -O probes.ere.txt
p_ace = read_probes("probes.ace.txt")
p_ere = read_probes("probes.ere.txt")
z = {'ace':p_ace, 'ere':p_ere}
with open("probes.txt", 'w') as fd: fd.write(str(z))
"""
