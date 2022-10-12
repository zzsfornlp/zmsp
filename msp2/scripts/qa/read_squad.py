#

# read in squad (and do tokenization)

import sys
from collections import Counter
from msp2.data.inst import Doc, Sent
from msp2.data.rw import WriterGetterConf
from msp2.utils import zlog, zwarn, zopen, default_json_serializer, OtherHelper
from nltk.tokenize import TreebankWordTokenizer, WordPunctTokenizer, PunktSentenceTokenizer

# --
class NTokenizer:
    def __init__(self):
        self.word_toker = TreebankWordTokenizer()
        self.sent_toker = PunktSentenceTokenizer()
        # --

    def tokenize(self, text: str):
        # first split sent
        sent_spans = list(self.sent_toker.span_tokenize(text))
        sents = [text[a:b] for a,b in sent_spans]
        # then split tokens
        char2posi = [None] * len(text)
        all_tokens = []
        mark_eos = []
        for sid, sent in enumerate(sents):
            if len(mark_eos) > 0:
                mark_eos[-1] = True
            # --
            # note: a trick to split hypen
            tmp_sent = sent.replace('–',';').replace('—',';').replace('-',';')
            # --
            tok_spans = list(self.word_toker.span_tokenize(tmp_sent))
            tokens = [sent[a:b] for a,b in tok_spans]
            for ii, (a, b) in enumerate(tok_spans):
                _offset = sent_spans[sid][0]
                _s0, _s1 = _offset+a, _offset+b
                char2posi[_s0:_s1] = [len(all_tokens)] * (b - a)
                all_tokens.append(tokens[ii])
                mark_eos.append(False)
        if len(mark_eos) > 0:
            mark_eos[-1] = True
        return all_tokens, char2posi, mark_eos

    def locate_tokens(self, span, char2tid, context_tokens, check_str, cc):
        tids = sorted(set(char2tid[z] for z in range(span[0], sum(span)) if char2tid[z] is not None))
        assert len(tids)>0 and tids == list(range(tids[0], tids[-1]+1))
        t_start, t_len = tids[0], tids[-1]-tids[0]+1
        # --
        # check
        t_toks = context_tokens[t_start:t_start+t_len]
        _c0, _c1 = ''.join(t_toks), ''.join(check_str.split())
        if _c0 != _c1:
            cc['all_ans_mis'] += 1
            zwarn(f"Mismatch span: {t_toks} vs {check_str}")
        # --
        return t_start, t_len

def main(input_file: str, output_file: str):
    cc = Counter()
    toker = NTokenizer()
    # --
    all_docs = []
    d = default_json_serializer.from_file(input_file)
    assert d['version'] == 'v2.0'
    for article in d['data']:
        cc['all_art'] += 1
        for para_id, paragraph in enumerate(article['paragraphs']):
            # first for the context
            context = paragraph['context']
            context_tokens, context_char2tid, context_eos = toker.tokenize(context)
            context_conts = [False] * len(context_tokens)  # whether continuous?
            cc['all_para'] += 1
            cc['all_context_tok'] += len(context_tokens)
            # then for the qa
            qas = []
            for qa in paragraph['qas']:
                cc['all_qa'] += 1
                cc[f"all_qa_A={len(qa['answers'])}"] += 1
                assert (len(qa['answers'])>0 and not qa['is_impossible']) or (len(qa['answers'])==0 and qa['is_impossible'])
                # question
                question = qa['question']
                question_tokens, _, _ = toker.tokenize(question)
                # answer
                answers_tids = []  # List([t_start, t_len])
                for ans in qa['answers']:
                    cc['all_ans'] += 1
                    c_start, c_len = ans['answer_start'], len(ans['text'])
                    assert context[c_start:c_start+c_len] == ans['text']
                    t_start, t_len = toker.locate_tokens((c_start, c_len), context_char2tid, context_tokens, ans['text'], cc)
                    if t_len>1:
                        context_conts[t_start:t_start+t_len-1] = [True] * (t_len-1)
                    answers_tids.append((t_start, t_len))
                qas.append((question_tokens, answers_tids))
            # split sent and convert things!
            tid2posi = []
            context_sents = []
            _cur_toks = []
            for tid, tok in enumerate(context_tokens):
                tid2posi.append((len(context_sents), len(_cur_toks)))  # [sidx, widx]
                _cur_toks.append(tok)
                # --
                if context_eos[tid]:
                    if context_conts[tid]:
                        cc['all_sent_badsplit'] += 1
                    else:
                        cc['all_sent'] += 1
                        assert len(_cur_toks) > 0
                        context_sents.append(_cur_toks)
                        _cur_toks = []
            assert len(_cur_toks)==0
            # remap answers
            remapped_qas = [(_q_toks, [tid2posi[_p0[0]] + (_p0[1],) for _p0 in _a_tids]) for _q_toks, _a_tids in qas]
            # =====
            doc_sents = []
            doc_id = f"{article['title']}-{para_id}"
            for _sid, _sent in enumerate(context_sents):
                doc_sents.append(Sent.create(_sent, id=f"{doc_id}-{_sid}"))
            for _qid, (_q_toks, _ans_posis) in enumerate(remapped_qas):
                doc_sents.append(Sent.create(_q_toks, id=paragraph['qas'][_qid]['id']))
                doc_sents[-1].info['answers'] = _ans_posis  # [sid, widx, wlen]
            doc = Doc.create(doc_sents, id=doc_id)
            doc.info['context_nsent'] = len(context_sents)
            all_docs.append(doc)
            # --
    # --
    if output_file:
        with WriterGetterConf().get_writer(output_path=output_file) as writer:
            writer.write_insts(all_docs)
    # --
    zlog("#--\nProcess them all:")
    OtherHelper.printd(cc)
    # --

# --
if __name__ == '__main__':
    main(*sys.argv[1:])

# --
"""
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
eval script: https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
# --
# tokenize
{
python3 -m msp2.scripts.qa.read_squad dev-v2.0.json squad.dev.json
python3 -m msp2.scripts.qa.read_squad train-v2.0.json squad.train.json
} |& tee _log.tok
# --
# parse
for wset in dev train; do
python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:en stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:squad.$wset.json output_path:squad.$wset.ud.json
done |& tee _log.parse
# --
# analyze questions

"""
