#

# a specific script to convert ontonotes' conll

import re
from collections import Counter
from mspx.data.inst import Sent, Doc
from mspx.data.rw import ReaderGetterConf, WriterGetterConf, FileStreamer
from mspx.utils import zlog, zopen, Conf, Random, zglobs, init_everything, ZHelper, zwarn

class MainConf(Conf):
    def __init__(self):
        self.input_path = []
        self.W = WriterGetterConf()
        # --
        self.make_doc = False  # check doc_id and make_doc!
        self.add_syntax = False  # only xpos and xparse!
        self.add_ner = False  # ner
        self.add_srl = False  # srl
        self.add_wordsense = False  # wordsense
        self.add_coref = False  # coref
        # --

# --
def read_spans(pieces):  # for args/ner/coref
    # read each one
    rets = []
    stack = []  # cur_start, cur_lab
    # --
    for one_widx, one_s in enumerate(pieces):
        _toks = [z for z in re.split(r"([*()|])", one_s)]
        _toks = [z for z in _toks if (z not in ['', '-', '*', '|']) and (not str.isspace(z))]
        _idx = 0
        while _idx < len(_toks):
            if _toks[_idx] == '(':  # open
                _lab = _toks[_idx+1]  # must have a label next!
                stack.append((one_widx, _lab))
                _idx += 2
            elif _toks[_idx] == ')':  # close
                _widx, _lab = stack.pop()
                rets.append((_widx, one_widx-_widx+1, _lab))  # widx, wlen, label
                _idx += 1
            else:
                assert _toks[_idx+1] == ')'  # special close for coref
                _clab = _toks[_idx]
                _pii = len(stack) - 1
                while _pii >= 0:
                    if stack[_pii][-1] == _clab:  # find it!
                        _widx, _lab = stack.pop(_pii)
                        rets.append((_widx, one_widx-_widx+1, _lab))  # widx, wlen, label
                        break
                    _pii -= 1
                assert _pii >= 0  # must find it!
                _idx += 2
    # --
    assert len(stack) == 0
    return rets

def yield_sents(streamer, conf, cc):
    for mline in streamer:
        lines = [z.strip() for z in mline.strip().split("\n")]
        lines = [z for z in lines if z and z[0] != '#']  # ignore comments
        fields = [z.split() for z in lines]
        if len(fields) == 0:
            zwarn(f"Meet empty one: {mline}")
            continue
        assert len(fields) > 0 and len(fields[0][0]) > 0
        # --
        cc['sent'] += 1
        cc['word'] += len(fields)
        doc_id = fields[0][0]
        assert all(z[0]==doc_id for z in fields)  # 0: doc_id
        assert all(z[1]==fields[0][1] for z in fields)  # 1: part_num
        assert [z[2] for z in fields] == [str(z) for z in range(len(fields))]  # 2: widx
        words, xpos, xparse, pred_lemma, pred_fid, pred_ws = [[z[ii] for z in fields] for ii in range(3, 9)]
        speaker = fields[0][9]
        assert all(z[9] == speaker for z in fields)  # 9: speaker
        ner_bits = [z[10] for z in fields]  # 10: ner
        coref_bits = [z[-1] for z in fields]  # -1: coref
        # --
        sent = Sent(words, make_singleton_doc=True)
        sent.info.update({"doc_id": doc_id})
        if conf.add_syntax:
            sent.info.update({"xpos": xpos, "xparse": xparse})
        if conf.add_coref:
            sent.info.update({"speaker": speaker})
        if conf.add_wordsense:
            sent.info.update({"word_lemma": pred_lemma, "word_sense": pred_ws})
        # srl
        valid_preds = [(ii, fid) for ii, fid in enumerate(pred_fid) if fid!='-']  # judge by Frame-ID
        for pii, (widx, fid) in enumerate(valid_preds):
            predicate = f"{pred_lemma[widx]}.{fid}"
            arg_strs = [z[11+pii] for z in fields]  # 11+: srl
            span_args = read_spans(arg_strs)
            cc['srl_frame'] += 1
            cc['srl_arg'] += len(span_args)
            if conf.add_srl:
                _pred = sent.make_frame(widx, 1, label=predicate, cate='srlP')
                for _widx, _wlen, _role in span_args:
                    _arg = sent.make_frame(_widx, _wlen, label='A', cate='srlA')
                    _pred.add_arg(_arg, _role)
        # ner
        span_ners = read_spans(ner_bits)
        cc['ner'] += len(span_ners)
        if conf.add_ner:
            for _widx, _wlen, _lab in span_ners:
                _ef = sent.make_frame(_widx, _wlen, label=_lab, cate='ef')
        # coref
        span_corefs = read_spans(coref_bits)
        cc['coref_m'] += len(span_corefs)
        if conf.add_coref:
            for _widx, _wlen, _lab in span_corefs:
                _m = sent.make_frame(_widx, _wlen, label=_lab, cate='coref')
        # --
        yield sent
    # --

def yield_docs(streamer, conf, cc):
    _make_doc = conf.make_doc
    curr = None
    for sent in yield_sents(streamer, conf, cc):
        if not _make_doc:
            yield sent.doc
        else:
            doc_id = sent.info['doc_id']
            if curr is not None and doc_id == curr[0]:  # append!
                curr[1].append(sent)
            else:
                if curr is not None:
                    # doc = Doc(curr[1], id=curr[0])
                    doc = Doc.merge_docs([z.doc for z in curr[1]], new_doc_id=curr[0])
                    yield doc
                curr = [doc_id, [sent]]  # make a new one!
    if curr is not None:
        doc = Doc(curr[1], id=curr[0])
        yield doc
    # --

# --
def main(args):
    conf: MainConf = init_everything(MainConf(), args)
    cc = Counter()
    # --
    input_paths = zglobs(conf.input_path)
    with conf.W.get_writer() as writer:
        for f in input_paths:
            cc['file'] += 1
            streamer = FileStreamer(f, mode='mline')
            for doc in yield_docs(streamer, conf, cc):
                writer.write_inst(doc)
                cc['doc'] += 1
    # --
    zlog(f"Finish reading-onto: {ZHelper.resort_dict(cc)}")
    # --

# python3 -m mspx.scripts.data.onto.prep_onto input_path:??
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
