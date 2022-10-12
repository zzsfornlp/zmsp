#

# do coref

from collections import Counter, OrderedDict
from msp2.data.inst import yield_sents, set_ee_heads, Doc, Sent, Mention, HeadFinder
from msp2.utils import Conf, zlog, zopen, zwarn, init_everything, default_json_serializer, OtherHelper, zglob
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from tqdm import tqdm

class MainConf(Conf):
    def __init__(self):
        self.input_file = ""  # input data
        self.output_file = ""  # output data
        self.sdist = 2  # sdist <= this

def do_coref(nlp, sent, conf, cc):
    _sdist = conf.sdist
    _c_sent_idx = sent.sid
    ctx_sents = sent.doc.sents[max(0, _c_sent_idx-_sdist):_c_sent_idx+_sdist+1]
    tokens = sum([z.tokens for z in ctx_sents], [])
    # --
    hf = HeadFinder("NOUN")
    _res = nlp([t.word for t in tokens])
    assert len(_res) == len(tokens)
    _info_coref = {}  # tok-id -> [others ...]
    for cluster in _res._.coref_clusters:
        _chain = []
        for span in cluster.mentions:
            _toks0 = [tokens[z] for z in range(span.start, span.end)]
            _toks = [z for z in _toks0 if z.sent is _toks0[0].sent]
            if len(_toks) < len(_toks0):
                zwarn(f"Cross-sent mention: {span} in {cluster}")
            # find head word
            _m = Mention.create(_toks[0].sent, _toks[0].widx, (_toks[-1].widx-_toks[0].widx+1))
            hf.set_head_for_mention(_m)
            _shead_tok = _m.shead_token
            _m_toks = [_shead_tok]
            # check conj
            for tok2 in _shead_tok.ch_toks:
                if tok2.deplab.split(":")[0] == 'conj' and tok2.widx >= _m.widx and tok2.widx < _m.wridx:
                    if tok2 not in _m_toks:
                        _m_toks.append(tok2)
            # --
            _chain.append(_m_toks)
        # --
        for ts in _chain:
            if all(t.sent is sent for t in ts):
                cc['coref'] += 1
                for t0 in ts:
                    cc['corefT'] += 1
                    _tid = t0.get_indoc_id(True)
                    if _tid in _info_coref:
                        zwarn(f"Skip since repeated corefs: {t0} in {_chain}")
                        # breakpoint()
                    else:
                        _info_coref[_tid] = sum([[t.get_indoc_id(True) for t in ts2] for ts2 in _chain if ts2 is not ts], [])
    # --
    sent.info["info_coref"] = _info_coref
    # --

def main(*args):
    # setup
    import spacy
    nlp = spacy.load('en')
    import neuralcoref
    neuralcoref.add_to_pipe(nlp)
    nlp.tokenizer = nlp.tokenizer.tokens_from_list
    # --
    conf: MainConf = init_everything(MainConf(), args, add_nn=False)
    zlog(f"Read from {conf.input_file}")
    reader = ReaderGetterConf().get_reader(input_path=conf.input_file)
    all_insts = list(reader)
    cc = Counter()
    for inst in tqdm(all_insts):
        cc['inst'] += 1
        for sent in yield_sents(inst):
            cc['sent'] += 1
            cc['token'] += len(sent)
            do_coref(nlp, sent, conf, cc)
    # --
    zlog(f"Do coref for {conf.input_file} -> {conf.output_file}: {cc}")
    if conf.output_file:
        with WriterGetterConf().get_writer(output_path=conf.output_file) as writer:
            writer.write_insts(all_insts)
    # --

# python3 -m msp2.tasks.zmtl3.scripts.srl.zz_coref ...
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# setup
"""
# env
conda create -n neuralcoref python=3.7
conda activate neuralcoref
git clone https://github.com/huggingface/neuralcoref.git
cd neuralcoref
pip install -r requirements.txt
pip install -e .
python -m spacy download en
conda install numpy scipy cython pybind11 pandas pip
"""
