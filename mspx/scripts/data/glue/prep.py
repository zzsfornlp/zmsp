#

# simply prepare glue with "datasets"

import sys
import os
import datasets
from mspx.data.inst import Sent, NLTKTokenizer
from mspx.data.rw import WriterGetterConf
from mspx.utils import mkdir_p, zlog

# --
def get_data(data: str, wset: str, do_toker: bool, t_names, cache_dir: str):
    dataset = datasets.load_dataset('glue', data, split=wset, cache_dir=cache_dir)
    # --
    if t_names is None:  # "sentence?"
        s_prefix = 'sentence'
        if (s_prefix+"1") in dataset.features:
            t_names = (s_prefix+"1", s_prefix+"2")
        else:
            t_names = (s_prefix, )
    # --
    insts = []
    toker = NLTKTokenizer() if do_toker else None
    toker_f = (lambda _x: toker.tokenize(_x, split_sent=False)) if do_toker else (lambda _x: _x.split())
    for one in dataset:
        info = {'idx': one['idx'], 'label': one['label']}
        if len(t_names) == 1:
            tokens = toker_f(one[t_names[0]].strip())
        else:
            n1, n2 = t_names
            t1, t2 = toker_f(one[n1].strip()), toker_f(one[n2].strip())
            info['len0'] = len(t1)
            tokens = t1 + t2  # note: simply concat!
        sent = Sent(tokens)
        sent.info.update(info)
        insts.append(sent.make_singleton_doc())
    zlog(f"Load {data}({wset})[{do_toker}, {t_names}]: {len(insts)}")
    return insts
# --

DATA = {
    'cola': (['train', 'validation', 'test'], True, None),
    'sst2': (['train', 'validation', 'test'], False, None),
    'mrpc': (['train', 'validation', 'test'], False, None),
    'qqp': (['train', 'validation', 'test'], True, ('question1', 'question2')),
    'stsb': (['train', 'validation', 'test'], True, None),
    'mnli': (['train', 'validation_matched', 'test_matched', 'validation_mismatched', 'test_mismatched'],
             True, ('premise', 'hypothesis')),
    'qnli': (['train', 'validation', 'test'], True, ('question', 'sentence')),
    'rte': (['train', 'validation', 'test'], True, None),
    'wnli': (['train', 'validation', 'test'], True, None),
    'ax': (['test'], True, ('premise', 'hypothesis'))
}

def main(trg_dir='.'):
    cache_dir = os.path.join(trg_dir, "_cache")
    mkdir_p(trg_dir)
    mkdir_p(cache_dir)
    # --
    for data, (wsets, do_toker, t_names) in DATA.items():
        for wset in wsets:
            insts = get_data(data, wset, do_toker, t_names, cache_dir)
            wset = wset.replace('validation', 'dev').replace('matched', 'm')
            output_path = os.path.join(trg_dir, f"{data}.{wset}.json")
            with WriterGetterConf().get_writer(output_path=output_path) as writer:
                writer.write_insts(insts)
    # --

# --
# python3 -m mspx.scripts.data.glue.prep
if __name__ == '__main__':
    main(*sys.argv[1:])
