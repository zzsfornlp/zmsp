#

# read atis data

from collections import Counter
from mspx.data.inst import Sent, Doc, NLTKTokenizer
from mspx.data.rw import ReaderGetterConf, WriterGetterConf, FileStreamer
from mspx.data.vocab import SeqSchemeHelperStr
from mspx.utils import zlog, zopen, Conf, Random, zglobs, init_everything, default_json_serializer

class MainConf(Conf):
    def __init__(self):
        self.data = 'atis'  # atis or snips
        self.input_path = []
        self.W = WriterGetterConf()
        # --
        self.filter_data = ""
        # --

# --
def main(args):
    conf: MainConf = init_everything(MainConf(), args)
    cc = Counter()
    # --
    input_paths = zglobs(conf.input_path)
    all_insts = []
    toker = NLTKTokenizer({'atis': 'simple', 'snips': 'default'}[conf.data])
    for f in input_paths:
        cc['file'] += 1
        # --
        data = default_json_serializer.from_file(f)
        if conf.data == 'atis':
            for inst in data['rasa_nlu_data']['common_examples']:
                text = inst['text']
                # tokenization
                tokens, _, char2posi = toker.tokenize(text, split_sent=False, return_posi_info=True)
                # entities
                sent = Sent(tokens, make_singleton_doc=True)
                sent.info['intent'] = inst['intent']
                cc['sent'] += 1
                for ent in inst['entities']:
                    hit_tidxes = sorted(set([char2posi[z] for z in range(ent['start'], ent['end']) if z<len(text) and char2posi[z] is not None]))
                    i0, i1 = hit_tidxes[0], hit_tidxes[-1]
                    assert tokens[i0:i1+1] == ent['value'].split()
                    sent.make_frame(i0, i1-i0+1, ent['entity'], 'ef')
                    cc['ef'] += 1
                all_insts.append(sent.doc)
        elif conf.data == 'snips':
            assert len(data) == 1
            intent = list(data.keys())[0]
            for inst in data[intent]:
                tokens = []
                entities = []
                for piece in inst['data']:
                    p_toks = toker.tokenize(piece['text'], split_sent=False, return_posi_info=False)
                    if 'entity' in piece:
                        entities.append((len(tokens), len(p_toks), piece['entity']))
                    tokens.extend(p_toks)
                sent = Sent(tokens, make_singleton_doc=True)
                sent.info['intent'] = intent
                cc['sent'] += 1
                for widx, wlen, label in entities:
                    sent.make_frame(widx, wlen, label, 'ef')
                    cc['ef'] += 1
                all_insts.append(sent.doc)
        else:
            raise NotImplementedError(f"UNK data of {conf.data}")
        # --
    # --
    # filter?
    if conf.filter_data:
        # --
        def _get_key(_ts):
            return "".join(["".join(c for c in z if str.isalnum(c)) for z in _ts]).lower()
        # --
        cc0 = cc
        zlog(f"Stat before filtering: {cc0}")
        cc = Counter()
        filter_keys = Counter()
        with zopen(conf.filter_data) as fd:
            for line in fd:
                filter_keys[_get_key(line.split())] += 1
        zlog(f"Read filter {len(filter_keys)}/{sum(filter_keys.values())} from {conf.filter_data}")
        final_insts = []
        for one_inst in all_insts:
            _key = _get_key(one_inst.sent_single.seq_word.vals)
            if filter_keys[_key] > 0:
                final_insts.append(one_inst)
                filter_keys[_key] -= 1
                cc['inst_in'] += 1
                cc['sent'] += 1
                cc['ef'] += len(one_inst.get_frames())
            else:
                cc['inst_out'] += 1
        all_insts = final_insts
        if sum(filter_keys.values()) > 0:
            zlog(f"Unfound: {[k for k,v in filter_keys.items() if v>0]}")
    # --
    if conf.W.has_path():
        with conf.W.get_writer() as writer:
            writer.write_insts(all_insts)
    zlog(f"Read from {input_paths} to {conf.W.output_path}: {cc}", timed=True)
    # --

# python3 -m mspx.scripts.data.nlu.read_atis input_path:??
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
