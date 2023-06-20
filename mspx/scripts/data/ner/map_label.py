#

# map the labels

from collections import Counter
from mspx.data.rw import ReaderGetterConf, WriterGetterConf, FileStreamer
from mspx.utils import zlog

MAPPINGS = {
'cner_T': {  # trg2src
    'PER': 'researcher, person, writer, musicalartist, politician, scientist',
    'ORG': 'organisation, university, band, politicalparty',
    'LOC': 'country, location',
    'MISC': 'misc',
}
}

def main(input_path, output_path, code):
    # get the map
    _map = MAPPINGS[code]
    for k in list(_map.keys()):
        v = _map[k]
        if isinstance(v, str):
            v = [z.strip() for z in v.split(",")]
            _map[k] = set(v)
    if code.endswith("_T"):  # reverse
        _map2 = {}
        for k, v in _map.items():
            for k2 in v:
                assert k2 not in _map2
                _map2[k2] = k
        _map = _map2
    # --
    cc = Counter()
    cc2 = Counter()
    insts = list(ReaderGetterConf().get_reader(input_path=input_path))
    for inst in insts:
        cc['inst'] += 1
        for frame in inst.get_frames():
            trg_label = _map.get(frame.label)
            cc2[trg_label] += 1
            cc['frame'] += 1
            if trg_label is None:
                frame.del_self()
                cc['frame_del'] += 1
            elif trg_label != frame.label:
                frame.set_label(trg_label)
                cc['frame_reset'] += 1
            else:
                cc['frame_keep'] += 1
    zlog(f"Finish processing {input_path}: {cc} {cc2}")
    if output_path:
        with WriterGetterConf().get_writer(output_path=output_path) as writer:
            writer.write_insts(insts)
    # --

# python3 -m mspx.scripts.data.ner.map_label
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
