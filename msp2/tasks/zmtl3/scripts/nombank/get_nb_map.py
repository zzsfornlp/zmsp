#

# obtain a map from nb to pb
# note: simply using the 'source', but might not be exactly accurate

import sys
from collections import Counter
from msp2.utils import default_pickle_serializer, default_json_serializer, zlog, zwarn, OtherHelper

def main(input_file: str, output_file: str):
    fc = default_pickle_serializer.from_file(input_file)
    # --
    m = {}
    cc = Counter()
    for f in fc.frames:
        cc['frame'] += 1
        if f.name in m:
            zwarn(f"Name repeat: {f.name}")
        else:
            trg = f.info.get('source', '')
            if trg:
                cc['frame_hit'] += 1
                m[f.name] = trg
            else:
                cc['frame_miss'] += 1
    # --
    OtherHelper.printd(cc, sep=' || ')
    if output_file:
        default_json_serializer.to_file(m, output_file)
    # --

# --
# python3 -m msp2.tasks.zmtl3.scripts.nombank.get_nb_map frames.nb.pkl map.nb.json
if __name__ == '__main__':
    main(*sys.argv[1:])
