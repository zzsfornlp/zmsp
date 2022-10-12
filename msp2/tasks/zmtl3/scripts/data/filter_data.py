#

# filter data for certain types

import json
import sys
from collections import Counter
from msp2.data.resources import FramePresetHelper
from msp2.utils import zlog, default_json_serializer

# ..., ..., for example ace.-s5+*
def main(input_file: str, output_file: str, signature: str):
    insts = default_json_serializer.load_list(input_file)
    cc0 = Counter()  # yes
    cc1 = Counter()  # nope
    helper = FramePresetHelper(signature)
    for inst in insts:
        new_event_mentions = []
        for evt in inst["event_mentions"]:
            if helper.f(evt["event_type"]):
                new_event_mentions.append(evt)
                cc0[evt["event_type"]] += 1
            else:
                cc1[evt["event_type"]] += 1
        inst["event_mentions"] = new_event_mentions
    default_json_serializer.save_iter(insts, output_file)
    # --
    count_yes, count_nope = sum(cc0.values()), sum(cc1.values())
    zlog(f"Yes ones ({count_yes}): {cc0}")
    zlog(f"Nope ones ({count_nope}): {cc1}")
    # --

if __name__ == '__main__':
    main(*sys.argv[1:])
