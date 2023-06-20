#

# prepare alpaca data

import sys
import os
from mspx.utils import default_json_serializer

def main(data, output):
    if data.startswith("alpaca"):
        _tmp_file = f"_tmp{os.getpid()}.json"
        assert _tmp_file != output
        url = f"https://raw.githubusercontent.com/tloen/alpaca-lora/main/{data}.json"
        os.system(f"wget {url} -O {_tmp_file}")
        insts = default_json_serializer.from_file(_tmp_file)
        if output:
            default_json_serializer.save_iter(insts, output)
    # --

# --
if __name__ == '__main__':
    main(*sys.argv[1:])

"""
python3 -m mspx.znew.prompt.scripts.prep_alpaca alpaca_data alpaca_data.json
"""
