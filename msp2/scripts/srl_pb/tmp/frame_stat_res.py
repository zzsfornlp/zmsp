#

# print results for frame_stat

import json
import pandas as pd
from collections import OrderedDict
from msp2.utils import zlog, zopen, MyCounter
import functools

# --
def main(res_file="res.json", known_set="", *extra_names):
    # --
    with zopen(res_file) as fd:
        res = json.load(fd)
    # --
    def _get_data(_name: str):
        _keys = [k for k in res if _name in k]
        assert len(_keys)==1
        return res[_keys[0]]
    # --
    def _show_entry(_d: dict):
        return MyCounter(_d).summary_str(30,130)[9:]
    # -----
    all_datasets = OrderedDict([
        ("conll05", [f"conll05/{z}" for z in ["train", "dev", "test.wsj", "test.brown"]]),
        ("conll12", [f"conll12b/{z}" for z in ["train", "dev", "test"]] + ["conll12/train", "pb/ontonotes.train"]),
        ("ewt", [f"ewt.{z}" for z in ["train", "dev", "test"]]),
        ("fn15", [f"fn15_fulltext.{z}" for z in ["train", "dev", "test."]] + ["fn15_exemplars"]),
        ("fn17", [f"fn17_fulltext.{z}" for z in ["train", "dev", "test."]] + ["fn17_exemplars"]),
    ])
    all_groups = OrderedDict([
        ("basic", ["sent", "tok", "frame", "f/s", "f/t", "arg", "a/f", "a/(f*t/s)", "AO", "AO1"]),
        ("frame_wlen", ["frame_wlen"]),
        ("frame_trigger_pos", ["frame_trigger_pos"]),
        ("frame_type", ["frame_type"]),
        ("frame_type0", ["frame_type0"]),
        ("arg_wlen_m30", ["arg_wlen_m30"]),
        ("arg_role", ["arg_role"]),
        ("arg_repeat", ["arg_repeat"]),
        ("arg_repeatR", ["arg_repeatR"]),
    ])
    cc = OrderedDict()
    # --
    # first collect all
    all_data_names = sum(all_datasets.values(), []) if not known_set else all_datasets.get(known_set, [])
    all_data_names = all_data_names + list(extra_names)
    all_data = [_get_data(z) for z in all_data_names]
    # basic
    cc["sent"] = [z["sent"] for z in all_data]
    cc["tok"] = [z["tok"] for z in all_data]
    cc["frame"] = [z["frame"] for z in all_data]
    cc["f/s"] = [z["frame"]/z["sent"] for z in all_data]
    cc["f/t"] = [z["frame"]/z["tok"] for z in all_data]
    cc["arg"] = [z["arg"] for z in all_data]
    cc["a/f"] = [z["arg"] / z["frame"] for z in all_data]
    cc["a/(f*t/s)"] = [z["arg"] / (z["frame"]*z["tok"]/z["sent"]) for z in all_data]
    cc["AO"] = [z["arg_overlapped"]/z["arg"] for z in all_data]
    cc["AO1"] = [z["arg_overlapped_R1"]/z["arg_R1"] for z in all_data]
    # others
    cc["frame_wlen"] = [_show_entry(z["frame_wlen"]) for z in all_data]
    cc["frame_trigger_pos"] = [_show_entry(z["frame_trigger_pos"]) for z in all_data]
    cc["frame_type"] = [_show_entry(z["frame_type"]) for z in all_data]
    cc["frame_type0"] = [_show_entry(z["frame_type0"]) for z in all_data]
    cc["arg_wlen_m30"] = [_show_entry(z["arg_wlen_m30"]) for z in all_data]
    cc["arg_role"] = [_show_entry(z["arg_role"]) for z in all_data]
    cc["arg_repeat"] = [_show_entry(z["arg_repeat"]) for z in all_data]
    cc["arg_repeatR"] = [_show_entry(z["arg_repeatR"]) for z in all_data]
    # --
    def left_justified(df):
        try:
            formatters = {}
            for li in list(df.columns):
                max = df[li].str.len().max()
                form = "{{:<{}s}}".format(max)
                formatters[li] = functools.partial(str.format, form)
            return df.to_string(formatters=formatters)
        except:
            return df.to_string()
    # --
    dd = pd.DataFrame(cc, index=all_data_names)
    for group_name, group_keys in all_groups.items():
        zlog(f"#== GROUP {group_name}\n" + left_justified(dd[group_keys]))
    # breakpoint()

# PYTHONPATH=../src/ ptyhon3 frame_stat_res.py
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
