#

# read the special conllu from fipb

from collections import Counter
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import zlog, zopen

# --
def main(input_file: str, output_file: str):
    reader_conf = ReaderGetterConf().direct_update(input_format='conllufipb')
    reader_conf.validate()
    # --
    cc = Counter()
    arg_cc = Counter()
    all_insts = list(reader_conf.get_reader(input_path=input_file))
    for sent in all_insts:
        fields_args, fields_preds = sent.info[8], sent.info[9]
        assert len(fields_args) == len(fields_preds) and len(fields_args) == len(sent)
        # first collect preds
        all_preds = {}  # widx -> event
        for widx, vv in enumerate(fields_preds):
            pred_name = None
            for vv2 in vv.split("|"):
                if vv2.startswith("PBSENSE="):
                    assert pred_name is None
                    pred_name = vv2.split("=")[-1]
            if pred_name is not None:
                evt = sent.make_event(widx, 1, type=pred_name)
                assert widx not in all_preds
                all_preds[widx] = evt
        # then collect args
        for widx, vv in enumerate(fields_args):
            for vv2 in vv.split("|"):
                if ":" not in vv2:
                    continue
                tidx, aname = vv2.split(":", 1)
                tidx = int(tidx)
                role = None
                if aname.startswith("PBArg_"):
                    nn = aname[len("PBArg_"):]
                    role = f"ARG{nn}"
                elif aname.startswith("PBArgM_"):
                    _, nn = aname.split("_")
                    role = f"ARGM-{str.upper(nn)}"
                if role is not None:
                    evt = all_preds[tidx-1]
                    ef = sent.make_entity_filler(widx, 1, type="UNK")
                    evt.add_arg(ef, role)
                    arg_cc[role] += 1
        # --
        cc["sent"] += 1
        cc["frames"] += len(sent.events)
        cc["args"] += sum(len(z.args) for z in sent.events)
        # --
    # --
    with WriterGetterConf().get_writer(output_path=output_file) as writer:
        writer.write_insts(all_insts)
    # --
    zlog(f"Read fipb from {input_file} to {output_file}: {cc}")
    zlog(f"Role counts = {arg_cc}")
    # --

# --
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])

# --
# python3 read_fipb.py IN OUT
