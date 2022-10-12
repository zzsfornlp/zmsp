#

# merge pb's sents into docs using special info

import sys
from collections import Counter
from msp2.data.inst import Doc, Sent
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.utils import zlog

def new_doc(sents, writer, cc):
    cc['doc'] += 1
    cc['sent'] += len(sents)
    d = Doc.create(sents, id=sents[-1].info['doc_id'])
    if writer is not None:
        writer.write_inst(d)
    return d

def main(input_file: str, output_file: str):
    cc = Counter()
    # --
    reader = ReaderGetterConf().get_reader(input_path=input_file)
    with WriterGetterConf().get_writer(output_path=output_file) as writer:
        cur_sents = []
        for one in reader:
            assert isinstance(one, Sent)
            doc_id = one.info['doc_id']
            if len(cur_sents)>0 and cur_sents[-1].info['doc_id'] != doc_id:
                new_doc(cur_sents, writer, cc)
                cur_sents = []  # refresh!
            cur_sents.append(one)
        if len(cur_sents)>0:
            new_doc(cur_sents, writer, cc)
    # --
    zlog(f"pbs2d from {input_file} to {output_file}: {cc}")
    # --

# python3 -m msp2.tasks.zmtl3.scripts.srl.s4_pbs2d IN OUT
if __name__ == '__main__':
    main(*sys.argv[1:])
