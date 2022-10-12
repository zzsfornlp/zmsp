#

import sys
import os
from msp2.utils import zopen, zlog, mkdir_p, Random
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

# --
def main(input_file: str, output_piece: int, output_prefix: str):
    output_dir = os.path.dirname(output_prefix)
    if output_dir:
        mkdir_p(output_dir, raise_error=True)
    # --
    _gen = Random.get_generator('split')
    output_piece = int(output_piece)
    reader = ReaderGetterConf().get_reader(input_path=input_file)
    insts = list(reader)
    _gen.shuffle(insts)
    bins = [[z for ii2,z in enumerate(insts) if ii2%output_piece==ii] for ii in range(output_piece)]
    zlog(f"Split from {input_file} to {output_prefix} with [{output_piece}]{[len(z) for z in bins]}")
    # --
    # output
    all_iis = set(range(output_piece))
    _padn = len(str(output_piece))
    _pads = f"%0{_padn}d"
    for ii in range(output_piece):
        ii2 = (ii+1) % output_piece
        test_iis, dev_iis, train_iis = [ii], [ii2], sorted(all_iis - {ii, ii2})
        for iis, wset in zip([test_iis, dev_iis, train_iis], ['test', 'dev', 'train']):
            with WriterGetterConf().get_writer(output_path=f"{output_prefix}{_pads % ii}.{wset}.json") as writer:
                writer.write_insts(sum([bins[z] for z in iis], []))
    # --

# python3 -m msp2.tasks.zmtl3.mat.prep.split ...
if __name__ == '__main__':
    main(*sys.argv[1:])
