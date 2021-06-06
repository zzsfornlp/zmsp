#

# simply delete all frames

from msp2.data.inst import yield_sents
from msp2.data.rw import ReaderGetterConf, WriterGetterConf

# --
def main(file_in="", file_out=""):
    insts = list(ReaderGetterConf().get_reader(input_path=file_in))  # read from stdin
    with WriterGetterConf().get_writer(output_path=file_out) as writer:
        for inst in insts:
            for sent in yield_sents([inst]):
                sent.delete_frames("evt")
                sent.delete_frames("ef")
            writer.write_inst(inst)

# PYTHONPATH=../src/ python3 del_frames.py <IN >OUT
if __name__ == '__main__':
    import sys
    main(*sys.argv[1:])
