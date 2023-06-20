#

# sample/shuffle/split lines

from collections import Counter
from mspx.data.rw import ReaderGetterConf, WriterGetterConf
from mspx.utils import zlog, zopen, Conf, Random, zglobs, init_everything, ZHelper

# --
class MainConf(Conf):
    def __init__(self):
        self.use_inst = False  # use inst.rw rather simply lines
        self.R = ReaderGetterConf()
        self.W = WriterGetterConf()
        self.skip_blank = False  # skip blank line
        self.shuffle_times = 0  # shuffle times?
        self.rate = 1.  # <=1. as rate, >1 as number
        self.sample_size_f = 'None'  # inst or len
        self.output_key_pattern = 'ZZKEYZZ'
        self.split_sep = []  # specifically separate some splittings (#insts), name as *.s#N.*
        self.split_names = []  # names to specify (otherwise s#N)
        self.split_piece = 1  # for the remaining, number of pieces to split, name as *.#N.*

# --
def stat(cc, doc):
    cc['c_doc'] += 1
    cc['c_sent'] += len(doc.sents)
    cc['c_word'] += sum(len(s) for s in doc.sents)
    for cate in doc.get_frame_cates():
        for ff in doc.yield_frames(cates=cate):
            cc[f'c_f_{cate}'] += 1
            cc[f'c_a_{cate}'] += len(ff.args)
    # --

# --
def main(args):
    conf: MainConf = init_everything(MainConf(), args)
    cc = Counter()
    size_f = eval(conf.sample_size_f)
    # --
    # input
    zlog("Do read")
    if conf.use_inst:
        insts = list(conf.R.get_reader())
    else:
        _inputs = zglobs(conf.R.input_path)
        insts = []
        for _one in _inputs:
            with zopen(_one) as fd:
                if conf.skip_blank:
                    insts.extend(z for z in fd if len(z.strip())>0)
                else:
                    insts.extend(fd)
    cc['inst_orig'] = len(insts)
    # shuffle?
    if conf.shuffle_times > 0:
        zlog("Do shuffle")
        _gen = Random.get_generator('')
        for _ in range(conf.shuffle_times):
            _gen.shuffle(insts)
    # sample?
    if conf.rate != 1.:
        zlog("Do sample")
        final_size = int(0.999 + (conf.rate * len(insts) if conf.rate<=1. else conf.rate))
        if size_f is None:  # simply by count
            insts = insts[:final_size]
        else:
            remaining_budget = final_size
            new_insts = []
            for one_inst in insts:
                if remaining_budget <= 0:
                    break
                _size = size_f(one_inst)
                remaining_budget -= _size
                new_insts.append(one_inst)
            insts = new_insts
    # stat
    cc['inst_final'] = len(insts)
    if conf.use_inst:
        for inst in insts:
            stat(cc, inst)
    # output
    if conf.W.has_path():
        _path0 = conf.W.output_path
        _split_names = list(conf.split_names)
        buckets = []
        _pre, _post = _path0.rsplit(conf.output_key_pattern, 1) if conf.output_key_pattern in _path0 else _path0.rsplit(".", 1)
        _pre, _post = _pre.rstrip('.'), _post.lstrip('.')
        _cur_ss = 0
        for _ii, _ss in enumerate([int(z) for z in conf.split_sep]):
            _mid = _split_names.pop(0) if len(_split_names)>0 else f"s{_ii}"
            buckets.append(('.'.join([_pre, _mid, _post]), insts[_cur_ss:(_cur_ss+_ss)]))
            _cur_ss += _ss
        if _cur_ss > 0:
            insts = insts[_cur_ss:]
        if conf.split_piece > 1:
            _mids = ZHelper.pad_strings(list(range(conf.split_piece)), '0')
            buckets2 = []
            for _ii2 in range(conf.split_piece):
                _mid = _split_names.pop(0) if len(_split_names)>0 else _mids[_ii2]
                buckets2.append(('.'.join([_pre, _mid, _post]), []))
            for ii, inst in enumerate(insts):  # note: interleavely split
                buckets2[ii%conf.split_piece][1].append(inst)
        else:
            buckets2 = [(_path0, insts)]
        buckets.extend(buckets2)
        # --
        zlog("Do write")
        for one_path, one_insts in buckets:
            zlog(f"Write {one_path}: L={len(one_insts)}")
            if conf.use_inst:
                with conf.W.get_writer(output_path=one_path) as writer:
                    writer.write_insts(one_insts)
            else:
                with zopen(one_path, 'w') as fd:
                    for line in one_insts:
                        fd.write(line)
    # --
    zlog(f"sample_shuffle: {conf.R.input_path} => {conf.W.output_path} [split={conf.split_sep}+{conf.split_piece}]: {cc}")

# --
# python3 -m mspx.scripts.tools.sample_shuffle input_path:?? output_path:??
# python3 sample_shuffle.py input_path:?? output_path:??
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
