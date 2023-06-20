#

# artificially create noisy data

from collections import Counter, defaultdict
from mspx.data.inst import yield_frames, yield_sents
from mspx.data.rw import ReaderGetterConf, WriterGetterConf
from mspx.utils import zlog, zopen, Conf, Random, zglobs, init_everything

class MainConf(Conf):
    def __init__(self):
        self.R = ReaderGetterConf()
        self.W = WriterGetterConf()
        self.frame_cates = ['ef']
        self.group_f = "lambda f: f.mention.get_words(concat=True)"
        self.trg_recall = 0.5
        self.trg_precision = 0.9
        self.adding_range = [1, 4]  # adding span of [a,b)

def my_choice(_gen, cands):
    ii = _gen.randint(len(cands))
    return cands[ii]

# --
def main(args):
    conf: MainConf = init_everything(MainConf(), args)
    # --
    # read
    reader = conf.R.get_reader()
    insts = list(reader)
    all_frames = list(yield_frames(insts, cates=conf.frame_cates))
    curr_all_frames = all_frames
    cc_labs = Counter(z.cate_label for z in all_frames)
    all_labs = list(cc_labs.keys())
    zlog(f"Read from {reader}: frames[{len(all_frames)}]: {cc_labs}")
    # --
    # noise
    _gen = Random.get_generator()
    # - first to target recall
    if conf.trg_recall < 1.:
        _group_f = eval(conf.group_f)
        frame_groups = defaultdict(list)
        for frame in all_frames:
            key = _group_f(frame)
            frame_groups[key].append(frame)
        all_group_keys = list(frame_groups.keys())
        to_remove = min(int(len(all_frames) * (1. - conf.trg_recall)), len(all_frames))
        _gen.shuffle(all_group_keys)
        for kk in all_group_keys:
            if to_remove <= 0: break
            for ff in frame_groups[kk]:
                ff.del_self()
                to_remove -= 1
            frame_groups[kk] = []
        r_frames = sum(frame_groups.values(), [])
        zlog(f"After adjusting recall: frames[{len(r_frames)}:{len(r_frames)/len(all_frames):.3f}]:"
             f" {Counter(z.cate_label for z in r_frames)}")
        curr_all_frames = r_frames
    # - then to target precision
    if conf.trg_precision < 1.:
        assert conf.trg_precision > 0
        to_add = int(len(curr_all_frames) * (1./conf.trg_precision - 1.))
        adding_frames = []
        all_sents = list(yield_sents(insts))
        _l0, _l1 = conf.adding_range
        while 1:
            if to_add <= 0: break
            one_sent = my_choice(_gen, all_sents)
            ok_flags = [1] * len(one_sent)
            for _f in one_sent.get_frames(cates=conf.frame_cates):
                _widx, _wlen = _f.mention.get_span()
                ok_flags[_widx:_widx+_wlen] = [0] * _wlen
            _r_wlen = _gen.randint(_l0, _l1)
            _cand_starts = [ii for ii in range(len(one_sent))
                            if all((z<len(ok_flags) and ok_flags[z]) for z in range(ii, ii+_r_wlen))]
            if len(_cand_starts) == 0:
                continue  # no valid cands
            new_frame = one_sent.make_frame(my_choice(_gen, _cand_starts), _r_wlen,
                                            label=my_choice(_gen, all_labs), cate=None)
            adding_frames.append(new_frame)
            to_add -= 1
        f_frames = curr_all_frames + adding_frames
        zlog(f"After adjusting precision: frames[{len(f_frames)}:{len(curr_all_frames)/len(f_frames):.3f}]:"
             f" {Counter(z.cate_label for z in f_frames)}")
    # --
    # write
    if conf.W.has_path():
        with conf.W.get_writer() as writer:
            writer.write_insts(insts)
    # --

# python3 -m mspx.scripts.data.ner.noise_data input_path:?? output_path:??
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
