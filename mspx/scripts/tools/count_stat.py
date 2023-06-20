#

# simple counting of stats

from collections import Counter, defaultdict
from mspx.data.rw import ReaderGetterConf, WriterGetterConf
from mspx.utils import zlog, zopen, Conf, Random, zglobs, init_everything, MyCounter, ZHelper

# --
class MainConf(Conf):
    def __init__(self):
        self.R = ReaderGetterConf()
        self.report_batch_interval = 10000
        self.count_types = False
        self.alink_cates = ["", ""]  # H->T
        self.span_cates = []  # check span overlaps?

# --
def stat(cc, tcc, doc, alink_cates, span_cates):
    cc['c_doc'] += 1
    cc['c_sent'] += len(doc.sents)
    cc['c_word'] += sum(len(s) for s in doc.sents)
    for cate in doc.get_frame_cates():
        for ff in doc.yield_frames(cates=cate):
            cc[f'c_F{cate}'] += 1
            cc[f'c_A{cate}'] += len(ff.args)
            tcc[f"F{cate}"][ff.label] += 1
            for arg in ff.args:
                tcc[f"A{cate}"][arg.label] += 1
    # --
    if span_cates:
        for sent in doc.sents:
            frames = sent.get_frames(cates=span_cates)
            for f0 in frames:
                has_overlap, has_nearby, has_nearbyL = 0, 0, 0
                _widx0, _wlen0 = f0.mention.get_span()
                for f1 in frames:
                    if f1 is f0: continue
                    _widx1, _wlen1 = f1.mention.get_span()
                    if f0.mention.overlap_tokens(f1.mention):
                        has_overlap += 1
                    if _widx0 == _widx1 + _wlen1 or _widx1 == _widx0 + _wlen0:
                        has_nearby += 1
                        if f0.label == f1.label:
                            has_nearbyL += 1
                cc['span'] += 1
                cc[f'span_overlap={has_overlap}'] += 1
                cc[f'span_nearby={has_nearby}'] += 1
                cc[f'span_nearbyL={has_nearbyL}'] += 1
    # --
    cateH, cateT = alink_cates
    if cateH and cateT:
        cateH, cateT = cateH.split(':'), cateT.split(':')
        for sent in doc.sents:
            frame_hs, frame_ts = sent.get_frames(cates=cateH), sent.get_frames(cates=cateT)
            hit_ts = {id(f) for f in frame_ts}
            for frame in frame_hs:
                cc['cp_cands'] += len(frame_ts) - int(id(frame) in hit_ts)  # excluding self!
                for alink in frame.args:
                    cc['cp_valid'] += (id(alink.arg) in hit_ts)
                    cc['cp_loop'] += any(frame is z.arg for z in alink.arg.args)
    # --

# --
def main(args):
    conf: MainConf = init_everything(MainConf(), args)
    cc = Counter()
    tcc = defaultdict(MyCounter)
    # --
    ii = 0
    for inst in conf.R.get_reader():
        stat(cc, tcc, inst, conf.alink_cates, conf.span_cates)
        ii += 1
        if ii % conf.report_batch_interval == 0:
            zlog(f"Stat progress: {cc}", timed=True)
    zlog(f"Stat Finish: {ZHelper.resort_dict(cc)}", timed=True)
    if conf.count_types:
        for key, one_tcc in tcc.items():
            zlog(f"#--\nTypes[{key}]: {one_tcc.summary_str()}")
    # --

# python3 -m mspx.scripts.tools.count_stat input_path:??
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
