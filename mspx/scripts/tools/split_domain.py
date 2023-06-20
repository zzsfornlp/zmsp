#

# split file (of insts) by domain
# -- (similar to sample_shuffle)

from collections import Counter, defaultdict
from mspx.data.rw import ReaderGetterConf, WriterGetterConf
from mspx.utils import zlog, zopen, Conf, Random, zglobs, init_everything, ZHelper

# --
class MainConf(Conf):
    def __init__(self):
        self.use_inst = True
        self.skip_blank = False  # skip blank line
        self.R = ReaderGetterConf()
        self.W = WriterGetterConf()
        # --
        self.domain_key_f = "lambda x: x.id"
        self.output_key_pattern = 'ZZKEYZZ'
        # --

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
# helper for ACE-EN
def judge_ace_genre(doc_id: str):
    cols = [
       ('bc', ["CNN_CF", "CNN_IP", "CNN_LE"]),
       ('bn', ["CNN_ENG", "CNNHL_ENG"]),
       ('cts', ["fsh"]),
       ('nw', ["AFP", "APW", "NYT", "XIN"]),
       ('un', ["alt", "aus", "Austin", "Integritas", "marcellapr", "misc", "rec", "seattle", "soc", "talk", "uk"]),
       ('wl', ["AGGRESSIVEVOICEDAILY", "BACONSREBELLION", "FLOPPINGACES", "GETTINGPOLITICAL", "HEALINGIRAQ", "MARKBACKER", "MARKETVIEW", "OIADVANTAGE", "TTRACY"]),
    ]
    for gg, pp in cols:
        if any(doc_id.startswith(p) for p in pp):
            return gg
    raise RuntimeError()
# --

# --
def main(args):
    conf: MainConf = init_everything(MainConf(), args)
    cc = Counter()
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
    # --
    # split domain
    key_f = ZHelper.eval_ff(conf.domain_key_f, 'x', globals().copy(), locals().copy())
    domains = defaultdict(list)
    for inst in insts:
        domain_key = key_f(inst)
        domains[domain_key].append(inst)
    domains = {k: domains[k] for k in sorted(domains.keys())}  # simply sort by name!
    # stat
    for k, vs in domains.items():
        cc[f'instC_{k}'] = len(vs)
        cc2 = Counter()
        for v in vs:
            stat(cc2, v)
        zlog(f"Key={k}: {cc2}")
    # --
    # output
    if conf.W.has_path():
        _path0 = conf.W.output_path
        if conf.output_key_pattern in _path0:
            _pre, _post = _path0.rsplit(conf.output_key_pattern, 1)
        else:
            _pre, _post = _path0.rsplit(".", 1)
            _pre, _post = _pre+".", "."+_post
        zlog("Do write")
        for k, one_insts in domains.items():
            one_path = _pre + str(k) + _post
            zlog(f"Write {one_path}: L={len(one_insts)}")
            if conf.use_inst:
                with conf.W.get_writer(output_path=one_path) as writer:
                    writer.write_insts(one_insts)
            else:
                with zopen(one_path, 'w') as fd:
                    for line in one_insts:
                        fd.write(line)
    # --
    zlog(f"split_domain: {conf.R.input_path} => {conf.W.output_path}: {ZHelper.resort_dict(cc)}")
    # --

# --
# python3 -m mspx.scripts.tools.split_domain input_path:?? output_path:??
if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
