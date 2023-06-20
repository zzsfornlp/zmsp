#

# extraction task

__all__ = [
    "ZTaskExtConf", "ZTaskExt", "ZModExtConf", "ZModExt"
]

from mspx.data.inst import yield_frames
from mspx.data.vocab import Vocab
from mspx.proc.eval import FrameEvalConf
from mspx.utils import zlog, ConfEntryChoices
from mspx.nn import BK, ZTaskConf, ZModConf, ZTaskSbConf, ZTaskSb, ZModSbConf, ZModSb, ZRunCache

# --

@ZTaskConf.rd('ext')
class ZTaskExtConf(ZTaskSbConf):
    def __init__(self):
        super().__init__()
        self.mod = ZModExtConf.get_entries()
        self.eval_ = FrameEvalConf().direct_update(
            weight_frame=1., weight_arg=0., bd_frame_lines=50, _rm_names=['frame_cate'])
        self.frame_cate = ['ef']  # frame category
        # --
        self.lab_nil = "_NIL_"  # special tag for NIL
        self.lab_unk = "_UNK_"  # special tag for UNK (PA)

@ZTaskExtConf.conf_rd()
class ZTaskExt(ZTaskSb):
    def __init__(self, conf: ZTaskExtConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZTaskExtConf = self.conf
        # --
        self.eval = [conf.eval_.make_node(frame_cate=z) for z in conf.frame_cate]
        if len(self.eval) == 1:
            self.eval = self.eval[0]
        # --

    @property
    def frame_cate(self):
        return self.conf.frame_cate

    # some special processings to handle special marks in AL
    def process_label(self, lab: str):
        sep = '___'
        lab0 = None
        if sep in lab:
            lab0, lab = lab.split(sep, 1)
        if lab.startswith("**"):
            lab = lab[2:]
        if lab0 is not None:
            lab = lab0 + sep + lab
        return lab

    def get_frame_label(self, frame):
        conf: ZTaskExtConf = self.conf
        _multi_cates = len(conf.frame_cate) > 1
        _special_labs = [conf.lab_nil, conf.lab_unk]
        if not _multi_cates or self.process_label(frame.label) in _special_labs:
            ret = self.process_label(frame.label)
        else:
            ret = self.process_label(frame.cate_label)
        return ret

    def build_vocab(self, datasets):
        conf: ZTaskExtConf = self.conf
        _cates = conf.frame_cate
        _ignore_labs = [conf.lab_nil, conf.lab_unk]
        # --
        voc = Vocab.build_empty(f"voc_{self.name}")
        for dataset in datasets:
            if dataset.name.startswith('train'):
                for frame in yield_frames(dataset.yield_insts(), True, cates=_cates):
                    flab = self.get_frame_label(frame)
                    if flab not in _ignore_labs:  # ignore special!
                        voc.feed_one(flab)
        voc.build_sort()
        zlog(f"Finish building for: {voc}")
        return (voc, )
        # --

# just an abstract base class!
class ZModExtConf(ZModSbConf):
    def __init__(self):
        super().__init__('bmod2')
        self.remain_toks = True  # overwrite!
        # --

    @staticmethod
    def get_entries():  # note: allow different extractors!
        from .m_seq import ExtSeqlabConf
        return ConfEntryChoices({'seq': ExtSeqlabConf()}, 'seq')

class ZModExt(ZModSb):
    def __init__(self, conf: ZModExtConf, ztask: ZTaskExt, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZModExtConf = self.conf
        assert conf.remain_toks
        self.frame_cate = self.ztask.conf.frame_cate
        self.voc = ztask.vpack[0]
        # --
