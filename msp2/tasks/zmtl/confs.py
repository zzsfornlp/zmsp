#

# common confs

from typing import List, Dict
from msp2.utils import Conf, ConfEntryChoices
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.proc import SVConf, FrameEvalConf, MyFNEvalConf, MyPBEvalConf, TRConf, DparEvalConf

# --
# Data Conf
class DConf(Conf):
    def __init__(self):
        # reading and writing formats
        # self.R = ReaderGetterConf()
        from .run import MyDataReaderConf
        self.R = MyDataReaderConf()  # reader + optional extra functionalities
        self.W = WriterGetterConf()
        # --
        # data paths: train/dev can be multiple!
        self.train: List[str] = []
        self.train_props: List[Dict] = []  # extra properties for R
        self.dev: List[str] = []
        self.test = ""
        self.test_do_score = False  # do score instead of predict!!
        self.cache_data = True  # turn off if large data
        self.cache_shuffle = True  # shuffle cache if using cache
        self.dict_dir = "./"
        self.dict_frame_file = ""  # read frame file for building types
        # -----
        # for multi-source training (len(train)>1); note: currently three should be enough
        self.ms_stop_idx = 0  # with which to consider the end of an epoch
        self.ms_train_budget0 = SVConf().direct_update(val=1., which_idx="eidx", mode="none", min_val=0.)
        self.ms_train_budget1 = SVConf().direct_update(val=1., which_idx="eidx", mode="none", min_val=0.)
        self.ms_train_budget2 = SVConf().direct_update(val=1., which_idx="eidx", mode="none", min_val=0.)
        # -----
        # save name for trainer not here!!
        self.model_load_name = "zmodel.best.m"  # load name
        self.model_load_strict = True  # strict loading?
        self.output = "zout.json"
        self.eval_conf = ConfEntryChoices(
            {'frame': FrameEvalConf(), 'fn': MyFNEvalConf(), 'pb': MyPBEvalConf(), 'dpar': DparEvalConf(), 'none': None}, 'frame')
        self.eval2_conf = ConfEntryChoices(  # secondary eval
            {'frame': FrameEvalConf(), 'fn': MyFNEvalConf(), 'pb': MyPBEvalConf(), 'dpar': DparEvalConf(), 'none': None}, 'none')
        # pretrain
        self.pretrain_wv_file = ""
        self.pretrain_scale = 1.0
        # (not used anymore) self.pretrain_init_nohit = 1.0
        self.pretrain_hits_outf = ""  # output a by-product of filtered embeddings
        # extra for testing
        self.test_extra_pretrain_wv_files = []
        # thresholds for word
        self.word_rthres = 200000  # rank <= this
        self.word_fthres = 1  # freq >= this

    def get_ms_train_budgets(self):
        cands = [self.ms_train_budget0, self.ms_train_budget1, self.ms_train_budget2]
        assert len(cands) >= len(self.train)
        return cands[:len(self.train)]

    def get_train_props(self):
        cands = self.train_props.copy()
        for _ in range(len(self.train) - len(cands)):
            cands.append({})  # by default no props!
        return cands

# --
# Training/Testing Conf
class TConf(Conf):
    def __init__(self):
        # =====
        # training
        self.tr_conf = TRConf()
        self.train_batch_size = 32  # batch size
        self.train_maxibatch_size = 20  # number of batches to collect for the batcher
        # --
        self.train_stream_reshuffle_times = 0  # sent_stream shuffle (most time already shuffled with cache!)
        self.train_stream_reshuffle_bsize = 1000  # sent_stream shuffle
        self.train_batch_shuffle_times = 2  # b_stream shuffle
        # --
        self.train_stream_mode = "sent"  # stream sents in training
        self.train_count_mode = "sent"  # counting for batch_size
        self.train_min_length = 0
        self.train_max_length = 100
        self.train_skip_noevt_rate = 0.  # the rate to skip sentences without any event in training
        self.train_skip_batch_noevt = False  # skip batches with no frames
        # --
        self.no_build_dict = False
        self.load_model = False
        self.load_process = False
        # =====
        # testing
        self.test_count_mode = "sent"
        self.test_batch_size = 16
        self.test_do_oracle_batching = False  # first run once and determine batch by exit_layers
        # =====
        # general
        self.train_use_cache = True
        self.dev_use_cache = True
        self.cache_shuffle_times = 2
        # =====
        # special for CL
        from .run import CLHelperConf
        self.cl_helper = CLHelperConf()

# the overall conf
class OverallConf(Conf):
    def __init__(self):
        from .model import ZmtlModelConf
        self.dconf = DConf()  # data conf
        self.tconf = TConf()  # train/test conf
        self.mconf = ConfEntryChoices({"mtl": ZmtlModelConf()}, 'mtl')

# get model
def build_model(conf: OverallConf, *args, **kwargs):
    from .model import ZmtlModelConf, ZmtlModel
    mconf = conf.mconf
    if isinstance(mconf, ZmtlModelConf):
        model = ZmtlModel(mconf, *args, **kwargs)
    else:
        raise NotImplementedError(f"Error: UNK model conf type {mconf}")
    return model
