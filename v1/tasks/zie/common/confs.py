#

# common configurations for the IE-Event project

from msp import utils, nn
from msp.nn import NIConf
from msp.utils import Conf, Logger, zopen, zfatal, zwarn, zlog
from msp.zext.process_train import SVConf

from msp.nn.modules.berter import Berter, BerterConf

from .eval import MyIEEvalerConf

class DConf(Conf):
    def __init__(self):
        # data paths
        self.train = ""
        self.dev = ""
        self.test = ""
        self.cache_data = True          # turn off if large data
        self.cache_shuffle = True  # shuffle cache if using cache
        self.dict_dir = "./"
        # -----
        # for multi-source training (the list has to be hard_coded!!)
        # currently three should be enough
        self.ms_train = []  # replacing "train"
        self.ms_stop_idx = 0  # with which to consider the end of an epoch
        self.ms_train_budget0 = SVConf().init_from_kwargs(val=1., which_idx="eidx", mode="none", min_val=0.)
        self.ms_train_budget1 = SVConf().init_from_kwargs(val=1., which_idx="eidx", mode="none", min_val=0.)
        self.ms_train_budget2 = SVConf().init_from_kwargs(val=1., which_idx="eidx", mode="none", min_val=0.)
        # -----
        # extra aux inputs (sentence level), must be strictly aligned with the corresponded data
        self.aux_repr_train = ""
        self.aux_repr_dev = ""
        self.aux_repr_test = ""
        # save name for trainer not here!!
        self.model_load_name = "zmodel.best"  # load name
        self.output_file = "zout.json"
        # format
        self.input_format = "json"
        self.output_format = "json"
        self.eval_conf = MyIEEvalerConf()
        # special loading
        self.noef_link0 = False  # do not load (for all purposes: train/eval) Entity or Fillers with link==0
        # pretrain
        self.pretrain_file = ""
        self.init_from_pretrain = False
        self.pretrain_scale = 1.0
        self.pretrain_init_nohit = 1.0
        self.output_pretrain_filter = ""  # output a by-product of filtered embeddings
        # thresholds for word
        self.word_rthres = 50000  # rank <= this
        self.word_sthres = 2  # the threshold of considering singleton treating （freq<=this)
        self.word_fthres = 1  # freq >= this
        self.use_label0 = True  # using only first-level ud label
        # testing mode extra embeddings
        self.test_extra_pretrain_files = []
        # =====
        # berter features
        self.use_bert = False
        # self.use_bert_precompute = False
        self.bconf = BerterConf().init_from_kwargs(bert_batch_size=8, bert_root_mode=1)
        # =====
        # max evt layer
        self.max_evt_layers = 100  # doesn't matter if larger

    @property
    def ms_train_budget_list(self):
        return [self.ms_train_budget0, self.ms_train_budget1, self.ms_train_budget2]

# the overall conf
class OverallConf(Conf):
    def __init__(self, model_type, args):
        # top-levels
        self.model_type = model_type
        self.conf_output = ""
        self.log_file = Logger.MAGIC_CODE
        self.msp_seed = 9341
        #
        self.niconf = NIConf()      # nn-init-conf
        self.dconf = DConf()        # data-conf
        # parser-conf
        if model_type == "simple":
            from ..models2.model import MySimpleIEModelConf
            self.mconf = MySimpleIEModelConf()
        elif model_type == "m3":
            from ..models3.model import M3IEModelConf
            self.mconf = M3IEModelConf()
        elif model_type == "m3a":
            from ..models3.modelA import M3AIEModelConf
            self.mconf = M3AIEModelConf()
        elif model_type == "m3r":
            from ..models3.modelR import M3RIEModelConf
            self.mconf = M3RIEModelConf()
        else:
            zfatal(f"Unknown model type: {model_type}, please provide correct type!")
        # =====
        #
        if args is not None:
            self.update_from_args(args)
            self.validate()

# get model
def build_model(model_type, conf, vpack):
    mconf = conf.mconf
    model = None
    if model_type == "simple":
        from ..models2.model import MySimpleIEModel
        model = MySimpleIEModel(mconf, vpack)
    elif model_type == "m3":
        from ..models3.model import M3IEModel
        model = M3IEModel(mconf, vpack)
    elif model_type == "m3a":
        from ..models3.modelA import M3AIEModel
        model = M3AIEModel(mconf, vpack)
    elif model_type == "m3r":
        from ..models3.modelR import M3RIEModel
        model = M3RIEModel(mconf, vpack)
    else:
        zfatal(f"Unknown model type: {model_type}, please provide correct type!")
    return model

# get berter
def get_berter(bconf: BerterConf):
    zlog("Get bert model ...")
    bmodel = Berter(None, bconf)
    return bmodel

# the start of everything
def init_everything(args, ConfType=None):
    # search for basic confs
    all_argv, basic_argv = Conf.search_args(args, ["model_type", "conf_output", "log_file", "msp_seed"],
                                            [str, str, str, int], [None, None, Logger.MAGIC_CODE, None])
    # for basic argvs
    model_type = basic_argv["model_type"]
    conf_output = basic_argv["conf_output"]
    log_file = basic_argv["log_file"]
    msp_seed = basic_argv["msp_seed"]
    if conf_output:
        with zopen(conf_output, "w") as fd:
            for k,v in all_argv.items():
                # todo(note): do not save this one
                if k != "conf_output":
                    fd.write(f"{k}:{v}\n")
    utils.init(log_file, msp_seed)
    # real init of the conf
    if model_type is None:
        utils.zlog("Using the default model type = simple!")
        model_type = "simple"
    if ConfType is None:
        conf = OverallConf(model_type, args)
    else:
        conf = ConfType(model_type, args)
    nn.init(conf.niconf)
    return conf
