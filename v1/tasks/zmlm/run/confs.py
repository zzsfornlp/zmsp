#

# common configurations

import json
from msp import utils, nn
from msp.nn import NIConf
from msp.utils import Conf, Logger, zopen, zfatal, zwarn

from .vocab import MLMVocabPackageConf

# todo(note): when testing, unless in the default setting, need to specify: "dict_dir" and "model_load_name"
class DConf(Conf):
    def __init__(self):
        # vocab
        self.vconf = MLMVocabPackageConf()
        # data paths
        self.train = ""
        self.dev = ""
        self.test = ""
        self.cache_data = True          # turn off if large data
        self.to_cache_shuffle = False
        self.dict_dir = "./"
        # cuttings for training (simply for convenience without especially preparing data...)
        self.cut_train = -1  # <0 means no cut!
        self.cut_dev = -1
        # save name for trainer not here!!
        self.model_load_name = "zmodel.best"         # load name
        self.output_file = "zout"
        # format (conllu, plain, json)
        self.input_format = "conllu"
        self.dev_input_format = ""  # special mixing case
        self.output_format = "json"
        # special processing
        self.lower_case = False
        self.norm_digit = False  # norm digits to 0
        # todo(note): deprecated; do not mix things in this way. Use pos field and word-thresh to control things!
        self.backoff_pos_idx = -1  # <0 means nope, when idx>=this, then backoff to pos ones

    def do_validate(self):
        if len(self.dev_input_format)==0:
            self.dev_input_format = self.input_format

# the overall conf
class OverallConf(Conf):
    def __init__(self, args):
        # top-levels
        self.conf_output = ""
        self.log_file = Logger.MAGIC_CODE
        self.msp_seed = 9341
        #
        self.niconf = NIConf()      # nn-init-conf
        self.dconf = DConf()        # data-conf
        # model
        from ..model.mtl import MtlMlmModelConf
        self.mconf = MtlMlmModelConf()
        # =====
        #
        self.update_from_args(args)
        self.validate()

# get model
def build_model(conf, vpack):
    mconf = conf.mconf
    from ..model.mtl import MtlMlmModel
    model = MtlMlmModel(mconf, vpack)
    return model

# the start of everything
def init_everything(args, ConfType=None, quite=False):
    # search for basic confs
    all_argv, basic_argv = Conf.search_args(args, ["conf_output", "log_file", "msp_seed"],
                                            [str, str, int], [None, Logger.MAGIC_CODE, None])
    # for basic argvs
    conf_output = basic_argv["conf_output"]
    log_file = basic_argv["log_file"]
    msp_seed = basic_argv["msp_seed"]
    if conf_output:
        with zopen(conf_output, "w") as fd:
            for k,v in all_argv.items():
                # todo(note): do not save this one
                if k != "conf_output":
                    fd.write(f"{k}:{v}\n")
    utils.init(log_file, msp_seed, quite=quite)
    # real init of the conf
    if ConfType is None:
        conf = OverallConf(args)
    else:
        conf = ConfType(args)
    nn.init(conf.niconf)
    return conf
