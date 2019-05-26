#

# common configurations

import json
from msp import utils, nn
from msp.nn import NIConf
from msp.utils import Conf, Logger, zopen, zfatal

from ..graph.parser import GraphParserConf, GraphParser

#
class DConf(Conf):
    def __init__(self):
        # data paths
        self.train = ""
        self.dev = ""
        self.test = ""
        self.cache_data = True          # turn off if large data
        self.dict_dir = "./"
        # save name for trainer not here!!
        self.model_load_name = "zmodel.best"         # load name
        self.output_file = "zout"
        # format (conllu, plain, json)
        self.input_format = "conllu"
        self.output_format = "conllu"
        # pretrain
        self.pretrain_file = ""
        self.init_from_pretrain = False
        self.pretrain_scale = 1.0
        self.pretrain_init_nohit = 1.0
        # thresholds for word
        self.word_rthres = 50000    # rank <= this
        self.word_sthres = 2        # the threshold of considering singleton treating （freq<=this)
        self.word_fthres = 1        # freq >= this
        # special processing
        self.lower_case = False
        self.norm_digit = False     # norm digits to 0
        self.use_label0 = False     # using only first-level label
        # =====
        # for multi-lingual processing (another option is to pre-processing suitable data)
        # language code (empty str for no effects)
        self.code_train = ""
        self.code_dev = ""
        self.code_test = ""
        self.code_pretrain = ""
        # testing mode extra embeddings
        self.test_extra_pretrain_files = []
        self.test_extra_pretrain_codes = []


# the overall conf
class DepParserConf(Conf):
    def __init__(self, partype, args):
        # top-levels
        self.partype = partype
        self.conf_output = ""
        self.log_file = Logger.MAGIC_CODE
        self.msp_seed = 9341
        #
        self.niconf = NIConf()      # nn-init-conf
        self.dconf = DConf()        # data-conf
        # parser-conf
        if partype == "graph":
            self.pconf = GraphParserConf()
        else:
            zfatal("Unknown parser type: %s, please provide correct type with the option `partype:{graph,}`")
        # =====
        #
        self.update_from_args(args)
        self.validate()

# get model
def build_model(partype, conf, vpack):
    pconf = conf.pconf
    parser = None
    if partype == "graph":
        parser = GraphParser(pconf, vpack)
    else:
        zfatal("Unknown parser type: %s")
    return parser

# the start of everything
def init_everything(args):
    # search for basic confs
    all_argv, basic_argv = Conf.search_args(args, ["partype", "conf_output", "log_file", "msp_seed"],
                                            [str, str, str, int], [None, None, Logger.MAGIC_CODE, None])
    # for basic argvs
    parser_type = basic_argv["partype"]
    conf_output = basic_argv["conf_output"]
    log_file = basic_argv["log_file"]
    msp_seed = basic_argv["msp_seed"]
    if conf_output:
        with zopen(conf_output, "w") as fd:
            for k,v in all_argv.items():
                fd.write(f"{k}:{v}\n")
    utils.init(log_file, msp_seed)
    # real init of the conf
    conf = DepParserConf(parser_type, args)
    nn.init(conf.niconf)
    return conf
