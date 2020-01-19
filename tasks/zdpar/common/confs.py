#

# common configurations

import json
from msp import utils, nn
from msp.nn import NIConf
from msp.utils import Conf, Logger, zopen, zfatal, zwarn

# todo(note): when testing, unless in the default setting, need to specify: "dict_dir" and "model_load_name"
class DConf(Conf):
    def __init__(self):
        # whether allow multi-source mode
        self.multi_source = False  # split file names for inputs (use multi_reader)
        # data paths
        self.train = ""
        self.dev = ""
        self.test = ""
        self.cache_data = True          # turn off if large data
        self.to_cache_shuffle = False
        self.dict_dir = "./"
        # cuttings for training (simply for convenience without especially preparing data...)
        self.cut_train = ""
        self.cut_dev = ""
        # extra aux inputs, must be aligned with the corresponded data
        self.aux_repr_train = ""
        self.aux_repr_dev = ""
        self.aux_repr_test = ""
        self.aux_score_train = ""
        self.aux_score_dev = ""
        self.aux_score_test = ""
        # save name for trainer not here!!
        self.model_load_name = "zmodel.best"         # load name
        self.output_file = "zout"
        # format (conllu, plain, json)
        self.input_format = "conllu"
        self.output_format = "conllu"
        # pretrain
        self.pretrain_file = []
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
        self.code_pretrain = []
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
            from ..graph.parser import GraphParserConf
            self.pconf = GraphParserConf()
        elif partype == "td":
            from ..transition.topdown.parser import TdParserConf
            self.pconf = TdParserConf()
        elif partype == "ef":
            from ..ef.parser import EfParserConf
            self.pconf = EfParserConf()
        elif partype == "g1":
            from ..ef.parser import G1ParserConf
            self.pconf = G1ParserConf()
        elif partype == "g2":
            from ..ef.parser import G2ParserConf
            self.pconf = G2ParserConf()
        elif partype == "s2":
            from ..ef.parser import S2ParserConf
            self.pconf = S2ParserConf()
        else:
            zfatal(f"Unknown parser type: {partype}, please provide correct type with the option.")
        # =====
        #
        self.update_from_args(args)
        self.validate()

# get model
def build_model(partype, conf, vpack):
    pconf = conf.pconf
    parser = None
    if partype == "graph":
        # original first-order graph with various output constraints
        from ..graph.parser import GraphParser
        parser = GraphParser(pconf, vpack)
    elif partype == "td":
        # re-implementation of the top-down stack-pointer parser
        zwarn("Warning: Current implementation of td-mode is deprecated and outdated.")
        from ..transition.topdown.parser import TdParser
        parser = TdParser(pconf, vpack)
    elif partype == "ef":
        # generalized easy-first parser
        from ..ef.parser import EfParser
        parser = EfParser(pconf, vpack)
    elif partype == "g1":
        # first-order graph parser
        from ..ef.parser import G1Parser
        parser = G1Parser(pconf, vpack)
    elif partype == "g2":
        # higher-order graph parser
        from ..ef.parser import G2Parser
        parser = G2Parser(pconf, vpack)
    elif partype == "s2":
        # two-stage parser
        from ..ef.parser import S2Parser
        parser = S2Parser(pconf, vpack)
    else:
        zfatal("Unknown parser type: %s")
    return parser

# the start of everything
def init_everything(args, ConfType=None):
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
                # todo(note): do not save this one
                if k != "conf_output":
                    fd.write(f"{k}:{v}\n")
    utils.init(log_file, msp_seed)
    # real init of the conf
    if ConfType is None:
        conf = DepParserConf(parser_type, args)
    else:
        conf = ConfType(parser_type, args)
    nn.init(conf.niconf)
    return conf
