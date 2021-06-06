#

# Extractors for a sequence of spans + labels
# -- since Direct and Sequential ones are quite different, we make them into two classes

from .base import *
from .direct import *
from .seq import *
from .soft import *
from .anchor import *

# ==
class ExtractorGetter:
    @staticmethod
    def make_conf_entry(ftag: str):  # make default entries
        from msp2.utils import ConfEntryChoices
        direct_conf = DirectExtractorConf()
        seq_conf = SeqExtractorConf()
        soft_conf = SoftExtractorConf()
        anchor_conf = AnchorExtractorConf()
        all_confs = [direct_conf, seq_conf, soft_conf, anchor_conf]
        for cc in all_confs:
            cc.ftag = ftag
        if ftag == "evt":  # simply use shead-mode for evt!!
            for cc in all_confs:
                cc.core_span_mode = "shead"
                cc.lab_conf.labeler_conf.e_dim = 512
                # cc.lab_conf.labeler_conf.sconf.pls_conf.hid_dim = 300
                # cc.lab_conf.labeler_conf.sconf.pls_conf.hid_nlayer = 1  # give one layer of hidden
                cc.lab_conf.labeler_conf.e_tie_weights = True  # tie weights for pred&lookup!
            direct_conf.span_conf.max_width = 1
        elif ftag == "ef" or ftag == "arg":
            pass
        else:
            raise NotImplementedError()
        # special for pair-scorer
        if ftag == "arg":
            for cc in all_confs:  # avoid large biaffine-tensor
                cc.lab_conf.labeler_conf.sconf.pas_conf.direct_update(use_biaffine=False, use_ff2=True)
        # --
        return ConfEntryChoices(
            {"direct": direct_conf, "seq": seq_conf, "soft": soft_conf, "anchor": anchor_conf}, 'direct')

    @staticmethod
    def make_extractor(conf: BaseExtractorConf, vocab, **kwargs):
        if isinstance(conf, DirectExtractorConf):
            return DirectExtractor(conf, vocab, **kwargs)
        elif isinstance(conf, SeqExtractorConf):
            return SeqExtractor(conf, vocab, **kwargs)
        elif isinstance(conf, SoftExtractorConf):
            return SoftExtractor(conf, vocab, **kwargs)
        elif isinstance(conf, AnchorExtractorConf):
            return AnchorExtractor(conf, vocab, **kwargs)
        else:
            raise NotImplementedError()
