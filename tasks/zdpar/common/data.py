#

# data for dependency parsing

import json
from typing import Iterable
import numpy as np

from msp.utils import zfatal, zopen, zwarn
from msp.data import Instance, FileOrFdStreamer, VocabHelper, MultiHelper
from msp.zext.dpar import ConlluReader, write_conllu, ConlluParse
from msp.zext.seq_data import InstanceHelper, SeqFactor, InputCharFactor

#
def get_aug_words(ws, aug_code):
    return [MultiHelper.aug_word_with_prefix(w, aug_code) for w in ws]

class ParseInstance(Instance):
    #
    ROOT_SYMBOL = VocabHelper.convert_special_pattern("r")

    def __init__(self, words, poses=None, heads=None, labels=None, code=""):
        super().__init__()
        # todo(0): always include special ROOT symbol
        _tmp_root_list = [ParseInstance.ROOT_SYMBOL]
        self.code = code
        if code:
            aug_words = get_aug_words(words, code)
        else:
            aug_words = words
        self.words = SeqFactor(_tmp_root_list + aug_words)
        self.chars = InputCharFactor([""] + words)      # empty pad chars
        #
        if poses is not None:
            poses = _tmp_root_list + poses
        if heads is not None:
            heads = [0] + heads
        if labels is not None:
            labels = _tmp_root_list + labels
        # todo(warn): calculate the unproj situations: 0 proj edge, 1 unproj edge
        if heads is not None:
            unprojs = [0] + ConlluParse.calc_crossed(heads[1:])
        else:
            unprojs = None
        #
        self.poses = SeqFactor(poses)
        self.heads = SeqFactor(heads)
        self.labels = SeqFactor(labels)
        #
        self.children_mask_arr = None
        #
        self.unprojs = SeqFactor(unprojs)
        # predictions (optional prediction probs)
        self.pred_poses = SeqFactor(None)
        self.pred_pos_scores = SeqFactor(None)
        self.pred_heads = SeqFactor(None)
        self.pred_labels = SeqFactor(None)
        self.pred_par_scores = SeqFactor(None)
        # for real length
        self.length = InstanceHelper.check_equal_length([self.words, self.chars, self.poses, self.heads, self.labels]) - 1

    def __len__(self):
        return self.length

    # for h-local-loss
    def get_children_mask_arr(self, add_self_if_leaf=True):
        if self.children_mask_arr is None:
            # on need
            heads = self.heads.vals
            the_len = len(heads)
            masks = np.zeros([the_len, the_len], dtype=np.float32)
            # exclude root
            for m, h in enumerate(heads[1:], 1):
                masks[h, m] = 1.        # this one is [h,m]
            if add_self_if_leaf:
                for i in range(the_len):
                    if sum(masks[i]) == 0.:
                        masks[i, i] = 1.
            self.children_mask_arr = masks
        return self.children_mask_arr

    #
    def get_real_values_select(self, selections):
        ret = []
        for name in selections:
            zv = getattr(self, name)
            if zv.has_vals():
                ret.append(zv.vals[1:])
            else:
                ret.append(None)
        return ret

    def get_real_values_all(self):
        ret = {}
        for zn, zv in vars(self).items():
            if isinstance(zv, SeqFactor):
                if zv.has_vals():
                    ret[zn] = zv.vals[1:]
                else:
                    ret[zn] = None
        return ret

# ===== Data Reader
def get_data_reader(file_or_fd, input_format, aug_code, use_la0):
    if input_format == "conllu":
        return ParseConlluReader(file_or_fd, aug_code, use_la0=use_la0)
    elif input_format == "plain":
        return ParseTextReader(file_or_fd, aug_code)
    elif input_format == "json":
        return ParseJsonReader(file_or_fd, aug_code, use_la0=use_la0)
    else:
        zfatal("Unknown input_format %s, should select from {conllu,plain,json}" % input_format)

# read from conllu file
class ParseConlluReader(FileOrFdStreamer):
    def __init__(self, file_or_fd, aug_code, use_xpos=False, use_la0=False):
        super().__init__(file_or_fd)
        self.aug_code = aug_code
        self.reader = ConlluReader()
        #
        if use_xpos:
            self.pos_f = lambda t: t.xpos
        else:
            self.pos_f = lambda t: t.upos
        if use_la0:
            self.label_f = lambda t: t.label0
        else:
            self.label_f = lambda t: t.label

    def _next(self):
        parse = self.reader.read_one(self.fd)
        if parse is None:
            return None
        else:
            tokens = parse.get_tokens()
            one = ParseInstance([t.word for t in tokens], [self.pos_f(t) for t in tokens],
                                [t.head for t in tokens], [self.label_f(t) for t in tokens], code=self.aug_code)
            one.init_idx = self.count()
            return one

# read raw text from plain file, no annotations (mainly for prediction)
class ParseTextReader(FileOrFdStreamer):
    def __init__(self, file_or_fd, aug_code, skip_empty_line=False, sep=None):
        super().__init__(file_or_fd)
        self.aug_code = aug_code
        self.skip_empty_line = skip_empty_line
        self.sep = sep

    def _next(self):
        while True:
            line = self.fd.readline()
            if len(line)==0:
                return None
            line = line.strip()
            if self.skip_empty_line and len(line)==0:
                continue
            break
        words = line.split(self.sep)
        one = ParseInstance(words, code=self.aug_code)
        return one

# read json, one line per instance
class ParseJsonReader(FileOrFdStreamer):
    def __init__(self, file_or_fd, aug_code, use_la0=False):
        super().__init__(file_or_fd)
        self.aug_code = aug_code
        self.use_la0 = use_la0

    def _next(self):
        line = self.fd.readline()
        if len(line)==0:
            return None
        vv = json.loads(line)
        # todo(+1): more flexible fields
        the_labels = vv.get("label", None)
        if the_labels and self.use_la0:
            the_labels = [z.split(":")[0] for z in the_labels]
        one = ParseInstance(words=vv.get("word", None), poses=vv.get("pos", None),
                            heads=vv.get("head", None), labels=the_labels, code=self.aug_code)
        return one

# =====
# Data Writer

def get_data_writer(file_or_fd, output_format):
    if output_format == "conllu":
        return ParserConlluWriter(file_or_fd)
    elif output_format == "plain":
        zwarn("May be meaningless to write plain files for parses!")
        return ParserPlainWriter(file_or_fd)
    elif output_format == "json":
        return ParserJsonWriter(file_or_fd)
    else:
        zfatal("Unknown output_format %s, should select from {conllu,plain,json}" % output_format)

#
class ParserWriter:
    def __init__(self, file_or_fd):
        if isinstance(file_or_fd, str):
            self.fd = zopen(file_or_fd, "w")
        else:
            self.fd = file_or_fd

    def write_one(self, inst: ParseInstance):
        raise NotImplementedError()

    def write(self, insts):
        if isinstance(insts, Iterable):
            for one in insts:
                self.write_one(one)
        else:
            self.write_one(insts)

class ParserConlluWriter(ParserWriter):
    def __init__(self, file_or_fd):
        super().__init__(file_or_fd)
        self.obtain_names = ["words", "poses", "pred_heads", "pred_labels"]

    def write_one(self, inst: ParseInstance):
        values = inst.get_real_values_select(self.obtain_names)
        write_conllu(self.fd, *values)

class ParserPlainWriter(ParserWriter):
    def __init__(self, file_or_fd):
        super().__init__(file_or_fd)

    def write_one(self, inst: ParseInstance):
        self.fd.write(" ".join(inst.get_real_values_select(["words"])))
        self.fd.write("\n")

class ParserJsonWriter(ParserWriter):
    def __init__(self, file_or_fd):
        super().__init__(file_or_fd)

    def write_one(self, inst: ParseInstance):
        values = inst.get_real_values_all()
        self.fd.write(json.dumps(values))
