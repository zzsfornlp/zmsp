#

# data for dependency parsing

import json
from typing import Iterable, List, Set, Dict
import numpy as np
import pickle

from msp.utils import zfatal, zopen, zwarn, Random, Helper, MathHelper, zcheck
from msp.data import Instance, FileOrFdStreamer, VocabHelper, MultiHelper, AdapterStreamer, MultiJoinStreamer
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
        self.poses = SeqFactor(poses)
        self.heads = SeqFactor(heads)
        self.labels = SeqFactor(labels)         # todo(warn): for td, no processing, directly use 0 as padding!!
        # =====
        # for top-down parsing, but deprecated now
        self.children_mask_arr: np.ndarray = None       # [N, N]
        # self.children_se: List[Set] = None
        self.children_list: List[List] = None           # list of children
        self.descendant_list: List[List] = None         # list of all descendants
        #
        self.free_dist_alpha: float = None
        # =====
        # other helpful info (calculate in need)
        self._unprojs = None
        self._sibs = None
        self._gps = None
        # all children should be aranged l2r
        self._children_left = None
        self._children_right = None
        self._children_all = None
        # predictions (optional prediction probs)
        self.pred_poses = SeqFactor(None)
        self.pred_pos_scores = SeqFactor(None)
        self.pred_heads = SeqFactor(None)
        self.pred_labels = SeqFactor(None)
        self.pred_par_scores = SeqFactor(None)
        self.pred_miscs = SeqFactor(None)
        # for real length
        self.length = InstanceHelper.check_equal_length([self.words, self.chars, self.poses, self.heads, self.labels]) - 1
        # extra inputs, for example, those from mbert
        self.extra_features = {"aux_repr": None}
        # extra preds info
        self.extra_pred_misc = {}

    def __len__(self):
        return self.length

    # =====
    # helpful functions
    @staticmethod
    def get_children(heads):
        cur_len = len(heads)
        children_left, children_right = [[] for _ in range(cur_len)], [[] for _ in range(cur_len)]
        for i in range(1, cur_len):
            h = heads[i]
            if i < h:
                children_left[h].append(i)
            else:
                children_right[h].append(i)
        return children_left, children_right

    @staticmethod
    def get_sibs(children_left, children_right):
        # get sibling list: sided nearest sibling (self if single)
        sibs = [-1] * len(children_left)
        for vs in children_left:
            if len(vs) > 0:
                prev_s = vs[-1]
                for cur_m in reversed(vs):
                    sibs[cur_m] = prev_s
                    prev_s = cur_m
        for vs in children_right:
            if len(vs) > 0:
                prev_s = vs[0]
                for cur_m in vs:
                    sibs[cur_m] = prev_s
                    prev_s = cur_m
        # todo(+2): how about 0? currently set to 0
        sibs[0] = 0
        return sibs

    @staticmethod
    def get_gps(heads):
        # get grandparent list
        # todo(+2): how about 0? currently will be 0
        return [heads[x] for x in heads]

    # =====
    # other properties
    @property
    def unprojs(self):
        # todo(warn): calculate the unproj situations: 0 proj edge, 1 unproj edge
        if self._unprojs is None:
            self._unprojs = [0] + ConlluParse.calc_crossed(self.heads.vals[1:])
        return self._unprojs

    @property
    def sibs(self):
        if self._sibs is None:
            self._sibs = ParseInstance.get_sibs(self.children_left, self.children_right)
        return self._sibs

    @property
    def gps(self):
        if self._gps is None:
            heads = self.heads.vals
            self._gps = ParseInstance.get_gps(heads)
        return self._gps

    @property
    def children_left(self):
        if self._children_left is None:
            heads = self.heads.vals
            self._children_left, self._children_right = ParseInstance.get_children(heads)
        return self._children_left

    @property
    def children_right(self):
        if self._children_right is None:
            heads = self.heads.vals
            self._children_left, self._children_right = ParseInstance.get_children(heads)
        return self._children_right

    @property
    def children_all(self):
        if self._children_all is None:
            self._children_all = [a+b for a,b in zip(self.children_left, self.children_right)]
        return self._children_all

    # =====
    # special processing for training
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

    # set once when reading!
    def set_children_info(self, oracle_strategy, label_ranking_dict:Dict=None, free_dist_alpha:float=0.):
        heads = self.heads.vals
        the_len = len(heads)
        # self.children_set = [set() for _ in range(the_len)]
        self.children_list = [[] for _ in range(the_len)]
        tmp_descendant_list = [None for _ in range(the_len)]
        # exclude root
        for m, h in enumerate(heads[1:], 1):
            # self.children_set[h].add(m)
            self.children_list[h].append(m)     # l2r order
        # re-arrange list order (left -> right)
        if oracle_strategy == "i2o":
            for h in range(the_len):
                self.children_list[h].sort(key=lambda x: -x if x<h else x)
        elif oracle_strategy == "label":
            # todo(warn): only use first level!
            level0_labels = [z.split(":")[0] for z in self.labels.vals]
            for h in range(the_len):
                self.children_list[h].sort(key=lambda x: label_ranking_dict[level0_labels[x]])
        elif oracle_strategy == "n2f":
            self.shuffle_children_n2f()
        elif oracle_strategy == "free":
            self.free_dist_alpha = free_dist_alpha
            self.shuffle_children_free()
        else:
            assert oracle_strategy == "l2r"
            pass
        # todo(+N): does the order of descendant list matter?
        # todo(+N): depth-first or breadth-first? (currently select the latter)
        # recursively get descendant list: do this
        # =====
        def _recursive_add(cur_n):
            cur_children = self.children_list[cur_n]            # List[int]
            for i in cur_children:
                _recursive_add(i)
            new_dlist = [cur_children]
            cur_layer = 0
            while True:
                another_layer = Helper.join_list(tmp_descendant_list[i][cur_layer]
                                                 if cur_layer<len(tmp_descendant_list[i]) else [] for i in cur_children)
                if len(another_layer) == 0:
                    break
                new_dlist.append(another_layer)
                cur_layer += 1
            tmp_descendant_list[cur_n] = new_dlist
        # =====
        _recursive_add(0)
        self.descendant_list = [Helper.join_list(tmp_descendant_list[i]) for i in range(the_len)]

    # =====
    # todo(warn): does not shuffle descendant list since this can disturb the depth-order

    # shuffle once before each running for free mode
    def shuffle_children_free(self):
        alpha = self.free_dist_alpha
        if alpha <= 0.:
            for one_list in self.children_list:
                if len(one_list)>1:
                    Random.shuffle(one_list)
        else:
            for i, one_list in enumerate(self.children_list):
                if len(one_list)>1:
                    values = [abs(i-z)*alpha for z in one_list]
                    # TODO(+N): is it correct to use Gumble for ranking
                    logprobs = np.log(MathHelper.softmax(values))
                    G = np.random.random_sample(len(logprobs))
                    ranking_values = np.log(-np.log(G)) - logprobs
                    self.children_list[i] = [one_list[z] for z in np.argsort(ranking_values)]

    INST_RAND = Random.stream(Random.random_sample)
    # shuffle for n2f mode
    def shuffle_children_n2f(self):
        rr = ParseInstance.INST_RAND
        for i, one_list in enumerate(self.children_list):
            if len(one_list) > 1:
                # todo(warn): use small random to break tie
                values = [abs(i-z)+next(rr) for z in one_list]
                self.children_list[i] = [one_list[z] for z in np.argsort(values)]
    # =====
    # todo(warn): exclude artificial root node

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
def get_data_reader(file_or_fd, input_format, aug_code, use_la0, aux_repr_file=None, aux_score_file=None, cut=None):
    cut = -1 if (cut is None or len(cut)==0) else int(cut)
    if input_format == "conllu":
        r = ParseConlluReader(file_or_fd, aug_code, use_la0=use_la0, cut=cut)
    elif input_format == "plain":
        r = ParseTextReader(file_or_fd, aug_code, cut=cut)
    elif input_format == "json":
        r = ParseJsonReader(file_or_fd, aug_code, use_la0=use_la0, cut=cut)
    else:
        zfatal("Unknown input_format %s, should select from {conllu,plain,json}" % input_format)
        r = None
    if aux_repr_file is not None and len(aux_repr_file)>0:
        r = AuxDataReader(r, aux_repr_file, "aux_repr")
    if aux_score_file is not None and len(aux_score_file)>0:
        r = AuxDataReader(r, aux_score_file, "aux_score")
    return r

# pre-computed auxiliary data
class AuxDataReader(AdapterStreamer):
    def __init__(self, base_streamer, aux_repr_file, aux_name):
        super().__init__(base_streamer)
        self.file = aux_repr_file
        self.fd = None
        self.aux_name = aux_name

    def __del__(self):
        if self.fd is not None:
            self.fd.close()

    def _restart(self):
        self.base_streamer_.restart()
        if isinstance(self.file, str):
            if self.fd is not None:
                self.fd.close()
            self.fd = zopen(self.file, mode='rb', encoding=None)
        else:
            zcheck(self.restart_times_ == 0, "Cannot restart a FdStreamer")

    def _next(self):
        one = self.base_streamer_.next()
        if self.base_streamer_.is_eos(one):
            return None
        res = pickle.load(self.fd)
        # todo(warn): specific checks
        if isinstance(res, (tuple, list)):
            assert len(res) == 2
            assert all(len(z)==len(one)+1 for z in res), "Unmatched length for the aux_score arr"
        else:
            assert len(res) == len(one)+1, "Unmatched length for the aux_repr arr"
        one.extra_features[self.aux_name] = res
        return one

# read from conllu file
class ParseConlluReader(FileOrFdStreamer):
    def __init__(self, file_or_fd, aug_code, use_xpos=False, use_la0=False, cut=-1):
        super().__init__(file_or_fd)
        self.aug_code = aug_code
        self.reader = ConlluReader()
        self.cut = cut
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
        if self.count_ == self.cut:
            return None
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
    def __init__(self, file_or_fd, aug_code, skip_empty_line=False, sep=None, cut=-1):
        super().__init__(file_or_fd)
        self.aug_code = aug_code
        self.skip_empty_line = skip_empty_line
        self.sep = sep
        self.cut = cut

    def _next(self):
        if self.count_ == self.cut:
            return None
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
        one.init_idx = self.count()
        return one

# read json, one line per instance
class ParseJsonReader(FileOrFdStreamer):
    def __init__(self, file_or_fd, aug_code, use_la0=False, cut=-1):
        super().__init__(file_or_fd)
        self.aug_code = aug_code
        self.use_la0 = use_la0
        self.cut = cut

    def _next(self):
        if self.count_ == self.cut:
            return None
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
        one.init_idx = self.count()
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
        self.obtain_names = ["words", "poses", "pred_heads", "pred_labels", "pred_miscs"]

    def write_one(self, inst: ParseInstance):
        values = inst.get_real_values_select(self.obtain_names)
        # record ROOT info in headlines
        headlines = []
        if inst.pred_miscs.has_vals():
            headlines.append(json.dumps({"ROOT-MISC": inst.pred_miscs.vals[0]}))
        if len(inst.extra_pred_misc) > 0:
            headlines.append(json.dumps(inst.extra_pred_misc))
        write_conllu(self.fd, *values, headlines=headlines)

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

# =====
# multi-source reader
# split the srings by ","
# todo(note): best if input data are balanced, otherwise mix and shuffle data should be better
def get_multisoure_data_reader(file, input_format, aug_code, use_la0, aux_repr_file="", aux_score_file="", cut="", sep=","):
    # -----
    def _get_and_pad(num, s, pad):
        results = s.split(sep) if len(s)>0 else []
        if len(results) < num:
            results.extend([pad] * (num - len(results)))
        return results
    # -----
    list_file = file.split(sep)
    num_files = len(list_file)
    list_aug_code = _get_and_pad(num_files, aug_code, "")
    list_aux_repr_file = _get_and_pad(num_files, aux_repr_file, "")
    list_aux_score_file = _get_and_pad(num_files, aux_score_file, "")
    list_cut = _get_and_pad(num_files, cut, "")
    # prepare all streams
    list_streams = [get_data_reader(f, input_format, ac, use_la0, arf, asf, cc)
                    for f, ac, arf, asf, cc in zip(list_file, list_aug_code, list_aux_repr_file, list_aux_score_file, list_cut)]
    # join them
    return MultiJoinStreamer(list_streams)
