#

# represent one dataset
# -- note: currently load them all in-mem
"""
# two versions of data:
1) static raw ones: Doc/Sent, stored here and used for build vocab and preprocess
2) runtime InputItem: msent ones, prepared and processed by Batcher and ZTask at run time
"""

__all__ = ["ZDatasetConf", "ZDataset", "ZDataPreprocessor"]

import os
import re
from typing import List, Callable
from collections import OrderedDict
from msp2.utils import Conf, zlog, Random, Registrable
from msp2.data.inst import Sent
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.proc import SVConf
from .run import *

# --
class ZDataPreprocessor(Registrable):
    def __call__(self, insts):
        return insts

class ZDatasetConf(Conf):
    def __init__(self):
        # ==
        # top/group-level info (used at outside, put here for convenience)
        self.group_name = ""
        self.group_files = []  # List: # -> "input_file" or Dict: sub_name -> "input_file"
        self.group_tasks = []  # tasks to perform! note: allow sub-name!
        self.group_info = {}  # extra info?
        self.group_joint = False  # join all these into one dataset?
        # train (train)
        self.group_sample_rate = SVConf().direct_update(val=1., which_idx="cidx", mode="none")  # outside_sample by rate
        self.group_sample_alpha = 0.  # inside_sample by len(inst)**alpha
        # eval (test/dev)
        self.group_eval_weight = 1.  # weight for final eval
        # ==
        # (static) io
        self.R = ReaderGetterConf()
        self.W = WriterGetterConf()
        # - paths (further we have default ones for "*_gold", "*_output" if not provided in extras)
        self.input_dir = "./"  # if needed
        self.input_file = ""
        self.gold_file = ""  # by default the same as input_file
        self.output_dir = "./"  # if needed
        self.output_file = ""
        self.output_prefix = "_zout"  # default output prefix, full will be "{this}.{wset}.json"
        # - special
        self.preprocessors = []  # need to slightly modify the data?
        # self.approx_prev_next = False  # approx. setting of prev & next when loading, note: deprecated
        self.presample = 1.0  # (>1=N,<1=Rate) random sample how much at the very beginning, as pre-processing for convenience!
        self.presample_shuffle = False  # whether shuffle in presample?
        self.presample_reverse = False  # from back to start (for convenience)
        # ==
        # runtime
        self.convert_conf = ZIConverterConf()
        self.batch_conf = ZIBatcherConf()

    def get_input_path(self):
        return os.path.join(self.input_dir, self.input_file) if self.input_dir else self.input_file

    def get_gold_path(self):
        fname = self.gold_file if self.gold_file else self.input_file
        return os.path.join(self.input_dir, fname) if self.input_dir else fname

    def get_output_path(self, default_sig: str):
        # first get output's file name (can be an auto name)
        fname = self.output_file if self.output_file else f"{self.output_prefix}.{default_sig}.json"
        return os.path.join(self.output_dir, fname) if self.output_dir else fname

    @classmethod
    def _get_type_hints(cls):
        return {"input_dir": "zglob1", "output_dir": "zglob1"}  # easier finding!

class ZDataset:
    # --
    # note: to be extended!
    _GUESSED_LANG_SET = {"en", "zh", "ar", "fi", "fr", "de", "it", "pt", "es"}
    # --

    def __init__(self, conf: ZDatasetConf, name: str, wset: str, _no_load=False):
        self.conf = conf
        self.name = name
        self.wset = wset
        self.lang = self._guess_lang(conf.input_file)  # guess lang from file name!
        self.info = conf.group_info.copy()  # extra info
        # note: data processings are all greedy!!
        # precessors
        self._preprocessors = [ZDataPreprocessor.try_load_and_lookup(z).T for z in conf.preprocessors]
        # read & prepare insts
        self._gold_insts = None  # used for eval, lazy loaded!
        self._presample_indexes = None  # used for loading gold_insts
        if _no_load:
            self.insts = []
            self.items = []
        else:
            self.insts = self._load_insts()  # store raw data (can be Doc or Sent, read from file)
            self.items = self._prepare_items()  # input_item
        # --
        # tasks needed to perform: as an ordered collection
        self._tasks = OrderedDict()
        self._dec_tasks = OrderedDict()
        for t in ["enc"] + conf.group_tasks:  # note: first everyone has "enc"!!
            t0, t1 = (t.split(".", 1) + [""])[:2]
            self._tasks[t0] = t1
        for t in conf.group_tasks:
            t0, t1 = (t.split(".", 1) + [""])[:2]
            self._dec_tasks[t0] = t1
        # --

    def _guess_lang(self, s: str):
        s = s.split("/")[-1]
        for f in re.split(r"\W+", s):
            if f in ZDataset._GUESSED_LANG_SET:
                return f
        return "UNK"

    def __repr__(self):
        return f"Dataset({self.name}:{self.wset})"

    def _load_insts(self):
        conf = self.conf
        full_path = conf.get_input_path()
        insts = list(conf.R.get_reader(input_path=full_path))  # note: read them all!!
        for pp in self._preprocessors:
            insts = pp(insts)
        # --
        original_len = len(insts)
        if conf.presample != 1.:
            insts, idxes = ZDataset.do_presample(insts, conf.presample, conf.presample_shuffle, conf.presample_reverse)
            self._presample_indexes = idxes
        zlog(f"Read data {self.name}:{self.wset} from {full_path}(s={conf.presample}): {original_len}=>{len(insts)} instances.")
        return insts

    def _prepare_items(self):
        converter = ZIConverter(self.conf.convert_conf)
        items = list(converter.convert(self.insts))  # simply convert them all
        return items

    # --
    def set_insts(self, insts):
        self.insts = insts
        self.items = self._prepare_items()  # further refresh

    def set_items(self, items):
        self.items = items

    @property
    def gold_insts(self):
        if self._gold_insts is None:  # load when needed!!
            conf = self.conf
            _gold_insts = list(conf.R.get_reader(input_path=conf.get_gold_path()))
            for pp in self._preprocessors:
                _gold_insts = pp(_gold_insts)
            if self._presample_indexes is not None:
                _gold_insts = [_gold_insts[z] for z in self._presample_indexes]
            self._gold_insts = _gold_insts
        return self._gold_insts

    @property
    def tasks(self):
        return self._tasks

    @property
    def dec_tasks(self):  # decoding (true) tasks
        return self._dec_tasks

    def write_insts(self, output_path: str = None):
        conf = self.conf
        # --
        # get output path name
        if output_path is None:
            sig = ''.join([(c if c!='/' else '_') for c in f"{self.name}_{self.wset}"])
            full_path = conf.get_output_path(default_sig=sig)
        else:  # override by outside!!
            full_path = output_path
        # --
        with conf.W.get_writer(output_path=full_path) as writer:
            writer.write_insts(self.insts)
            zlog(f"Write data {self.name}:{self.wset} to {full_path}: {len(self.insts)} instances.")
        # --

    def yield_batches(self, external_batcher: ZIBatcher=None, loop=None, filter_f=None):
        if loop is None:  # auto decide it by wset name
            loop = (self.wset == "train")
        batcher = ZIBatcher(self.conf.batch_conf) if external_batcher is None else external_batcher
        for items in batcher.yield_batches(self.items, loop=loop, filter_f=filter_f):
            yield InputBatch(items, self)

    @staticmethod
    def do_presample(insts: List, s: float, shuffle: bool, reverse: bool):
        assert s>0
        if s < 1.:
            s = len(insts)*s
        s = int(s+0.99999)
        # --
        ret_idxes = list(range(len(insts)))
        if reverse:
            ret_idxes = list(reversed(ret_idxes))
        if shuffle:
            _gen = Random.get_generator('presample')
            _gen.shuffle(ret_idxes)
        ret_idxes = ret_idxes[:s]
        return [insts[z] for z in ret_idxes], ret_idxes

    @staticmethod
    def join_datasets(conf: ZDatasetConf, name: str, wset: str, datasets: List['ZDataset']):
        ret = ZDataset(conf, name, wset, _no_load=True)  # make an empty one
        # note: simply directly set!
        insts = sum([d.insts for d in datasets], [])
        items = sum([d.items for d in datasets], [])
        if wset == "train":  # note: can directly use themselves as gold!
            gold_insts = insts
        else:
            gold_insts = sum([d.gold_insts for d in datasets], [])
        zlog(f"Build jonit dataset {ret} with D={len(datasets)},inst={len(insts)},item={len(items)}")
        ret.insts = insts
        ret.items = items
        ret._gold_insts = gold_insts
        return ret
