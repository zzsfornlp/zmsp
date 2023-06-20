#

# represent one dataset (a data pipeline)

__all__ = [
    "ZDatasetConf", "ZDataset",
]

import re
import os
from typing import Union, List
from collections import OrderedDict
from mspx.data.rw import ReaderGetterConf, WriterGetterConf
from mspx.utils import Conf, Configurable, zglob1, zglobs, ConfEntryCallback, zlog, Timer
from .data_prep import *
from .data_batch import *

# --
class ZDatasetConf(Conf):
    def __init__(self):
        self.name = ""  # to be auto-filled!
        self.tasks = []  # tasks to do for this data
        self.lang = ""  # lang?
        self.info = {}  # extra data info
        # io
        self.R = ReaderGetterConf()
        self.W = WriterGetterConf()
        # - paths (further we have default ones for "*_gold", "*_output" if not provided in extras)
        self.input_dir = ""  # glob1
        self.input_file = []  # or str
        self.gold_file = ""  # by default the same as input_file
        self.output_dir = ""  # glob1
        self.output_file = ""
        self.output_prefix = "_zout"  # default output prefix, full will be "{this}.{wset}.json"
        # pipeline
        self.preps0 = ConfEntryCallback(lambda s: self.callback_entries(s, T=DataPreperConf))  # preps before inits
        self.do_cache_insts = True  # store the insts!
        self.preps1 = ConfEntryCallback(lambda s: self.callback_entries(s, T=DataPreperConf))  # preps after insts
        self.batcher = ConfEntryCallback((lambda s: self.callback_entry(s, T=DataBatcherConf)), default_s='plain')
        # special
        self.test_with_loss = 0  # when testing, if >0 calculating loss instead of actual decoding
        self.test_streaming = 0  # when testing, do streaming processing (batch-size) rather than loading all

    @staticmethod
    def compose_input_paths(path: Union[str, List[str]], dir=''):
        _paths = [path] if isinstance(path, str) else path
        if dir:
            _dir = zglob1(dir)
            _paths = [os.path.join(_dir, z) for z in _paths]
        rets = zglobs(_paths, err_act='err')
        return rets

    def get_input_paths(self):
        return ZDatasetConf.compose_input_paths(self.input_file, self.input_dir)

    def get_gold_paths(self):
        _gold_path = self.gold_file
        _ret = self.get_input_paths()
        if _gold_path:
            if _gold_path.startswith(":s/"):  # special sub!
                _, a, b = self.gold_file.strip().split("/")
                _ret = [z.replace(a,b) for z in _ret]
            else:
                _ret = ZDatasetConf.compose_input_paths(_gold_path, self.input_dir)
        return _ret

    def get_output_path(self, default_sig=None):  # note: only one output path
        # first get output's file name (can be an auto name)
        if default_sig is None:
            default_sig = self.name  # use the name!
        _path = self.output_file if self.output_file else f"{self.output_prefix}.{default_sig}.json"
        if self.output_dir:
            _dir = zglob1(self.output_dir)
            _path = os.path.join(_dir, _path)
        return _path

# --
class ZDataset(Configurable):
    GUESSED_LANG_SET = {"en", "zh", "ar", "fi", "fr", "de", "it", "pt", "es", "ja", "cs"}

    def __init__(self, conf: ZDatasetConf, preset_insts=None, **kwargs):
        super().__init__(conf, **kwargs)
        conf: ZDatasetConf = self.conf
        if conf.test_streaming > 0:
            conf.do_cache_insts = False  # no caching!
        # --
        self.name = conf.name
        self.info = conf.info.copy()
        self.lang = conf.lang if conf.lang else self.guess_lang(conf.input_file)  # guess lang from file name!
        # prepers
        self.preps0 = [] if conf.preps0 is None else [getattr(conf, z[0]).make_node() for z in conf.preps0]
        self.preps1 = [] if conf.preps1 is None else [getattr(conf, z[0]).make_node() for z in conf.preps1]
        # read & prepare insts
        self._gold_insts = None  # used for eval, lazy loaded!
        if preset_insts is not None:  # allow preset!
            self._gold_insts = []  # note: no using of this!!
            self.insts = preset_insts
        else:
            if conf.do_cache_insts:
                self.insts = list(self.yield_insts0())  # store raw data (can be Doc or Sent, read from file)
            else:
                self.insts = None
        zlog(f"Create {self}: => {len(self)} instances")
        self.my_batcher = conf.batcher.make_node()
        self.cur_batchers = []  # first in last out?
        # --
        self._tasks = OrderedDict()
        for t in conf.tasks:
            t0, t1 = (t.split(".", 1) + [""])[:2]
            self._tasks[t0] = t1
        # --

    @staticmethod
    def guess_lang(s: str):
        if isinstance(s, (list, tuple)):
            s = s[0] if len(s)>0 else ""
        s = s.split("/")[-1]
        for f in re.split(r"\W+", s):
            if f in ZDataset.GUESSED_LANG_SET:
                return f
        return "UNK"

    def __repr__(self): return f"Dataset({self.name})"
    def __len__(self): return len(self.insts) if self.insts else 0
    @property
    def tasks(self): return self._tasks

    # reader -> preps0 ->
    def yield_insts0(self, input_path=None):
        conf: ZDatasetConf = self.conf
        if input_path is None:
            input_path = conf.get_input_paths()
        reader = conf.R.get_reader(input_path=input_path)
        stream = reader
        for pp in self.preps0:
            stream = pp.prep_insts(stream)
        yield from stream

    def yield_insts(self):
        if self.insts is not None:
            yield from self.insts  # already preloaded at creation time!
        else:
            yield from self.yield_insts0()

    def set_insts(self, insts):
        self.insts = insts  # note: simply set it!

    @property
    def gold_insts(self):
        conf: ZDatasetConf = self.conf
        if self._gold_insts is None:
            assert conf.do_cache_insts, "For simplicity, do caching when we need gold instances!"
            hit_idxes = [z.read_idx for z in self.yield_insts()]
            reader = conf.R.get_reader(input_path=conf.get_gold_paths())
            stream = yield_insts_with_idxes(reader, hit_idxes)
            # note: currently no applying pps!
            # for pp in self.preps0:
            #     if not isinstance(pp, DataSampler):  # note: especially exclude sampler!
            #         stream = pp.prep_insts(stream)
            self._gold_insts = list(stream)
            assert len(hit_idxes) == len(self._gold_insts), \
                f"Length mismatch for inst({len(hit_idxes)}) vs gold_inst({len(self._gold_insts)})"
        return self._gold_insts

    def get_writer(self, output_path: str = None):
        conf: ZDatasetConf = self.conf
        if output_path is None:
            full_path = conf.get_output_path(default_sig=self.name)
        else:  # override by outside!!
            full_path = output_path
        return conf.W.get_writer(output_path=full_path)

    def write_insts(self, output_path: str = None, insts=None):
        if insts is None:
            insts = self.insts
        with self.get_writer(output_path) as writer:
            writer.write_insts(insts)
            zlog(f"Write data {self} with {writer}: {len(insts)} instances.")
        # --

    def yield_batches(self, loop=False, external_batcher=None, external_stream=None, quiet=False):
        conf: ZDatasetConf = self.conf
        batcher = self.my_batcher if external_batcher is None else external_batcher
        self.cur_batchers.append(batcher)
        _ii = 0
        has_es = (external_stream is not None)
        while True:
            with Timer(info=f"Dataset[{self}]: Epoch[{_ii}]", quite=quiet):
                stream = external_stream if has_es else self.yield_insts()
                for pp in self.preps1:
                    stream = pp.prep_insts(stream)
                for items in batcher.batch_insts(stream, no_cache=has_es):
                    yield InputBatch(items, self)
            _ii += 1
            if not loop:
                break
        assert self.cur_batchers[-1] is batcher
        self.cur_batchers.pop(-1)

# --
# b mspx/proc/run/data:
