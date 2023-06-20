#

# a simpler dataset (consisting of instances)

from collections import OrderedDict
from mspx.utils import Conf, Configurable, default_json_serializer, zlog, Timer
from copy import deepcopy
from tqdm.auto import tqdm
from .inst import DataConf, DataInst, obtain_data
from .batch import BatcherConf, Batcher, InputBatch

# --
class DatasetConf(Conf):
    def __init__(self):
        self.wset = ""  # to be auto-filled!
        self.tasks = []  # tasks to do for this data
        self.info = {}  # extra data info
        # --
        # paths
        self.data = DataConf()
        self.output_path = ""
        self.batcher = BatcherConf()
        # --
        self.do_cache_insts = True  # store the insts!
        self.store_gold = False  # store for gold insts
        # self.iter_show_progress = False  # print progress when iter?

    def has_data(self):
        return bool(self.data.path)

class MyDataset(Configurable):
    def __init__(self, conf: DatasetConf, **kwargs):
        super().__init__(conf, **kwargs)
        conf: DatasetConf = self.conf
        # --
        self.wset = conf.wset
        self.info = conf.info.copy()
        # --
        # read & prepare insts
        self._gold_insts = None  # used for eval, deepcopy if cache_insts
        if conf.do_cache_insts:
            self.insts = list(self.yield_insts0())  # store raw data (can be Doc or Sent, read from file)
            if conf.store_gold:
                self._gold_insts = deepcopy(self.insts)
        else:
            self.insts = None
        # --
        # batchers and tasks
        self.my_batcher = Batcher(conf.batcher)
        self.cur_batchers = []  # first in last out?
        self._tasks = OrderedDict()
        for t in conf.tasks:
            t0, t1 = (t.split(".", 1) + [""])[:2]
            self._tasks[t0] = t1
        # --

    @staticmethod
    def make_fake_dataset(**kwargs):
        conf = DatasetConf.direct_conf(do_cache_insts=False, **kwargs)
        ret = MyDataset(conf)
        return ret

    def __repr__(self): return f"Dataset({self.wset})"
    def __len__(self): return len(self.insts) if self.insts else 0
    @property
    def tasks(self): return self._tasks
    @property
    def is_train(self): return 'train' in self.wset

    def yield_insts0(self):
        conf: DatasetConf = self.conf
        stream = obtain_data(conf.data)
        yield from stream

    def yield_insts(self, processors=()):
        conf: DatasetConf = self.conf
        if self.insts is not None:
            stream = self.insts  # already preloaded at creation time!
        else:
            stream = self.yield_insts0()
        for inst in stream:
            for p in processors:
                inst = p(inst, self)
            yield inst

    def set_insts(self, insts):
        self.insts = insts  # note: simply set it!

    # writing
    def write_insts(self, output_path: str = None, insts=None):
        conf: DatasetConf = self.conf
        if insts is None:
            insts = self.insts
        if not output_path:
            output_path = conf.output_path
        if output_path and insts is not None:
            default_json_serializer.save_iter(insts, output_path)
            zlog(f"Write data {self} to {output_path}: {len(insts)} instances.")
        else:
            zlog(f"No writing since NOT ({output_path} and {insts is not None})")
        # --

    def yield_batches(self, loop=False, processors=(), external_batcher=None, external_stream=None, quiet=True):
        conf: DatasetConf = self.conf
        batcher = self.my_batcher if external_batcher is None else external_batcher
        self.cur_batchers.append(batcher)
        _ii = 0
        has_es = (external_stream is not None)
        while True:
            with Timer(info=f"Dataset[{self}]: Epoch[{_ii}]", quite=quiet):
                stream = external_stream if has_es else self.yield_insts(processors=processors)
                for items in batcher.batch_insts(stream, no_cache=has_es):
                    yield InputBatch(items, self)
            _ii += 1
            if not loop:
                break
        assert self.cur_batchers[-1] is batcher
        self.cur_batchers.pop(-1)

# --
# b mspx/znew/prompt/data/dataset:
