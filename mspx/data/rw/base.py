#

# basic reader/writer
# note: separate these interactions: inst <-> python-object(usually str) <-> files

__all__ = [
    "DataReader", "DataWriter",
    "FileStreamer", "FileDumper",
    "ReaderGetterConf", "WriterGetterConf", "get_reader", "get_writer",
]

from typing import Union, IO, Iterable, Callable, List, Type
import os
import sys
from mspx.utils import Conf, mkdir_p, zopen, zglobs, zwarn, zlog
from mspx.data.inst import DataInst, Doc, Sent
from mspx.data.stream import Streamer, FIterStreamer, MultiZipStreamer, \
    Dumper, WrapperDumper,  yield_lines, yield_files, yield_multilines
from .formats import *

# =====
# note: Reader and Writer are simply compositions of streamer/dumper and formator!

class DataReader(FIterStreamer):
    def __init__(self, base_streamer, formator: Union[DataFormator, str], yield_wrappers: List, max_count=-1, sel_idxes=None):
        super().__init__(self._yield)
        self.base_streamer = base_streamer
        self.formator = DataFormator.key2cls(formator)() if isinstance(formator, str) else formator
        self.yield_wrappers = yield_wrappers
        self.max_count = max_count
        self.sel_idxes = set([int(z) for z in sel_idxes]) if sel_idxes else None

    def __repr__(self):
        return f"Reader({self.base_streamer})"

    def _yield(self):
        gen = self.formator.yield_objs(iter(self.base_streamer))
        for yw in self.yield_wrappers:
            gen = yw(gen)
        for ii, inst in enumerate(gen):
            if ii == self.max_count:
                break
            if self.sel_idxes is not None and ii not in self.sel_idxes:
                continue
            inst.set_read_idx(self.count())  # note: mark reading-order idx!
            yield inst

class DataWriter(WrapperDumper):
    def __init__(self, base_dumper: Dumper, formator: Union[DataFormator, str]):
        super().__init__(base_dumper)
        self.formator = DataFormator.key2cls(formator)() if isinstance(formator, str) else formator

    def __repr__(self):
        return f"Writer({self._base_dumper})"

    def write_inst(self, inst: DataInst): self.dump_one(inst)
    def write_insts(self, insts: Iterable[DataInst]): self.dump_iter(insts)

    def dump_one(self, inst: DataInst):
        obj = self.formator.to_obj(inst)
        self._base_dumper.dump_one(obj)

# =====
# File related streamers

class FileStreamer(FIterStreamer):
    def __init__(self, files: Iterable[str], mode='line'):
        if isinstance(files, (str, IO)) or (files is sys.stdin):
            files = [files]
        self.files = list(files)
        self.mode = mode
        # --
        _f = getattr(self, f'_yield_{mode}s')
        super().__init__(_f)
        # --

    def __repr__(self):
        return f"FS{self.files}"

    def _yield_lines(self):
        for f in self.files:
            yield from yield_lines(f)

    def _yield_mlines(self):
        for f in self.files:
            yield from yield_multilines(f)

    def _yield_files(self):
        yield from yield_files(self.files)

# =====
# File related dumpers

# write one line at one time
class FileDumper(Dumper):
    def __init__(self, fd_or_path: Union[IO, str], end='\n'):
        super().__init__()
        self.fd_or_path = fd_or_path
        self.end = end
        if isinstance(fd_or_path, str):
            self.should_close = True
            self.fd = zopen(fd_or_path, 'w')
        else:
            self.should_close = False
            self.fd = fd_or_path

    def __repr__(self):
        return f"FD={self.fd_or_path}"

    def close(self):
        if self.should_close:  # only close if that is what we opened
            self.fd.close()

    def dump_one(self, obj: object):
        self.fd.write(f"{obj}{self.end}")

# =====
# Overall Facade Interface!

class ReaderGetterConf(Conf):
    def __init__(self):
        self.input_path = []  # allow multiple ones!
        self.input_format = "zjson"
        self.input_fmode = 'line'
        self.input_allow_std = True  # allow "" or "-" to mean stdin
        self.input_wrappers = []  # further wrappers
        self.input_split_ddp = False  # split for ddp?
        self.input_max_count = -1  # read first?
        self.sel_idxes = []
        # bitext options
        self.is_bitext = False  # bitext?
        self.bitext_src_suffix = ""
        self.bitext_trg_suffix = ""

    def has_path(self): return len(self.input_path)>0

    def __repr__(self):
        return f"Input={self.input_path}({self.input_format})"

    def get_reader(self, _clone=True, **kwargs):  # directly get a reader; need to clone if want to reuse
        return get_reader(self.copy() if _clone else self, **kwargs)

# some wrapping generators
def yw_s2d(base):
    last_doc_id = None
    to_merge = []
    for x in base:
        cur_doc_id = x.sent_single.info.get("doc_id")  # todo(+N): may have better methods
        if len(to_merge)>0 and (cur_doc_id is None or cur_doc_id != last_doc_id):
            doc = Doc.merge_docs(to_merge, new_doc_id=last_doc_id)
            yield doc
            to_merge = []
        # --
        to_merge.append(x)
        last_doc_id = cur_doc_id
    if len(to_merge) > 0:
        doc = Doc.merge_docs(to_merge, new_doc_id=last_doc_id)
        yield doc

def get_reader(conf: ReaderGetterConf = None, **kwargs):
    conf = ReaderGetterConf.direct_conf(conf, **kwargs)
    # --
    def _split_ddp(_x):
        from mspx.nn import BK
        if BK.use_ddp():
            _rank, _wsize = BK.ddp_rank(), BK.ddp_world_size()
            if len(_x) % _wsize != 0:
                zwarn("Uneven split for file reading!")
            _y = [p for i,p in enumerate(_x) if (i%_wsize==_rank)]
            return _y
        else:
            zwarn("DDP not used in BK!")
            return _x
    # --
    # note: special checks
    if isinstance(conf.input_path, str):  # if directly configured
        conf.input_path = [conf.input_path]
    if conf.input_format.startswith('conll'):
        conf.input_fmode = 'mline'
    # --
    if conf.is_bitext:  # special bitext mode!
        _s0, _s1 = conf.bitext_src_suffix, conf.bitext_trg_suffix
        src_paths = zglobs([p+_s0 for p in conf.input_path], err_act='err')
        if conf.input_split_ddp:
            src_paths = _split_ddp(src_paths)
        trg_paths = [z[:-len(_s0)]+_s1 for z in src_paths]
        streamer0, streamer1 = FileStreamer(src_paths, mode=conf.input_fmode), \
                               FileStreamer(trg_paths, mode=conf.input_fmode)
        streamer = MultiZipStreamer([streamer0, streamer1], auto_mode='strict')
        zlog(f"Read (bitext) from {src_paths}[{_s0},{_s1}]")
    else:
        if conf.input_allow_std and len(conf.input_path)==0:
            input_paths = [sys.stdin]
        else:
            input_paths = zglobs(conf.input_path, err_act='err')
            if conf.input_split_ddp:
                input_paths = _split_ddp(input_paths)
        streamer = FileStreamer(input_paths, mode=conf.input_fmode)
        zlog(f"Read from {input_paths}")
    # --
    yield_wrappers = [eval(ww) for ww in conf.input_wrappers]
    reader = DataReader(streamer, formator=conf.input_format, yield_wrappers=yield_wrappers, max_count=conf.input_max_count, sel_idxes=conf.sel_idxes)
    return reader

class WriterGetterConf(Conf):
    def __init__(self):
        self.output_path = ""
        self.output_format = "zjson"
        self.output_allow_std = True  # allow " " or "-" to mean stdin
        self.output_auto_mkdir = False  # auto mkdir if not existing

    def has_path(self): return len(self.output_path)>0

    def __repr__(self):
        return f"Output={self.output_path}({self.output_format})"

    def get_writer(self, _clone=True, **kwargs):
        return get_writer(self.copy() if _clone else self, **kwargs)

def get_writer(conf: WriterGetterConf=None, **kwargs):
    conf = WriterGetterConf.direct_conf(conf, **kwargs)
    output_path = sys.stdout if (conf.output_allow_std and conf.output_path in ["", "-"]) else conf.output_path
    if conf.output_auto_mkdir and isinstance(output_path, str):
        dir_name = os.path.dirname(output_path)
        if dir_name and not os.path.exists(dir_name):
            mkdir_p(dir_name)
    zlog(f"Write to {output_path}")
    dumper = FileDumper(output_path)
    writer = DataWriter(dumper, formator=conf.output_format)
    return writer

# --
# b mspx/data/rw/base:??
