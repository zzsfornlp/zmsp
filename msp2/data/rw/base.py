#

# basic reader/writer
# todo(note): separate these interactions: inst <-> python-object(usually str) <-> files

__all__ = [
    "DataReader", "DataWriter",
    "LineStreamer", "MultiFileStreamer", "MultiLinerHelper", "MultiLineStreamer",
    "FileDumper", "DirDumper",
    "StreamerInputConf", "DumperOutputConf", "get_text_streamer", "get_text_dumper",
    "ReaderGetterConf", "WriterGetterConf", "get_reader", "get_writer",
    "s2d_wrapper", "d2s_wrapper",
]

from typing import Union, IO, Iterable, Callable, List, Type
import os
import sys
from msp2.utils import Conf, mkdir_p, zopen
from msp2.data.inst import DataInstance, Doc, Sent, yield_sents
from msp2.data.stream import Streamer, WrapperStreamer, FIterStreamer, yield_lines, yield_files, yield_filenames, \
    Dumper, WrapperDumper, FListWrapperStreamer
from .formats import DataFormator, ZJsonDataFormator

# =====
class DataReader(WrapperStreamer):
    def __init__(self, base_stream: Streamer, formator: Union[DataFormator, str]):
        super().__init__(base_stream)
        self.formator = DataFormator.try_load_and_lookup(formator).T() if isinstance(formator, str) else formator

    @staticmethod
    def get_zjson_reader(cls: Type, base_stream: Streamer):
        return DataReader(base_stream, ZJsonDataFormator(cls))

    def _next(self):
        obj, _iseos = self._base_streamer.next_and_check()
        if _iseos:
            return self.eos
        else:
            inst = self.formator.from_obj(obj)
            inst.set_read_idx(self.count())
            return inst

class DataWriter(WrapperDumper):
    def __init__(self, base_dumper: Dumper, formator: Union[DataFormator, str]='zjson'):
        super().__init__(base_dumper)
        self.formator = DataFormator.try_load_and_lookup(formator).T() if isinstance(formator, str) else formator

    def write_inst(self, inst: DataInstance): self.dump_one(inst)
    def write_insts(self, insts: Iterable[DataInstance]): self.dump_iter(insts)

    def dump_one(self, inst: DataInstance):
        obj = self.formator.to_obj(inst)
        self._base_dumper.dump_one(obj)

# =====
# File related streamers

# read one line at one time: useful for jsonlines files
class LineStreamer(FIterStreamer):
    def __init__(self, fd_or_path: Union[IO, str]):
        super().__init__(self._f)  # simply put self's method
        self.fd_or_path = fd_or_path

    def _f(self):
        return yield_lines(self.fd_or_path)

# read one file at one time: useful for one FILE one instance
# or read one from multiple files
class MultiFileStreamer(FIterStreamer):
    def __init__(self, files: Iterable[Union[IO, str]], yield_lines=False):
        super().__init__(self._f)  # simply put self's method
        self.files = list(files)  # simply store them all!!
        self.yield_lines = yield_lines

    def _f(self):
        return yield_files(self.files, self.yield_lines)

    @staticmethod
    def build_from_dir(dir: str, yield_lines=False, **dir_kwargs):
        filename_iter = yield_filenames(dir, **dir_kwargs)
        return MultiFileStreamer(filename_iter, yield_lines=yield_lines)

# helper for read multiple line
class MultiLinerHelper:
    _FUNC_REPOS = {
        # sep: separator lines
        "sep_empty": lambda x: len(x.rstrip()) == 0,
        "sep_empty_strict": lambda x: x=="\n",
        # ignore: ignore lines
        "ignore_nope": lambda x: False,
        "ignore_#": lambda x: x.lstrip().startswith("#")
    }
    @staticmethod
    def _get_func(f: Union[str, Callable]):
        if isinstance(f, str):
            return MultiLinerHelper._FUNC_REPOS[f]
        else:
            assert isinstance(f, Callable)
            return f

    def __init__(self, sep_f: Union[str, Callable] = "sep_empty", ignore_f: Union[str, Callable] = "ignore_nope",
                 no_empty=True, eos=None):
        self.sep_f = MultiLinerHelper._get_func(sep_f)
        self.ignore_f = MultiLinerHelper._get_func(ignore_f)
        self.no_empty = no_empty
        self.eos = eos

    # return lines
    def read_multiline(self, fd: Iterable[str]):
        _iseos = False
        while not _iseos:
            lines = []
            while True:  # read multilines until sep_f
                # get one
                try:
                    line = next(fd)
                except StopIteration:
                    _iseos = True
                    break
                # first check ignore!!
                if self.ignore_f(line):
                    continue
                # then check sep
                if self.sep_f(line):
                    break
                lines.append(line)
            # skip empty struct
            if self.no_empty and len(lines)==0:
                pass
            else:  # return str
                return ''.join(lines)
        return self.eos  # EOS

class MultiLineStreamer(WrapperStreamer):
    def __init__(self, line_streamer: Streamer, mhelper: MultiLinerHelper=None):
        super().__init__(line_streamer)
        self.mhelper = MultiLinerHelper(eos=self.eos) if mhelper is None else mhelper

    def _next(self):
        lines = self.mhelper.read_multiline(self._base_streamer)
        return lines

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

    def close(self):
        if self.should_close:  # only close if that is what we opened
            self.fd.close()

    def dump_one(self, obj: object):
        self.fd.write(f"{obj}{self.end}")

# dump one to one file
class DirDumper(Dumper):
    def __init__(self, path: str, dir_file_f: Callable = (lambda x: x.id), dir_file_suffix=".json", end="\n"):
        super().__init__()
        self.path = path
        self.dir_file_f = dir_file_f
        self.dir_file_suffix = dir_file_suffix
        self.end = end
        mkdir_p(path, True)  # mkdir

    def dump_one(self, obj: object):
        new_file = os.path.join(self.path, self.dir_file_f(obj) + self.dir_file_suffix)
        with zopen(new_file, 'w') as fd:
            fd.write(f"{obj}{self.end}")

# =====
# some useful shortcuts

class StreamerInputConf(Conf):
    def __init__(self):
        # read dir
        self.read_dir = False  # path is dir
        self.dir_entry_pat: str = None  # re-pattern for files in dir
        self.dir_entry_sorted = False
        # read file: multiline
        self.use_multiline = False  # path is file, but yield multiline
        self.mtl_sep_f = "'sep_empty'"
        self.mtl_ignore_f = "'ignore_nope'"
        # read file: line
        # ...

    @classmethod
    def _get_type_hints(cls):
        return {"dir_entry_pat": str}

def get_text_streamer(fd_or_path: Union[IO, str], conf: StreamerInputConf=None, **kwargs):
    conf = StreamerInputConf.direct_conf(conf, **kwargs)
    # --
    if conf.read_dir:  # read files from dir: one file at one time!
        return MultiFileStreamer.build_from_dir(fd_or_path, re_pat=conf.dir_entry_pat, sorted=conf.dir_entry_sorted)
    else:  # read from single file
        lstream = LineStreamer(fd_or_path)
        if conf.use_multiline:
            return MultiLineStreamer(lstream, MultiLinerHelper(
                sep_f=eval(conf.mtl_sep_f), ignore_f=eval(conf.mtl_ignore_f)))
        else:
            return lstream

class DumperOutputConf(Conf):
    def __init__(self):
        # write dir
        self.write_dir = False  # path is dir
        self.dir_file_f = "lambda x: x.id"  # eval() as function
        self.dir_file_suffix = ".json"
        self.write_end = "\n"

def get_text_dumper(fd_or_path: Union[IO, str], conf: DumperOutputConf=None, **kwargs):
    conf = DumperOutputConf.direct_conf(conf, **kwargs)
    # --
    if conf.write_dir:
        return DirDumper(fd_or_path, dir_file_f=eval(conf.dir_file_f),
                         dir_file_suffix=conf.dir_file_suffix, end=conf.write_end)
    else:
        return FileDumper(fd_or_path, end=conf.write_end)

# =====
# further more
class ReaderGetterConf(Conf):
    def __init__(self):
        self.input_path = ""
        self.input_format = "zjson"
        self.input_conf = StreamerInputConf()
        self.input_allow_std = True  # allow " " or "-" to mean stdin
        self.input_wrapper = ""  # empty means nope

    def _do_validate(self):
        # note: force them here to make it convenient!
        if self.input_format.startswith("conll"):
            self.input_conf.direct_update(use_multiline=True, mtl_ignore_f="'ignore_#'")
        # --

    def __repr__(self):
        return f"Input={self.input_path}({self.input_format})"

    def get_reader(self, _clone=True, **kwargs):  # directly get a reader; need to clone if want to reuse
        return get_reader(self.copy() if _clone else self, **kwargs)

def get_reader(conf: ReaderGetterConf=None, **kwargs):
    conf = ReaderGetterConf.direct_conf(conf, **kwargs)
    # --
    input_path = sys.stdin if (conf.input_allow_std and conf.input_path in ["", "-"]) else conf.input_path
    reader = DataReader(get_text_streamer(input_path, conf=conf.input_conf), formator=conf.input_format)
    if conf.input_wrapper:
        reader = eval(conf.input_wrapper)(reader)  # wrap another one!!
    return reader

class WriterGetterConf(Conf):
    def __init__(self):
        self.output_path = ""
        self.output_format = "zjson"
        self.output_conf = DumperOutputConf()
        self.output_allow_std = True  # allow " " or "-" to mean stdin

    def __repr__(self):
        return f"Output={self.output_path}({self.output_format})"

    def get_writer(self, _clone=True, **kwargs):
        return get_writer(self.copy() if _clone else self, **kwargs)

def get_writer(conf: WriterGetterConf=None, **kwargs):
    conf = WriterGetterConf.direct_conf(conf, **kwargs)
    # --
    output_path = sys.stdout if (conf.output_allow_std and conf.output_path in ["", "-"]) else conf.output_path
    writer = DataWriter(get_text_dumper(output_path, conf=conf.output_conf), formator=conf.output_format)
    return writer

# =====
# extra one: Sent <=> Doc

# Doc -> Sent
def d2s_wrapper(stream: Streamer):
    return FListWrapperStreamer(stream, lambda d: yield_sents([d]))

# --
class _D2sWrapper(WrapperStreamer):
    def __init__(self, base):
        super().__init__(base)

    def _next(self):
        _base_streamer = self._base_streamer
        last_doc_id = None
        cur_sents = []
        while True:
            x, _iseos = _base_streamer.next_and_check()
            if _iseos:
                break
            cur_doc_id = x.info.get("doc_id")  # todo(+W): may have better methods
            if len(cur_sents)>0 and (cur_doc_id is None or cur_doc_id != last_doc_id):
                _base_streamer.put(x, -1)  # put it back!
                break
            # --
            cur_sents.append(x)
            last_doc_id = cur_doc_id
        # return
        if len(cur_sents) == 0:
            return self.eos
        doc = Doc.create(cur_sents, id=last_doc_id)
        return doc

def s2d_wrapper(stream: Streamer):
    return _D2sWrapper(stream)
