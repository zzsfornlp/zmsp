# about files and loggings

import gzip, bz2, platform, time, sys, logging

# files and loggings

def zopen(filename, mode='r', encoding="utf-8"):
    suffix = "" if ('b' in mode) else "t"
    if 'b' in mode:
        encoding = None
    if filename.endswith('.gz'):
        # "t" for text mode of gzip
        return gzip.open(filename, mode+suffix, encoding=encoding)
    elif filename.endswith('.bz2'):
        return bz2.open(filename, mode+suffix, encoding=encoding)
    else:
        return open(filename, mode, encoding=encoding)

def zlog(s, func="plain", end='\n', flush=True, timed=False, level=3):
    Logger._instance._log(str(s), func, end, flush, timed, level)

printing = zlog

class Logger(object):
    _instance = None
    _logger_heads = {
        "plain": "--", "time": "##", "io": "==", "result": ">>", "report": "**",
        "warn": "!!", "error": "ER", "fatal": "KI", "config": "CC",
        "debug": "DE", "nope": "",
    }
    @staticmethod
    def _get_ch(func):  # get code & head
        if func not in Logger._logger_heads:
            return func, func
        else:
            return func, Logger._logger_heads[func]
    MAGIC_CODE = "?THE-NATURE-OF-HUMAN-IS?"  # REPEATER?

    @staticmethod
    def init(files):
        s = "LOG-%s-%s.txt" % (platform.uname().node, '-'.join(time.ctime().split()[-4:]))
        s = '-'.join(s.split(':'))      # ':' raise error in Win
        log_files = []
        for f in files:
            if f is not None:
                log_files.append(f if f!=Logger.MAGIC_CODE else s)
        ff = dict((f, True) for f in log_files)
        lf = dict((l, True) for l in Logger._logger_heads)
        Logger._instance = Logger(ff, lf)
        # zlog("START!!", func="plain")

    # =====
    def __init__(self, file_filters, func_filters, level=3):
        self.file_filters = file_filters
        self.func_filters = func_filters
        self.fds = {}
        # the managing of open files (except outside handlers like stdio) is by this one
        for f in self.file_filters:
            if isinstance(f, str):
                self.fds[f] = zopen(f, mode="w")
            else:
                self.fds[f] = f
        self.level = level

    def __del__(self):
        for f in self.file_filters:
            if isinstance(f, str):
                self.fds[f].close()

    def set_level(self, level):
        zlog(f"Change log level from {self.level} to {level}")
        self.level = level

    def _log(self, s, func, end, flush, timed, level):
        if level >= self.level:
            func, head = Logger._get_ch(func)
            if self.func_filters[func]:
                if timed:
                    ss = f"{head}[{'-'.join(time.ctime().split()[-4:])}, L{level}] {s}"
                else:
                    ss = f"{head}[L{level}] {s}"
                for f in self.fds:
                    if self.file_filters[f]:
                        print(ss, end=end, file=self.fds[f], flush=flush)

    # todo(1): register or filter files & codes

    # =====
    # system's logging module

    cached_loggers = {}

    @staticmethod
    def get_logger(name, level=logging.INFO, handler=sys.stderr,
                   formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
        if name not in Logger.cached_loggers:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter(formatter)
            stream_handler = logging.StreamHandler(handler)
            stream_handler.setLevel(level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
            Logger.cached_loggers[name] = logger
        return Logger.cached_loggers[name]
