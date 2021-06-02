#

from msp.utils import zlog, zopen, Helper

class DataReader:
    def __init__(self, conf):
        self.lc = conf.lc
        self.thr_minlen = conf.thr_minlen
        self.thr_maxlen = conf.thr_maxlen
        #
        self.stats = {"orig_sent": 0, "orig_tok": 0, "sent": 0, "tok": 0}
        self.report_freq = 5

    def yield_data(self, files):
        #
        if not isinstance(files, (list, tuple)):
            files = [files]
        #
        cur_num = 0
        for f in files:
            cur_num += 1
            zlog("-----\nDataReader: [#%d] Start reading file %s." % (cur_num, f))
            with zopen(f) as fd:
                for z in self._yield_tokens(fd):
                    yield z
            if cur_num % self.report_freq == 0:
                zlog("** DataReader: [#%d] Summary till now:" % cur_num)
                Helper.printd(self.stats)
        zlog("=====\nDataReader: End reading ALL (#%d) ==> Summary ALL:" % cur_num)
        Helper.printd(self.stats)

    def _yield_tokens(self, fin):
        ss = self.stats
        for line in fin:
            line = line.strip()
            if len(line) > 0:
                # todo(note): sep by seq of spaces!
                tokens = line.split()
                tok_num = len(tokens)
                ss["orig_sent"] += 1
                ss["orig_tok"] += tok_num
                if tok_num <= self.thr_maxlen and tok_num >= self.thr_minlen:
                    ss["sent"] += 1
                    ss["tok"] += tok_num
                    # todo(note): current only processing is lowercase
                    if self.lc:
                        tokens = [c.lower() for c in tokens]
                    yield tokens
