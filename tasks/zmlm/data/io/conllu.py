#

# read CoNLL's column format

from msp.utils import Conf, FileHelper
from msp.data import FileOrFdStreamer
from msp.zext.dpar import ConlluReader
from ..insts import GeneralSentence, SeqField, DepTreeField

# read from column styled file and return Dict
class ColumnReader(FileOrFdStreamer):
    def __init__(self, file_or_fd, sep_line_f=lambda x: len(x.strip()) == 0, sep_field_t="\t"):
        super().__init__(file_or_fd)
        self.sep_line_f = sep_line_f
        self.sep_field_t = sep_field_t

    #
    def _get(self, fd):
        lines = FileHelper.read_multiline(fd, self.sep_line_f)
        if lines is None:
            return None
        ret = {}
        headlines = []
        for one_line in lines:
            if one_line.startswith("#"):
                headlines.append(one_line)
                continue
            fileds = one_line.strip("\n").split(self.sep_field_t)
            if len(ret) == 0:
                for i in range(len(fileds)):
                    ret[i] = []  # the start
            else:
                assert len(ret) == len(fileds), "Fields length um-matched!"
            for i, x in enumerate(fileds):
                ret[i].append(x)
        ret["headlines"] = headlines
        return ret

    # not from self.fd
    def yield_insts(self, file_or_fd):
        if isinstance(file_or_fd, str):
            with open(file_or_fd) as fd:
                yield from self.yield_insts(fd)
        else:
            while True:
                z = self._get(file_or_fd)
                if z is None:
                    break
                yield z

    def _next(self):
        return self._get(self.fd)

#
class ConlluParseReader(FileOrFdStreamer):
    def __init__(self, file_or_fd, aug_code="", use_xpos=False, use_la0=True, cut=-1):
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
            # one = ParseInstance([t.word for t in tokens], [self.pos_f(t) for t in tokens],
            #                     [t.head for t in tokens], [self.label_f(t) for t in tokens], code=self.aug_code)
            # one.init_idx = self.count()
            one = GeneralSentence.create([t.word for t in tokens])
            one.add_item("pos_seq", SeqField([self.pos_f(t) for t in tokens]))
            one.add_item("dep_tree", DepTreeField([0]+[t.head for t in tokens], ['']+[self.label_f(t) for t in tokens]))
            one.add_info("sid", self.count())
            one.add_info("aug_code", self.aug_code)
            return one

# BIO NER tags reader
class ConllNerReader(FileOrFdStreamer):
    def __init__(self, file_or_fd, aug_code="", cut=-1):
        super().__init__(file_or_fd)
        self.aug_code = aug_code
        self.cut = cut

    def _next(self):
        if self.count_ == self.cut:
            return None
        lines = FileHelper.read_multiline(self.fd)
        if lines is None:
            return None
        else:
            tokens, tags = [], []
            for line in lines:
                one_tok, one_tag = line.split()
                if one_tag != "O":  # checking
                    t0, t1 = one_tag.split("-")
                    assert t0 in "BI"
                tokens.append(one_tok)
                tags.append(one_tag)
            # one = ParseInstance([t.word for t in tokens], [self.pos_f(t) for t in tokens],
            #                     [t.head for t in tokens], [self.label_f(t) for t in tokens], code=self.aug_code)
            # one.init_idx = self.count()
            one = GeneralSentence.create(tokens)
            one.add_item("ner_seq", SeqField(tags))
            one.add_info("sid", self.count())
            one.add_info("aug_code", self.aug_code)
            return one
