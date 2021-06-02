#

from msp.utils import zcheck, FileHelper

class ConllRW(object):
    def __init__(self, num_fields=None, sep=None, sep_struct_f=lambda x: len(x.strip()) == 0, ignore_line_f=lambda x: False):
        self.num_fileds = num_fields
        self.separator = sep                    # field sep
        self.sep_struct_f = sep_struct_f        # inst sep
        self.ignore_line_f = ignore_line_f

    def split_line(self, line):
        return line.strip().split(self.separator)

    def read_lines(self, fd):
        lines = FileHelper.read_multiline(fd, self.sep_struct_f, self.ignore_line_f)
        return lines

    # read from a file-like and return one instance, None if ending
    def read_fields(self, fd):
        lines = self.read_lines(fd)
        if lines is None:
            return None
        else:
            if self.num_fileds is None:     # first line of data
                self.num_fileds = len(self.split_line(lines[0]))
            ret = [[] for i in range(self.num_fileds)]        # list of list of fields
            for one in lines:
                fields = self.split_line(one)
                zcheck(len(fields)==self.num_fileds, "READ: Unmatched number of fields.")
                for i, f in enumerate(fields):
                    ret[i].append(f)
            return ret

    def write_fields(self, fd, fields):
        zcheck(len(fields)==self.num_fileds, "Write: Unmatched number of fields.")
        sep = "\t" if self.separator is None else self.separator
        length = len(fields[0])
        zcheck(all(len(f)==length for f in fields), "Write: Unmatched length.")
        for idx in range(length):
            cur_line_fs = [str(z[idx]) for z in fields]
            cur_line = sep.join(cur_line_fs)
            fd.write(cur_line+"\n")
        fd.write("\n")
