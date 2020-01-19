#

#
class ZTab:
    @staticmethod
    def read(ss, fsep='\t', lsep='\n'):
        pieces = [z.split(fsep) for z in ss.split(lsep)]
        num_col = len(pieces[0]) - 1
        num_row = len(pieces) - 1
        row_names = [z[0] for z in pieces[1:]]
        col_names = pieces[0][1:]
        ret = {}
        # add entries
        for one_row in pieces[1:]:
            k = one_row[0]
            assert k not in ret
            cur_m = {}
            assert len(one_row)-1 == num_col
            for n,v in zip(col_names, one_row[1:]):
                cur_m[n] = v
            ret[k] = cur_m
        return row_names, col_names, ret

    @staticmethod
    def write(row_names, col_names, value_f, fsep='\t', lsep='\n'):
        pieces = []
        # add head line
        pieces.append([""] + col_names)
        # add each rows
        for r in row_names:
            pieces.append([r] + [value_f(r, c) for c in col_names])
        # finally concat
        ss = lsep.join([fsep.join(z) for z in pieces])
        return ss
