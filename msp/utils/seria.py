# serialization

import json
import pickle
from typing import List
from .log import zopen, zlog

# todo(note):
# for json serialization: customed to_builtin, has simple __dict__ or self is builtin
# otherwise, use pickle serialization

# io
# builtin types will not use this!
class _MyJsonEncoder(json.JSONEncoder):
    def default(self, one):
        if hasattr(one, "to_builtin"):
            return one.to_builtin()
        else:
            return one.__dict__

class JsonRW:
    # update from v and return one if one is not builtin, directly return otherwise
    @staticmethod
    def _update_return(one, v):
        if hasattr(one, "from_builtin"):
            one.from_builtin(v)
        # todo(2): is this one OK?
        elif hasattr(one, "__dict__"):
            one.__dict__.update(v)
        else:
            return v
        return one

    @staticmethod
    def from_file(one, fn_or_fd):
        if isinstance(fn_or_fd, str):
            with zopen(fn_or_fd, 'r') as fd:
                return JsonRW.from_file(one, fd)
        else:
            return JsonRW._update_return(one, json.load(fn_or_fd))

    @staticmethod
    def to_file(one, fn_or_fd):
        if isinstance(fn_or_fd, str):
            with zopen(fn_or_fd, 'w') as fd:
                JsonRW.to_file(one, fd)
        else:
            json.dump(one, fn_or_fd, cls=_MyJsonEncoder)

    @staticmethod
    def from_str(one, s):
        v = json.loads(s)
        return JsonRW._update_return(one, v)

    @staticmethod
    def to_str(one):
        return json.dumps(one, cls=_MyJsonEncoder)

    @staticmethod
    def save_list(ones: List, fn_or_fd):
        if isinstance(fn_or_fd, str):
            with zopen(fn_or_fd, 'w') as fd:
                JsonRW.save_list(ones, fd)
        else:
            for one in ones:
                fn_or_fd.write(JsonRW.to_str(one) + "\n")

    @staticmethod
    def yield_list(fn_or_fd, max_num=-1):
        if isinstance(fn_or_fd, str):
            with zopen(fn_or_fd) as fd:
                for one in JsonRW.yield_list(fd, max_num):
                    yield one
        else:
            c = 0
            while c!=max_num:
                line = fn_or_fd.readline()
                if len(line) == 0:
                    break
                yield JsonRW.from_str(None, line)
                c += 1

    @staticmethod
    def load_list(fn_or_fd, max_num=-1):
        return list(JsonRW.yield_list(fn_or_fd, max_num))

#
class PickleRW:
    @staticmethod
    def from_file(fn_or_fd):
        if isinstance(fn_or_fd, str):
            with zopen(fn_or_fd, 'rb') as fd:
                return PickleRW.from_file(fd)
        else:
            return pickle.load(fn_or_fd)

    @staticmethod
    def to_file(one, fn_or_fd):
        if isinstance(fn_or_fd, str):
            with zopen(fn_or_fd, 'wb') as fd:
                PickleRW.to_file(one, fd)
        else:
            pickle.dump(one, fn_or_fd)

    @staticmethod
    def save_list(ones: List, fn_or_fd):
        if isinstance(fn_or_fd, str):
            with zopen(fn_or_fd, 'wb') as fd:
                PickleRW.save_list(ones, fd)
        else:
            for one in ones:
                pickle.dump(one, fn_or_fd)

    @staticmethod
    def yield_list(fn_or_fd, max_num=-1):
        if isinstance(fn_or_fd, str):
            with zopen(fn_or_fd, 'rb') as fd:
                for one in PickleRW.yield_list(fd, max_num):
                    yield one
        else:
            c = 0
            while c!=max_num:
                yield pickle.load(fn_or_fd)
                c += 1

    @staticmethod
    def load_list(fn_or_fd, max_num=-1):
        ret = []
        try:
            for one in PickleRW.yield_list(fn_or_fd, max_num):
                ret.append(one)
        except EOFError:
            pass
        return ret
