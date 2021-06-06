#

# the opposite of streamer

from typing import Iterable

# just make it simple
class Dumper:
    def __init__(self):
        pass

    def dump_one(self, obj: object):
        raise NotImplementedError()

    def dump_iter(self, iter: Iterable):
        for one in iter:
            self.dump_one(one)

    def close(self):
        pass

    def __enter__(self): return self
    def __exit__(self, exc_type, exc_val, exc_tb): self.close()
    def __del__(self): self.close()

# with a wrapper
class WrapperDumper(Dumper):
    def __init__(self, base_dumper: Dumper):
        super().__init__()
        self._base_dumper = base_dumper

    def close(self):
        self._base_dumper.close()
