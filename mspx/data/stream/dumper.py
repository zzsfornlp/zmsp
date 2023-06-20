#

# the opposite of streamer

__all__ = [
    "Dumper", "WrapperDumper", "FWrapperDumper", "MultiDumper",
]

from typing import Iterable, Callable

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

class FWrapperDumper(WrapperDumper):
    def __init__(self, base_dumper: Dumper, func: Callable, inplaced=False):
        super().__init__(base_dumper)
        self.func = func
        self.inplaced = inplaced

    def dump_one(self, obj: object):
        z = self.func(obj)
        out = obj if self.inplaced else z
        self._base_dumper.dump_one(out)

# inverse of zip
class MultiDumper(Dumper):
    def __init__(self, base_dumpers: Iterable[Dumper]):
        super().__init__()
        # --
        self._base_dumpers = list(base_dumpers)

    def close(self):
        for d in self._base_dumpers:
            d.close()

    def dump_one(self, obj: Iterable):
        objs = list(obj)
        assert len(objs) == len(self._base_dumpers)
        for v, d in zip(objs, self._base_dumpers):
            d.dump_one(v)
