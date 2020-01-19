#

from typing import List

#
class Agenda:
    def is_end(self):
        raise NotImplementedError()

#
class Searcher:
    def refresh(self, *args, **kwargs):
        raise NotImplementedError()

    def go(self, ags: List[Agenda]):
        raise NotImplementedError()

#
class Component:
    def refresh(self, *args, **kwargs):
        raise NotImplementedError()

    def __repr__(self):
        return f"<{self.__class__.__name__}>"
