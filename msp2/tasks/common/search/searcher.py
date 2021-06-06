#

# the general searcher
# the three layers: State(one state), Agenda(one inst), Searcher(all insts if batched)

__all__ = [
    "SearcherAgenda", "SearcherCache", "SearcherComponent", "Searcher",
]

from typing import List, Callable

# searching agenda for one instance
class SearcherAgenda:
    def is_all_end(self):
        raise NotImplementedError()

# batched cache for the current search
class SearcherCache:
    pass

# one component
class SearcherComponent:
    def refresh(self, *args, **kwargs):
        raise NotImplementedError()

# the searcher itself
class Searcher:
    def refresh(self, *args, **kwargs):
        raise NotImplementedError()

    def go(self, ags: List[SearcherAgenda], cache: SearcherCache):
        raise NotImplementedError()
