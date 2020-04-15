#

# data streamer from multiple sources

from typing import List
from msp.data import Streamer
from msp.utils import Random, zlog
from .data import get_data_reader

#
class MultiSpecialJoinStream(Streamer):
    def __init__(self, base_streams: List[Streamer], budgets: List, stop_sidx: int):
        super().__init__()
        # -----
        assert len(base_streams) == len(budgets)
        assert len(base_streams) > 0
        self.base_streams = base_streams
        self.budgets = budgets
        self.stop_sidx = stop_sidx
        self.random_sampler = Random.stream(Random.random_sample)
        # status
        self.current_ptr = len(base_streams)-1
        self.current_budget = 0.
        self.stats = [0 for _ in self.base_streams]

    def _next(self):
        if not self.is_active():
            return None  # make sure requiring restart!
        # first go to the one with budget!!
        cur_ptr = self.current_ptr
        cur_budget = self.current_budget
        while cur_budget<=0. or next(self.random_sampler)>=cur_budget:
            cur_ptr = (cur_ptr+1) % len(self.base_streams)
            cur_budget = float(self.budgets[cur_ptr])
        # find it
        self.current_ptr = cur_ptr
        self.current_budget = cur_budget - 1  # cost one for the current step (or decreasing for empty streamer)
        # get one
        cur_streamer = self.base_streams[cur_ptr]
        read_times = 0
        while read_times<=2:  # avoid loop with empty streamer
            one = cur_streamer.next()
            read_times += 1
            if cur_streamer.is_eos(one):
                if cur_ptr == self.stop_sidx:  # actually restart
                    zlog(f"From the multi-streamer, this epoch stats: {self.stats}")
                    self.stats = [0 for _ in self.stats]
                    return None  # stop here (need restart)
                else:
                    cur_streamer.restart()  # restart right now!
            else:
                self.stats[cur_ptr] += 1
                return one
        # when we get here, it means that there are emtpy loops, thus tail-calling the function again
        return self._next()

    def _restart(self):
        if self.restart_times_ == 0:
            for one in self.base_streams:
                one.restart()
        else:
            # only restart the stop_sidx one
            self.base_streams[self.stop_sidx].restart()
            self.current_ptr = len(self.base_streams)-1
            self.current_budget = 0.

# b tasks/zie/common/data_multi:44
