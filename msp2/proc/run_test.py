#

# for the testing process

from typing import List, Iterable
from msp2.utils import Conf, Constants, StatRecorder, zlog, default_json_serializer, zwarn, Timer, Random, OtherHelper
from msp2.data.stream import Streamer
from .help import SVConf, ScheduledValue, TrainingProgressRecord, ResultRecord
from .model import ZModel

class TestingRunner:
    def __init__(self, model: ZModel, test_stream: Streamer):
        self.model = model
        self.test_recorder = StatRecorder(timing=True)
        self.test_stream = test_stream

    def run(self):
        rec = self.test_recorder
        with Timer(info="Run-test", print_date=True):
            for insts in self.test_stream:
                # results are stored in insts themselves
                with rec.go():
                    res = self._run_batch(insts)
                rec.record(res)
            res = self._run_end()
        return res

    # =====
    # template method to be implemented

    # end the testing & eval & return sumarized results
    def _run_end(self):
        raise NotImplementedError()

    # run and record for one batch
    def _run_batch(self, insts: List):
        raise NotImplementedError()
