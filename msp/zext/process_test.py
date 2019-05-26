#

from msp import utils
from msp.utils import StatRecorder, Timer

# common practice for testing
class TestingRunner(object):
    def __init__(self, model):
        self.model = model
        self.test_recorder = StatRecorder(True)

    def run(self, stream):
        rec = self.test_recorder
        with Timer(tag="Run-test", info="", print_date=True):
            for insts in stream:
                # results are stored in insts
                with rec.go():
                    res = self._run_batch(insts)
                rec.record(res)
        res = self._run_end()
        return res

    # to be implemented
    # end the testing & eval & return sumarized results
    def _run_end(self):
        raise NotImplementedError()

    # run and record for one batch
    def _run_batch(self, insts):
        raise NotImplementedError()

# eval and write results
class ResultManager(object):
    pass
