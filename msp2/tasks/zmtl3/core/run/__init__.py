#

# since this time we can have quite different training schemes, especially modeling runners more

from .common import InputBatch, DataItem

def get_batcher_options():
    from msp2.utils import ConfEntryChoices
    from .plain import PlainBatcherConf
    from .frame import FrameBatcherConf
    entries = {'plain': PlainBatcherConf(), 'frame': FrameBatcherConf()}
    ret = ConfEntryChoices(entries, 'plain')
    return ret
