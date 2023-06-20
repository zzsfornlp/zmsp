#

# task specific helper

from collections import Counter
from mspx.utils import Conf, zlog

# task table
TASK_INFO = {
'eae': {
    'data_map': {'label': (lambda x: x['role'])},  # {key: function}
},
'sst': {
    'data_map': {
        'sent': (lambda x: x['sentence'].strip()),
        # 'label': (lambda x: ['negative', 'positive'][x['label']]),
        'label': (lambda x: ['Negative', 'Positive'][x['label']]),
    },
    # 'label_options': ['negative', 'positive'],
    'label_options': ['Negative', 'Positive'],
},
}

class TaskConf(Conf):
    def __init__(self):
        self.task = ""

class TaskHelper:
    def __init__(self, conf: TaskConf):
        self.info = TASK_INFO[conf.task]

    def map_data(self, inst):
        data_map = self.info.get('data_map')
        if data_map:
            for k, f in data_map.items():
                inst[k] = f(inst)  # modify inplace!
        return inst

    def get_label_options(self, data_pool):
        if 'label_options' in self.info:
            label_options = self.info['label_options']
        else:
            cc = Counter([z['label'] for z in data_pool])
            label_options = list(cc.keys())
            zlog(f"Obtain label options {label_options}: {cc}")
        return label_options
