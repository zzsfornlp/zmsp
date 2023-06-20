#

# data and data-format

from mspx.utils import Conf, zglob1, default_json_serializer, zlog, Random

class IclDataConf(Conf):
    def __init__(self):
        self.path = ""  # path
        self.shuffle_times = 0  # shuffle data?
        self.sample_k = -1  # shuffle and sample?

def load_data(path: str):
    if path.startswith(":"):  # use datasets
        import datasets
        _path, _name, _split = path.split(":")[1:]
        _load = datasets.load_dataset(_path, _name, split=_split)
    else:
        _path = zglob1(path)
        _load = default_json_serializer.yield_iter(_path)
    # return a list
    ret = list(_load)
    return ret

def prepare_icl_data(conf: IclDataConf, task_helper=None, **kwargs):
    conf: IclDataConf = IclDataConf.direct_conf(conf, **kwargs)
    # --
    zlog("Start to prepare data ...")
    # load data
    cur_data = load_data(conf.path)
    zlog(f"Load {len(cur_data)} instances from {conf.path}!")
    # down-sample at loading time?
    if conf.shuffle_times > 0:
        _gen = Random.get_generator('shuffle_data')
        for _ in range(conf.shuffle_times):
            _gen.shuffle(cur_data)
        zlog(f"Shuffle data: {len(cur_data)}")
    if conf.sample_k > 0:
        cur_data = cur_data[:conf.sample_k]
        zlog(f"Sample data {conf.sample_k}: {len(cur_data)}")
    # map data
    if task_helper is not None:
        cur_data = [task_helper.map_data(z) for z in cur_data]
    return cur_data
