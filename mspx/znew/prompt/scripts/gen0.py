#

# streaming generation

import sys
import torch
from tqdm.auto import tqdm
from mspx.utils import Conf, init_everything, zwarn, zlog, default_json_serializer, default_pickle_serializer
from ..proc.train import MainConf as BaseConf
from ..proc.run import MyRunner
from ..data import MyDataset, InputBatch, DataInst, DatasetConf

class MainConf(BaseConf):
    def __init__(self):
        super().__init__()
        # --
        self.use_gr = False  # otherwise simply through CMD
        self.gr_host = "0.0.0.0"
        self.gr_port = 27018  # which port?
        self.use_file = ""
        self.do_pred = True
        self.do_enc = False
        self.output_file = ""
        self.quiet = False
        # --
        self.do_batch_enc = False  # another mode for batch-encoding
        self.do_clf0 = False  # instruction running with clf tasks
        self.clf0_task = ''
        self.binput = DatasetConf()

# --
class Generator:
    def __init__(self, model, task, conf):
        self.model = model
        self.task = task
        self.conf = conf
        # --
        self.fake_dataset = MyDataset.make_fake_dataset(wset='test')

    def generate(self, inst):
        conf = self.conf
        with torch.autocast("cuda" if torch.cuda.is_available() else 'cpu'):
        # if 1:
            with torch.no_grad():
                inst = self.task.preprocess_inst(inst, self.fake_dataset)
                ibatch = InputBatch([inst], dataset=self.fake_dataset)
                model_inputs = self.task.collate_fn(ibatch)
                model_output = self.task.model_forward(**model_inputs, do_test=conf.do_pred, do_enc=conf.do_enc, debug_print=(not conf.quiet))
                self.task.pred(ibatch, model_output, None)
                if conf.do_enc:
                    inst['arr_enc'] = model_output['t_enc'].cpu().numpy()
        return inst

def yield_input_stream():
    while 1:
        s_instruction = input("Instruction >> ").strip()
        s_input = input("Input >> ").strip()
        inst = DataInst({'instruction': s_instruction, 'input': s_input, 'output': ''})
        yield inst

# --
def main(args):
    conf: MainConf = init_everything(MainConf(), args)
    accelerator = conf.accelerate_conf.get_accelerator()
    # --
    # model & task
    model = conf.model_conf.make_node()
    task = conf.task_conf.make_node(model=model)
    # load model
    runner = MyRunner(conf.run_conf, model, task, accelerator)
    if conf.run_conf.model_load_name:
        runner.load_model(model, conf.run_conf.model_load_name)
    else:
        zwarn("No model loading for testing, is this OK?")
    # --
    generator = Generator(model, task, conf)
    if conf.use_gr:
        raise NotImplementedError()
    elif conf.do_clf0:
        # from ..eval import EvalConf, Evaler
        # # --
        # data_test0 = MyDataset(conf.binput, name='test0')
        # loader_test0 = task.get_dataloader(data_test0, False)
        # evaler = Evaler(None, metrics=['z_f1'])
        # with torch.autocast("cuda" if torch.cuda.is_available() else 'cpu'):
        #     with torch.no_grad():
        #         progress_bar = tqdm(range(len(list(loader_test0))))
        #         for step, cur_data in enumerate(loader_test0):
        #             cur_ibatch = cur_data['ibatch']
        #             outputs = model(task=task, do_test=True, **cur_data)
        #             _arr = outputs['t_enc'].cpu().numpy()
        #             for _ii, _inst in enumerate(cur_ibatch.items):
        #                 _inst['arr_enc'] = _arr[_ii]
        #                 # breakpoint()
        #             task.pred(cur_ibatch, outputs, None)
        #             progress_bar.update(1)
        #             # --
        #             # decode
        #             breakpoint()
        #             # --
        # data_test0.write_insts()
        # ret = evaler.get_res()
        pass
    elif conf.do_batch_enc:
        data_test0 = MyDataset(conf.binput, name='test0')
        loader_test0 = task.get_dataloader(data_test0, False)
        all_insts = []
        with torch.autocast("cuda" if torch.cuda.is_available() else 'cpu'):
            with torch.no_grad():
                progress_bar = tqdm(range(len(list(loader_test0))))
                for step, cur_data in enumerate(loader_test0):
                    cur_ibatch = cur_data['ibatch']
                    outputs = model(task=task, do_test=True, do_enc=True, **cur_data)
                    _arr = outputs['t_enc'].cpu().numpy()
                    for _ii, _inst in enumerate(cur_ibatch.items):
                        _inst['arr_enc'] = _arr[_ii]
                        # breakpoint()
                    task.pred(cur_ibatch, outputs, None)
                    progress_bar.update(1)
                    all_insts.extend(cur_ibatch.items)
        if conf.output_file:
            default_pickle_serializer.to_file(all_insts, conf.output_file)
    else:
        if conf.use_file:
            inst_stream = tqdm(DataInst(z) for z in default_json_serializer.yield_iter(conf.use_file))
        else:
            inst_stream = yield_input_stream()
        all_insts = []
        for inst0 in inst_stream:
            inst = generator.generate(inst0)
            if conf.output_file:
                all_insts.append(inst)
            if not conf.quiet:
                zlog(f"Result instance is: {inst}")
                zlog(f"Output=\n\n{inst['output_pred']}")
        if conf.output_file:
            default_pickle_serializer.to_file(all_insts, conf.output_file)
    # --

# python -mpdb -m mspx.znew.prompt.scripts.gen0
if __name__ == '__main__':
    main(sys.argv[1:])

# --
def sbert_emb(ds):
    from sentence_transformers import SentenceTransformer
    _sbert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = _sbert_model.encode([z['instruction'] for z in ds])
    return embeddings

def get_cluster(filename, use_sbert=False, n_clu=10, n_sample=0, take_first=None):
    import numpy as np
    from mspx.utils import default_pickle_serializer
    _ds = list(default_pickle_serializer.from_file(filename))
    if take_first:
        _ds = _ds[:take_first]
    if use_sbert:
        _arr = sbert_emb(_ds)
    else:
        _arr = np.stack([d['arr_enc'] for d in _ds], axis=0)
    import sklearn.cluster
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clu, random_state=0, n_init="auto").fit(_arr)
    groups = [(kmeans.labels_ == z).nonzero()[0] for z in range(n_clu)]
    distances = []
    for ii in range(len(groups)):
        _center = kmeans.cluster_centers_[ii]
        _dist = ((_arr[groups[ii]] - _center) ** 2).sum(-1)
        distances.append(_dist)
    sample_groups = None
    if n_sample > 0:
        np.random.seed(12345)
        sample_groups = []
        for ii in range(len(groups)):
            one_sample = np.random.choice(groups[ii], size=n_sample, replace=False, p=distances[ii]/distances[ii].sum())
            sample_groups.append(one_sample)
    return _ds, groups, distances, sample_groups

# --
"""
reses0 = [get_cluster(f'run_instr0508_{z}/dA.pkl', take_first=1000) for z in range(4)]
reses1 = [get_cluster(f'run_instr0508_{z}/dA.pkl') for z in range(4)]
import pprint
pprint.pprint([reses0[0][0][z]['instruction'] for z in reses0[0][1][0]])
# --
from mspx.tools.algo.eval_cluster import *
eval_cluster(reses[0][1], reses[1][1])
# ->

"""
