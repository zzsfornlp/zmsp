#

# prediction

import numpy as np
import tqdm
from mspx.utils import Conf, zlog, ZHelper

class IclPredConf(Conf):
    def __init__(self):
        self.score_mode = 'avg'  # sum or avg or first
        self.batch_size = 16

def pred_results(queries, model, conf: IclPredConf, task_helper=None, **kwargs):
    conf: IclPredConf = IclPredConf.direct_conf(conf, **kwargs)
    _score_mode = conf.score_mode
    # --
    zlog("Start to do prediction ...")
    stream = (d for query in tqdm.tqdm(queries) for d in query['data'])
    all_res, all_info = [], []
    for batch in ZHelper.yield_batches(stream, conf.batch_size):
        inputs0, inputs1 = [z[0] for z in batch], [z[1] for z in batch]
        one_res, one_info = model.run_logprob(inputs0, inputs1)
        # --
        if _score_mode == 'avg':
            one_fres = [np.mean(z).item() for z in one_res]
        elif _score_mode == 'sum':
            one_fres = [sum(z) for z in one_res]
        elif _score_mode == 'first':
            one_fres = [z[0] for z in one_res]
        else:
            raise NotImplementedError(f"UNK score_mode = {_score_mode}")
        # --
        all_res.extend(one_fres)
        if one_info is not None:
            all_info.extend(one_info)
    # read the results and predict
    cur_res_idx = 0
    ret = []
    for query in queries:
        num_data = len(query['data'])
        num_map = len(query['map'])
        if num_data == num_map:  # plain mode
            scores = all_res[cur_res_idx:cur_res_idx+num_data]
        elif num_data == 2*num_map:  # calibrated mode
            scores = [a-b for a,b in zip(all_res[cur_res_idx:cur_res_idx+num_map], all_res[cur_res_idx+num_map:cur_res_idx+num_data])]
        else:
            raise RuntimeError(f"Strange number: {num_data} vs {num_map}")
        one_ret = {'pred': {'scores': scores}, 'label': query['map'][np.argmax(scores)]}
        if len(all_info) > 0:
            one_ret['pred']['info'] = all_info[cur_res_idx:cur_res_idx+num_data]
        ret.append(one_ret)
        cur_res_idx += num_data
    assert cur_res_idx == len(all_res)
    return ret

# --
# b mspx/znew/icl/pred:25
