#

# assign embeddings from pretrained models

__all__ = [
    "ZTaskAssignEmbConf", "ZTaskAssignEmb", "ZModAssignEmbConf", "ZModAssignEmb",
]

import pandas
import numpy as np
from mspx.cli.main import main as cli_main
from mspx.utils import Timer
from mspx.nn import BK, ZTaskConf, ZModConf, ZTaskSbConf, ZTaskSb, ZModSbConf, ZModSb, ZRunCache

@ZTaskConf.rd('assign_emb')
class ZTaskAssignEmbConf(ZTaskSbConf):
    def __init__(self):
        super().__init__()
        self.mod = ZModAssignEmbConf()

@ZTaskAssignEmbConf.conf_rd()
class ZTaskAssignEmb(ZTaskSb):
    pass

@ZModConf.rd('assign_emb')
class ZModAssignEmbConf(ZModSbConf):
    def __init__(self):
        super().__init__('bmod2')
        self.remain_toks = True  # overwrite!
        # --
        self.max_seq_len = 256  # larger!
        self.ftype = 'float16'
        self.name_prefix = ''
        self.assign_hid = True  # hidden repr [1+L, D]
        self.assign_att = False  # att scores [1+L, 1+L, D']

@ZModAssignEmbConf.conf_rd()
class ZModAssignEmb(ZModSb):
    def __init__(self, conf: ZModAssignEmbConf, ztask, zmodel, **kwargs):
        super().__init__(conf, ztask, zmodel, **kwargs)
        conf: ZModAssignEmbConf = self.conf
        # --

    def do_loss(self, rc: ZRunCache, *args, **kwargs):
        raise NotImplementedError("No usage for training!")

    def do_predict(self, rc: ZRunCache, *args, **kwargs):
        conf: ZModAssignEmbConf = self.conf
        # --
        assert not conf.assign_att, "Currently not using this!!"
        res = self._do_forward(rc)
        arr_hid, arr_att = None, None
        if conf.assign_att:
            t_hid, t_att = res
            arr_att = BK.get_value(t_att)
        else:
            t_hid = res
        if conf.assign_hid:
            arr_hid = BK.get_value(t_hid)
        # assign
        _, _, _, arr_toks = rc.get_cache((self.name, 'input'))  # [*, L]
        for _sidx, _item in enumerate(rc.ibatch.items):
            _sent = _item.sent
            _len = len(_sent)
            # read token mapping: note: ii=0 means CLS!
            index_src, index_trg = [0], [0]
            for _ii, _tok in enumerate(arr_toks[_sidx]):
                if _tok is not None and _tok.sent is _sent:
                    assert _ii > 0
                    index_src.append(_ii)
                    index_trg.append(1+_tok.widx)  # note: extra one for CLSsss
            index_src, index_trg = np.asarray(index_src), np.asarray(index_trg)
            if arr_hid is not None:
                one_arr_hid = np.full([1+_len, arr_hid.shape[-1]], 0., dtype=conf.ftype)
                one_arr_hid[index_trg] = arr_hid[_sidx][index_src]
                _sent.arrs[conf.name_prefix+"repr_hid"] = one_arr_hid
            if arr_att is not None:
                one_arr_att = np.full([1+_len, 1+_len, arr_att.shape[-1]], 0., dtype=conf.ftype)
                one_arr_att[index_trg[:, np.newaxis], index_trg] = arr_att[_sidx][index_src[:, np.newaxis], index_src]
                _sent.arrs[conf.name_prefix+"repr_att"] = one_arr_att
        # --
        return {}

# --
# main

def conf_getter(bert_name: str, task_name='AE', **kwargs):
    args = []
    # task & model
    args += f"tcs:{task_name}:assign_emb".split()
    if bert_name:
        args += f"{task_name}.sconf:{bert_name}".split()
    # data
    for wset in ["test0"]:
        args += f"{wset}.group_files: {wset}.tasks:{task_name} {wset}.batch_size:512".split()
        if bert_name:
            args += f"{wset}.len_f:subword:{bert_name}".split()
    # extra
    args += f"fs:test".split()
    return args

def main(args):
    cli_main(args, sbase_getter=conf_getter)

# python3 -m mspx.tools.misc.assign_emb ...
if __name__ == '__main__':
    import sys
    with Timer(info=f"Main", print_date=True) as et:
        main(sys.argv[1:])

"""
# b mspx/tools/misc/assign_emb:78
# examples
python3 -m mspx.tools.misc.assign_emb conf_sbase:bert_name:xlm-roberta-base::task_name:zz device:0 model_load_name:
bert_lidx:8 assign_att:1 'att_lidx:range(12)'
"""
