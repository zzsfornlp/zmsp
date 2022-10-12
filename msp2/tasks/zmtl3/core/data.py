#

# represent one dataset
# note: mostly from zmtl2.core.data
# -- note: currently load them all in-mem
"""
# two versions of data:
1) static raw ones: Doc/Sent, stored here and used for build vocab and preprocess
2) runtime prepared ones: prepared and processed, then get batched
"""

__all__ = ["ZDatasetConf", "ZDataset", "ZDataPreprocessor"]

import os
import re
from typing import List, Callable
from collections import OrderedDict, defaultdict, Counter
from msp2.utils import Conf, zlog, Random, Registrable, ConfEntryDict
from msp2.data.inst import Sent, yield_frames, yield_sents, set_ee_heads
from msp2.data.rw import ReaderGetterConf, WriterGetterConf
from msp2.data.resources import FramePresetHelper
from msp2.proc import SVConf
import pandas as pd
from .run import InputBatch, get_batcher_options

# --
class ZDataPreprocessor(Registrable):
    def __call__(self, insts):
        return insts

    def get_rng_rands(self, rng):  # check rng signature
        return rng.randint(100,size=22).tolist()

# --
# for example: 1) 'ace.-s5,+*:5;seed:0', 2) 'ace.+*:5;seed:0', 3) 'ace.+s5,-*:0;ace.+*:5;seed:0'
class FrameSamplerPrep(ZDataPreprocessor):
    def __init__(self, fs_str: str):
        self.fs = []  # List[(Filter,Num)]
        ds = ConfEntryDict.dict_getv(fs_str, str)
        _seed = 12345 + int(ds.get('seed', 0))
        for sig, rr in ds.items():
            if sig not in ["seed", ""]:
                ff = FramePresetHelper(sig)
                _range = [int(z) for z in rr.split(",")]
                if len(_range) == 1:
                    _range = [0] + _range
                self.fs.append((ff, _range))
        self.fs_str = fs_str
        self.seed = _seed
        self.verbose = ds.get('verbose', 0)
        # --

    def __call__(self, insts):
        if len(self.fs) > 0:
            fs_str, _seed = self.fs_str, self.seed
            rng = Random.get_np_generator(_seed)  # make sure to create the same data!
            zlog(f"Start rng with {fs_str}(seed={_seed}): {self.get_rng_rands(rng)}")
            # collect them
            all_frames = defaultdict(list)  # type -> List[Evt]
            for evt in yield_frames(insts):
                all_frames[evt.type].append(evt)
            # shuffle
            for k in all_frames.keys():
                rng.shuffle(all_frames[k])
            # simply get k-previous
            output_data = {'type': [], 'all': [], 'sample': [], 'lastid': []}
            sel_data = OrderedDict()
            for evt_type in sorted(all_frames.keys(), key=(lambda x: (-len(all_frames[x]), x))):
                evts = all_frames[evt_type]
                sample_num = len(evts)  # by default all
                _low, _high = 0, sample_num  # [low, high)
                for ff, _range in self.fs:
                    if ff.f(evt_type):  # if hit!
                        _low = max(0, _range[0])
                        _high = min(_high, _range[1])
                # delete the extra ones!
                for one in evts[:_low]:
                    one.sent.delete_frame(one, 'evt')
                for one in evts[_high:]:
                    one.sent.delete_frame(one, 'evt')
                sel_data[evt_type] = sorted(evts[_low:_high], key=(lambda x: x.id))  # simply sort by id
                output_data['type'].append(evt_type)
                output_data['all'].append(len(evts))
                output_data['sample'].append(_high - _low)
                output_data['lastid'].append(str(evts[_high-1].id))
            # report
            df = pd.DataFrame(output_data)
            df['all_ratio'] = df['all'] / df['all'].sum()
            df['sample_ratio'] = df['sample'] / df['all']
            zlog(f"#-- With FrameSampler ({self.get_rng_rands(rng)})\n{df.to_string()}\n{df.sum()}")
            zlog("Selected details are:")
            if self.verbose:
                for ii, (k, vs) in enumerate(sel_data.items()):
                    zlog(f"#{ii}: {vs}")
            # --
        return insts
# --

# --
# fake frames with lemma
class FrameLemmaPrep(ZDataPreprocessor):
    def __init__(self, fl_str: str):
        self.upos_set = set(fl_str.split(",")) if fl_str else set()

    def __call__(self, insts):
        # --
        if len(self.upos_set) > 0:
            cc = Counter()
            count_lemma = Counter()
            for sent in yield_sents(insts):
                sent.clear_events()  # clean the original ones!
                upos_vals = sent.seq_upos.vals
                lemma_vals = sent.seq_lemma.vals
                for widx, (pp, ll) in enumerate(zip(upos_vals, lemma_vals)):
                    cc['tok_all'] += 1
                    if ll is not None and pp in self.upos_set:
                        cc['tok_add'] += 1
                        count_lemma[ll] += 1
                        sent.make_event(widx, 1, type=str.lower(ll))
            cc['count_lemma'] = len(count_lemma)
            zlog(f"#-- With FrameLemma: {cc}")
        # --
        return insts

# --
# fake frames with syn (dependency & upos)
class FrameDepPrep(ZDataPreprocessor):
    def __init__(self, fd_str: str):
        self.fd_str = fd_str
        ds = ConfEntryDict.dict_getv(fd_str, str)
        self.enabled = ('yes' in ds)
        if self.enabled:
            self.ftrg = FramePresetHelper("upos." + ds.get('ftrg', ''))  # frame targets (UPOS)
            self.rtrg = FramePresetHelper("udep." + ds.get('rtrg', ''))  # role targets (DEP relations)
            self.level = int(ds.get('level', 2))  # by default including subtype

    def __call__(self, insts):
        if self.enabled:
            cc = Counter()
            for sent in yield_sents(insts):
                sent.clear_events()  # clean the original ones!
                frames = [None] * len(sent)
                for widx, upos in enumerate(sent.seq_upos.vals):
                    if self.ftrg.f(upos):
                        frames[widx] = sent.make_event(widx, 1, type=f'{upos}')
                        cc['frame1'] += 1
                    else:
                        cc['frame0'] += 1
                dep_heads, dep_labs = sent.tree_dep.seq_head.vals, sent.tree_dep.seq_label.vals
                for m, (h, lab) in enumerate(zip(dep_heads, dep_labs)):
                    h = h - 1
                    lab_fields = lab.split(":")
                    if h>=0 and self.rtrg.f(lab_fields[0]) and frames[h] is not None:  # note: for this judgement, use L0
                        ef = sent.make_entity_filler(m, 1)
                        frames[h].add_arg(ef, role=':'.join(lab_fields[:self.level]))
                        cc['arg1'] += 1
                    else:
                        cc['arg0'] += 1
            zlog(f"#-- With FrameDep: {cc}")
            # breakpoint()
        return insts

# --
# fake a balanced dataset
# for example: 'fb_str:ratio:1.;upos:NOUN,VERB;seed:0' -> used in combination with test?.inst_f:sent_evt2
class FrameBalancePrep(ZDataPreprocessor):
    def __init__(self, fb_str: str):
        raise RuntimeError("Deprecated by sampling on the run ...")
# --

# --
class ZDatasetConf(Conf):
    def __init__(self):
        # ==
        # top/group-level info (used at outside, put here for convenience)
        self.group_name = ""
        self.group_files = []  # List: # -> "input_file" or Dict: sub_name -> "input_file"
        self.group_tasks = []  # tasks to perform! note: allow sub-name!
        self.group_info = {}  # extra info?
        self.group_joint = False  # join all these into one dataset?
        # train (train)
        self.group_sample_rate = SVConf().direct_update(val=1., which_idx="cidx", mode="none")  # outside_sample by rate
        self.group_sample_betas = [1.]
        self.group_sample_alpha = 0.  # inside_sample by beta * len(inst)**alpha
        # eval (test/dev)
        self.group_eval_weight = 1.  # weight for final eval
        # ==
        # (static) io
        self.R = ReaderGetterConf()
        self.W = WriterGetterConf()
        # - paths (further we have default ones for "*_gold", "*_output" if not provided in extras)
        self.input_dir = ""
        self.input_file = ""
        self.gold_file = ""  # by default the same as input_file
        self.output_dir = ""
        self.output_file = ""
        self.output_prefix = "_zout"  # default output prefix, full will be "{this}.{wset}.json"
        # - special
        self.fs_str = ''  # see 'FrameSamplerPrep'
        self.fl_str = ''  # see 'FrameLemmaPrep'
        self.fd_str = ''  # see 'FrameDepPrep'
        self.preprocessors = []  # need to slightly modify the data?
        # self.approx_prev_next = False  # approx. setting of prev & next when loading, note: deprecated
        self.presample = 1.0  # (>1=N,<1=Rate) random sample how much at the very beginning, as pre-processing for convenience!
        self.presample_shuffle = True  # whether shuffle in presample?
        self.presample_seed = 0
        self.presample_reverse = False  # from back to start (for convenience)
        # --
        self.batcher = get_batcher_options()
        self.test_with_loss = 0  # when testing, if >0 calculating loss instead of actual decoding
        self.special_test = False  # special testing with whole dataset as input!
        # --

    def get_input_path(self):
        return os.path.join(self.input_dir, self.input_file) if self.input_dir else self.input_file

    def get_gold_path(self):
        if self.gold_file.startswith(":s/"):  # special sub!
            _, a, b = self.gold_file.strip().split("/")
            fname = self.input_file.replace(a, b)
        else:
            fname = self.gold_file if self.gold_file else self.input_file
        return os.path.join(self.input_dir, fname) if self.input_dir else fname

    def get_output_path(self, default_sig: str):
        # first get output's file name (can be an auto name)
        fname = self.output_file if self.output_file else f"{self.output_prefix}.{default_sig}.json"
        return os.path.join(self.output_dir, fname) if self.output_dir else fname

    @classmethod
    def _get_type_hints(cls):
        return {"input_dir": "zglob1", "output_dir": "zglob1"}  # easier finding!

class ZDataset:
    # --
    # note: to be extended!
    _GUESSED_LANG_SET = {"en", "zh", "ar", "fi", "fr", "de", "it", "pt", "es"}
    # --

    def __init__(self, conf: ZDatasetConf, name: str, wset: str, _no_load=False, sample_beta=1.):
        self.conf = conf
        self.name = name
        self.wset = wset
        self.lang = self._guess_lang(conf.input_file)  # guess lang from file name!
        self.info = conf.group_info.copy()  # extra info
        self.sample_beta = sample_beta
        # note: data processings are all greedy!!
        # precessors
        self._preprocessors = [ZDataPreprocessor.try_load_and_lookup(z).T for z in conf.preprocessors]
        self._preprocessors.append(FrameSamplerPrep(conf.fs_str))
        self._preprocessors.append(FrameLemmaPrep(conf.fl_str))
        self._preprocessors.append(FrameDepPrep(conf.fd_str))
        # read & prepare insts
        self._gold_insts = None  # used for eval, lazy loaded!
        self._presample_indexes = None  # used for loading gold_insts
        if _no_load:
            self.insts = []
        else:
            self.insts = self._load_insts()  # store raw data (can be Doc or Sent, read from file)
        self.cur_batchers = []  # first in last out?
        # --
        # tasks needed to perform: as an ordered collection
        self._tasks = OrderedDict()
        for t in conf.group_tasks:
            t0, t1 = (t.split(".", 1) + [""])[:2]
            self._tasks[t0] = t1
        # --

    def _guess_lang(self, s: str):
        s = s.split("/")[-1]
        for f in re.split(r"\W+", s):
            if f in ZDataset._GUESSED_LANG_SET:
                return f
        return "UNK"

    def __repr__(self):
        return f"Dataset({self.name}:{self.wset})" + ("" if self.sample_beta==1 else f"[beta={self.sample_beta}]")

    def __len__(self):  # number of instances
        return len(self.insts)

    def _load_insts(self):
        conf = self.conf
        full_path = conf.get_input_path()
        insts = list(conf.R.get_reader(input_path=full_path))  # note: read them all!!
        for pp in self._preprocessors:
            insts = pp(insts)
        # --
        original_len = len(insts)
        if conf.presample != 1.:
            insts, idxes = ZDataset.do_presample(
                insts, conf.presample, conf.presample_shuffle, conf.presample_reverse, (12345+conf.presample_seed))
            self._presample_indexes = idxes
        zlog(f"Read data {self.name}:{self.wset}[beat={self.sample_beta}] "
             f"from {full_path}(s={conf.presample}): {original_len}=>{len(insts)} instances.")
        return insts

    # --
    def set_insts(self, insts):
        self.insts = insts

    @property
    def gold_insts(self):
        if self._gold_insts is None:  # load when needed!!
            conf = self.conf
            _gold_insts = list(conf.R.get_reader(input_path=conf.get_gold_path()))
            for pp in self._preprocessors:
                _gold_insts = pp(_gold_insts)
            if self._presample_indexes is not None:
                _gold_insts = [_gold_insts[z] for z in self._presample_indexes]
            self._gold_insts = _gold_insts
            # --
            assert len(self._gold_insts) == len(self.insts)
            for gi, pi in zip(self._gold_insts, self.insts):
                gi._cache['inst_pred'] = pi  # for some special usage!
                pi._cache['inst_gold'] = gi  # for some special usage!
            # --
        return self._gold_insts

    @property
    def tasks(self):
        return self._tasks

    def write_insts(self, output_path: str = None):
        conf = self.conf
        # --
        # get output path name
        if output_path is None:
            sig = ''.join([(c if c!='/' else '_') for c in f"{self.name}_{self.wset}"])
            full_path = conf.get_output_path(default_sig=sig)
        else:  # override by outside!!
            full_path = output_path
        # --
        with conf.W.get_writer(output_path=full_path) as writer:
            writer.write_insts(self.insts)
            zlog(f"Write data {self.name}:{self.wset} to {full_path}: {len(self.insts)} instances.")
        # --

    @staticmethod
    def do_presample(insts: List, s: float, shuffle: bool, reverse: bool, seed: int):
        assert s>0
        if s < 1.:
            s = len(insts)*s
        s = max(1, int(s+0.99999))  # at least one
        # --
        ret_idxes = list(range(len(insts)))
        if reverse:
            ret_idxes = list(reversed(ret_idxes))
        if shuffle:
            # _gen = Random.get_generator('presample')
            _gen = Random.get_np_generator(seed)
            _gen.shuffle(ret_idxes)
        ret_idxes = ret_idxes[:s]
        return [insts[z] for z in ret_idxes], ret_idxes

    @staticmethod
    def join_datasets(conf: ZDatasetConf, name: str, wset: str, datasets: List['ZDataset']):
        ret = ZDataset(conf, name, wset, _no_load=True)  # make an empty one
        # note: simply directly set!
        insts = sum([d.insts for d in datasets], [])
        if wset == "train":  # note: can directly use themselves as gold!
            gold_insts = insts
        else:
            gold_insts = sum([d.gold_insts for d in datasets], [])
        zlog(f"Build jonit dataset {ret} with D={len(datasets)},inst={len(insts)}")
        ret.insts = insts
        ret._gold_insts = gold_insts
        return ret

    def yield_batches(self, external_batcher=None, loop=None):
        if loop is None:  # auto decide it by wset name
            loop = (self.wset == "train")
        batcher = self.conf.batcher.get_batcher() if external_batcher is None else external_batcher
        self.cur_batchers.append(batcher)
        for items in batcher.yield_batches(self.insts, loop=loop):
            yield InputBatch(items, self)
        assert self.cur_batchers[-1] is batcher
        self.cur_batchers.pop(-1)

# --
# b msp2/tasks/zmtl3/core/data:249
