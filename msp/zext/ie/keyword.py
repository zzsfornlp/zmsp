#

# from gensim.models import TfidfModel
# with TfidfModel as the reference

import glob
from typing import Dict, List, Tuple
import numpy as np
# from multiprocessing import Process
# from threading import Thread, Lock
from multiprocessing.pool import ThreadPool, Pool
from msp.utils import Conf, zopen, Helper, PickleRW, zlog

#
class KeyWordConf(Conf):
    def __init__(self):
        # how to build
        self.build_files = []
        self.build_num_core = 4
        self.save_file = ""
        self.load_file = ""

class KeyWordModel:
    def __init__(self, conf: KeyWordConf, num_doc=0, w2f=None, w2d=None):
        self.conf = conf
        # -----
        self.num_doc = num_doc
        self.w2f = {} if w2f is None else w2f  # word -> freq
        self.w2d = {} if w2d is None else w2d  # word -> num-doc (dfs)
        # -----
        self._calc()
        zlog(f"Create(or Load) {self}")

    #
    def __repr__(self):
        return f"KeyWordModel: #doc={self.num_doc}, #wordtype={len(self.w2f)}"

    #
    def get_rank(self, w):
        RANK_INF = len(self.w2i)
        return self.w2i.get(w, RANK_INF)

    #
    def _calc(self):
        self.i2w = sorted(self.w2f.keys(), key=lambda x: (-self.w2f[x], x))  # rank by freq
        self.w2i = {w:i for i,w in enumerate(self.i2w)}  # reversely word -> rank

    # save and load
    def save(self, f):
        PickleRW.to_file([self.num_doc, self.w2f, self.w2d], f)
        zlog(f"Save {self}")

    @staticmethod
    def load(f, conf=None):
        if conf is None:
            conf = KeyWordConf()
        num_doc, w2f, w2d = PickleRW.from_file(f)
        m = KeyWordModel(conf, num_doc, w2f, w2d)
        return m

    @staticmethod
    def collect_freqs(file):
        zlog(f"Starting dealing with {file}")
        MIN_TOK_PER_DOC = 100  # minimum token per doc
        MIN_PARA_PER_DOC = 5  # minimum line-num (paragraph)
        word2info = {}  # str -> [count, doc-count]
        num_doc = 0
        with zopen(file) as fd:
            docs = fd.read().split("\n\n")
            for one_doc in docs:
                tokens = one_doc.split()  # space or newline
                if len(tokens)>=MIN_TOK_PER_DOC and len(one_doc.split("\n"))>=MIN_PARA_PER_DOC:
                    num_doc += 1
                    # first raw counts
                    for t in tokens:
                        t = str.lower(t)  # todo(note): lowercase!!
                        if t not in word2info:
                            word2info[t] = [0, 0]
                        word2info[t][0] += 1
                    # then doc counts (must be there)
                    for t in set(tokens):
                        t = str.lower(t)
                        word2info[t][1] += 1
        return num_doc, word2info

    @staticmethod
    def build(conf: KeyWordConf):
        # collect all files
        all_build_files = []
        for one in conf.build_files:
            all_build_files.extend(glob.glob(one))
        # using multiprocessing
        pool = Pool(conf.build_num_core)
        # pool = ThreadPool(conf.build_num_core)
        results = pool.map(KeyWordModel.collect_freqs, all_build_files)
        pool.close()
        pool.join()
        # then merge all together
        word2info = {}  # str -> [count, doc-count]
        num_doc = 0
        for one_num_doc, one_w2i in results:
            num_doc += one_num_doc
            for k, info in one_w2i.items():
                if k not in word2info:
                    word2info[k] = [0, 0]
                word2info[k][0] += info[0]
                word2info[k][1] += info[1]
        # build the model
        m = KeyWordModel(conf, num_doc=num_doc, w2f={w:z[0] for w,z in word2info.items()},
                         w2d={w:z[1] for w,z in word2info.items()})
        return m

    # =====
    # sort and return range ([start, end)) for "the freq of freq"
    def _freq2range(self, freqs: List[int]) -> List[Tuple[int, int]]:
        sorting_items = [(f,i) for i,f in enumerate(freqs)]
        sorting_items.sort(reverse=True)
        ranges = [None for _ in freqs]
        prev_v, prev_idxes = None, []
        cur_rank = 0
        for cur_v, cur_idx in sorting_items+[(None, None)]:
            if cur_v != prev_v:
                # close previous ones
                one_range = (cur_rank-len(prev_idxes), cur_rank)
                for one_idx in prev_idxes:
                    assert ranges[one_idx] is None
                    ranges[one_idx] = one_range
                prev_idxes.clear()
            prev_v = cur_v
            prev_idxes.append(cur_idx)
            cur_rank += 1
        assert all(x is not None for x in ranges)
        return ranges

    def extract(self, tokens: List[str], n_tf='n', n_df='t'):
        # get local vocabs
        local_w2f = {}
        for t in tokens:
            t = str.lower(t)  # todo(note): lowercase!!
            local_w2f[t] = local_w2f.get(t, 0.) + 1
        # get tf/idf
        f_tf = {
            "n": lambda tf: tf,
            "l": lambda tf: 1+np.log(tf)/np.log(2),
            "a": lambda tf: 0.5 + (0.5 * tf / tf.max(axis=0)),
            "b": lambda tf: tf.astype('bool').astype('int'),
            "L": lambda tf: (1 + np.log(tf) / np.log(2)) / (1 + np.log(tf.mean(axis=0) / np.log(2))),
        }[n_tf]
        totaldocs = self.num_doc
        f_idf = {
            "n": lambda df: df,
            "t": lambda df: np.log(1.0 * totaldocs / df) / np.log(2),
            "p": lambda df: np.log((1.0 * totaldocs - df) / df) / np.log(2),
        }[n_df]
        # tf-idf
        local_words = sorted(local_w2f.keys(), key=lambda x: (-local_w2f[x], x))
        cur_tf = f_tf(np.array([local_w2f[w] for w in local_words]))
        cur_idf = f_idf(np.array([self.w2d.get(w, 0)+1. for w in local_words]))  # +1 smoothing
        cur_tf_idf = cur_tf * cur_idf
        # comparing the rankings
        local_freqs = [local_w2f[x] for x in local_words]
        global_freqs = [self.w2f.get(x, 0) for x in local_words]
        local_ranges = self._freq2range(local_freqs)
        global_ranges = self._freq2range(global_freqs)
        # ranking range center jump forward
        cur_range_jump = [(r2[0]+r2[1]-r1[0]-r1[1])/2. for r1,r2 in zip(local_ranges, global_ranges)]
        # return local_words, global_rank, cur_tf, cur_tf_idf, cur_range_jump
        global_ranks = [self.get_rank(x) for x in local_words]
        ret = [(w,fr,gr,s1,s2) for w,fr,gr,s1,s2 in zip(local_words, global_ranks, cur_tf, cur_tf_idf, cur_range_jump)]
        return ret

if __name__ == '__main__':
    import sys
    conf = KeyWordConf()
    conf.update_from_args(sys.argv[1:])
    if len(conf.build_files)>0:
        m = KeyWordModel.build(conf)
        if conf.save_file:
            m.save(conf.save_file)
    else:
        m = KeyWordModel.load(conf.load_file, conf)
        # test it
        testing_file = "1993.01.tok"
        with zopen(testing_file) as fd:
            docs = fd.read().split("\n\n")
            for one_doc in docs:
                tokens = one_doc.split()  # space or newline
                one_result = m.extract(tokens)
                res_sort1 = sorted(one_result, key=lambda x: -x[-2])
                res_sort2 = sorted(one_result, key=lambda x: -x[-1])
                res_sort0 = sorted(one_result, key=lambda x: -x[-3])
                zzzzz = 0

# PYTHONPATH=../src/ python3 k2.py build_files:./*.tok build_num_core:10 save_file:./nyt.voc.pic
# PYTHONPATH=../src/ python3 -m pdb k2.py build_files:[] load_file:./nyt.voc.pic
