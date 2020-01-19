#

# hierachical labeling vocab
# the hierarchical labeling system
# todo(note): the labeling str format is L1.L2.L3..., each layer has only 26-characters

from typing import List, Tuple, Iterable, Dict
import numpy as np
from collections import defaultdict

from msp.utils import Conf, Random, zlog
from msp.data import Vocab, WordVectors
from msp.zext.seq_helper import DataPadder

from .utils import split_camel, LABEL2LEX_MAP

#
class HLabelIdx:
    def __init__(self, types: List, idxes: List = None):
        # todo(note): input idxes is readable only!
        self.types = []
        for i, z in enumerate(types):
            if z is not None:
                self.types.append(z)
            else:
                assert all(zz is None for zz in types[i:])  # once None, must all be None
                break
        self.idxes: Tuple = (None if idxes is None else tuple(idxes))

    def is_nil(self):
        return len(self.types)==0

    @staticmethod
    def construct_hlidx(label: str, layered: bool):
        if label is None:
            types = []
        else:
            if layered:
                types = [z for z in label.split(".") if len(z)>0]  # List[str]
            else:
                types = [label]
        return HLabelIdx(types)

    def pad_types(self, num_layer) -> Tuple:
        if len(self.types) >= num_layer:
            ret = self.types
        else:
            ret = self.types + [None] * (num_layer - len(self.types))
        return tuple(ret)

    def get_idx(self, layer):
        return self.idxes[layer]

    def __eq__(self, other):
        assert self.idxes is not None
        return self.idxes == other.idxes

    def __hash__(self):
        assert self.idxes is not None
        return self.idxes

    def __len__(self):
        return len(self.types)

    # directly using __str__ to get the original string label
    def __repr__(self):
        return ".".join(self.types)

# configuration for HLabelVocab and HLabelModel
class HLabelConf(Conf):
    def __init__(self):
        # part 1: layering
        self.layered = False  # whether split layers for the labels
        # part 2: associating types (how to get/combine one type's embedding)
        self.pool_split_camel = False
        self.pool_sharing = "nope"  # nope: no sharing, layered: sharing for each layer, shared: fully shared on lexicon

# vocab: for each layer: 0 as NIL, [1,n] as types, n+1 as UNK
# two parts of label embeddings:
# pools(flattened embed pool) -> layer-label(hierarchical structured one mapping to one or sum of the ones in the pool)
# todo(+N): can the link be soften? like with trainable linking parameters?
# =====
# todo(note): summary of this complex class's fields
class HLabelVocab:
    def __init__(self, name, conf: HLabelConf, keys: Dict[str, int], nil_as_zero=True):
        assert nil_as_zero, "Currently assume nil as zero for all layers"
        self.conf = conf
        self.name = name
        # from original vocab
        self.orig_counts = {k: v for k,v in keys.items()}
        keys = sorted(set(keys.keys()))  # for example, can be "vocab.trg_keys()"
        max_layer = 0
        # =====
        # part 1: layering
        v = {}
        keys = [None] + keys  # ad None as NIL
        for k in keys:
            cur_idx = HLabelIdx.construct_hlidx(k, conf.layered)
            max_layer = max(max_layer, len(cur_idx))
            v[k] = cur_idx
        # collect all the layered types and put idxes
        self.max_layer = max_layer
        self.layered_v = [{} for _ in range(max_layer)]  # key -> int-idx for each layer
        self.layered_k = [[] for _ in range(max_layer)]  # int-idx -> key for each layer
        self.layered_prei = [[] for _ in range(max_layer)]  # int-idx -> int-idx: idx of prefix in previous layer
        self.layered_hlidx = [[] for _ in range(max_layer)]  # int-idx -> hlidx for each layer
        for k in keys:
            cur_hidx = v[k]
            cur_types = cur_hidx.pad_types(max_layer)
            assert len(cur_types) == max_layer
            cur_idxes = []  # int-idxes for each layer
            # assign from 0 to max-layer
            for cur_layer_i in range(max_layer):
                cur_layered_v, cur_layered_k, cur_layered_prei, cur_layered_hlidx = \
                    self.layered_v[cur_layer_i], self.layered_k[cur_layer_i], \
                    self.layered_prei[cur_layer_i], self.layered_hlidx[cur_layer_i]
                # also put empty classes here
                for cur_layer_types in [cur_types[:cur_layer_i]+(None,), cur_types[:cur_layer_i+1]]:
                    if cur_layer_types not in cur_layered_v:
                        new_idx = len(cur_layered_k)
                        cur_layered_v[cur_layer_types] = new_idx
                        cur_layered_k.append(cur_layer_types)
                        cur_layered_prei.append(0 if cur_layer_i==0 else cur_idxes[-1])  # previous idx
                        cur_layered_hlidx.append(HLabelIdx(cur_layer_types, None))  # make a new hlidx, need to fill idxes later
                # put the actual idx
                cur_idxes.append(cur_layered_v[cur_types[:cur_layer_i+1]])
            cur_hidx.idxes = cur_idxes  # put the idxes (actually not useful here)
        self.nil_as_zero = nil_as_zero
        if nil_as_zero:
            assert all(z[0].is_nil() for z in self.layered_hlidx)  # make sure each layer's 0 is all-Nil
        # put the idxes for layered_hlidx
        self.v = {}
        for cur_layer_i in range(max_layer):
            cur_layered_hlidx = self.layered_hlidx[cur_layer_i]
            for one_hlidx in cur_layered_hlidx:
                one_types = one_hlidx.pad_types(max_layer)
                one_hlidx.idxes = [self.layered_v[i][one_types[:i+1]] for i in range(max_layer)]
                self.v[str(one_hlidx)] = one_hlidx  # todo(note): further layers will over-written previous ones
        self.nil_idx = self.v[""]
        # =====
        # (main) part 2: representation
        # link each type representation to the overall pool
        self.pools_v = {None: 0}  # NIL as 0
        self.pools_k = [None]
        self.pools_hint_lexicon = [[]]  # hit lexicon to look up in pre-trained embeddings： List[pool] of List[str]
        # links for each label-embeddings to the pool-embeddings: List(layer) of List(label) of List(idx-in-pool)
        self.layered_pool_links = [[] for _ in range(max_layer)]
        # masks indicating local-NIL(None)
        self.layered_pool_isnil = [[] for _ in range(max_layer)]
        for cur_layer_i in range(max_layer):
            cur_layered_pool_links = self.layered_pool_links[cur_layer_i]  # List(label) of List
            cur_layered_k = self.layered_k[cur_layer_i]  # List(full-key-tuple)
            cur_layered_pool_isnil = self.layered_pool_isnil[cur_layer_i]  # List[int]
            for one_k in cur_layered_k:
                one_k_final = one_k[cur_layer_i]  # use the final token at least for lexicon hint
                if one_k_final is None:
                    cur_layered_pool_links.append([0])  # todo(note): None is always zero
                    cur_layered_pool_isnil.append(1)
                    continue
                cur_layered_pool_isnil.append(0)
                # either splitting into pools or splitting for lexicon hint
                one_k_final_elems: List[str] = split_camel(one_k_final)
                # adding prefix according to the strategy
                if conf.pool_sharing == "nope":
                    one_prefix = f"L{cur_layer_i}-" + ".".join(one_k[:-1]) + "."
                elif conf.pool_sharing == "layered":
                    one_prefix = f"L{cur_layer_i}-"
                elif conf.pool_sharing == "shared":
                    one_prefix = ""
                else:
                    raise NotImplementedError(f"UNK pool-sharing strategy {conf.pool_sharing}!")
                # put in the pools (also two types of strategies)
                if conf.pool_split_camel:
                    # possibly multiple mappings to the pool, each pool-elem gets only one hint-lexicon
                    cur_layered_pool_links.append([])
                    for this_pool_key in one_k_final_elems:
                        this_pool_key1 = one_prefix + this_pool_key
                        if this_pool_key1 not in self.pools_v:
                            self.pools_v[this_pool_key1] = len(self.pools_k)
                            self.pools_k.append(this_pool_key1)
                            self.pools_hint_lexicon.append([self.get_lexicon_hint(this_pool_key)])
                        cur_layered_pool_links[-1].append(self.pools_v[this_pool_key1])
                else:
                    # only one mapping to the pool, each pool-elem can get multiple hint-lexicons
                    this_pool_key1 = one_prefix + one_k_final
                    if this_pool_key1 not in self.pools_v:
                        self.pools_v[this_pool_key1] = len(self.pools_k)
                        self.pools_k.append(this_pool_key1)
                        self.pools_hint_lexicon.append([self.get_lexicon_hint(z) for z in one_k_final_elems])
                    cur_layered_pool_links.append([self.pools_v[this_pool_key1]])
        assert self.pools_v[None] == 0, "Internal error!"
        # padding and masking for the links (in numpy)
        self.layered_pool_links_padded = []  # List[arr(#layered-label, #elem)]
        self.layered_pool_links_mask = []  # List[arr(...)]
        padder = DataPadder(2, mask_range=2)  # separately for each layer
        for cur_layer_i in range(max_layer):
            cur_arr, cur_mask = padder.pad(self.layered_pool_links[cur_layer_i])  # [each-sublabel, padded-max-elems]
            self.layered_pool_links_padded.append(cur_arr)
            self.layered_pool_links_mask.append(cur_mask)
        self.layered_prei = [np.asarray(z) for z in self.layered_prei]
        self.layered_pool_isnil = [np.asarray(z) for z in self.layered_pool_isnil]
        self.pool_init_vec = None
        #
        zlog(f"Build HLabelVocab {name} (max-layer={max_layer} from pools={len(self.pools_k)}): " + "; ".join([f"L{i}={len(self.layered_k[i])}" for i in range(max_layer)]))

    # set npvec to init HLNode
    def set_pool_init(self, npvec: np.ndarray):
        assert len(npvec) == len(self.pools_k)
        self.pool_init_vec = npvec

    # filter embeddings for init pool (similar to Vocab.filter_embeds)
    def filter_pembed(self, wv: WordVectors, init_nohit=0., scale=1.0, assert_all_hit=True, set_init=True):
        if init_nohit <= 0.:
            get_nohit = lambda s: np.zeros((s,), dtype=np.float32)
        else:
            get_nohit = lambda s: (Random.random_sample((s,)).astype(np.float32)-0.5) * (2*init_nohit)
        #
        ret = [np.zeros((wv.embed_size,), dtype=np.float32)]  # init NIL is zero
        record = defaultdict(int)
        for ws in self.pools_hint_lexicon[1:]:
            res = np.zeros((wv.embed_size,), dtype=np.float32)
            for w in ws:
                hit, norm_name, norm_w = wv.norm_until_hit(w)
                if hit:
                    value = np.asarray(wv.get_vec(norm_w, norm=False), dtype=np.float32)
                    record[norm_name] += 1
                else:
                    value = get_nohit(wv.embed_size)
                    record["no-hit"] += 1
                res += value
            ret.append(res)
        #
        assert not assert_all_hit or record["no-hit"]==0, f"Filter-embed error: assert all-hit but get no-hit of {record['no-hit']}"
        zlog(f"Filter pre-trained Pembed: {record}, no-hit is inited with {init_nohit}.")
        ret = np.asarray(ret, dtype=np.float32) * scale
        if set_init:
            self.set_pool_init(ret)
        return ret

    # =====
    # query
    def val2idx(self, item: str) -> HLabelIdx:
        if item is None:
            item = ""
        return self.v[item]

    def idx2val(self, idx: HLabelIdx) -> str:
        return str(idx)

    def get_hlidx(self, idx: int, eff_max_layer: int):
        return self.layered_hlidx[eff_max_layer-1][idx]

    def val2count(self, item: str) -> int:
        return self.orig_counts[item]

    # transform single unit into lexicon
    @staticmethod
    def get_lexicon_hint(key: str):
        k0 = key.lower()
        return LABEL2LEX_MAP.get(k0, k0)
