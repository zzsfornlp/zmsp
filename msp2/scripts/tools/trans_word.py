#

# translate words
# bilingual dictionary induction with pre-trained and aligned word vectors
# adopted from "fastText/alignment/*"

# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import collections
import argparse
import json
import time
import os
import numpy as np
import torch

# =====
# utils.py

def load_vectors(fname: str, maxload: int, norm=True, center=False, verbose=True):
    if verbose:
        print("Loading vectors from %s" % fname)
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    if maxload > 0:
        n = min(n, maxload)
    x = np.zeros([n, d])
    words = []
    for i, line in enumerate(fin):
        if i >= n:
            break
        tokens = line.rstrip().split(' ')
        words.append(tokens[0])
        v = np.array(tokens[1:], dtype=float)
        x[i, :] = v
    if norm:
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    if center:
        x -= x.mean(axis=0)[np.newaxis, :]
        x /= np.linalg.norm(x, axis=1)[:, np.newaxis] + 1e-8
    if verbose:
        print("%d word vectors loaded" % (len(words)))
    return words, x

def idx(words):
    w2i = {}
    for i, w in enumerate(words):
        if w not in w2i:
            w2i[w] = i
    return w2i

def load_input(filename, words_src, verbose=True):
    f = io.open(filename, 'r', encoding='utf-8')
    idx_src = idx(words_src)
    all_words = []
    all_idxes, hit_idxes = [], []
    for line in f:
        # one_word, = line.split()
        one_word = line.strip()
        one_idx = idx_src.get(one_word, None)
        all_words.append(one_word)
        all_idxes.append(one_idx)
        if one_idx is not None:
            hit_idxes.append(one_idx)
    if verbose:
        coverage = len(hit_idxes) / float(len(all_idxes))
        print(f"Coverage of source vocab: {len(hit_idxes)}/{len(all_idxes)}={coverage}")
    return all_words, all_idxes, hit_idxes  # List[int]

def find_by_csls(input_idxes, x_src, x_tgt, topk: int, reverse_lookup: bool, k=10, bsz=1024, device=-1):
    print(f"{time.ctime()} find_by_csls: {len(input_idxes)} （{len(x_src)} => {len(x_tgt)})")
    _device = torch.device('cpu') if device<0 else torch.device(device)
    x_src, x_tgt = torch.as_tensor(x_src, device=_device), torch.as_tensor(x_tgt, device=_device)
    input_idxes = torch.as_tensor(input_idxes, device=_device)
    # =====
    print(f"{time.ctime()} Start sc2")
    all_sc2 = []
    for i in range(0, x_tgt.shape[0], bsz):
        j = min(i + bsz, x_tgt.shape[0])
        sc_batch = torch.matmul(x_tgt[i:j, :], x_src.T)
        all_sc2.append(sc_batch.topk(k, -1)[0].mean(-1))
    sc2 = torch.cat(all_sc2, 0)  # [tgt_size]
    print(f"{time.ctime()} Finish sc2")
    # =====
    all_results = []
    for bidx in range(0, len(input_idxes), bsz):
        input_idxes_slice = input_idxes[bidx:bidx+bsz]
        # csls
        sr = x_src[input_idxes_slice]  # [bs, D]
        sc = torch.matmul(sr, x_tgt.T)  # [bs, tgt_size]
        similarities = 2 * sc - sc2
        cur_res = similarities.topk(topk, -1)[1]  # [bs, topk]
        all_results.append(cur_res)
        print(f"-- {time.ctime()} finish batch: {len(all_results)}")
    results = torch.cat(all_results, 0).cpu().numpy()  # [len, topk]
    if reverse_lookup:  # from target back to source
        reverse_results, _ = find_by_csls(results.flatten(), x_tgt, x_src, topk, False, k, bsz, device=device)
        reverse_results = reverse_results.reshape(results.shape + (-1,))
    else:
        reverse_results = None
    return results, reverse_results

# =====
def main():
    # arg
    parser = argparse.ArgumentParser(description='Evaluation of word alignment')
    parser.add_argument("--src_emb", type=str, default='', help="Load source embeddings")
    parser.add_argument("--tgt_emb", type=str, default='', help="Load target embeddings")
    parser.add_argument('--center', action='store_true', help='whether to center embeddings or not')
    parser.add_argument("--maxload", type=int, default=300000)
    parser.add_argument("-i", "--input", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-k", "--topk", type=int, default=5)
    parser.add_argument("--ignore_no_hit", type=int, default=1)
    parser.add_argument("--check_backward", type=int, default=0)
    parser.add_argument("--output_format", type=str, default="plain")
    parser.add_argument("--device", type=int, default=-1)  # -1 means cpu
    params = parser.parse_args()
    # eval
    print(f"Translate words: {params.input} => {params.output}")
    words_tgt, x_tgt = load_vectors(params.tgt_emb, maxload=params.maxload, center=params.center)
    words_src, x_src = load_vectors(params.src_emb, maxload=params.maxload, center=params.center)
    input_all_words, input_all_idxes, input_hit_idxes = load_input(params.input, words_src)
    output_trg_topk, output_trg_reverse_topk = find_by_csls(
        input_hit_idxes, x_src, x_tgt, params.topk, params.check_backward, device=params.device)  # [ilen, topk], [ilen, topk, topk]
    # decide the translations
    results = []
    idx_hit = 0
    cc = collections.Counter()
    for w, x in zip(input_all_words, input_hit_idxes):
        cc['all'] += 1
        if x is None:
            cc['src0'] += 1
            results.append((w, None))
        else:
            assert x == input_hit_idxes[idx_hit]
            _tgt_words = []
            if output_trg_reverse_topk is not None:
                for z, r_z in zip(output_trg_topk[idx_hit], output_trg_reverse_topk[idx_hit]):
                    if x in r_z:  # hit in both directions!
                        _tgt_words.append(words_tgt[z])
            else:
                _tgt_words = [words_tgt[z] for z in output_trg_topk[idx_hit]]
            cc[f'trg={len(_tgt_words)}'] += 1
            if len(_tgt_words) > 0:
                cc['all_finalH'] += 1  # final hit
            results.append((w, _tgt_words))
            idx_hit += 1
    print(f"Final results: {cc}")
    # --
    if params.ignore_no_hit:
        results = [(a,b) for a,b in results if b is not None and len(b)>0]
    with open(params.output, 'w') as fd:
        if params.output_format == 'plain':
            for a, b in results:
                if len(b) > 0:
                    fd.write(f"{a}\t{b[0]}\n")
        else:
            fd.write(json.dumps(results))
    # --

# --
if __name__ == '__main__':
    main()

# go
"""
# example as debug: en-es
wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.en.align.vec
wget https://dl.fbaipublicfiles.com/fasttext/vectors-aligned/wiki.es.align.vec
python3 -m msp2.scripts.tools.trans_word --src_emb wiki.es.align.vec --tgt_emb wiki.en.align.vec -i IN -o OUT --maxload 1000
"""
