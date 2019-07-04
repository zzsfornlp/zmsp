### For the CrossLingual SelfAtt+Graph Parser

Hi, this describes a Re-Implementation of the `SelfAtt+Graph Parser` for our NAACL-2019 paper: "On Difficulties of Cross-Lingual Transfer with Order Differences: A Case Study on Dependency Parsing".

The original implementation can be found [[here]](https://github.com/uclanlp/CrossLingualDepParser), which contains both Graph-Parser and TopDownStackPointer Parser and is the ones with which we performed our experiments in the paper. Here, we implement the Graph-Parser.

Please refer to the paper for more details: [[paper]](https://www.aclweb.org/anthology/N19-1253) [[bib]](https://aclweb.org/anthology/papers/N/N19/N19-1253.bib) [[arxiv]](https://arxiv.org/abs/1811.00570)

### Repo

Please use the repo at this commit [`here`](https://github.com/zzsfornlp/zmsp/commit/ecf5dc2d87abed430f52f154c16c42e9c809c844), since in later versions of this repo, there may be slight changes (for example, default hyper-parameter change or hyper-parameter name change) so that the old scripts might need modifications to be runnable.

### Environment

As those of the main `msp` package:

	python>=3.6
	dependencies: pytorch>=0.4.1, numpy, scipy, gensim, cython

### Data Preparation

Please refer to the scripts in DIR [`scripts/dpar/prep/`](../scripts/dpar/prep/) for data preparation. Generally, we need data files and pre-trained aligned embeddings files. Data formats will be standard CoNLL-U format and we use fasttext embeddings aligned with this [toolkit](https://github.com/Babylonpartners/fastText_multilingual).

*Note that the data format is original CoNLL-U format which is slightly different than the ones adopted in our original implementation, and the difference is the column position of UPOS.*

### Running

Please refer to the script of [`scripts/dpar/zrun0/zrun.sh`](../scripts/dpar/zrun0/zrun.sh) for an example of full running.

*Note that the paths (of data files and embeddings files) in that script may need to be modified to be runnable).*
