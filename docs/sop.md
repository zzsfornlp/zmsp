### For the Some Other Parsers (sop)

Hi, this describes our implementation for some other (higher-order) parsers, including higher-order graph parsers with approximate decoding algorithms and a generalized easy-first transition based parser.

### Repo

When we were carrying out our experiments, we used the repo at the commit named `Add some other parsers.` (for main codes) and its following `Update scripts.` (for scripts). In later versions of this repo, there may be slight changes (for example, default hyper-parameter change or hyper-parameter name change).

### Environment

As those of the main `msp` package:

	python>=3.6
	dependencies: pytorch>=0.4.1, numpy, scipy, gensim, cython, pybind11

### Data Preparation

Data preparation is similar to those in [`emp_graph.md`](./emp_graph.md), please refer it for data preparation.

### Compiling (for certain functionalities)

To include full functionalities (especially high-order approximate decoding), compiling (with cython) is required. This can be done by simply running [`tasks/compile.sh`](../tasks/compile.sh).

	cd tasks; bash compile.sh; cd ..

### Running

Please refer to the scripts in [`scripts/dpar/zrun2/*`](../scripts/dpar/zrun2) for examples of running. Here, `train_ef.sh` is for training easy-first parser, `train_g1.sh` is for training basic first-order graph parser, which is needed as the base of higher-order graph parsers and `train_g2.sh` is for training higher-order graph parsers.

(Note: some paths might need to be changed for actual running)

(Note: to train `g2`, we also need to use [`score_go.py`](../tasks/zdpar/main/score_go.py) to pre-compute first-order scores for pruning for higher-order models, check this python script for more details.)

(Note: the script [`scripts/dpar/zrun2/runm.py`](../scripts/dpar/zrun2/runm.py) also provides a reference for multi-lingual training for some of these parsers.)
