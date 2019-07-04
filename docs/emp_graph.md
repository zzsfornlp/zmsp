### For the Empirical Graph Parser

Hi, this describes our implementation for our ACL-2019 paper: "An Empirical Investigation of Structured Output Modeling for Graph-based Neural Dependency Parsing". 

Please refer to the paper for more details: [[paper]](TODO) [[bib]](TODO)

### Repo

When we were carrying out our experiments for this work, we used the repo at this commit [`here`](https://github.com/zzsfornlp/zmsp/commit/ecf5dc2d87abed430f52f154c16c42e9c809c844). In later versions of this repo, there may be slight changes (for example, default hyper-parameter change or hyper-parameter name change).

### Environment

As those of the main `msp` package:

	python>=3.6
	dependencies: pytorch>=0.4.1, numpy, scipy, gensim, cython

### Data Preparation

Please refer to the scripts in DIR [`scripts/dpar/prep/`](../scripts/dpar/prep/) for data preparation. Generally, we need data files and pre-trained embeddings files. Please refer to `ptb.py` for PTB, `ctb.py` for CTB and `zprep_ud23.py` for UDv2.3. Alternatively, feel free to prepare the data by yourself, as long as it is in CONLL-U format (as in UD files). (We use the POS tags at column 3 for all data, including PTB and CTB, so be sure to put them at the right places.)

### Compiling (for certain functionalities)

To include full functionalities of different normalization methods, compiling (with cython) is required. This can be done by simply running [`tasks/compile.sh`](../tasks/compile.sh).

	cd tasks; bash compile.sh; cd ..

### Running

We have 4 normalization modes: `single/local/unproj/proj`, and 2 kinds of losses: `prob/hinge`.

	# =====
	# base options
	base_opt="conf_output:_conf partype:graph init_from_pretrain:1 lrate.init_val:0.001 drop_embed:0.33 dropmd_embed:0.33 singleton_thr:2 fix_drop:1 max_epochs:300"
	# possible methods
	single_m="dec_algorithm:unproj output_normalizing:single loss_single_sample:2."
	local_m="dec_algorithm:unproj output_normalizing:local"
	global_unproj_m="dec_algorithm:unproj output_normalizing:global"
	global_proj_m="dec_algorithm:proj output_normalizing:global"
	# possible losses
	hinge_loss="loss_function:hinge margin.init_val:2.0"
	prob_loss="loss_function:prob margin.init_val:0."

Please provide the corresponding cmd options for running, including the `normalization mode`, `loss function`, and correct `train/dev/test/embed` paths. For example, if we want to run with `unproj`-global normalization and `prob` loss function. Then, we can run:

	# =====
	# specify paths
	SRC_DIR=../zmsp/
	method=${global_unproj_m}
	loss=${prob_loss}
	# further specify `train/dev/test/embed` paths
	# ...
	# =====
	# running
	# on GPU 0
	CUDA_VISIBLE_DEVICES=0 PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.train train:${DATA_TRAIN} dev:${DATA_DEV} test:${DATA_TEST} pretrain_file:${DATA_EMBED} device:0 ${base_opt} ${method} ${loss}
	# on CPU
	CUDA_VISIBLE_DEVICES= PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.train train:${DATA_TRAIN} dev:${DATA_DEV} test:${DATA_TEST} pretrain_file:${DATA_EMBED} device:-1 ${base_opt} ${method} ${loss}

Please refer to the script of [`scripts/dpar/zrun1/ztrain.sh`](../scripts/dpar/zrun1/ztrain.sh) for an example of full running.
