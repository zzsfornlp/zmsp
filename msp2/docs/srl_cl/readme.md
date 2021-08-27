### For Using Syntactic Supervision for Cross-lingual SRL

Hi, this describes our implementation for our paper: "On the Benefit of Syntactic Supervision for Cross-lingual Transfer in Semantic Role Labeling".

Please refer to the paper for more details: [[paper]](TODO) [[bib]](TODO)

### Repo

When we were carrying out our experiments for this work, we used the repo at this commit [`here`](TODO). In later versions of this repo, there may be slight changes (for example, default hyper-parameter change or hyper-parameter name change).

Clone this repo:

	git clone https://github.com/zzsfornlp/zmsp/ src

### Environment

Prepare the environment using conda:

	conda create -n p20 python=3.8
	conda activate p20
	conda install pytorch=1.5.0 -c pytorch
	conda install numpy scipy cython pybind11 pandas pip
	pip install transformers==3.1.0

Please refer to [`conda_p20.yml`](./conda_p20.yml) for our specific conda environment and you can also specify conda environment using this.

### Data Preparation

There are three extra steps (extra data) to prepare before the final preparation:

- For **PATB_INT**: follow [`UD_Arabic-NYUAD`](https://github.com/UniversalDependencies/UD_Arabic-NYUAD/tree/r2.7#data) to prepare PATB data.
- For **EWT_FOLDER**: use this script [`prepP.sh`](../../scripts/srl_pb/prepP.sh) to prepare EWT PropBank dataset.
- For **CONLL12D**: Please first use [`srl_span/prep.sh`](../srl_span/prep.sh) to prepare OntoNotes data for further use here.
- For **CONLL09**, **PTB3**, **CTB6**, they should point to the corresponding LDC's dataset folders.
- In addition, `udapy` is also needed, please install it following [here](https://github.com/udapi/udapi-python).
- In addition, `stanza` is also needed, please install it following [here](https://stanfordnlp.github.io/stanza/). And further download stanza models for `[en,zh,ar]`.

Then, please use [`prep.sh`](./prep.sh) to prepare data, refer to it for more details.

For example, run it with:

	export PYTHONPATH=/your/path/to/src
	export PATB_INT=/your/path/to/PATB/integrated
	export EWT_FOLDER=/your/path/to/processed_EWT 
	export CONLL12D=/your/path/to/data/pb/conll12d 
	export CONLL09=/your/path/to/CONLL09 
	export PTB3=/your/path/to/treebank_3 
	export CTB6=/your/path/to/ctb_v6 
	bash prep.sh

After all of these (probably takes several hours), you can get the data files prepared in `data/ud/cl{0,1,2,3}/*`, corresponding to the data for different experiments. (cl0=EWT/UPB, cl1=EWT/FiPB, cl2=OntoNotes, cl3=CONLL09)

### Running

Please use [`go.py`](./go.py) for running:

	# create a new dir
	mkdir run; cd run;
	# ln data
	ln -s /your/path/to/prepared//data/ud/ ./ud/
	# copy go.py & examples.py
	cp /your/path/to/src//msp2/docs/srl_cl/{examples.py,go.py} .
	# use the example confs to train/test, for example, here we run the `cl0_base_xlmr` instance with GPU=0.
	python3 -m pdb go.py settings:clsrl0 tune_table_file:examples.py shuffle:0 gpus:0 tune_name:cl0_base_xlmr

* Notice that in the above example running of `cl0_base_xlmr`, the program will run in the background in a new sub-folder called `run_cl0_base_xlmr_0`. By default, it will not print to the stdout, but we can check `run_cl0_base_xlmr_0/_log_*` for the loggings.
* Notice that we need different `settings:??` for running different experiments(datasets), corresponding to the data folder names, they are: `clsrl0,clsrl1,clsrl2,clsrl3`. And for `clsrl2` and `clsrl3`, we need an additional one `cl2_lang:??` or `cl3_lang:??` to specify the target language.

Here are more runnings and explanations for the confs with the example file:

	# `cl0_base_xlmr`: 
	# Running the UPB zero-resource experiment (cl0) with the baseline model trained on English-SRL and directly tested on all languages.
	python3 -m pdb go.py settings:clsrl0 tune_table_file:examples.py shuffle:0 gpus:0 tune_name:cl0_base_xlmr

	# `cl0_syn_xlmr`: 
	# Running the UPB zero-resource experiment (cl0) with the baseline model trained on English-SRL and as well as syntactic trees on all languages.
	python3 -m pdb go.py settings:clsrl0 tune_table_file:examples.py shuffle:0 gpus:0 tune_name:cl0_syn_xlmr

	# `cl1_both`: 
	# Running the EWT/FiPB experiment (cl1) with English + 1k Finnish-SRL.
	python3 -m pdb go.py settings:clsrl1 tune_table_file:examples.py shuffle:0 gpus:0 tune_name:cl1_both

	# `cl1_syn`: 
	# Running the EWT/FiPB experiment (cl1) with English + 1k Finnish-SRL, as well as syntactic trees.
	python3 -m pdb go.py settings:clsrl1 tune_table_file:examples.py shuffle:0 gpus:0 tune_name:cl1_syn

	# `cl2_zh_both`: 
	# Running the OntoNotes experiment (cl2) with English + 1k Chinese SRL.
	python3 -m pdb go.py settings:clsrl2 tune_table_file:examples.py shuffle:0 gpus:0 tune_name:cl2_zh_both cl2_lang:zh

	# `cl2_zh_syn`: 
	# Running the OntoNotes experiment (cl2) with English + 1k Chinese SRL, as well as syntactic trees.
	python3 -m pdb go.py settings:clsrl2 tune_table_file:examples.py shuffle:0 gpus:0 tune_name:cl2_zh_syn cl2_lang:zh

	# `cl3_es_both`: 
	# Running the CoNLL09 experiment (cl3) with English + 1k Spanish SRL.
	python3 -m pdb go.py settings:clsrl3 tune_table_file:examples.py shuffle:0 gpus:0 tune_name:cl3_es_both cl3_lang:es

	# `cl3_es_syn`: 
	# Running the CoNLL09 experiment (cl3) with English + 1k Spanish SRL, as well as syntactic trees.
	python3 -m pdb go.py settings:clsrl3 tune_table_file:examples.py shuffle:0 gpus:0 tune_name:cl3_es_syn cl3_lang:es

Alternatively, if not using the configuration files, we can also concat all the configuration strings by using `train_extras:...` or `test_extras:...`, for example, still running the `cl0_base_xlmr` example. In this case, the running dir will directly be the current one and the loggings will be printed to stderr:

	# directly run one
	python3 go.py settings:clsrl0 gpus:0 'train_extras:bert_model:xlm-roberta-base reg0.mod_trg:Menc.bert reg0.reg_method:update reg0.l2_reg:1'

Please refer to the running and example files for more details, here are some brief explanations:

- We have a `msp2.tasks.zmtl2.drive.data_center.DataCenter` which contains various groups of train/dev/test datasets.
- Each group, `train*/dev*/test*`, will be configurated by a `msp2.tasks.zmtl2.core.data.ZDatasetConf` item, whose `group_files` indicate the input file paths of this dataset group.
- For example, when specifying `train1.group_files:en.json,zh.json`, we assign these two data files to this group.
- Each group also have: `group_tasks` which specifies the tasks (we use `udep` for dependency parsing and `pb1` for propbank SRL parsing) to be performed for this group of datasets, `input_folder` which is the common folder of the files in this group (for conveniency), `presample` denotes how many instances we want to down-sample for a dataset.
- By default, each group and each dataset inside a group will be sampled uniformly, the ratio can be changed with `group_sample_rate` and `group_sample_alpha`.
- There are other configurations for the datasets, please refer to the class `ZDatasetConf` for more details.
- Generally, by specifying the input files and tasks of each `ZDataset` group, we can control what to train and test.
