### For Transfer Learning from SRL to EAE

Hi, this describes our implementation for our EMNLP-22 paper: "Transfer Learning from Semantic Role Labeling to Event Argument Extraction with Template-based Slot Querying".

Please refer to the paper for more details: [[paper]](https://aclanthology.org/2022.emnlp-main.169/)

### Repo

When we were carrying out our experiments for this work, we used the repo at this commit [`here`](https://github.com/zzsfornlp/zmsp/commit/e5da0ee640614b3420fd92bd2c704d060506f84f). In later versions of this repo, there may be slight changes (for example, default hyper-parameter change or hyper-parameter name change).

Clone this repo:

	git clone https://github.com/zzsfornlp/zmsp/ src

### Environment

Prepare the environment using conda:

	conda create -n p21 python=3.8
	conda activate p21
	conda install pytorch=1.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
	conda install numpy scipy cython pybind11 pandas pip
	pip install transformers==3.1.0

Please refer to [`conda_p21.yml`](./conda_p21.yml) for our specific conda environment and you can also specify conda environment using this.

Before running anything, make sure to export the src directory to your `$PYTHONPATH`:

    export PYTHONPATH=/your/path/to/src

### Data Preparation

There are some extra steps (extra data) to prepare before the final preparation:
- For **PropBank**: please follow [`prepPB.sh`](./prepPB.sh) to prepare PropBank data.
- For **FrameNet**: please follow [`prepFN.sh`](./prepFN.sh) to prepare FrameNet data.
- For **NomBank**: please follow [`prepNB.sh`](./prepNB.sh) to prepare NomBank data.
- For **QA**: please follow [`prepQA.sh`](./prepQA.sh) to prepare QA data.

Finally, for the event data, please first utilize `OneIE` to preprocess data, we utilize `oneie-v0.4.8` (http://blender.cs.illinois.edu/software/oneie/). After obtaining the json files with the OneIE formats, please further use this command to convert it into our own format (and use stanza to parse it), for example, for the ACE-dev set:

    python3 -m msp2.tasks.zmtl3.scripts.data.oneie2mine dev.oneie.json en.ace.dev.json
    python3 -m msp2.cli.annotate 'stanza' stanza_use_gpu:0 stanza_lang:en stanza_processors:tokenize,pos,lemma,depparse stanza_input_mode:tokenized ann_batch_size:1 input_path:en.ace.dev.json output_path:en.ace.dev.ud2.json

After all of these (probably takes several hours), you can get the data files prepared. For easier running, please put all the prepared data into one dir called `data` and put this data dir in a place where `../data` can be found according to the running dir. In the following, we assume that the data are available at the sub-dirs: `../data/data_pb`, `../data/data_fn`, `../data/data_nb`, `../data/data_qa`, `../data/data_evt`.

### Running

Please use [`go.py`](./go.py) for training:

	# create a new dir
	mkdir run; cd run;
    # make sure data are available
    ls -lh ../data/
	# copy go.py & examples.py
	cp /your/path/to/src//msp2/docs/srl2arg/{examples.py,go.py} .
	# use the example confs to training, for example, here we run the `train_srl` instance with GPU=0.
	python3 go.py tune_table_file:examples.py shuffle:0 do_loop:0 gpus:0 tune_name:train_srl

* Notice that in the above example running of `train_srl`, the program will run in the background in a new sub-folder called `train_srl_0`. By default, it will not print to the stdout, but we can check `train_srl_0/_log_*` for the loggings.

Here are more runnings and explanations for the confs with the example file:

#### Training a supervised EAE model:

To train a supervised EAE model, we need the corresponding EAE data files. For example, for training with ACE data, we assume that we have data files of `../data/data_evt/en.ace.{train,dev,test}.json`.

Run this for training & testing:

    python3 go.py tune_table_file:examples.py shuffle:0 do_loop:0 gpus:0 tune_name:train_super

#### Training an SRL model and test on EAE:

To train an SRL model, we need the corresponding SRL data files. Assume that we have prepared them at `../data/data_pb`, `../data/data_fn` and `../data/data_nb`.

Run this for training:

    python3 go.py tune_table_file:examples.py shuffle:0 do_loop:0 gpus:0 tune_name:train_srl

And run this for testing on ACE(dev) data:

    CUDA_VISIBLE_DEVICES=0 python3 -m msp2.tasks.zmtl3.main.test device:0 'conf_sbase:data:ace;task:arg' "model_load_name:run_train_srl_0/zmodel.best.m###DMarg0.qmod.emb_frame,DMarg0.qmod.emb_role" arg0.arg_mode:tpl arg0.mix_evt_ind:0.5 test0.input_dir:data/data_evt/ test0.group_files:en.ace.dev.json

#### Training a QA model and test on EAE:

To train a QA model, we need the corresponding QA data files. Assume that we have prepared them at `../data/data_qa`.

Run this for training:
    
    python3 go.py tune_table_file:examples.py shuffle:0 do_loop:0 gpus:0 tune_name:train_qa

And run this for testing on ACE(dev) data:

    CUDA_VISIBLE_DEVICES=0 python3 -m msp2.tasks.zmtl3.main.test device:0 "conf_sbase:data:ace;task:arg" "model_load_name:run_train_qa_0/zmodel.best.m###DMarg0.qmod.emb_frame,DMarg0.qmod.emb_role" arg0.arg_mode:mrc arg0.mix_evt_ind:0.5 test0.input_dir:data/data_evt/ test0.group_files:en.ace.dev.json

#### Training an augmented SRL model and test on EAE:

Similar to the training of SRL model, we need the SRL data files. Furthermore, we first utilize the above QA model to obtain (soft) labels for them (outputting to `../data/qadistill`):

    CCMD="python3 -m msp2.tasks.zmtl3.main.stream device:0 'conf_sbase:data:pbfn;task:arg' arg0.pred_store_scores:1 arg0.pred_no_change:1 arg0.extend_span:0 arg0.np_getter:fn 'arg0.filter_noncore:+spec,-*' arg0.build_oload:pbfn arg0.arg_mode:mrc"
    CCMD_Q1="$CCMD 'model_load_name:run_train_qa_0/zmodel.best.m###DMarg0.qmod.emb_frame,DMarg0.qmod.emb_role' arg0.mix_evt_ind:0.5"
    mkdir -p ../data/qadistill
    # predict
    for ff in ../data/data_pb/ewt.dev.conll.ud.json ../data/data_pb/ontonotes.train.conll.ud.json ../data/data_pb/ewt.train.conll.ud.json ../data/data_fn/parsed/fn17_exemplars.filtered.json ../data/data_nb/nb_f0.train.ud.json; do
      ff2=`basename $ff`
      CUDA_VISIBLE_DEVICES=0 $CCMD_Q1 stream_input:$ff stream_output:../data/qadistill/${ff2%.json}.q1.json
    done
    # get calibration tau with ewt-dev => obtaining 2.511 with our latest run
    python3 -m msp2.scripts.calibrate.ts2 ../data/qadistill/ewt.dev.conll.ud.q1.json

Then do training and testing as in previous:

    python3 go.py tune_table_file:examples.py shuffle:0 do_loop:0 gpus:0 tune_name:train_aug_srl
    CUDA_VISIBLE_DEVICES=0 python3 -m msp2.tasks.zmtl3.main.test device:0 'conf_sbase:data:ace;task:arg' "model_load_name:run_train_aug_srl_0/zmodel.best.m###DMarg0.qmod.emb_frame,DMarg0.qmod.emb_role" arg0.arg_mode:tpl arg0.mix_evt_ind:0.5 test0.input_dir:data/data_evt/ test0.group_files:en.ace.dev.json

#### More details on the scripts:

The running scripts are very similar to those in [`srl_cl`](../srl_cl/readme.md), they share the same code-base and the running conventions are very similar. Please refer to it (especially the last several paragraphs in its README) for more details.
