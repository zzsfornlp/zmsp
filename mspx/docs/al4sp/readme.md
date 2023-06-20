### Active Learning for Structured Prediction

Hi, this describes our implementation for our work of active learning for structured prediction.


Clone this repo:

	git clone https://github.com/zzsfornlp/zmsp/ src

### Environment

Prepare the environment using conda:

    conda create -n p22 python=3.8
    conda install numpy scipy cython pybind11 pandas pip nltk
    pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    pip install transformers==4.8.2 datasets
    pip install stanza
    pip install -U scikit-learn

Please refer to [`conda_p22.yml`](./conda_p22.yml) for our specific conda environment and you can also specify conda environment using this.

Before running anything, make sure to export the src directory to your `$PYTHONPATH`:

    export PYTHONPATH=/your/path/to/src

### Data Preparation

#### Prepare English-EWT from UD

Data files will be `data/en_ewt.{train,dev,test}.json`.

    mkdir -p data
    wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4758/ud-treebanks-v2.10.tgz
    tar -zxvf ud-treebanks-v2.10.tgz
    for ccl in en_ewt; do
      for wset in train dev test; do
        python3 -m mspx.cli.change_format "R.input_path:ud-treebanks-v2.10/UD_*/${ccl}-ud-${wset}.conllu" R.input_format:conllu W.output_path:data/${ccl}.${wset}.json
      done
    done

#### Prepare CoNLL-03 English NER

Transform from the CoNLL format (`${CONLL_ORIG_FOLDER}/en.{train,dev,test}.txt`) into our json format, the final data files will be `data/en.{train,dev,test}.json`.

    mkdir -p data/
    for f in ${CONLL_ORIG_FOLDER}/*.txt; do
      python3 -m mspx.scripts.data.ner.prep_conll input_path:$f output_path:${f%.txt}.json
    done

### Running

#### Run NER

    # run in a new dir
    mkdir run
    cd run
    # build vocab
    dd=../vocabs/ner_en/
    mkdir -p $dd
    python3 -m mspx.tasks.zext.main vocab_save_dir:$dd log_file:$dd/_log conf_sbase:bert_name:roberta-base train0.group_files:__data/en.*.json fs:build
    # common arguments
    COMMON_ARGS="log_stderr:1 al_task:zext frame_cate:ef al_task.name:ext0 setup_dataU:__data/ner/data/en.train.ud2.json setup_dataD:__data/ner/data/en.dev.json specs_base:vocab_load_dir:__vocabs/ner_en/ stop:14 budget:4000 sampler.sample_s:4000 query_selv2:1 query_i0_partial:0 strg_bias:1. train_st:2 query_use_t0:1 sel_seed:0 sampler.seed:0"
    COMMON_ARGS2="specs_train:conf_sbase:bert_name:roberta-base model_save_suffix_curr: dev0.group_files:__al.dev.json test0.group_files:__data/ner/data/en.dev.json,__data/ner/data/en.test.json,__al.dev.json train0.inst_f:sentF seed0:0 msp_seed:0 ext0.train_cons:1 ext0.pred_cons:1"
    # run with FA (full annotation)
    python3 -m mspx.tools.al.main ${COMMON_ARGS} "${COMMON_ARGS2}" query_partial:0
    # or Run with PA (partial annotation)
    python3 -m mspx.tools.al.main ${COMMON_ARGS} "${COMMON_ARGS2}" 'specs_trainST:train0.group_files:_train.json train0.info:strgR:1' train_st:3 specs_trainST1:train1.group_files: selv2_arM:0

#### Run DPAR

    # run in a new dir
    mkdir run
    cd run
    # build vocab
    dd=../vocabs/dpar_en/
    mkdir -p $dd
    python3 -m mspx.tasks.zdpar.main vocab_save_dir:$dd log_file:$dd/_log conf_sbase:bert_name:roberta-base train0.group_files:__data/en_ewt.*.json fs:build
    # common arguments
    COMMON_ARGS="log_stderr:1 al_task:zdpar al_task.name:dpar0 setup_dataU:__data/ud/data/en_ewt.train.json setup_dataD:__data/ud/data/en_ewt.dev.json specs_base:vocab_load_dir:__vocabs/dpar_en/ stop:19 budget:4000 sampler.sample_s:4000 query_selv2:1 query_i0_partial:0 strg_bias:1. train_st:2 query_use_t0:1 sel_seed:0 sampler.seed:0"
    COMMON_ARGS2="specs_train:conf_sbase:bert_name:roberta-base model_save_suffix_curr: dev0.group_files:__al.dev.json test0.group_files:__data/ud/data/en_ewt.dev.json,__data/ud/data/en_ewt.test.json,__al.dev.json train0.inst_f:sentDP dpar0.dist_clip:-10 seed0:0 msp_seed:0 ext0.train_cons:1 ext0.pred_cons:1"
    # run with FA (full annotation)
    python3 -m mspx.tools.al.main ${COMMON_ARGS} "${COMMON_ARGS2}" query_partial:0
    # or Run with PA (partial annotation)
    python3 -m mspx.tools.al.main ${COMMON_ARGS} "${COMMON_ARGS2}" 'specs_trainST:train0.group_files:_train.json train0.info:strgR:1' train_st:3 specs_trainST1:train1.group_files: selv2_arM:0

### File Structures

    ├── al.conf.json            # conf file
    ├── al.dev.json             # dev set
    ├── al.log                  # output loggings
    ├── al.record.json          # record file
    ├── al.ref.json             # reference file (for simulation running)
    ├── iter00                  # files and models for each iterration
    │    ├── data.init.json     # data available (L+U) at the start of this iter
    │    ├── data.query.json    # data(subset) for query in this iter
    │    ├── data.ann.json      # data(subset) annotated in this iter
    │    ├── data.comb.json     # combined data for training
    │    ├── t0                 # training folder for each self-train stage
    │    │   ├── _conf          # model conf file
    │    │   ├── zmodel.best.m  # best model after training
    │    │   ├── _log           # training log file
    │    │   └── _log_test      # testing log file
    │    └── tz                 # training folder for the final self-train stage
    │        ├── _conf
    │        ├── zmodel.best.m  
    │        ├── _log
    │        └── _log_test
    ├── iter01                  # more iterations
    ├── iter02
    ├── ...
