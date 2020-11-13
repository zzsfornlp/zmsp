### For the Implicit Event Argument Detection

Hi, this describes our implementation for our ACL-2020 paper: "A Two-Step Approach for Implicit Event Argument Detection".

Please refer to the paper for more details: [[paper]](https://www.aclweb.org/anthology/2020.acl-main.667.pdf) [[bib]](https://www.aclweb.org/anthology/2020.acl-main.667.bib)

### Repo

When we were carrying out our experiments for this work, we used the repo at this commit [`here`](https://github.com/zzsfornlp/zmsp/commit/80a7fa85d9e7c12d50df460d8bc0029d2c6cf40b). In later versions of this repo, there may be slight changes (for example, default hyper-parameter change or hyper-parameter name change).

### Environment

As those of the main `msp` package:

	python>=3.6
	dependencies: pytorch>=1.0.0, numpy, scipy, gensim, cython, transformers, stanfordnlp=0.2.0, ...

Please refer to [`scripts/ie/iarg/conda_env_zie.yml`](../scripts/ie/iarg/conda_env_zie.yml) for an example of specific conda environment and you can specify conda environment using this.

(**note**: Please try to specify these versions of the corresponding libraries, since we find that there may be errors if using versions newer than these: `python3.6 + pytorch1.0.0 + transformers2.8.0`)

### Data Preparation

Please refer to [`scripts/ie/iarg/prepare_data.sh`](../scripts/ie/iarg/prepare_data.sh) for more details. To be noted, for easier processing, we transformed from RAMS's format to our own format.

### Data Format

Our own data format is also in json (jsonlines), where each line is a dict for one doc, it roughly contains the following fields:

    ## updated json format
    {"doc_id": ~, "dataset": ~, "source": str, "lang": str[2],
    "sents": List[Dict],
    extra fields for these: posi, score=0., extra_info={}, is_aug=False
    "fillers": [{id, offset, length, type}],
    "entity_mentions": [id, gid, offset, length, type, mtype, (head)],
    "relation_mentions": [id, gid, type, rel_arg1{aid, role}, rel_arg2{aid, role}],
    "event_mentions": [id, gid, type, trigger{offset, length}, em_arg[{aid, role}]]}

Please refer to [`tasks/zie/common/data.py`](../tasks/zie/common/data.py) or loading the data in python for more details. We think that the namings of the json fields are relatively straightforward.

We also provide a script [`scripts/ie/iarg/doc2rams.py`](../scripts/ie/iarg/doc2rams.py) to transform from our format back to RAMS's.

### Running

Please refer to [`scripts/ie/iarg/go20.sh`](../scripts/ie/iarg/go20.sh) for an example of training and testing.

For training, please make a new dir (for example, "run") and run `go20.sh` in it. (Maybe need to fix some paths.)

For testing, simply run the following after training (`RGPU` is gpu id and `SRC_DIR` is the top path of the source code); here, input/output can be specified with: `test:[input]` and `output_file:[output]`:

`CUDA_VISIBLE_DEVICES=${RGPU} PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zie.main.test _conf device:0 test:../RAMS_1.0/data/en.rams.dev.json output_file:zout.dev.json`

### Eval

Our codes directly do evaluations and print the results (assume annotations are already in the input), you can also convert the outputs back to RAMS format and use the scorer in RAMS's package for evaluations (in our experiments, the results are the same as ours).

    # convert back to RAMS format
    PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/scripts/ie/iarg/doc2rams.py zout.dev.json zout.dev.rams eng
    python3 ../RAMS_1.0/scorer/scorer.py --gold_file ../RAMS_1.0/data/dev.jsonlines --pred_file zout.dev.rams --ontology_file ../RAMS_1.0/scorer/event_role_multiplicities.txt --do_all
