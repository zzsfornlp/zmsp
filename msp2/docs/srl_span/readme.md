### For Comparing Span Extraction Methods for SRL

Hi, this describes our implementation for our paper: "Comparing Span Extraction Methods for Semantic Role Labeling".

Please refer to the paper for more details: [TODO]()

### Repo

When we were carrying out our experiments for this work, we used the repo at this commit [TODO](). In later versions of this repo, there may be slight changes (for example, default hyper-parameter change or hyper-parameter name change).

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

Please use [`prep.sh`](./prep.sh) to prepare data, refer to it for more details. Notice that there should be some preparations before running this script, see the `step 0: prepare` part of this script. Running with:

	PYTHONPATH=/your/path/to/src PTB3=/your/path/to/TREEBANK_3 ONTO5=/your/path/to/ontonotes-release-5.0 bash prep.sh

After this, you will get the data files in `data/pb/conll*/*`.

### Extra Preparation

Please also prepare pre-trained static embeddings for running:

	mkdir voc; cd voc;
	wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
	unzip crawl-300d-2M-subword.zip
	EMBED_FILE=crawl-300d-2M-subword.vec
	python3 -m msp2.tasks.zsfp.main.build train:../data/pb/conll05/train.conll.ud.json dev:../data/pb/conll05/dev.conll.ud.json,../data/pb/conll05/test.wsj.conll.ud.json,../data/pb/conll05/test.brown.conll.ud.json pretrain_hits_outf:hits_pb05.vec pretrain_wv_file:${EMBED_FILE} |& tee _log.voc_pb05
	python3 -m msp2.tasks.zsfp.main.build train:../data/pb/conll12b/train.conll.ud.json dev:../data/pb/conll12b/dev.conll.ud.json,../data/pb/conll12b/test.conll.ud.json pretrain_hits_outf:hits_pb12.vec pretrain_wv_file:${EMBED_FILE} |& tee _log.voc_pb12

### Running

Please use [`go.py`](./go.py) for running (training and testing):

	# create a new dir
	mkdir run; cd run;
	# copy go.py
	cp /your/path/to/src//msp2/docs/srl_span/go.py .	
	# training
	python3 go.py rgpu:0 dataset:$DATASET arg_mode:$ARGMODE use_bert_input:1
	# test-given-predicates
	python3 go.py rgpu:0 dataset:$DATASET do_train:0 do_test_all:1 log_prefix:_logG out_prefix:_zoutG "test_extras:evt_conf.pred_use_posi:1"

Here, `DATASET` can be one of `{pb05, pb12, pb12c, pb12cl3}` for experiments with `{conll05, conll12, conll12-cross-genre, conll12-cross-lingual}`. And `ARGMODE` can be one of `{seq0, seq, span, head, anchor}` for experiments with `BIO(w/o CRF), BIO(w/ CRF), Span, HeadSyntax, HeadAuto`.
