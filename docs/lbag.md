### For the Local Bag pre-training method

Hi, this describes our implementation for our work: "An Empirical Exploration of Local Ordering Pre-training for Structured Prediction".

Please refer to the paper for more details: [[paper]](TODO) [[bib]](TODO)

### Repo

When we were carrying out our experiments for this work, we used the repo at this commit [`here`](TODO). In later versions of this repo, there may be slight changes (for example, default hyper-parameter change or hyper-parameter name change).

### Environment

As those of the main `msp` package:

	python>=3.6
	dependencies: pytorch>=1.0.0, numpy, scipy, gensim, cython, transformers, ...

### Data

- Pre-training data: any large corpus can be utilized, we use a random subset of wikipedia. (The format is simply one sentence per line, but **need to be tokenized (separated by spaces)!!**)
- Task data: The dependency parsing data are in CoNLL-U format, which are available from the official UD website. The NER data should otherwise be in the format like those in CoNLL03.

### Running

- Step 0: Setup

Assume we are at a new DIR, and please download this repo into a DIR called `src`: `git clone https://github.com/zzsfornlp/zmsp src` and specify some ENV variables (for convenience):

	SRC_DIR: Root dir of this repo
	CUR_LANG: Lang id of the current language (for example en)
	WIKI_PRETRAIN_SIZE: Pretraining size
	UD_TRAIN_SIZE: Task training size

- Step 1: Build dictionary with pre-trained data

Before this, we should have the data prepared (for pre-training and task-training).

Assume that we have UD files at `data/UD_RUN/ud24s/${CUR_LANG}_train.${UD_TRAIN_SIZE}.conllu`, and pre-training (wiki) files at `data/UD_RUN/wikis/wiki_${CUR_LANG}.${WIKI_PRETRAIN_SIZE}.txt`. 

Assmuing now we are at DIR `data/UD_RUN/vocabs/voc_${CUR_LANG}`, we first create vocabulary for this setting with:

	PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zmlm.main.vocab_utils train:../../wikis/wiki_en.${WIKI_PRETRAIN_SIZE}.txt input_format:plain norm_digit:1 >vv.list

The `vv_*` files at this dir will be the vocabularies that will be utilized for the remaining steps.

- Step 2: Do pre-training

Assuming now we are at DIR `data/..`

Simply use the script of `${SRC_DIR}/scripts/lbag/run.py` for pre-training.

	python3 ${SRC_DIR}/scripts/lbag/run.py -l ${CUR_LANG} --rgpu 0 --run_dir run_orp_${CUR_LANG} --enc_type trans --run_mode pre --pre_mode orp --train_size ${WIKI_PRETRAIN_SIZE} --do_test 0

Note that by default, the data dirs are already pre-set as the ones in step 1, the paths can also be specified, please refer to the script for more details.

There are various modes for pre-training, the most typical ones are: orp (or lbag, our local reordering strategy), mlm (masked LM), om (orp+mlm). Please use `--pre_mode` to specify.

This may take a while (it took us three days to pretrain with 1M data on a single GPU). After this, we get the pre-trained models at `run_orp_${CUR_LANG}`.

- Step 3: Fine-tuning on specific tasks

Finally, training (fine-tuning) on specific tasks (here on Dep+Pos with UD data) with the pre-trained model. We can still use the script of `${SRC_DIR}/scripts/lbag/run.py`, simply change the `--run_mode` to `ppp1`, together with other information.

	python3 ${SRC_DIR}/scripts/lbag/run.py -l ${CUR_LANG} --rgpu 0 --cur_run 1 --run_dir run_ppp1_${CUR_LANG} --run_mode ppp1 --train_size ${UD_TRAIN_SIZE} --preload_prefix ../run_orp_${CUR_LANG}/zmodel.c200

Here, we use the `checkpoint@200` model from the pre-trained dir, other checkpoints can also be specified (only providing the unambigous model prefix will be enough).

Again, paths are by default the ones we setup from Step 1, if using other paths, things can be also specified with various `--*_dir`.
