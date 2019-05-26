## How to run the programs:

### Requirements

	python>=3.6
	dependencies: pytorch>=0.4.1, numpy, scipy, gensim, cython

### Run Individual Programs (`tasks`)

Firstly, the main dir `zmsp` should be in `sys.path` since most of the programs depend on the tools in `msp`.

For individual tasks, the running should invoke [`tasks/cmd.py`](../tasks/cmd.py) and with the relative package name of the further actual running file as the first cmd argument. For example, to run [`tasks/zdpar/main/train.py`](../tasks/zdpar/main/train.py), then that will be:

	SRC_DIR="../zmsp/"
	PYTHONPATH=${SRC_DIR} python3 ${SRC_DIR}/tasks/cmd.py zdpar.main.train (other-args)...

