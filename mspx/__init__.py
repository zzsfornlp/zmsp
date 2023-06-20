#

# The Mingled Structured Prediction v3 (zmspx where x=3)
# author: zzs
# time(v1): 2018.02 - 2020.06
# time(v2): 2020.06 - 2022.04
# time(v3): 2022.04 - now!!

# --
"""
# env:
conda create -n p22 python=3.8
conda install numpy scipy cython pybind11 pandas pip nltk
pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers==4.8.2 datasets
# (datasets==1.11.0)
pip install stanza
pip install -U scikit-learn
# apex
wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
bash ./cuda_11.1.0_455.23.05_linux.run --silent --toolkit --installpath=`pwd`/cuda
git clone https://github.com/NVIDIA/apex
cd apex; git checkout dcb02fcf805524b4df52e31d26953d852bbeb291; CUDA_HOME=`pwd`/../cuda pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./; cd ..
"""
# --

# new env
# --
"""
# env:
conda create -n s23 python=3.10
pip install torch==2.0.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -U git+https://github.com/huggingface/transformers
pip install -U git+https://github.com/huggingface/peft
pip install numpy scipy cython pybind11 pandas pip nltk stanza scikit-learn sentencepiece
pip install datasets accelerate evaluate 
# pip install bitsandbytes
# --
pip list | grep cuda
wget https://developer.download.nvidia.com/compute/cuda/11.7.1/local_installers/cuda_11.7.1_515.65.01_linux.run
bash ./cuda_11.7.1_515.65.01_linux.run --silent --toolkit --installpath=`pwd`/cuda
export LD_LIBRARY_PATH="`pwd`/cuda/lib64/:$LD_LIBRARY_PATH"
export PATH="`pwd`/cuda/bin/:$PATH"
git clone https://github.com/TimDettmers/bitsandbytes
cd bitsandbytes
CUDA_HOME=../cuda/ CUDA_VERSION=117 make cuda11x
CUDA_VERSION=117 python setup.py install
"""
# --

VERSION_MAJOR = 0
VERSION_MINOR = 3
VERSION_PATCH = 0
VERSION_STATUS = "dev"

def version(level=3):
    return (VERSION_MAJOR, VERSION_MINOR, VERSION_PATCH, VERSION_STATUS)[:level]

def msp_which_x():
    return 3

__version__ = ".".join(str(z) for z in version())

# file structures
"""
mspx:
# core
- data: Data Instances & ReaderWriter & Vocabs
- utils: utils
# peripherals
- tools: some useful tools
# others (not packages)
- cli: command line runners
- scripts: various helping scripts (for pre-processing or post-processing ...)
"""

# formats of todos
"""
-> TODO(!): immediate todos
-> todo(+N): difficult todos
-> todo(+[int]): plain ones with levels
-> note: just noting
"""
