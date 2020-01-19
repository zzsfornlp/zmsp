#!/bin/bash

# dir
PREV_DIR=`pwd`
RUNNING_DIR="$( cd "$( dirname ${BASH_SOURCE[0]}  )" && pwd )"
cd $RUNNING_DIR

# get ad3 (specific version)
git clone https://github.com/andre-martins/AD3/
cd AD3
git checkout 22131c7457614dd159546500cd1a0fd8cdf2d282
cd ..
mv AD3/ad3 .
mv AD3/Eigen .
rm -rf AD3

# fix mem-leak?
sed -i '437s/if (j == k) continue//' ad3/GenericFactor.cpp

# compile
c++ -O3 -Wall -Wno-sign-compare -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` -I./ad3/ -I./Eigen -I. ./ad3/*.cpp algo.cpp parser2.cpp -o parser2`python3-config --extension-suffix`
mv parser2`python3-config --extension-suffix` ..
cd $PREV_DIR
