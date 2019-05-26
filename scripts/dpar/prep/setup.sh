#!/usr/bin/env bash

# setup the data for parsing & tagging

# initial steps: setup these original files and link into the DATA_HOME
# todo(warn): PTB: unrar+7z/mount+cp, CTB: unrar, UD: tar
# apt-get download + dpkg-deb -xv *.deb .
PTB_ROOT="."
CTB_ROOT="."
UD_ROOT="."     # https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2515/ud-treebanks-v2.1.tgz

#
RUNNING_DIR="$( cd "$( dirname ${BASH_SOURCE[0]}  )" && pwd )"
DP_HOME=`pwd`       # parsing root
TOOLS_HOME=${DP_HOME}/tools/
DATA_HOME=${DP_HOME}/data/

# stanford parser 3.3.0 + CoreNLP 3.8.0
echo "Download tools ..."
mkdir -p ${TOOLS_HOME}
cd ${TOOLS_HOME}
if [ ! -d stanford-parser-full-2013-11-12 ]; then
    wget -nc https://nlp.stanford.edu/software/stanford-parser-full-2013-11-12.zip
    unzip stanford-parser-full-2013-11-12.zip
    ln -s stanford-parser-full-2013-11-12/stanford-parser.jar .
fi
if [ ! -d stanford-corenlp-full-2017-06-09 ]; then
    wget -nc https://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip
    unzip stanford-corenlp-full-2017-06-09.zip
    ln -s stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar .
fi
#
wget -nc https://stp.lingfil.uu.se/~nivre/research/Penn2Malt.jar
wget -nc https://stp.lingfil.uu.se/~nivre/research/headrules.txt
wget -nc https://stp.lingfil.uu.se/~nivre/research/chn_headrules.txt

# prepare data
echo "Prepare data ..."
mkdir -p ${DATA_HOME}
cd ${DATA_HOME}
#ln -s ${PTB_ROOT} PTB
#ln -s ${CTB_ROOT} CTB
#ln -s ${UD_ROOT} UD

#
function new-data
{
    echo "Creating and goto dir of $1"
    NEW_DIR=$1
    mkdir -p ${NEW_DIR}; cd ${NEW_DIR}
}

# PTB_SD -> stanford converter 3.3.0
new-data ${DATA_HOME}/PTB_SD
python3 ${RUNNING_DIR}/ptb.py sd 0
# PTB_P2M -> Penn2Malt
new-data ${DATA_HOME}/PTB_P2M
python3 ${RUNNING_DIR}/ptb.py p2m 0
# CTB
new-data ${DATA_HOME}/CTB_RUN
python3 ${RUNNING_DIR}/ctb.py
# UD
new-data ${DATA_HOME}/UD_RUN
python3 ${RUNNING_DIR}/ud.py
#new-data ${DATA_HOME}/UD23_RUN
#python3 ${RUNNING_DIR}/ud23.py
# POS_SD
new-data ${DATA_HOME}/POS_SD
bash ${RUNNING_DIR}/pos.sh ../PTB_SD
# POS_P2M
new-data ${DATA_HOME}/POS_P2M
bash ${RUNNING_DIR}/pos.sh ../PTB_P2M
