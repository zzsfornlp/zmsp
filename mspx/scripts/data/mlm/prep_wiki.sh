#!/usr/bin/env bash

# get into lang dir
if [[ -z "${WIKI_LANG}" ]];
then
echo "Please provide the lang to deal with";
exit 1;
fi

# which version
if [[ -z "${WIKI_VER}" ]];
then
WIKI_VER="latest"
exit 1;
fi

# prepare the tools
if ! [[ -d wikiextractor ]]; then
  git clone https://github.com/attardi/wikiextractor
  cd wikiextractor; git checkout v3.0.6; cd ..
fi

# download
echo "Preparing for the version of ${WIKI_LANG}-${WIKI_VER}"
wget -nc https://dumps.wikimedia.org/${WIKI_LANG}wiki/${WIKI_VER}/${WIKI_LANG}wiki-${WIKI_VER}-pages-articles.xml.bz2

# extract
EXT_OUT=${WIKI_LANG}_ext
mkdir -p ${EXT_OUT}
#PYTHONPATH=wikiextractor python3 -m wikiextractor.WikiExtractor ${WIKI_LANG}wiki-${WIKI_VER}-pages-articles.xml.bz2 -c -b 200M -o ${EXT_OUT} --no-templates --processes 8 |& tee ${EXT_OUT}/log  # "-c" seems to induce errs ...
PYTHONPATH=wikiextractor python3 -m wikiextractor.WikiExtractor ${WIKI_LANG}wiki-${WIKI_VER}-pages-articles.xml.bz2 -b 200M -o ${EXT_OUT} --processes 8 |& tee ${EXT_OUT}/log
