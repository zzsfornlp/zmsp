#!/usr/bin/env bash

PTB_DIR=$1
echo "Get *.pos files from DIR ${PTB_DIR}"

#python3 -c print(",".join(["%.2d" % i for i in range(0,19)]))
cat ${PTB_DIR}/{00,01,02,03,04,05,06,07,08,09,10,11,12,13,14,15,16,17,18}.pos > train.pos
cat ${PTB_DIR}/{19,20,21}.pos > dev.pos
cat ${PTB_DIR}/{22,23,24}.pos > test.pos
