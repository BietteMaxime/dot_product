#!/bin/sh

N=$1
shift

for i in $(seq $N)
do
    taskset -c 2 $@
done | grep 'StaticVec 02' | cut -d ' ' -f 9 | python $(dirname $0)/stat_desc.py
