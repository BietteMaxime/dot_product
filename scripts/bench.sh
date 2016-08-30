#!/bin/sh

N=$1
shift

for i in $(seq $N)
do
    taskset -c 2 $@
done | python $(dirname $0)/stat_desc.py
