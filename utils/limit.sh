#!/bin/bash

size=$1
app=$2

taskset -c 15 echo $size > /sys/fs/cgroup/lxc.payload.$app/.lxc/memory.toptier.high

