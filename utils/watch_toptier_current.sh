#!/bin/bash

app=$1
watch -n 0.5 "cat /sys/fs/cgroup/lxc.payload.${app}/.lxc/memory.toptier.current"
