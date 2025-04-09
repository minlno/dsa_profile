#!/bin/bash

sudo mount -t cgroup -o memory memory /cgroup/memory

sudo mkdir /cgroup/memory/app1
sudo mkdir /cgroup/memory/app2
sudo mkdir /cgroup/memory/app3
sudo mkdir /cgroup/memory/app4
