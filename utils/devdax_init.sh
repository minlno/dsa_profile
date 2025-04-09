#!/bin/bash

sudo echo offline > /sys/devices/system/memory/auto_online_blocks
sudo ndctl create-namespace -fe namespace0.0 -m devdax
sudo daxctl reconfigure-device -m system-ram dax0.0
