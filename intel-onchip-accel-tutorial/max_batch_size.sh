#!/bin/bash

sudo ./dsa-setup.sh -d dsa0
echo 1024 > /sys/devices/pci0000:6a/0000:6a:01.0/dsa0/wq0.0/max_batch_size
sudo ./dsa-setup.sh -d dsa0 -w 1 -m s -e 4
