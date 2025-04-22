#!/bin/bash

sudo ./dsa-setup.sh -d dsa0
echo 1024 > /sys/devices/pci0000:6a/0000:6a:01.0/dsa0/wq0.0/max_batch_size
echo 1024 > /sys/devices/pci0000:6a/0000:6a:01.0/dsa2/wq2.0/max_batch_size
sudo ./dsa-setup.sh -d dsa0 -w 1 -m d -e 4
#sudo ./dsa-setup.sh -d dsa2 -w 2 -m d -e 4
