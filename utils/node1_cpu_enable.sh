#!/bin/bash

#for ((i = 29; i < 48; i = i + 1))
for ((i = 24; i < 48; i = i + 1))
do
	sudo echo 1 > /sys/devices/system/cpu/cpu$i/online
	echo "CPU$i online:"
	cat /sys/devices/system/cpu/cpu$i/online
done
