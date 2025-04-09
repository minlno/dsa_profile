#!/bin/bash

sudo echo 1 > /sys/kernel/mm/numa/demotion_enabled
sudo echo 2 > /proc/sys/kernel/numa_balancing

echo "demotion_enabled: "
echo `cat /sys/kernel/mm/numa/demotion_enabled`
echo "numa_balancing: "
echo `cat /proc/sys/kernel/numa_balancing`
