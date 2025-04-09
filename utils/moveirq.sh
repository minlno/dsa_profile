#!/bin/bash
source ../config.sh
echo "Set irq affinity"

start_cpu=${CPU["irq"]}
end_cpu=$((start_cpu + NCPU["irq"] - 1))

index=0
for (( cpu=start_cpu; cpu<=end_cpu; cpu++ ))
do
	sudo echo $cpu > /proc/irq/${irq_vector[$index]}/smp_affinity_list

	echo " irq ${irq_vector[$index]} : CPU `cat /proc/irq/${irq_vector[$index]}/smp_affinity_list`"
	index=$((index+1))
done
#sudo echo 16 > /proc/irq/210/smp_affinity_list
#cat /proc/irq/217/smp_affinity_list
