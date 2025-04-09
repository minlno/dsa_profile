#!/bin/bash

# NUMA node 1의 CPU 목록을 얻어옵니다.
CPUS=$(cat /sys/devices/system/node/node1/cpulist)

# 범위나 쉼표로 구분된 CPU 목록을 배열로 변환합니다.
IFS=',' read -ra ADDR <<< "$CPUS"
for i in "${ADDR[@]}"; do
    # 범위(예: 2-4)로 표현된 경우 처리
    if [[ $i =~ - ]]; then
        IFS='-' read -ra RANGE <<< "$i"
        for j in $(seq ${RANGE[0]} ${RANGE[1]}); do
            echo 0 | sudo tee /sys/devices/system/cpu/cpu$j/online
        done
    else
        echo 0 | sudo tee /sys/devices/system/cpu/cpu$i/online
    fi
done

