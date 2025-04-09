#!/bin/bash

# 모든 CPU에 대해 반복
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    # C-state가 지원되는지 확인
    if [ -d "$cpu/cpuidle" ]; then
        # 각 C-state를 비활성화
        for state in $cpu/cpuidle/state[0-9]*; do
            echo 1 | sudo tee $state/disable
        done
    fi
done


