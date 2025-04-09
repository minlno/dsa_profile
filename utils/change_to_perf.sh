#!/bin/bash

# 모든 CPU에 대해 반복
for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
    # governor 파일이 있는지 확인
    if [ -f "$cpu/cpufreq/scaling_governor" ]; then
        # P-state를 performance로 설정
        echo "performance" | sudo tee $cpu/cpufreq/scaling_governor
    fi
done

