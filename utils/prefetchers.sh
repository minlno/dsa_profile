#!/bin/bash
#https://software.intel.com/en-us/articles/disclosure-of-hw-prefetcher-control-on-some-intel-processors

sudo modprobe msr

if [[ -z $(which rdmsr) ]]; then
    echo "msr-tools is not installed. Run 'sudo apt-get install msr-tools' to install it." >&2
    exit 1
fi

lsmod | grep msr >& /dev/null
if [ $? -ne 0 ]; then
    echo "Run 'sudo modprobe msr' to load the MSR module"
    exit 1
fi

if [[ ! -z $1 && $1 != "enable" && $1 != "disable" ]]; then
    echo "Invalid argument: $1" >&2
    echo ""
    echo "Usage: $(basename $0) [disable|enable]"
    exit 1
fi

cores=$(cat /proc/cpuinfo | grep processor | awk '{print $3}')
for core in $cores; do
    if [[ $1 == "disable" ]]; then
        sudo wrmsr -p${core} 0x1a4 0xf
    fi
    if [[ $1 == "enable" ]]; then
        sudo wrmsr -p${core} 0x1a4 0x0
    fi
    state=$(sudo rdmsr -p${core} 0x1a4 -f 3:0)
    if [[ $state == "f" ]]; then
		echo "core ${core}: disabled"
    else
        echo "core ${core}: enabled"
	fi
done

