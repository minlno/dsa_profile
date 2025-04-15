#!/bin/bash

bin="./src/dsa_perf_micros"
raw_data_dir="./raw_data"
data_dir="./data"

batch_size_arr=(1 2 4 8 16 32 64 128 256 512 1024)
#xfer_size_arr=("64" "256" "1k" "4k" "64k" "512k")
xfer_size_arr=("8")
delta_rate_arr=(0 20 40 60 80 100)

run_delta_dsa() {
	local op=$1
	local batch_size=$2
	local xfer_size=$3
	local delta_rate=$4
	local raw_data_file=$5

	$bin -o $op -b $batch_size -s $xfer_size -D $delta_rate -i 100 -w 0 -zF,F > $raw_data_file

	Throughput=$(grep "GB per sec" $raw_data_file | awk -F'= ' '{print $2}' | awk '{print $1}')
}



#echo "batch_size,xfer_size,delta_rate,throughput" > $data_dir/cr_delta.csv
#echo "batch_size,xfer_size,delta_rate,throughput" > $data_dir/ap_delta.csv

for bs in "${batch_size_arr[@]}"
do
for xs in "${xfer_size_arr[@]}"
do
for dr in "${delta_rate_arr[@]}"
do
	run_delta_dsa 7 $bs $xs $dr $raw_data_dir/temp.dat
	echo "$bs,$xs,$dr,$Throughput" >> $data_dir/cr_delta.csv
	#run_delta_dsa 8 $bs $xs $dr $raw_data_dir/temp.dat
	#echo "$bs,$xs,$dr,$Throughput" >> $data_dir/ap_delta.csv
done
done
done
