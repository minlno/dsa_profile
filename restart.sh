#!/bin/bash

cd utils
./change_to_perf.sh
./cstate_disable.sh
./node1_cpu_disable.sh

cd ../
./max_batch_size.sh
