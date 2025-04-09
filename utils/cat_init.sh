#!/bin/bash
source ../config.sh
echo "Cache partitioning"

sudo pqos -e "llc:1=${CAT[LC]};llc:2=${CAT[BE]};"
sudo pqos -a "llc:1=${CPU[LC]};llc:2=${CPU[BE]};llc:1=${CPU[irq]}"
sudo pqos -s

