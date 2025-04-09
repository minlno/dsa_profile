#!/bin/bash

watch -n 0.5 "cat /proc/vmstat |grep -e promote -e demote"
