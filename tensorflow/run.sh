#!/bin/bash

# Enable all 8 cores
CORES=(
	# 0
	# 1
	# 2
	# 3
	4
	5
	6
	7 
)
for i in ${CORES[@]};
do 
	chcpu -e $i
done

python3 /usr/src/app/bisenetv2_tensorflow_depthconv.py